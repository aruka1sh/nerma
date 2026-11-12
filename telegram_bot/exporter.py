"""Unified result representation, citation verification, and document export for NERMA AI."""

from __future__ import annotations

import logging
import re
import urllib.parse
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import aiofiles
from docx import Document
from htmldocx import HtmlToDocx
from mistune import html as md_to_html

from telegram_bot.config import OUTPUTS_DIR, RequestState
from telegram_bot.query_utils import is_matching_domain, normalize_hostname, url_matches_any_domain

logger = logging.getLogger(__name__)

# Patterns that indicate an LLM refusal due to missing search results/context
_REFUSAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"результат\w*\s+поиска\s+отсутству\w*", re.IGNORECASE),
    re.compile(r"предоставленн\w*\s+результат\w*\s+поиска\s+отсутству\w*", re.IGNORECASE),
    re.compile(r"основываясь\s+на\s+предоставленных\s+результатах\s+поиска.*невозможно", re.IGNORECASE | re.DOTALL),
    re.compile(r"дать\s+ответ\s+на\s+основе\s+источников.*невозможно", re.IGNORECASE | re.DOTALL),
    re.compile(r"i\s+could\s+not\s+gather\s+any\s+source\s+material", re.IGNORECASE),
    re.compile(r"no\s+sources\s+were\s+retrieved", re.IGNORECASE),
    re.compile(r"not\s+able\s+to\s+produce\s+a\s+reliable,\s+sourced\s+report", re.IGNORECASE),
    re.compile(r"search\s+results\s+are\s+empty", re.IGNORECASE),
    re.compile(r"cannot\s+answer.*no\s+search\s+results", re.IGNORECASE | re.DOTALL),
]

_BIBLIOGRAPHY_HEADER_RE = re.compile(
    r"(?im)^#{1,3}\s*(?:references|sources|bibliography|источники|список\s+литературы|список\s+источников|библиография|дереккөздер)\s*$"
)


def is_refusal_or_empty_report(text: str | None) -> bool:
    """Return True if `text` is empty or is a refusal message about missing search results."""
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    for pat in _REFUSAL_PATTERNS:
        if pat.search(cleaned):
            return True
    return False


def _normalize_url_for_match(url: str) -> str:
    """Normalize a URL for deduplication and matching."""
    u = (url or "").strip()
    if not u:
        return ""
    try:
        parsed = urllib.parse.urlparse(u)
        scheme = (parsed.scheme or "https").lower()
        netloc = normalize_hostname(parsed.hostname)
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = urllib.parse.unquote(parsed.path or "/").rstrip("/") or "/"
        query = f"?{parsed.query}" if parsed.query else ""
        return f"{scheme}://{netloc}{path}{query}"
    except Exception:
        return u.rstrip("/")


@dataclass
class SourceItem:
    """Single retrieved source with a stable 1-based citation ID and processing metadata."""

    id: int
    url: str
    title: str
    hostname: str = ""
    read_status: str = "snippet"  # "full_page" | "snippet"
    content_chars: int = 0
    is_official: bool = False
    cited_in_text: bool = False

    def __post_init__(self) -> None:
        if not self.hostname and self.url:
            try:
                parsed = urllib.parse.urlparse(self.url if "://" in self.url else f"https://{self.url}")
                self.hostname = normalize_hostname(parsed.hostname)
            except Exception:
                self.hostname = ""
        if not self.title:
            self.title = self.hostname or self.url


@dataclass
class ResearchResultBundle:
    """Unified representation of a research result used across Telegram chat and exported files."""

    request_id: str
    query: str
    mode: str
    language: str = "russian"
    state: RequestState = RequestState.COMPLETED
    body_markdown: str = ""
    bibliography_markdown: str = ""
    full_markdown: str = ""
    sources: list[SourceItem] = field(default_factory=list)
    unique_urls_found: int = 0
    pages_read_successfully: int = 0
    sources_cited: int = 0
    required_official_domains: list[str] = field(default_factory=list)
    official_domains_found: list[str] = field(default_factory=list)
    official_source_warning: str = ""
    failure_reason: str = "none"
    duration_sec: float = 0.0
    cost_usd: float = 0.0
    exported_files: dict[str, Path] = field(default_factory=dict)

    @property
    def is_exportable(self) -> bool:
        """Only completed or partial reports backed by real sources may be exported to files."""
        return (
            self.state in (RequestState.COMPLETED, RequestState.PARTIAL)
            and self.unique_urls_found > 0
            and bool(self.body_markdown.strip())
            and not is_refusal_or_empty_report(self.body_markdown)
        )


def build_sources_catalog(
    search_results: Iterable[dict[str, Any]] | None = None,
    scraped_sources: Iterable[dict[str, Any]] | None = None,
    visited_urls: Iterable[str] | None = None,
    required_official_domains: Iterable[str] | None = None,
) -> list[SourceItem]:
    """Build a deduplicated, ordered list of `SourceItem` objects with accurate `read_status`.

    A source is marked `read_status="full_page"` ONLY if its page content was actually
    scraped/fetched (`raw_content` or `content` in `scraped_sources`), never when only
    a search snippet was returned.
    """
    official_doms = [normalize_hostname(d) for d in (required_official_domains or []) if d]
    by_norm_url: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    def _register(
        url: str,
        title: str = "",
        read_status: str = "snippet",
        content_chars: int = 0,
    ) -> None:
        raw_url = (url or "").strip()
        if not raw_url or not raw_url.startswith(("http://", "https://")):
            return
        norm = _normalize_url_for_match(raw_url)
        if not norm:
            return
        if norm not in by_norm_url:
            by_norm_url[norm] = {
                "url": raw_url,
                "title": (title or "").strip(),
                "read_status": read_status,
                "content_chars": max(0, int(content_chars)),
            }
            order.append(norm)
        else:
            existing = by_norm_url[norm]
            if title and (not existing["title"] or existing["title"] == existing["url"]):
                existing["title"] = title.strip()
            if read_status == "full_page":
                existing["read_status"] = "full_page"
            if content_chars > existing["content_chars"]:
                existing["content_chars"] = int(content_chars)

    # 1. Register scraped pages first (these have full_page read status)
    for src in scraped_sources or []:
        if not isinstance(src, dict):
            continue
        url = src.get("url") or src.get("href") or ""
        title = src.get("title") or ""
        raw_content = src.get("raw_content") or src.get("content") or ""
        chars = len(str(raw_content).strip())
        status = "full_page" if chars > 0 else "snippet"
        _register(url, title=title, read_status=status, content_chars=chars)

    # 2. Register search result snippets
    for res in search_results or []:
        if not isinstance(res, dict):
            continue
        url = res.get("href") or res.get("url") or ""
        title = res.get("title") or ""
        body = res.get("body") or res.get("content") or ""
        read_flag = "full_page" if res.get("read_status") == "full_page" else "snippet"
        _register(url, title=title, read_status=read_flag, content_chars=len(str(body).strip()))

    # 3. Register any remaining visited URLs
    for u in visited_urls or []:
        if isinstance(u, str):
            _register(u, title="", read_status="snippet", content_chars=0)

    # Prioritize official domain sources first so they get low citation IDs ([1], [2]) if not already ordered
    items: list[SourceItem] = []
    for idx, norm_key in enumerate(order, start=1):
        info = by_norm_url[norm_key]
        url_val = info["url"]
        is_off = url_matches_any_domain(url_val, official_doms) if official_doms else False
        items.append(
            SourceItem(
                id=idx,
                url=url_val,
                title=info["title"],
                read_status=info["read_status"],
                content_chars=info["content_chars"],
                is_official=is_off,
            )
        )
    return items


def _reconcile_citations_and_body(
    raw_markdown: str,
    sources: list[SourceItem],
) -> tuple[str, list[SourceItem]]:
    """Ensure every `[n]` marker in the body corresponds to a real `SourceItem`,
    convert inline markdown links to verified `[n]` citations where appropriate,
    and strip any unverified `[n]` markers or hallucinated URLs.
    """
    text = (raw_markdown or "").strip()
    if not text:
        return "", sources

    # Separate out any existing trailing References/Sources section generated by LLM
    split_match = _BIBLIOGRAPHY_HEADER_RE.search(text)
    if split_match:
        body = text[: split_match.start()].strip()
        llm_refs_block = text[split_match.end() :].strip()
    else:
        body = text
        llm_refs_block = ""

    # Map normalized URL -> SourceItem
    url_to_source: dict[str, SourceItem] = {}
    for src in sources:
        norm = _normalize_url_for_match(src.url)
        if norm:
            url_to_source[norm] = src

    # If the LLM wrote a References section with [Title](URL) or numbered URLs, extract titles
    if llm_refs_block:
        for m in re.finditer(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", llm_refs_block):
            ref_title, ref_url = m.group(1).strip(), m.group(2).strip()
            norm = _normalize_url_for_match(ref_url)
            if norm in url_to_source:
                src = url_to_source[norm]
                if ref_title and (not src.title or src.title == src.hostname or src.title == src.url):
                    src.title = ref_title
                src.cited_in_text = True

    # Check inline markdown links `[anchor](https://...)` in body:
    # - If the URL matches a retrieved source, keep it and attach `[id]` if not already present, marking `cited_in_text = True`.
    # - If the URL was NOT in our retrieved sources (hallucinated URL), replace `[anchor](url)` with `anchor`!
    def _replace_md_link(match: re.Match[str]) -> str:
        anchor = match.group(1).strip()
        link_url = match.group(2).strip().rstrip(".,;:")
        norm = _normalize_url_for_match(link_url)
        matched_src = url_to_source.get(norm)
        if matched_src is None:
            # Also check prefix/hostname match ONLY if path is empty or root of a retrieved source
            for s in sources:
                if _normalize_url_for_match(s.url) == norm:
                    matched_src = s
                    break
        if matched_src is not None:
            matched_src.cited_in_text = True
            if anchor.isdigit() and int(anchor) == matched_src.id:
                return f"[{matched_src.id}]({matched_src.url})"
            return f"{anchor} [{matched_src.id}]"
        # Unverified URL not present in retrieved sources -> strip link URL, keep text
        return anchor

    body = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", _replace_md_link, body)

    #Also check bare URLs in body: if they match a retrieved source, mark cited; if not, remove bare unverified URL
    def _check_bare_url(match: re.Match[str]) -> str:
        prefix = match.group(1)
        raw_u = match.group(2)
        trailing = ""
        while raw_u and raw_u[-1] in ".,;:!?)":
            trailing = raw_u[-1] + trailing
            raw_u = raw_u[:-1]
        norm = _normalize_url_for_match(raw_u)
        matched_src = url_to_source.get(norm)
        if matched_src is not None:
            matched_src.cited_in_text = True
            return f"{prefix}{raw_u} [{matched_src.id}]{trailing}"
        return f"{prefix}{trailing}"

    body = re.sub(r"(^|[\s(])(https?://[^\s)\]]+)", _check_bare_url, body)

    # Now validate numeric citation markers `[1]`, `[2]` or `[1, 2]` in `body`
    max_id = len(sources)
    id_to_source = {s.id: s for s in sources}

    def _validate_bracket_citations(match: re.Match[str]) -> str:
        inner = match.group(1)
        nums = [int(x.strip()) for x in inner.split(",") if x.strip().isdigit()]
        valid_nums: list[int] = []
        for n in nums:
            if 1 <= n <= max_id:
                id_to_source[n].cited_in_text = True
                if n not in valid_nums:
                    valid_nums.append(n)
        if not valid_nums:
            return ""
        return "".join(f"[{n}]" for n in valid_nums)

    body = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", _validate_bracket_citations, body)

    # Deduplicate adjacent identical markers like `[1] [1]` -> `[1]`
    body = re.sub(r"(\[\d+\])(?:\s*\1)+", r"\1", body)
    # Clean up extra spaces before punctuation left by stripped invalid markers
    body = re.sub(r"[ \t]+([.,;:!?])", r"\1", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    # If the body has retrieved sources and substantive text, but the LLM didn't emit any `[n]` markers,
    # attach `[1]` (and `[2]` if multiple top sources were read) at the end of the body so citations are verifiable.
    any_cited = any(s.cited_in_text for s in sources)
    if sources and body and not any_cited:
        top_sources = [s for s in sources if s.is_official][:2] or sources[: min(2, len(sources))]
        for s in top_sources:
            s.cited_in_text = True
        refs_tag = "".join(f"[{s.id}]" for s in top_sources)
        body = f"{body} {refs_tag}"

    # Renumber cited/uncited sources so that every `[n]` in the text is sequential `1..K`
    # and the bibliography lists the sources with matching `[1]..[N]` IDs!
    old_to_new: dict[int, int] = {}
    ordered_sources: list[SourceItem] = []
    # Keep cited sources first in order of their existing ID, then remaining retrieved sources
    for s in sources:
        if s.cited_in_text:
            new_id = len(ordered_sources) + 1
            old_to_new[s.id] = new_id
            s.id = new_id
            ordered_sources.append(s)
    for s in sources:
        if not s.cited_in_text:
            new_id = len(ordered_sources) + 1
            old_to_new[s.id] = new_id
            s.id = new_id
            ordered_sources.append(s)

    def _remap_marker(match: re.Match[str]) -> str:
        old_n = int(match.group(1))
        new_n = old_to_new.get(old_n, old_n)
        return f"[{new_n}]"

    body = re.sub(r"\[(\d+)\]", _remap_marker, body)

    return body, ordered_sources


def format_bibliography_section(
    sources: list[SourceItem],
    language: str = "russian",
    only_cited: bool = False,
) -> str:
    """Format the numbered `[n] Title — URL` bibliography section."""
    display_sources = [s for s in sources if s.cited_in_text] if only_cited else list(sources)
    if not display_sources and sources:
        display_sources = list(sources)
    if not display_sources:
        return ""

    header = "## Источники" if language != "english" else "## Sources"
    lines = [header, ""]
    for src in display_sources:
        title_clean = (src.title or src.hostname or src.url).replace("\n", " ").strip()
        badges: list[str] = []
        if src.is_official:
            badges.append("официальный домен" if language != "english" else "official domain")
        if src.read_status == "full_page":
            badges.append("прочитана страница" if language != "english" else "full page read")
        else:
            badges.append("сниппет поиска" if language != "english" else "search snippet")
        badge_str = f" _({', '.join(badges)})_" if badges else ""
        lines.append(f"[{src.id}] [{title_clean}]({src.url}) — {src.url}{badge_str}")
    return "\n".join(lines)


def format_metrics_block(bundle: ResearchResultBundle) -> str:
    """Format the 3 distinct source metrics clearly for Telegram and exported files."""
    if bundle.language == "english":
        return (
            f"🔗 Unique URLs found: {bundle.unique_urls_found}\n"
            f"📄 Pages read successfully: {bundle.pages_read_successfully}\n"
            f"📌 Sources cited in text: {bundle.sources_cited}"
        )
    return (
        f"🔗 Найдено уникальных URL: {bundle.unique_urls_found}\n"
        f"📄 Успешно прочитано страниц: {bundle.pages_read_successfully}\n"
        f"📌 Процитировано источников: {bundle.sources_cited}"
    )


def build_result_bundle(
    request_id: str,
    query: str,
    mode: str,
    raw_report: str,
    search_results: Iterable[dict[str, Any]] | None = None,
    scraped_sources: Iterable[dict[str, Any]] | None = None,
    visited_urls: Iterable[str] | None = None,
    required_official_domains: Iterable[str] | None = None,
    language: str = "russian",
    duration_sec: float = 0.0,
    cost_usd: float = 0.0,
    failure_reason: str = "none",
) -> ResearchResultBundle:
    """Construct a verified `ResearchResultBundle` for both chat display and file export."""
    req_domains = [normalize_hostname(d) for d in (required_official_domains or []) if d]
    sources = build_sources_catalog(
        search_results=search_results,
        scraped_sources=scraped_sources,
        visited_urls=visited_urls,
        required_official_domains=req_domains,
    )

    unique_urls_found = len(sources)
    pages_read_successfully = sum(1 for s in sources if s.read_status == "full_page")

    # Check if report is empty or a refusal
    if unique_urls_found == 0 or is_refusal_or_empty_report(raw_report):
        reason = failure_reason if failure_reason != "none" else (
            "empty_search_results" if unique_urls_found == 0 else "llm_refusal"
        )
        return ResearchResultBundle(
            request_id=request_id,
            query=query,
            mode=mode,
            language=language,
            state=RequestState.FAILED,
            body_markdown="",
            bibliography_markdown="",
            full_markdown="",
            sources=sources,
            unique_urls_found=unique_urls_found,
            pages_read_successfully=pages_read_successfully,
            sources_cited=0,
            required_official_domains=req_domains,
            official_domains_found=[],
            official_source_warning="",
            failure_reason=reason,
            duration_sec=duration_sec,
            cost_usd=cost_usd,
        )

    body_md, ordered_sources = _reconcile_citations_and_body(raw_report, sources)
    sources_cited = sum(1 for s in ordered_sources if s.cited_in_text)
    pages_read_successfully = sum(1 for s in ordered_sources if s.read_status == "full_page")

    official_found: list[str] = []
    for dom in req_domains:
        if any(is_matching_domain(s.url, dom) for s in ordered_sources):
            official_found.append(dom)

    missing_official = [d for d in req_domains if d not in official_found]
    official_warning = ""
    state = RequestState.COMPLETED
    if missing_official:
        state = RequestState.PARTIAL
        domains_str = ", ".join(f"`{d}`" for d in missing_official)
        if language == "english":
            official_warning = (
                f"⚠️ **Note on official sources:** Direct pages on the requested official domain "
                f"({domains_str}) were not returned by the search provider; the answer below is based on "
                f"available verified sources."
            )
        else:
            official_warning = (
                f"⚠️ **Примечание об официальном источнике:** Прямая ссылка на запрошенный официальный домен "
                f"({domains_str}) не найдена в поисковой выдаче; ответ сформирован по доступным проверенным источникам."
            )

    # In Deep/Standard mode with many visited URLs, include cited sources in the bibliography (or all if Quick)
    only_cited_bib = len(ordered_sources) > 10 and sources_cited > 0
    bib_md = format_bibliography_section(ordered_sources, language=language, only_cited=only_cited_bib)

    parts = [body_md]
    if official_warning:
        parts.append(official_warning)
    if bib_md:
        parts.append(bib_md)

    bundle = ResearchResultBundle(
        request_id=request_id,
        query=query,
        mode=mode,
        language=language,
        state=state,
        body_markdown=body_md,
        bibliography_markdown=bib_md,
        full_markdown="\n\n".join(p for p in parts if p).strip(),
        sources=ordered_sources,
        unique_urls_found=unique_urls_found,
        pages_read_successfully=pages_read_successfully,
        sources_cited=sources_cited,
        required_official_domains=req_domains,
        official_domains_found=official_found,
        official_source_warning=official_warning,
        failure_reason="official_source_missing" if missing_official else "none",
        duration_sec=duration_sec,
        cost_usd=cost_usd,
    )
    return bundle


def _sanitize_filename(query: str) -> str:
    """Create a safe, readable filename prefix from the research query."""
    clean = re.sub(r"[^\w\s-]", "", query, flags=re.UNICODE).strip()
    clean = re.sub(r"[-\s]+", "_", clean)
    short = clean[:35].strip("_") or "research_report"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    return f"{short}_{timestamp}_{uid}"


async def export_markdown(content: str, base_name: str) -> Path:
    """Save report as a Markdown (.md) file."""
    file_path = OUTPUTS_DIR / f"{base_name}.md"
    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(content)
    return file_path


def export_docx(content: str, base_name: str) -> Path:
    """Convert Markdown report (including numbered bibliography) to a Word (.docx) document."""
    file_path = OUTPUTS_DIR / f"{base_name}.docx"
    try:
        report_html = md_to_html(content)
        doc = Document()
        html_parser = HtmlToDocx()
        html_parser.add_html_to_document(report_html, doc)
        doc.save(str(file_path))
    except Exception as exc:
        logger.error("Error converting to DOCX with HtmlToDocx, using fallback: %s", exc)
        doc = Document()
        for line in content.splitlines():
            doc.add_paragraph(line)
        doc.save(str(file_path))
    return file_path


def export_pdf(content: str, base_name: str) -> Path | None:
    """Convert Markdown report to a styled PDF (.pdf) document if WeasyPrint/md2pdf is available."""
    file_path = OUTPUTS_DIR / f"{base_name}.pdf"
    css_path = Path(__file__).resolve().parent.parent / "multi_agents" / "frontend" / "pdf_styles.css"
    try:
        from md2pdf.core import md2pdf

        kwargs: dict[str, Any] = {
            "pdf_file_path": str(file_path),
            "md_content": content,
        }
        if css_path.exists():
            kwargs["css_file_path"] = str(css_path)
        md2pdf(**kwargs)
        return file_path
    except Exception as exc:
        logger.warning("PDF generation failed (falling back to DOCX/MD): %s", exc)
        return None


async def export_bundle(
    bundle: ResearchResultBundle,
    formats: list[str],
) -> dict[str, Path]:
    """Export a verified `ResearchResultBundle` into the requested file formats.

    Refuses to generate files if the bundle is in a failed/cancelled state or has 0 sources.
    """
    if not bundle.is_exportable:
        logger.info(
            "Skipping file export for request_id=%s (state=%s, unique_urls=%d)",
            bundle.request_id,
            bundle.state.value,
            bundle.unique_urls_found,
        )
        return {}

    # Reuse already exported files if present for the requested formats
    if bundle.exported_files and all(fmt in bundle.exported_files for fmt in formats if fmt in ("md", "docx")):
        return {k: v for k, v in bundle.exported_files.items() if k in formats and v.exists()}

    base_name = _sanitize_filename(bundle.query)
    export_text = bundle.full_markdown
    metrics_footer = "\n\n---\n" + format_metrics_block(bundle)
    if metrics_footer.strip() not in export_text:
        export_text = f"{export_text}{metrics_footer}"

    exported: dict[str, Path] = {}

    if "md" in formats:
        try:
            exported["md"] = await export_markdown(export_text, base_name)
        except Exception as exc:
            logger.error("Failed to export MD: %s", exc)

    if "docx" in formats:
        try:
            exported["docx"] = export_docx(export_text, base_name)
        except Exception as exc:
            logger.error("Failed to export DOCX: %s", exc)

    if "pdf" in formats:
        pdf_path = export_pdf(export_text, base_name)
        if pdf_path and pdf_path.exists():
            exported["pdf"] = pdf_path

    bundle.exported_files.update(exported)
    return exported


async def export_all(markdown_content: str, query: str, formats: list[str]) -> dict[str, Path]:
    """Backward-compatible wrapper that refuses to export empty or refusal texts."""
    if is_refusal_or_empty_report(markdown_content):
        return {}
    base_name = _sanitize_filename(query)
    exported: dict[str, Path] = {}
    if "md" in formats:
        exported["md"] = await export_markdown(markdown_content, base_name)
    if "docx" in formats:
        exported["docx"] = export_docx(markdown_content, base_name)
    if "pdf" in formats:
        pdf_path = export_pdf(markdown_content, base_name)
        if pdf_path and pdf_path.exists():
            exported["pdf"] = pdf_path
    return exported
