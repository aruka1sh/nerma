from __future__ import annotations

import logging
import urllib.parse
from collections import deque
from typing import Any

import requests

from telegram_bot.query_utils import (
    detect_query_region,
    extract_search_intent,
    get_curated_domain_results,
    is_matching_domain,
    normalize_hostname,
    url_matches_any_domain,
)

logger = logging.getLogger(__name__)


class Duckduckgo:
    """
    DuckDuckGo API Retriever with query sanitization, official domain targeting,
    strict hostname verification via urlparse, and bounded 2-step fallback recovery.
    """

    recent_search_events: deque[dict[str, Any]] = deque(maxlen=50)

    def __init__(self, query: str, query_domains: list[str] | None = None):
        from ddgs import DDGS

        self.ddgs_cls = DDGS
        try:
            self.ddgs = DDGS(timeout=10)
        except TypeError:
            self.ddgs = DDGS()
        self.query = query
        self.query_domains = [
            normalize_hostname(d) for d in (query_domains or []) if d and normalize_hostname(d)
        ] or None
        self.last_search_meta: dict[str, Any] = {}

    def _run_ddgs_text(
        self,
        search_query: str,
        region: str,
        max_results: int,
        backend: str | None = None,
        errors_out: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a single bounded DDGS text search call."""
        if not search_query.strip():
            return []
        kwargs: dict[str, Any] = {
            "region": region,
            "max_results": max_results,
        }
        if backend:
            kwargs["backend"] = backend
        try:
            raw = self.ddgs.text(search_query, **kwargs)
            return list(raw) if raw else []
        except TypeError:
            # Older/newer ddgs versions may not accept `backend`
            kwargs.pop("backend", None)
            try:
                raw = self.ddgs.text(search_query, **kwargs)
                return list(raw) if raw else []
            except Exception as exc:
                if errors_out is not None:
                    errors_out.append(f"ddgs:{type(exc).__name__}: {exc}")
                logger.warning("DDGS search error for '%s': %s", search_query, exc)
                return []
        except Exception as exc:
            if errors_out is not None:
                errors_out.append(f"ddgs:{type(exc).__name__}: {exc}")
            logger.warning("DDGS search error for '%s': %s", search_query, exc)
            return []

    def _wikipedia_fallback(
        self,
        search_query: str,
        region: str,
        max_results: int = 3,
        errors_out: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Bounded fallback to MediaWiki Search API (ru.wikipedia.org / en.wikipedia.org)."""
        if not search_query.strip():
            return []
        wiki_lang = "ru" if region == "ru-ru" else "en"
        api_url = f"https://{wiki_lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": search_query,
            "srlimit": max(1, min(max_results, 5)),
            "utf8": 1,
            "format": "json",
        }
        headers = {
            "User-Agent": "NermaAIBot/1.0 (Academic Research Assistant; +https://t.me/NermaAI_bot)"
        }
        try:
            resp = requests.get(api_url, params=params, headers=headers, timeout=6)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("query", {}).get("search", [])
            results: list[dict[str, str]] = []
            for item in items:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                # Strip HTML tags from snippet
                import re

                clean_snippet = re.sub(r"<[^>]+>", "", snippet).strip()
                if not title:
                    continue
                encoded_title = urllib.parse.quote(title.replace(" ", "_"))
                href = f"https://{wiki_lang}.wikipedia.org/wiki/{encoded_title}"
                results.append(
                    {
                        "title": f"{title} — Wikipedia",
                        "href": href,
                        "body": clean_snippet or title,
                    }
                )
            return results
        except Exception as exc:
            if errors_out is not None:
                errors_out.append(f"wikipedia_fallback:{type(exc).__name__}: {exc}")
            return []

    @staticmethod
    def _normalize_items(raw_items: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Normalize raw search items and deduplicate by URL."""
        normalized: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for item in raw_items or []:
            if not isinstance(item, dict):
                continue
            href = (item.get("href") or item.get("url") or "").strip()
            body = (item.get("body") or item.get("content") or item.get("snippet") or "").strip()
            title = (item.get("title") or "").strip()
            if not href or href in seen_urls:
                continue
            seen_urls.add(href)
            entry: dict[str, str] = {"href": href, "body": body}
            if title:
                entry["title"] = title
            normalized.append(entry)
        return normalized

    def search(self, max_results: int = 5) -> list[dict[str, str]]:
        """
        Perform bounded search with query sanitization, official domain targeting,
        and 2-step recovery on empty results.
        """
        intent = extract_search_intent(self.query)
        clean_query = intent.clean_query or self.query.strip()
        simplified_query = intent.simplified_query or clean_query
        region = intent.region or detect_query_region(clean_query)

        # Mandatory domains come from explicit `query_domains` or `site:domain` in the query
        mandatory_domains: list[str] = []
        for d in (self.query_domains or []) + intent.explicit_site_domains:
            norm_d = normalize_hostname(d)
            if norm_d.startswith("www."):
                norm_d = norm_d[4:]
            if norm_d and norm_d not in mandatory_domains:
                mandatory_domains.append(norm_d)

        # Target official domains include mandatory domains + natural-language requested official domains
        target_official_domains: list[str] = list(mandatory_domains)
        for d in intent.official_domains:
            norm_d = normalize_hostname(d)
            if norm_d.startswith("www."):
                norm_d = norm_d[4:]
            if norm_d and norm_d not in target_official_domains:
                target_official_domains.append(norm_d)

        errors: list[str] = []
        providers_used: list[str] = ["ddgs"]
        retries = 0
        fallback_used = False
        raw_candidates: list[dict[str, Any]] = []

        # Step 1: Primary search
        if mandatory_domains:
            for dom in mandatory_domains[:2]:
                site_q = f"site:{dom} {clean_query}".strip()
                raw_candidates.extend(
                    self._run_ddgs_text(site_q, region=region, max_results=max_results, errors_out=errors)
                )
        else:
            # If user requested an official domain in natural language, do a targeted search first
            if target_official_domains:
                for dom in target_official_domains[:2]:
                    site_q = f"site:{dom} {simplified_query}".strip()
                    raw_candidates.extend(
                        self._run_ddgs_text(site_q, region="us-en", max_results=max(2, max_results // 2), errors_out=errors)
                    )
            raw_candidates.extend(
                self._run_ddgs_text(clean_query, region=region, max_results=max_results, errors_out=errors)
            )

        normalized_before_filter = self._normalize_items(raw_candidates)
        before_filter_count = len(normalized_before_filter)

        # Filter by mandatory domains if specified (never drop mandatory domain constraint!)
        if mandatory_domains:
            filtered = [
                r for r in normalized_before_filter if url_matches_any_domain(r["href"], mandatory_domains)
            ]
        else:
            filtered = list(normalized_before_filter)

        # Step 2 (Bounded Recovery #1): Simplified query retry if empty
        if not filtered:
            retries += 1
            retry_candidates: list[dict[str, Any]] = []
            if mandatory_domains:
                for dom in mandatory_domains[:2]:
                    retry_q = f"site:{dom} {simplified_query}".strip()
                    retry_candidates.extend(
                        self._run_ddgs_text(
                            retry_q,
                            region="us-en",
                            max_results=max_results,
                            backend="html",
                            errors_out=errors,
                        )
                    )
            else:
                retry_candidates.extend(
                    self._run_ddgs_text(
                        simplified_query,
                        region=region,
                        max_results=max_results,
                        backend="html",
                        errors_out=errors,
                    )
                )
            norm_retry = self._normalize_items(retry_candidates)
            before_filter_count += len(norm_retry)
            if mandatory_domains:
                filtered = [r for r in norm_retry if url_matches_any_domain(r["href"], mandatory_domains)]
            else:
                filtered = norm_retry

        # Check if requested official domain is present in `filtered`
        official_found = [
            dom
            for dom in target_official_domains
            if any(is_matching_domain(r["href"], dom) for r in filtered)
        ]

        # Step 3 (Bounded Recovery #2): Fallback to curated official domain sources / Wikipedia API
        if target_official_domains and not official_found:
            curated_official = get_curated_domain_results(
                target_official_domains, clean_query, max_results=max_results
            )
            if mandatory_domains:
                curated_official = [
                    r for r in curated_official if url_matches_any_domain(r["href"], mandatory_domains)
                ]
            if curated_official:
                fallback_used = True
                if "curated_official_fallback" not in providers_used:
                    providers_used.append("curated_official_fallback")
                # Prepend official results so they are prioritized
                combined = self._normalize_items(curated_official + filtered)
                filtered = combined
                before_filter_count = max(before_filter_count, len(filtered))
                official_found = [
                    dom
                    for dom in target_official_domains
                    if any(is_matching_domain(r["href"], dom) for r in filtered)
                ]

        if not filtered and not mandatory_domains:
            fallback_used = True
            wiki_items = self._wikipedia_fallback(
                simplified_query, region=region, max_results=max_results, errors_out=errors
            )
            if wiki_items:
                providers_used.append("wikipedia_api")
                filtered = self._normalize_items(wiki_items)
                before_filter_count = max(before_filter_count, len(filtered))
            else:
                curated_any = get_curated_domain_results([], clean_query, max_results=max_results)
                if curated_any:
                    providers_used.append("curated_official_fallback")
                    filtered = self._normalize_items(curated_any)
                    before_filter_count = max(before_filter_count, len(filtered))

        # Prioritize matching official domains first when present
        if target_official_domains and filtered:
            official_items = [
                r for r in filtered if url_matches_any_domain(r["href"], target_official_domains)
            ]
            other_items = [
                r for r in filtered if not url_matches_any_domain(r["href"], target_official_domains)
            ]
            filtered = (official_items + other_items)[:max_results]
        else:
            filtered = filtered[:max_results]

        after_filter_count = len(filtered)
        if after_filter_count > 0:
            failure_reason = "none"
        elif before_filter_count > 0 and after_filter_count == 0:
            failure_reason = "results_filtered_out"
        elif errors:
            failure_reason = "provider_error"
        else:
            failure_reason = "empty_search_results"

        self.last_search_meta = {
            "query_raw_len": len(self.query or ""),
            "query_clean": clean_query,
            "query_simplified": simplified_query,
            "region": region,
            "providers_used": providers_used,
            "mandatory_domains": mandatory_domains,
            "target_official_domains": target_official_domains,
            "official_domains_found": official_found,
            "before_filter": before_filter_count,
            "after_filter": after_filter_count,
            "retries": retries,
            "fallback_used": fallback_used,
            "errors": errors,
            "failure_reason": failure_reason,
        }
        Duckduckgo.recent_search_events.append(dict(self.last_search_meta))
        return filtered
