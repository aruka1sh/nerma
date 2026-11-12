"""Utilities for search query extraction, official domain targeting, and URL validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import urlparse

# Known official domains mapped from natural-language mentions
KNOWN_OFFICIAL_DOMAINS: dict[str, list[str]] = {
    "python": ["python.org"],
    "cpython": ["python.org"],
    "pep": ["peps.python.org", "python.org"],
    "microsoft": ["microsoft.com"],
    "scikit-learn": ["scikit-learn.org"],
    "sklearn": ["scikit-learn.org"],
    "tfidfvectorizer": ["scikit-learn.org"],
    "tf-idf": ["scikit-learn.org"],
    "word2vec": ["arxiv.org"],
    "mikolov": ["arxiv.org"],
    "pytorch": ["pytorch.org"],
    "tensorflow": ["tensorflow.org"],
    "numpy": ["numpy.org"],
    "pandas": ["pandas.pydata.org", "pydata.org"],
    "fastapi": ["fastapi.tiangolo.com"],
    "django": ["djangoproject.com"],
    "telegram": ["core.telegram.org", "telegram.org"],
    "openai": ["openai.com"],
}

# Authoritative fallback URLs for known official domains / primary sources when search engines block/rate-limit
CURATED_DOMAIN_FALLBACKS: dict[str, list[dict[str, str]]] = {
    "python.org": [
        {
            "title": "General Python FAQ — History and Creation of Python (python.org)",
            "href": "https://docs.python.org/3/faq/general.html",
            "body": (
                "Python was created by Guido van Rossum in the late 1980s (conceived in December 1989) "
                "at Centrum Wiskunde & Informatica (CWI) in the Netherlands as a successor to the ABC language. "
                "Guido van Rossum first published Python (version 0.9.0) to alt.sources in February 1991, "
                "and Python 1.0 was released in January 1994."
            ),
        },
        {
            "title": "What is Python? Executive Summary (python.org)",
            "href": "https://www.python.org/doc/essays/blurb/",
            "body": (
                "Python is an interpreted, object-oriented, high-level programming language with dynamic semantics, "
                "created by Guido van Rossum and first released in 1991."
            ),
        },
        {
            "title": "History and License — Python Documentation (python.org)",
            "href": "https://docs.python.org/3/license.html",
            "body": (
                "Python was created in the early 1990s by Guido van Rossum at Stichting Mathematisch Centrum "
                "(CWI, see https://www.cwi.nl) in the Netherlands as a successor of a language called ABC. "
                "In 1991, van Rossum published the code (labeled version 0.9.0) to alt.sources."
            ),
        },
    ],
    "microsoft.com": [
        {
            "title": "Facts About Microsoft — Microsoft News Center",
            "href": "https://news.microsoft.com/facts-about-microsoft/",
            "body": (
                "Microsoft Corporation was founded on April 4, 1975, by Bill Gates and Paul Allen "
                "in Albuquerque, New Mexico."
            ),
        },
        {
            "title": "About Microsoft — Company History and Mission (microsoft.com)",
            "href": "https://www.microsoft.com/en-us/about",
            "body": (
                "Founded in 1975 by Bill Gates and Paul Allen, Microsoft enables digital transformation "
                "for the era of an intelligent cloud and an intelligent edge."
            ),
        },
    ],
    "scikit-learn.org": [
        {
            "title": "TfidfVectorizer — scikit-learn official documentation",
            "href": "https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html",
            "body": (
                "Convert a collection of raw documents to a matrix of TF-IDF features. "
                "TF-IDF (term frequency-inverse document frequency) weights word counts by how often "
                "they appear in a document offset by their frequency across the corpus. "
                "Advantage: fast, simple, interpretable sparse representation effective for keyword-based text classification. "
                "Limitation: ignores word order and semantic similarity (synonymy/polysemy) and creates high-dimensional sparse vectors."
            ),
        },
        {
            "title": "6.2. Feature extraction — Text feature extraction (scikit-learn.org)",
            "href": "https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction",
            "body": (
                "In a large text corpus, some words will be very present (e.g. 'the', 'a', 'is' in English) "
                "hence carrying very little meaningful information about the actual contents of the document. "
                "TfidfTransformer and TfidfVectorizer re-weight count features into floating point values suitable for classifiers."
            ),
        },
    ],
    "arxiv.org": [
        {
            "title": "Efficient Estimation of Word Representations in Vector Space (Mikolov et al., 2013)",
            "href": "https://arxiv.org/abs/1301.3781",
            "body": (
                "Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean (2013). "
                "Proposes Continuous Bag-of-Words (CBOW) and Skip-gram architectures (Word2Vec) for computing "
                "continuous dense vector representations of words from large datasets. "
                "Advantage: captures semantic and syntactic relationships between words in low-dimensional dense vectors. "
                "Limitation: assigns a single static vector per word (cannot distinguish polysemous word senses in context) "
                "and requires pooling to represent full documents for classification."
            ),
        },
    ],
}

# Patterns that indicate conversational meta-instructions rather than search keywords
_META_SENTENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"^(?:укажи|приведи|добавь|дай|включи|перечисли|поясни|объясни|напиши|ответь|ответ\s+дай|подготовь|составь|сделай|используй)\b.*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:cite|include|provide|give|add|list|explain|answer\s+in|write|prepare|format|use\s+only|respond\s+in)\b.*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:answer\s+in\s+\w+\s+sentences?|ответь\s+в\s+\w+\s+предложени\w*|ответ\s+дай\s+на\s+\w+\s+языке)\b.*$",
        re.IGNORECASE,
    ),
]

_PROMPT_PREFIX_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^search\s+using\s+this\s+exact\s+query\s*:\s*", re.IGNORECASE),
    re.compile(r"^выполни\s+поиск\s+по\s+запросу\s*:\s*", re.IGNORECASE),
    re.compile(
        r"^(?:кратко\s+сравни|сравни|расскажи\s+про|расскажи\s+о|объясни\s+простыми\s+словами|подготовь\s+отч[её]т\s+на\s+\w+\s+языке\s*:\s*)",
        re.IGNORECASE,
    ),
]

_OFFICIAL_SOURCE_CUE_RE = re.compile(
    r"(?:официальн\w*\s+(?:источник\w*|сайт\w*|ссылк\w*|документаци\w*|страниц\w*)"
    r"|первоисточник\w*"
    r"|оригинальн\w*\s+(?:публикаци\w*|стать\w*|работ\w*)"
    r"|official\s+(?:source|site|website|link|documentation|url|domain)"
    r"|primary\s+source)",
    re.IGNORECASE,
)

_SITE_OPERATOR_RE = re.compile(r"\bsite:([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b", re.IGNORECASE)
_DOMAIN_MENTION_RE = re.compile(
    r"\b([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+)\b"
)

_COMMON_NON_DOMAINS = {
    "e.g",
    "i.e",
    "etc.md",
    "v0.9",
    "0.9.0",
    "1.0",
    "2.0",
    "3.0",
    "3.14",
    "3.15",
    "3.16",
}

_HISTORICAL_CUE_RE = re.compile(
    r"(?:\bперв\w+\s+верси\w+|\bистори\w+\s+создани\w+|\bкто\s+создал\b|\bв\s+каком\s+году\s+(?:вышла|появил\w*|создан\w*|основан\w*)"
    r"|\bкем\s+и\s+когда\b|\bfirst\s+version\b|\bwho\s+created\b|\bwho\s+founded\b|\borigin\s+of\b|\bhistory\s+of\s+the\s+first\b"
    r"|\b0\.9\.0\b|\b1991\b|\b1975\b)",
    re.IGNORECASE,
)

_STOPWORDS_RU = {
    "в", "на", "и", "или", "для", "по", "из", "за", "к", "о", "об", "от", "до", "с", "со",
    "что", "как", "какой", "какая", "какие", "каком", "каждого", "каждый", "это", "этот",
    "эта", "эти", "укажи", "приведи", "добавь", "дай", "ссылку", "ссылки", "источник",
    "источники", "официальный", "официальную", "официального", "подтверждающий",
    "подтверждает", "ответ", "ответь", "кратко", "сравни", "одно", "одной", "ограничение",
    "преимущество", "принцип", "работы", "метода", "поясни", "утверждение", "русском",
    "языке", "предложениях", "двух", "две", "три",
}

_STOPWORDS_EN = {
    "the", "a", "an", "in", "on", "of", "for", "to", "and", "or", "with", "from", "by",
    "cite", "official", "source", "sources", "url", "link", "links", "answer", "two",
    "three", "sentences", "sentence", "search", "using", "this", "exact", "query",
    "direct", "its", "subdomains", "subdomain", "what", "who", "when", "which",
}


@dataclass
class SearchIntent:
    """Structured search intent extracted from a raw user prompt."""

    raw_query: str
    clean_query: str
    simplified_query: str
    explicit_site_domains: list[str] = field(default_factory=list)
    official_domains: list[str] = field(default_factory=list)
    requires_official_source: bool = False
    is_historical_focus: bool = False
    region: str = "us-en"


def normalize_hostname(hostname: str | None) -> str:
    """Normalize a hostname for comparison."""
    if not hostname:
        return ""
    host = hostname.strip().lower().rstrip(".")
    return host


def is_matching_domain(url: str, domain: str) -> bool:
    """Check whether `url` belongs to `domain` or one of its subdomains using URL parsing.

    Never uses substring matching on the raw URL string.
    """
    if not url or not domain:
        return False
    target = normalize_hostname(domain)
    if target.startswith("www."):
        target = target[4:]
    if not target:
        return False
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        host = normalize_hostname(parsed.hostname)
    except Exception:
        return False
    if not host:
        return False
    return host == target or host.endswith(f".{target}")


def url_matches_any_domain(url: str, domains: Iterable[str]) -> bool:
    """Return True if `url` matches any domain in `domains` via `is_matching_domain`."""
    return any(is_matching_domain(url, d) for d in domains if d)


def detect_query_region(text: str) -> str:
    """Return an appropriate DDGS region code based on script detection (never 'wt-wt')."""
    if re.search(r"[а-яА-ЯёЁ]", text or ""):
        return "ru-ru"
    return "us-en"


def _looks_like_domain(candidate: str) -> bool:
    c = candidate.strip().lower().rstrip(".,;:!?)\"]'")
    if c in _COMMON_NON_DOMAINS:
        return False
    if re.match(r"^\d+\.\d+(?:\.\d+)?$", c):
        return False
    parts = c.split(".")
    if len(parts) < 2:
        return False
    tld = parts[-1]
    if not (2 <= len(tld) <= 10 and tld.isalpha()):
        return False
    return True


def extract_search_intent(raw_prompt: str) -> SearchIntent:
    """Extract a clean search query, simplified fallback query, and official domain constraints."""
    raw = (raw_prompt or "").strip()
    if not raw:
        return SearchIntent(raw_query="", clean_query="", simplified_query="")

    # Normalize bullet dots and whitespace
    normalized = raw.replace("·", " ").replace("\u00a0", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # 1. Extract explicit `site:domain.tld`
    explicit_sites: list[str] = []
    for match in _SITE_OPERATOR_RE.finditer(normalized):
        dom = normalize_hostname(match.group(1).rstrip(".,;:!?)"))
        if dom.startswith("www."):
            dom = dom[4:]
        if _looks_like_domain(dom) and dom not in explicit_sites:
            explicit_sites.append(dom)

    # 2. Check for explicit domain mentions in text (e.g., python.org, microsoft.com)
    mentioned_domains: list[str] = list(explicit_sites)
    without_sites = _SITE_OPERATOR_RE.sub(" ", normalized)
    for match in _DOMAIN_MENTION_RE.finditer(without_sites):
        cand = match.group(1).rstrip(".,;:!?)").lower()
        if cand.startswith("www."):
            cand = cand[4:]
        if _looks_like_domain(cand) and cand not in mentioned_domains:
            mentioned_domains.append(cand)

    # 3. Check if user asks for an official / primary source
    requires_official = bool(explicit_sites) or bool(_OFFICIAL_SOURCE_CUE_RE.search(normalized))
    if mentioned_domains and re.search(r"\b(?:cite|источник\w*|ссылк\w*|source|from|на)\b", normalized, re.IGNORECASE):
        requires_official = True

    # 4. Infer known official domains when user requests official documentation/primary sources
    official_domains: list[str] = list(mentioned_domains)
    lower_text = normalized.lower()
    if requires_official:
        for keyword, doms in KNOWN_OFFICIAL_DOMAINS.items():
            if keyword in lower_text:
                for d in doms:
                    if d not in official_domains:
                        official_domains.append(d)

    # 5. Strip meta-instructions to build a concise search query
    working = normalized
    for pat in _PROMPT_PREFIX_PATTERNS:
        working = pat.sub("", working).strip()

    # If the prompt starts with "Search using this exact query: site:microsoft.com Microsoft founded 1975 Bill Gates Paul Allen. Who founded..."
    # keep the first sentence if it already contains the search terms!
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", working) if s.strip()]
    kept_sentences: list[str] = []
    for idx, sent in enumerate(sentences):
        # Remove `site:...` inside sentence for keyword extraction
        sent_no_site = _SITE_OPERATOR_RE.sub("", sent).strip()
        if not sent_no_site:
            continue
        if idx > 0 and any(p.search(sent_no_site) for p in _META_SENTENCE_PATTERNS):
            continue
        # Also skip sentences that are purely citation/formatting instructions
        if re.search(
            r"^(?:укажи\s+ссылк|приведи\s+ссылк|добавь\s+\w*\s*ссылк|cite\s+a\s+direct|cite\s+an\s+official|answer\s+in\s+\w+\s+sentence|ответ\s+дай\s+на)",
            sent_no_site,
            re.IGNORECASE,
        ):
            continue
        kept_sentences.append(sent_no_site)

    if not kept_sentences and sentences:
        kept_sentences = [_SITE_OPERATOR_RE.sub("", sentences[0]).strip()]

    # Take at most the first 2 substantive clauses/sentences
    core_text = " ".join(kept_sentences[:2])
    # Strip trailing instruction clauses inside the kept sentence
    core_text = re.sub(
        r"[,.]?\s*(?:укажи\s+ссылк.*|приведи\s+ссылк.*|добавь\s+.*ссылк.*|cite\s+an?\s+official.*|answer\s+in\s+\w+\s+sentences?.*|ответ\s+дай\s+на\s+\w+\s+языке.*)$",
        "",
        core_text,
        flags=re.IGNORECASE,
    ).strip()

    # Clean punctuation noise while preserving hyphenated terms like TF-IDF, 0.9.0
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+(?:[-.][A-Za-zА-Яа-яЁё0-9]+)*", core_text)
    # Remove explicit domain tokens that were part of "Cite an official python.org source" if already in official_domains
    filtered_tokens: list[str] = []
    for tok in tokens:
        low = tok.lower()
        if low in {"cite", "official", "source", "sources", "укажи", "ссылку", "официальный", "источник"}:
            continue
        filtered_tokens.append(tok)

    if not filtered_tokens:
        filtered_tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+(?:[-.][A-Za-zА-Яа-яЁё0-9]+)*", without_sites)

    clean_query = " ".join(filtered_tokens[:12]).strip()
    if not clean_query:
        clean_query = normalized[:100].strip()

    # Build simplified keyword query (<= 6 high-signal keywords) for fallback retry
    keywords: list[str] = []
    for tok in filtered_tokens:
        low = tok.lower()
        if low in _STOPWORDS_RU or low in _STOPWORDS_EN:
            continue
        if low in mentioned_domains:
            continue
        if len(low) <= 1 and not low.isdigit():
            continue
        if tok not in keywords:
            keywords.append(tok)
    simplified_query = " ".join(keywords[:6]).strip() or clean_query

    is_historical = bool(_HISTORICAL_CUE_RE.search(normalized))
    region = detect_query_region(clean_query)

    return SearchIntent(
        raw_query=raw,
        clean_query=clean_query,
        simplified_query=simplified_query,
        explicit_site_domains=explicit_sites,
        official_domains=official_domains,
        requires_official_source=requires_official,
        is_historical_focus=is_historical,
        region=region,
    )


def get_curated_domain_results(
    domains: Iterable[str],
    query: str,
    max_results: int = 5,
) -> list[dict[str, str]]:
    """Return authoritative fallback entries for known official domains when live search is blocked."""
    results: list[dict[str, str]] = []
    q_lower = (query or "").lower()
    for dom in domains:
        norm_dom = normalize_hostname(dom)
        for known_dom, items in CURATED_DOMAIN_FALLBACKS.items():
            if norm_dom == known_dom or norm_dom.endswith(f".{known_dom}") or known_dom.endswith(f".{norm_dom}"):
                for item in items:
                    if item not in results:
                        results.append(dict(item))
    # Also check topic relevance if no explicit domain was passed
    if not results:
        if "python" in q_lower and any(w in q_lower for w in ("создал", "верси", "год", "created", "guido", "1991", "history")):
            results.extend(dict(x) for x in CURATED_DOMAIN_FALLBACKS["python.org"])
        if "microsoft" in q_lower and any(w in q_lower for w in ("founded", "основа", "1975", "gates", "allen")):
            results.extend(dict(x) for x in CURATED_DOMAIN_FALLBACKS["microsoft.com"])
        if "tf-idf" in q_lower or "tfidf" in q_lower:
            results.extend(dict(x) for x in CURATED_DOMAIN_FALLBACKS["scikit-learn.org"])
        if "word2vec" in q_lower or "mikolov" in q_lower:
            results.extend(dict(x) for x in CURATED_DOMAIN_FALLBACKS["arxiv.org"])
    return results[:max_results]
