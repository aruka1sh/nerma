"""Structured diagnostics, telemetry, and secret-safe logging for NERMA AI requests."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("nerma.diagnostics")


class FailureReason(str, Enum):
    """Explicit failure/degradation taxonomy for search and research requests."""

    NONE = "none"
    EMPTY_SEARCH_RESULTS = "empty_search_results"
    PROVIDER_ERROR = "provider_error"
    PAGE_FETCH_FAILED = "page_fetch_failed"
    RESULTS_FILTERED_OUT = "results_filtered_out"
    EMPTY_LLM_CONTEXT = "empty_llm_context"
    OFFICIAL_SOURCE_MISSING = "official_source_missing"
    LLM_REFUSAL = "llm_refusal"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"


_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:bot)?\d{8,12}:[A-Za-z0-9_-]{30,50}"),  # Telegram bot token (standalone or in /bot<token>/)
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),  # Google API key
    re.compile(r"\bAQ\.[0-9A-Za-z_-]{20,}\b"),  # Google/Vertex token
    re.compile(r"\bsk-[0-9A-Za-z_-]{20,}\b"),  # OpenAI key
    re.compile(r"\bgsk_[0-9A-Za-z_-]{20,}\b"),  # Groq key
    re.compile(r"\btvly-[0-9A-Za-z_-]{15,}\b"),  # Tavily key
    re.compile(r"(?i)(?:api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[^\s'\"]{8,}['\"]?"),
]


def redact_secrets(text: str) -> str:
    """Remove any API keys, bot tokens, or sensitive credentials from log strings."""
    if not text:
        return ""
    sanitized = str(text)
    for pat in _SECRET_PATTERNS:
        sanitized = pat.sub("[REDACTED_SECRET]", sanitized)
    return sanitized


def format_query_for_log(query: str, verbose: bool | None = None) -> str:
    """Format a user query or search query for logging, respecting VERBOSE_QUERY_LOGS."""
    if verbose is None:
        verbose = os.getenv("VERBOSE_QUERY_LOGS", "false").lower() in ("1", "true", "yes")
    safe = redact_secrets(query or "").replace("\n", " ").strip()
    if verbose:
        return safe[:500]
    if len(safe) > 120:
        return f"{safe[:120]}... [len={len(safe)}]"
    return safe


@dataclass
class StageTimer:
    """Context manager to record stage duration in milliseconds."""

    diagnostics: "RequestDiagnostics"
    stage_name: str
    _start: float = 0.0

    def __enter__(self) -> "StageTimer":
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed_ms = round((time.monotonic() - self._start) * 1000.0, 2)
        prev = self.diagnostics.stage_durations_ms.get(self.stage_name, 0.0)
        self.diagnostics.stage_durations_ms[self.stage_name] = round(prev + elapsed_ms, 2)


@dataclass
class RequestDiagnostics:
    """Per-request diagnostic state and structured log builder."""

    request_id: str
    user_id: int
    selected_mode: str
    actual_mode: str = ""
    language: str = ""
    tone: str = ""
    raw_query_length: int = 0
    clean_search_query: str = ""
    search_queries: list[str] = field(default_factory=list)
    providers_used: list[str] = field(default_factory=list)
    required_official_domains: list[str] = field(default_factory=list)
    official_domains_found: list[str] = field(default_factory=list)
    results_before_filter: int = 0
    results_after_filter: int = 0
    unique_urls_found: int = 0
    pages_read_successfully: int = 0
    snippets_only_count: int = 0
    sources_cited: int = 0
    llm_context_chars: int = 0
    stage_durations_ms: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    timeouts: list[str] = field(default_factory=list)
    retries: int = 0
    fallback_used: bool = False
    failure_reason: FailureReason = FailureReason.NONE
    final_state: str = "queued"
    started_at: float = field(default_factory=time.monotonic)

    def stage(self, name: str) -> StageTimer:
        """Create a context manager to time a named pipeline stage."""
        return StageTimer(self, name)

    def record_search_event(
        self,
        query: str,
        provider: str,
        before_filter: int,
        after_filter: int,
        retries: int = 0,
        fallback_used: bool = False,
        error: str | None = None,
    ) -> None:
        """Record a search provider call and its filtering counts."""
        safe_q = format_query_for_log(query)
        if safe_q and safe_q not in self.search_queries:
            self.search_queries.append(safe_q)
        if provider and provider not in self.providers_used:
            self.providers_used.append(provider)
        self.results_before_filter += max(0, int(before_filter))
        self.results_after_filter += max(0, int(after_filter))
        self.retries += max(0, int(retries))
        if fallback_used:
            self.fallback_used = True
        if error:
            self.record_error(error)

    def record_error(self, error: str) -> None:
        """Record a redacted error string."""
        safe_err = redact_secrets(str(error))[:300]
        if safe_err and safe_err not in self.errors:
            self.errors.append(safe_err)

    def record_timeout(self, stage: str, timeout_sec: float) -> None:
        """Record a timeout occurrence on a specific stage."""
        entry = f"{stage}:{timeout_sec}s"
        if entry not in self.timeouts:
            self.timeouts.append(entry)
        self.failure_reason = FailureReason.TIMEOUT

    def classify_failure(self) -> FailureReason:
        """Determine the specific root-cause failure category when a request does not complete normally."""
        if self.failure_reason not in (FailureReason.NONE,):
            return self.failure_reason
        if self.timeouts:
            self.failure_reason = FailureReason.TIMEOUT
        elif self.results_before_filter == 0 and self.errors:
            self.failure_reason = FailureReason.PROVIDER_ERROR
        elif self.results_before_filter == 0:
            self.failure_reason = FailureReason.EMPTY_SEARCH_RESULTS
        elif self.results_before_filter > 0 and self.results_after_filter == 0:
            self.failure_reason = FailureReason.RESULTS_FILTERED_OUT
        elif self.results_after_filter > 0 and self.pages_read_successfully == 0 and self.snippets_only_count == 0:
            self.failure_reason = FailureReason.PAGE_FETCH_FAILED
        elif self.llm_context_chars == 0:
            self.failure_reason = FailureReason.EMPTY_LLM_CONTEXT
        return self.failure_reason

    def to_dict(self, event: str = "summary") -> dict[str, Any]:
        """Serialize diagnostic state as a structured dictionary safe for JSON logging."""
        total_ms = round((time.monotonic() - self.started_at) * 1000.0, 2)
        return {
            "event": event,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "selected_mode": self.selected_mode,
            "actual_mode": self.actual_mode or self.selected_mode,
            "language": self.language,
            "tone": self.tone,
            "raw_query_length": self.raw_query_length,
            "clean_search_query": format_query_for_log(self.clean_search_query),
            "search_queries": self.search_queries,
            "providers_used": self.providers_used,
            "required_official_domains": self.required_official_domains,
            "official_domains_found": self.official_domains_found,
            "results_before_filter": self.results_before_filter,
            "results_after_filter": self.results_after_filter,
            "unique_urls_found": self.unique_urls_found,
            "pages_read_successfully": self.pages_read_successfully,
            "snippets_only_count": self.snippets_only_count,
            "sources_cited": self.sources_cited,
            "llm_context_chars": self.llm_context_chars,
            "stage_durations_ms": self.stage_durations_ms,
            "total_duration_ms": total_ms,
            "retries": self.retries,
            "fallback_used": self.fallback_used,
            "errors": self.errors,
            "timeouts": self.timeouts,
            "failure_reason": self.failure_reason.value,
            "final_state": self.final_state,
        }

    def emit_log(self, event: str = "request_summary", level: int = logging.INFO) -> None:
        """Emit a structured JSON log entry."""
        payload = self.to_dict(event=event)
        logger.log(level, json.dumps(payload, ensure_ascii=False))
