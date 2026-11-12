"""Configuration and user settings management for the GPT Researcher Telegram Bot."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root .env
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# Configurable timeouts, concurrency, and mode budgets
QUEUE_TIMEOUT_SEC: float = _env_float("QUEUE_TIMEOUT_SEC", 30.0)
SEARCH_TIMEOUT_SEC: float = _env_float("SEARCH_TIMEOUT_SEC", 20.0)
SCRAPE_TIMEOUT_SEC: float = _env_float("SCRAPE_TIMEOUT_SEC", 12.0)
LLM_TIMEOUT_SEC: float = _env_float("LLM_TIMEOUT_SEC", 60.0)
REQUEST_TIMEOUT_QUICK_SEC: float = _env_float("REQUEST_TIMEOUT_QUICK_SEC", 60.0)
REQUEST_TIMEOUT_STANDARD_SEC: float = _env_float("REQUEST_TIMEOUT_STANDARD_SEC", 180.0)
REQUEST_TIMEOUT_DEEP_SEC: float = _env_float("REQUEST_TIMEOUT_DEEP_SEC", 240.0)

MAX_CONCURRENT_REQUESTS: int = _env_int("MAX_CONCURRENT_REQUESTS", 3)
MAX_SEARCH_RETRIES: int = _env_int("MAX_SEARCH_RETRIES", 1)

QUICK_MAX_RESULTS: int = _env_int("QUICK_MAX_RESULTS", 5)
QUICK_MAX_SCRAPE_PAGES: int = _env_int("QUICK_MAX_SCRAPE_PAGES", 3)
QUICK_MAX_WORDS: int = _env_int("QUICK_MAX_WORDS", 220)

DEEP_MAX_DEPTH: int = _env_int("DEEP_MAX_DEPTH", 1)
DEEP_MAX_BREADTH: int = _env_int("DEEP_MAX_BREADTH", 2)

VERBOSE_QUERY_LOGS: bool = _env_bool("VERBOSE_QUERY_LOGS", False)


class RequestState(str, Enum):
    """Explicit lifecycle states of a user research request."""

    QUEUED = "queued"
    SEARCHING = "searching"
    WRITING = "writing"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Available report modes (without unverified fixed speed promises)
REPORT_TYPES: dict[str, dict[str, str]] = {
    "quick_search": {
        "title": "⚡ Быстрый обзор (Quick)",
        "description": "Краткий ответ в чате с проверяемыми ссылками [n]; файлы только по кнопке",
        "gptr_type": "quick_search",
    },
    "research_report": {
        "title": "📊 Стандартный отчёт",
        "description": "Структурированный аналитический отчёт с проверенной библиографией",
        "gptr_type": "research_report",
    },
    "detailed_report": {
        "title": "📚 Подробный отчёт",
        "description": "Развёрнутый многосекционный анализ с источниками",
        "gptr_type": "detailed_report",
    },
    "deep": {
        "title": "🧠 Глубокое исследование (Deep Research)",
        "description": "Многошаговый поиск по целевым подзапросам в рамках исходной темы",
        "gptr_type": "deep",
    },
    "outline_report": {
        "title": "📝 План исследования (Outline)",
        "description": "Иерархическая структура и тезисный план по теме (не длинный отчёт)",
        "gptr_type": "outline_report",
    },
}

# Available tones mapped to gpt_researcher.utils.enum.Tone values
TONES: dict[str, dict[str, str]] = {
    "Objective": {
        "title": "⚖️ Объективный",
        "value": "Objective (impartial and unbiased presentation of facts and findings)",
    },
    "Analytical": {
        "title": "🔬 Аналитический",
        "value": "Analytical (critical evaluation and detailed examination of data and theories)",
    },
    "Formal": {
        "title": "🎓 Академический",
        "value": "Formal (adheres to academic standards with sophisticated language and structure)",
    },
    "Informative": {
        "title": "💡 Информативный",
        "value": "Informative (clear and comprehensive information on a topic)",
    },
    "Simple": {
        "title": "👶 Простой (для начинающих)",
        "value": "Simple (written for young readers, using basic vocabulary and short sentences)",
    },
}

# Available output languages
LANGUAGES: dict[str, dict[str, str]] = {
    "russian": {"title": "🇷🇺 Русский", "value": "russian"},
    "english": {"title": "🇬🇧 English", "value": "english"},
    "kazakh": {"title": "🇰🇿 Қазақша", "value": "kazakh"},
}

# Available export formats
EXPORT_FORMATS: dict[str, str] = {
    "pdf": "📕 PDF",
    "docx": "📘 Word (DOCX)",
    "md": "📄 Markdown (MD)",
}


def get_request_timeout_sec(report_type: str) -> float:
    """Return the configured total request timeout for the given report mode."""
    if report_type == "quick_search":
        return REQUEST_TIMEOUT_QUICK_SEC
    if report_type == "deep":
        return REQUEST_TIMEOUT_DEEP_SEC
    return REQUEST_TIMEOUT_STANDARD_SEC


@dataclass
class UserSettings:
    """Per-user configuration stored in memory."""

    report_type: str = "quick_search"
    tone: str = "Objective"
    language: str = "russian"
    export_formats: list[str] = field(default_factory=lambda: ["docx", "md"])
    last_query: str = ""
    last_request_id: str = ""

    def snapshot(self) -> "UserSettings":
        """Create an isolated deep copy of settings at request acceptance time."""
        return copy.deepcopy(self)


class UserStore:
    """Thread-safe in-memory store for user settings."""

    def __init__(self) -> None:
        self._users: dict[int, UserSettings] = {}

    def get(self, user_id: int) -> UserSettings:
        if user_id not in self._users:
            self._users[user_id] = UserSettings()
        return self._users[user_id]

    def snapshot(self, user_id: int) -> UserSettings:
        """Return an immutable snapshot copy of the user's current settings."""
        return self.get(user_id).snapshot()

    def reset(self, user_id: int) -> UserSettings:
        self._users[user_id] = UserSettings()
        return self._users[user_id]


user_store = UserStore()
