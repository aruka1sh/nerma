"""Main Telegram Bot module powered by GPT Researcher (NERMA AI)."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests
from bs4 import BeautifulSoup
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from gpt_researcher import GPTResearcher
from gpt_researcher.utils.enum import Tone
from gpt_researcher.utils.llm import create_chat_completion

from telegram_bot.config import (
    DEEP_MAX_BREADTH,
    DEEP_MAX_DEPTH,
    LANGUAGES,
    LLM_TIMEOUT_SEC,
    MAX_CONCURRENT_REQUESTS,
    QUEUE_TIMEOUT_SEC,
    QUICK_MAX_RESULTS,
    QUICK_MAX_SCRAPE_PAGES,
    QUICK_MAX_WORDS,
    REPORT_TYPES,
    SCRAPE_TIMEOUT_SEC,
    SEARCH_TIMEOUT_SEC,
    TELEGRAM_BOT_TOKEN,
    TONES,
    RequestState,
    UserSettings,
    get_request_timeout_sec,
    user_store,
)
from telegram_bot.diagnostics import FailureReason, RequestDiagnostics
from telegram_bot.exporter import (
    ResearchResultBundle,
    build_result_bundle,
    export_bundle,
    format_metrics_block,
    is_refusal_or_empty_report,
)
from telegram_bot.keyboards import (
    get_cancel_keyboard,
    get_failed_report_keyboard,
    get_formats_keyboard,
    get_language_keyboard,
    get_main_reply_keyboard,
    get_post_report_keyboard,
    get_report_type_keyboard,
    get_settings_keyboard,
    get_tone_keyboard,
)
from telegram_bot.progress_handler import TelegramProgressHandler
from telegram_bot.query_utils import (
    SearchIntent,
    extract_search_intent,
    is_matching_domain,
    url_matches_any_domain,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

TONE_ENUM_MAP = {
    "Objective": Tone.Objective,
    "Analytical": Tone.Analytical,
    "Formal": Tone.Formal,
    "Informative": Tone.Informative,
    "Simple": Tone.Simple,
}


@dataclass
class ActiveRequestContext:
    """Runtime handle for an in-flight research request supporting cancellation."""

    request_id: str
    user_id: int
    query: str
    settings: UserSettings
    diagnostics: RequestDiagnostics
    progress: TelegramProgressHandler | None = None
    task: asyncio.Task[Any] | None = None
    cancelled_event: asyncio.Event = field(default_factory=asyncio.Event)


# Active in-flight requests keyed by request_id, plus per-user active request_id
ACTIVE_REQUESTS: dict[str, ActiveRequestContext] = {}
USER_ACTIVE_REQUEST: dict[int, str] = {}

# Saved ResearchResultBundles keyed by request_id for on-demand Quick export without re-running research
SAVED_BUNDLES: dict[str, ResearchResultBundle] = {}
_MAX_SAVED_BUNDLES = 100

# Global concurrency semaphore
_CONCURRENCY_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


def _store_bundle(bundle: ResearchResultBundle) -> None:
    """Save a completed/partial bundle in bounded memory for button-triggered file export."""
    if len(SAVED_BUNDLES) >= _MAX_SAVED_BUNDLES:
        oldest_key = next(iter(SAVED_BUNDLES))
        SAVED_BUNDLES.pop(oldest_key, None)
    SAVED_BUNDLES[bundle.request_id] = bundle


def _format_settings_text(user_id: int, settings: UserSettings | None = None) -> str:
    s = settings or user_store.get(user_id)
    rt = REPORT_TYPES.get(s.report_type, {})
    tone = TONES.get(s.tone, {})
    lang = LANGUAGES.get(s.language, {})
    fmts = ", ".join(s.export_formats).upper() or "Не выбраны"

    return (
        "⚙️ *Текущие настройки исследования:*\n\n"
        f"📊 *Режим отчёта:* {rt.get('title', s.report_type)}\n"
        f"   └ _{rt.get('description', '')}_\n\n"
        f"🎭 *Тон изложения:* {tone.get('title', s.tone)}\n"
        f"🌐 *Язык ответа:* {lang.get('title', s.language)}\n"
        f"📁 *Форматы документов:* `{fmts}`\n\n"
        "💡 _Выберите параметр ниже для изменения или просто отправьте тему исследования текстом:_"
    )


def _fetch_page_excerpt_sync(url: str, timeout_sec: float = 6.0, max_chars: int = 2500) -> tuple[str, str]:
    """Fetch a web page and return `(title, extracted_text)`. Returns `("", "")` on failure."""
    if not url or not url.startswith(("http://", "https://")):
        return "", ""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout_sec, allow_redirects=True)
        if resp.status_code >= 400:
            return "", ""
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" not in ctype and "application/xhtml" not in ctype and "text/plain" not in ctype:
            return "", ""
        soup = BeautifulSoup(resp.text[:200_000], "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
            tag.decompose()
        title = (soup.title.string.strip() if soup.title and soup.title.string else "")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "li", "h1", "h2", "h3"])]
        text = "\n".join(p for p in paragraphs if len(p) > 25)
        if not text:
            text = soup.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        return title[:200], text[:max_chars]
    except Exception:
        return "", ""


async def _scrape_top_results_bounded(
    search_results: list[dict[str, Any]],
    max_pages: int = QUICK_MAX_SCRAPE_PAGES,
    timeout_sec: float = SCRAPE_TIMEOUT_SEC,
) -> list[dict[str, Any]]:
    """Concurrently fetch up to `max_pages` URLs with strict timeout to obtain real page content."""
    targets = (search_results or [])[:max_pages]
    if not targets:
        return []

    per_page_timeout = min(6.0, max(2.5, timeout_sec / max(1, len(targets))))

    async def _one(item: dict[str, Any]) -> dict[str, Any] | None:
        url = (item.get("href") or item.get("url") or "").strip()
        if not url:
            return None
        try:
            title, content = await asyncio.wait_for(
                asyncio.to_thread(_fetch_page_excerpt_sync, url, per_page_timeout),
                timeout=per_page_timeout + 1.0,
            )
        except Exception:
            title, content = "", ""
        if content and len(content) >= 60:
            return {
                "url": url,
                "title": title or item.get("title") or "",
                "raw_content": content,
                "content": content,
            }
        return None

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*[_one(it) for it in targets], return_exceptions=True),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        return []

    scraped: list[dict[str, Any]] = []
    for r in results:
        if isinstance(r, dict) and r.get("raw_content"):
            scraped.append(r)
    return scraped


def _localize_failure_message(
    reason: FailureReason,
    language: str,
    required_official_domains: list[str] | None = None,
) -> str:
    """Return a concise, user-friendly failure explanation in the user's selected language."""
    req_doms = ", ".join(f"`{d}`" for d in (required_official_domains or []))
    if language == "english":
        messages_en = {
            FailureReason.EMPTY_SEARCH_RESULTS: (
                "❌ *Could not complete the report: search returned 0 results.*\n"
                + (f"Requested official domain: {req_doms}\n" if req_doms else "")
                + "Try simplifying your query or switching to Standard mode."
            ),
            FailureReason.PROVIDER_ERROR: (
                "⚠️ *Search provider temporarily unavailable or rate-limited.*\n"
                "No error files were generated. Please retry in a few seconds or switch modes."
            ),
            FailureReason.RESULTS_FILTERED_OUT: (
                f"❌ *No sources matched the required domain constraint ({req_doms or 'official domain'}).*\n"
                "Search returned general pages, but none belonged to the requested official hostname."
            ),
            FailureReason.PAGE_FETCH_FAILED: (
                "⚠️ *Found search links, but target pages could not be loaded.*\n"
                "Please retry or switch to Standard mode."
            ),
            FailureReason.EMPTY_LLM_CONTEXT: (
                "⚠️ *Insufficient verified source context to generate a reliable answer.*\n"
                "We do not generate unverified answers without sources. Try rephrasing the query."
            ),
            FailureReason.LLM_REFUSAL: (
                "⚠️ *The model could not verify the answer from the retrieved sources.*\n"
                "Try running in Standard mode or simplifying the question."
            ),
            FailureReason.TIMEOUT: (
                "⏱ *Request timed out before a verified report could be completed.*\n"
                "Try Quick/Standard mode or a narrower query."
            ),
            FailureReason.CANCELLED: "🚫 *Request was cancelled.*",
        }
        return messages_en.get(
            reason,
            "⚠️ *Could not complete the research request.* Please retry or change the report mode.",
        )

    messages_ru = {
        FailureReason.EMPTY_SEARCH_RESULTS: (
            "❌ *Не удалось сформировать отчёт: поиск не вернул источников (0 результатов).*\n"
            + (f"Запрошенный официальный домен: {req_doms}\n" if req_doms else "")
            + "Попробуйте упростить формулировку запроса или выбрать «📊 Стандартный отчёт»."
        ),
        FailureReason.PROVIDER_ERROR: (
            "⚠️ *Поисковый провайдер временно вернул ошибку или ограничил частоту запросов.*\n"
            "Файлы с текстом ошибки не создавались. Повторите запрос или смените режим."
        ),
        FailureReason.RESULTS_FILTERED_OUT: (
            f"❌ *Официальный источник на домене {req_doms or 'указанном в запросе'} не найден.*\n"
            "Поисковая выдача не содержала страниц с подтверждённым hostname этого домена."
        ),
        FailureReason.PAGE_FETCH_FAILED: (
            "⚠️ *Ссылки найдены, но не удалось загрузить содержимое страниц.*\n"
            "Попробуйте повторить запрос или переключиться на Стандартный режим."
        ),
        FailureReason.EMPTY_LLM_CONTEXT: (
            "⚠️ *Не удалось собрать подтверждённый контекст для модели.*\n"
            "Бот не создаёт «Готовый отчёт» без проверяемых источников. Попробуйте уточнить запрос."
        ),
        FailureReason.LLM_REFUSAL: (
            "⚠️ *Недостаточно данных в найденных источниках для достоверного ответа.*\n"
            "Попробуйте повторить запрос или сменить режим исследования."
        ),
        FailureReason.TIMEOUT: (
            "⏱ *Превышено допустимое время ожидания ответа.*\n"
            "Попробуйте режим «⚡ Быстрый обзор» или сузьте тему запроса."
        ),
        FailureReason.CANCELLED: "🚫 *Запрос отменён пользователем.*",
    }
    return messages_ru.get(
        reason,
        "⚠️ *Не удалось выполнить исследование.* Попробуйте повторить запрос или сменить режим.",
    )


# ──────────────────────────────────────────────────────────────────────
# Command Handlers
# ──────────────────────────────────────────────────────────────────────


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    user = update.effective_user
    user_store.get(user.id)

    welcome_text = (
        f"👋 Привет, *{user.first_name}*!\n\n"
        "Я — *NERMA AI* 🤖, интеллектуальный исследовательский ассистент на базе *GPT Researcher*.\n\n"
        "🔍 *Что я умею:*\n"
        "• Искать и проверять информацию по реальным веб-источникам и официальной документации\n"
        "• Строго связывать ссылки `[1], [2]` в тексте с проверенным списком URL\n"
        "• Работать в режимах *Quick*, *Standard*, *Deep Research* и *Outline*\n"
        "• Экспортировать готовые отчёты в *DOCX*, *Markdown* и *PDF*\n\n"
        "✍️ *Просто напишите мне любой вопрос или тему исследования!*"
    )
    await update.message.reply_text(
        welcome_text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_main_reply_keyboard(),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    help_text = (
        "📖 *Справка по использованию бота NERMA AI*\n\n"
        "*Доступные команды:*\n"
        "• `/start` — Перезапустить бота и показать главное меню\n"
        "• `/settings` — Открыть панель настроек исследования\n"
        "• `/quick <тема>` — Быстрый ответ в чате с проверяемыми ссылками `[n]`\n"
        "• `/deep <тема>` — Глубокое исследование по целевым подзапросам\n"
        "• `/cancel` — Отменить текущее активное исследование\n"
        "• `/help` — Показать эту справку\n\n"
        "*Показатели источников в отчёте:*\n"
        "• *Найдено уникальных URL* — число уникальных ссылок, полученных из поиска после фильтрации.\n"
        "• *Успешно прочитано страниц* — число веб-страниц, чей полный текст был успешно загружен (сниппеты поисковика сюда не входят).\n"
        "• *Процитировано источников* — число источников `[1]..[n]`, на которые есть прямая ссылка в тексте ответа."
    )
    await update.message.reply_text(
        help_text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_main_reply_keyboard(),
    )


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /settings command."""
    user_id = update.effective_user.id
    settings = user_store.get(user_id)
    await update.message.reply_text(
        _format_settings_text(user_id, settings),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_settings_keyboard(settings),
    )


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /cancel command to stop the user's active request."""
    user_id = update.effective_user.id
    req_id = USER_ACTIVE_REQUEST.get(user_id)
    if not req_id or req_id not in ACTIVE_REQUESTS:
        await update.message.reply_text("ℹ️ Сейчас нет активного выполняющегося запроса.")
        return
    await _cancel_request_by_id(req_id)
    await update.message.reply_text(f"🛑 Запрос `[#{req_id}]` отменён.", parse_mode=ParseMode.MARKDOWN)


async def quick_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /quick <query> command."""
    query = " ".join(context.args).strip() if context.args else ""
    if not query:
        await update.message.reply_text(
            "⚡ Использование: `/quick <ваш запрос>`\nНапример: `/quick Кто создал язык Python?`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    await run_research(update, context, query, override_report_type="quick_search")


async def deep_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /deep <query> command."""
    query = " ".join(context.args).strip() if context.args else ""
    if not query:
        await update.message.reply_text(
            "🧠 Использование: `/deep <ваш запрос>`\nНапример: `/deep История создания первой версии Python`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    await run_research(update, context, query, override_report_type="deep")


# ──────────────────────────────────────────────────────────────────────
# Callback Query Handlers (Inline Buttons)
# ──────────────────────────────────────────────────────────────────────


async def _cancel_request_by_id(request_id: str) -> bool:
    """Cancel an active request by `request_id`, stopping its task and updating state."""
    active = ACTIVE_REQUESTS.get(request_id)
    if not active:
        return False
    active.cancelled_event.set()
    active.diagnostics.failure_reason = FailureReason.CANCELLED
    active.diagnostics.final_state = RequestState.CANCELLED.value
    if active.progress:
        await active.progress.set_state(
            RequestState.CANCELLED,
            "🚫 Остановлено пользователем"
            if active.settings.language != "english"
            else "🚫 Cancelled by user",
        )
    if active.task and not active.task.done():
        active.task.cancel()
    return True


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard button presses."""
    query_cb = update.callback_query
    await query_cb.answer()
    user_id = query_cb.from_user.id
    settings = user_store.get(user_id)
    data = query_cb.data or ""

    # Cancellation button
    if data.startswith("cancel:"):
        req_id = data.split(":", 1)[1]
        cancelled = await _cancel_request_by_id(req_id)
        if cancelled:
            msg = (
                f"🚫 *Запрос* `[#{req_id}]` *отменён.*\nОтправка отчёта остановлена."
                if settings.language != "english"
                else f"🚫 *Request* `[#{req_id}]` *cancelled.*"
            )
            try:
                await query_cb.edit_message_text(msg, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                pass
        return

    # On-demand file export button for Quick mode (uses saved bundle, does NOT re-run research!)
    if data.startswith("export:"):
        req_id = data.split(":", 1)[1]
        bundle = SAVED_BUNDLES.get(req_id)
        if not bundle or not bundle.is_exportable:
            await query_cb.message.reply_text(
                "⚠️ Сохранённый результат для экспорта не найден или устарел."
                if settings.language != "english"
                else "⚠️ Saved result for export was not found or expired."
            )
            return
        fmts = settings.export_formats or ["docx", "md"]
        exported = await export_bundle(bundle, fmts)
        if not exported:
            await query_cb.message.reply_text("⚠️ Не удалось сформировать файлы отчёта.")
            return
        await _send_exported_files(query_cb.message.chat_id, context, bundle, exported)
        return

    if data == "menu:main":
        await query_cb.edit_message_text(
            _format_settings_text(user_id, settings),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=get_settings_keyboard(settings),
        )
    elif data == "menu:report_type":
        await query_cb.edit_message_text(
            "📊 *Выберите режим исследования:*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=get_report_type_keyboard(settings.report_type),
        )
    elif data == "menu:tone":
        await query_cb.edit_message_text(
            "🎭 *Выберите тон изложения отчёта:*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=get_tone_keyboard(settings.tone),
        )
    elif data == "menu:language":
        await query_cb.edit_message_text(
            "🌐 *Выберите язык итогового отчёта:*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=get_language_keyboard(settings.language),
        )
    elif data == "menu:formats":
        await query_cb.edit_message_text(
            "📁 *Выберите форматы файлов для скачивания:*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=get_formats_keyboard(settings.export_formats),
        )
    elif data == "settings:reset":
        settings = user_store.reset(user_id)
        await query_cb.edit_message_text(
            "✅ *Настройки сброшены по умолчанию!*\n\n" + _format_settings_text(user_id, settings),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=get_settings_keyboard(settings),
        )
    elif data.startswith("set_rt:"):
        rt_key = data.split(":", 1)[1]
        if rt_key in REPORT_TYPES:
            settings.report_type = rt_key
        await query_cb.edit_message_text(
            _format_settings_text(user_id, settings),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=get_settings_keyboard(settings),
        )
    elif data.startswith("set_tone:"):
        tone_key = data.split(":", 1)[1]
        if tone_key in TONES:
            settings.tone = tone_key
        await query_cb.edit_message_text(
            _format_settings_text(user_id, settings),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=get_settings_keyboard(settings),
        )
    elif data.startswith("set_lang:"):
        lang_key = data.split(":", 1)[1]
        if lang_key in LANGUAGES:
            settings.language = lang_key
        await query_cb.edit_message_text(
            _format_settings_text(user_id, settings),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=get_settings_keyboard(settings),
        )
    elif data.startswith("toggle_fmt:"):
        fmt_key = data.split(":", 1)[1]
        if fmt_key in settings.export_formats:
            settings.export_formats.remove(fmt_key)
        else:
            settings.export_formats.append(fmt_key)
        await query_cb.edit_message_text(
            "📁 *Выберите форматы файлов для скачивания:*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=get_formats_keyboard(settings.export_formats),
        )
    elif data in ("action:rerun_deep", "action:rerun_outline", "action:rerun_standard", "action:retry_same"):
        if not settings.last_query:
            await query_cb.message.reply_text("⚠️ Нет сохранённого предыдущего запроса.")
            return
        override_map = {
            "action:rerun_deep": "deep",
            "action:rerun_outline": "outline_report",
            "action:rerun_standard": "research_report",
            "action:retry_same": settings.report_type,
        }
        await run_research(
            update,
            context,
            settings.last_query,
            override_report_type=override_map[data],
        )


# ──────────────────────────────────────────────────────────────────────
# Message & Research Execution Pipeline
# ──────────────────────────────────────────────────────────────────────


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle persistent keyboard buttons and free-form research queries."""
    text = (update.message.text or "").strip()
    user_id = update.effective_user.id
    settings = user_store.get(user_id)

    if text == "🔍 Новое исследование":
        await update.message.reply_text(
            "✍️ *Введите тему или вопрос для исследования:*\n\n"
            "Например:\n"
            "• _«В каком году вышла первая версия Python и кто создал этот язык?»_\n"
            "• _«Кратко сравни TF-IDF и Word2Vec для классификации текста»_\n"
            "• _«Who founded Microsoft and in what year? Cite an official microsoft.com source.»_",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    if text == "⚙️ Настройки":
        await settings_command(update, context)
        return
    if text == "📊 Режим отчёта":
        await update.message.reply_text(
            "📊 *Выберите режим исследования:*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=get_report_type_keyboard(settings.report_type),
        )
        return
    if text == "🌐 Язык ответа":
        await update.message.reply_text(
            "🌐 *Выберите язык итогового отчёта:*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=get_language_keyboard(settings.language),
        )
        return
    if text == "🎭 Тон изложения":
        await update.message.reply_text(
            "🎭 *Выберите тон изложения отчёта:*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=get_tone_keyboard(settings.tone),
        )
        return
    if text == "ℹ️ О боте / Помощь":
        await help_command(update, context)
        return

    await run_research(update, context, text)


def _ingest_retriever_telemetry(
    researcher: GPTResearcher,
    diagnostics: RequestDiagnostics,
) -> None:
    """Copy search telemetry collected on `researcher` into `RequestDiagnostics`."""
    telemetry_list = getattr(researcher, "search_telemetry", None) or []
    for meta in telemetry_list:
        if not isinstance(meta, dict):
            continue
        q_clean = meta.get("query_clean") or ""
        providers = meta.get("providers_used") or ["ddgs"]
        for prov in providers:
            diagnostics.record_search_event(
                query=q_clean,
                provider=prov,
                before_filter=meta.get("before_filter", 0),
                after_filter=meta.get("after_filter", 0),
                retries=meta.get("retries", 0),
                fallback_used=bool(meta.get("fallback_used", False)),
            )
        for err in meta.get("errors") or []:
            diagnostics.record_error(str(err))
        for dom in meta.get("official_domains_found") or []:
            if dom not in diagnostics.official_domains_found:
                diagnostics.official_domains_found.append(dom)
        reason_str = meta.get("failure_reason", "none")
        if reason_str != "none" and diagnostics.results_after_filter == 0:
            try:
                diagnostics.failure_reason = FailureReason(reason_str)
            except ValueError:
                pass


async def _execute_quick_mode(
    researcher: GPTResearcher,
    query: str,
    intent: SearchIntent,
    settings: UserSettings,
    diagnostics: RequestDiagnostics,
    progress: TelegramProgressHandler,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute Quick mode with a single search, bounded top-page scraping, and concise synthesis."""
    await progress.set_state(
        RequestState.SEARCHING,
        "⚡ Быстрый поиск проверенных источников..."
        if settings.language != "english"
        else "⚡ Searching verified sources...",
    )

    with diagnostics.stage("search"):
        search_results = await asyncio.wait_for(
            researcher.quick_search(
                query=query,
                query_domains=intent.explicit_site_domains or None,
                aggregated_summary=False,
            ),
            timeout=SEARCH_TIMEOUT_SEC + 5.0,
        )
    _ingest_retriever_telemetry(researcher, diagnostics)

    if not search_results:
        return "", [], []

    # Bounded scrape of top pages so `pages_read_successfully` reflects actually read pages
    with diagnostics.stage("scrape"):
        scraped_sources = await _scrape_top_results_bounded(
            search_results,
            max_pages=QUICK_MAX_SCRAPE_PAGES,
            timeout_sec=SCRAPE_TIMEOUT_SEC,
        )
    if scraped_sources:
        researcher.add_research_sources(scraped_sources)

    # Build numbered source context for the LLM
    scraped_by_url = {s["url"]: s["raw_content"] for s in scraped_sources if s.get("url")}
    context_blocks: list[str] = []
    for idx, item in enumerate(search_results[:QUICK_MAX_RESULTS], start=1):
        url = (item.get("href") or item.get("url") or "").strip()
        title = (item.get("title") or "").strip()
        full_text = scraped_by_url.get(url, "")
        excerpt = full_text[:1500] if full_text else (item.get("body") or item.get("content") or "")
        context_blocks.append(f"[{idx}] Title: {title}\nURL: {url}\nContent: {excerpt}")

    context_str = "\n\n".join(context_blocks).strip()
    diagnostics.llm_context_chars = len(context_str)
    if not context_str:
        diagnostics.failure_reason = FailureReason.EMPTY_LLM_CONTEXT
        return "", search_results, scraped_sources

    await progress.set_state(
        RequestState.WRITING,
        "✍️ Формирование краткого ответа со ссылками..."
        if settings.language != "english"
        else "✍️ Writing concise answer with citations...",
    )

    lang_instruction = (
        "Write the response in English."
        if settings.language == "english"
        else (
            "Жауапты қазақ тілінде жаз."
            if settings.language == "kazakh"
            else "Напиши ответ на русском языке."
        )
    )
    prompt = (
        f"You are a precise research assistant. Answer the user's question concisely (up to {QUICK_MAX_WORDS} words) "
        f"based ONLY on the numbered sources below.\n"
        f"- {lang_instruction}\n"
        f"- Tone: {settings.tone}.\n"
        f"- Cite facts inline using ONLY bracketed numbers like [1], [2] that correspond to the numbered sources below.\n"
        f"- Do NOT invent or guess any URLs not listed in the numbered sources.\n"
        f"- Do NOT append a separate References section at the end (it will be attached automatically).\n\n"
        f"User question: {query}\n\n"
        f"Numbered sources:\n{context_str}"
    )

    with diagnostics.stage("llm_write"):
        summary = await asyncio.wait_for(
            create_chat_completion(
                model=researcher.cfg.smart_llm_model,
                messages=[{"role": "user", "content": prompt}],
                llm_provider=researcher.cfg.smart_llm_provider,
                max_tokens=researcher.cfg.smart_token_limit,
                llm_kwargs=researcher.cfg.llm_kwargs,
                cost_callback=researcher.add_costs,
            ),
            timeout=LLM_TIMEOUT_SEC,
        )

    return summary, search_results, scraped_sources


async def _execute_outline_mode(
    researcher: GPTResearcher,
    query: str,
    intent: SearchIntent,
    settings: UserSettings,
    diagnostics: RequestDiagnostics,
    progress: TelegramProgressHandler,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute Outline mode: produce a structured hierarchical topic plan backed by sources (not a long essay)."""
    await progress.set_state(
        RequestState.SEARCHING,
        "🔎 Поиск источников для составления плана исследования..."
        if settings.language != "english"
        else "🔎 Searching sources for research outline...",
    )

    with diagnostics.stage("search"):
        search_results = await asyncio.wait_for(
            researcher.quick_search(
                query=query,
                query_domains=intent.explicit_site_domains or None,
                aggregated_summary=False,
            ),
            timeout=SEARCH_TIMEOUT_SEC + 5.0,
        )
    _ingest_retriever_telemetry(researcher, diagnostics)

    if not search_results:
        return "", [], []

    with diagnostics.stage("scrape"):
        scraped_sources = await _scrape_top_results_bounded(
            search_results,
            max_pages=QUICK_MAX_SCRAPE_PAGES,
            timeout_sec=SCRAPE_TIMEOUT_SEC,
        )
    if scraped_sources:
        researcher.add_research_sources(scraped_sources)

    scraped_by_url = {s["url"]: s["raw_content"] for s in scraped_sources if s.get("url")}
    context_blocks: list[str] = []
    for idx, item in enumerate(search_results[:QUICK_MAX_RESULTS], start=1):
        url = (item.get("href") or item.get("url") or "").strip()
        title = (item.get("title") or "").strip()
        full_text = scraped_by_url.get(url, "")
        excerpt = full_text[:1200] if full_text else (item.get("body") or item.get("content") or "")
        context_blocks.append(f"[{idx}] Title: {title}\nURL: {url}\nContent: {excerpt}")

    context_str = "\n\n".join(context_blocks).strip()
    diagnostics.llm_context_chars = len(context_str)
    if not context_str:
        diagnostics.failure_reason = FailureReason.EMPTY_LLM_CONTEXT
        return "", search_results, scraped_sources

    await progress.set_state(
        RequestState.WRITING,
        "📝 Формирование структуры и плана исследования (Outline)..."
        if settings.language != "english"
        else "📝 Generating structured research outline...",
    )

    lang_instruction = (
        "Write the outline in English."
        if settings.language == "english"
        else (
            "Жоспарды қазақ тілінде жаз."
            if settings.language == "kazakh"
            else "Составьте план исследования на русском языке."
        )
    )
    prompt = (
        f"You are an academic research planner. Create a concise, structured RESEARCH OUTLINE (plan of sections, "
        f"subsections, key research questions, and mapped sources [1], [2]) for the topic below.\n"
        f"IMPORTANT: Produce a hierarchical outline/plan with bullet points (I, II, III, IV...), NOT a long full-text essay.\n"
        f"- {lang_instruction}\n"
        f"- Reference the relevant numbered sources [1], [2] in each section of the outline.\n"
        f"- Do NOT invent URLs not present in the numbered sources.\n"
        f"- Do NOT append a separate bibliography section at the end.\n\n"
        f"Topic / Question: {query}\n\n"
        f"Numbered sources:\n{context_str}"
    )

    with diagnostics.stage("llm_write"):
        outline_md = await asyncio.wait_for(
            create_chat_completion(
                model=researcher.cfg.smart_llm_model,
                messages=[{"role": "user", "content": prompt}],
                llm_provider=researcher.cfg.smart_llm_provider,
                max_tokens=researcher.cfg.smart_token_limit,
                llm_kwargs=researcher.cfg.llm_kwargs,
                cost_callback=researcher.add_costs,
            ),
            timeout=LLM_TIMEOUT_SEC,
        )

    return outline_md, search_results, scraped_sources


async def _execute_full_or_deep_mode(
    researcher: GPTResearcher,
    query: str,
    intent: SearchIntent,
    settings: UserSettings,
    diagnostics: RequestDiagnostics,
    progress: TelegramProgressHandler,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute Standard (`research_report`), Detailed (`detailed_report`), or Deep (`deep`) mode."""
    await progress.set_state(
        RequestState.SEARCHING,
        "🔍 Поиск и чтение источников..."
        if settings.language != "english"
        else "🔍 Searching and reading sources...",
    )

    # If the user requested an official domain in natural language (e.g. python.org, scikit-learn.org, arxiv.org),
    # perform a targeted pre-search so official domain sources are guaranteed to be in `researcher`!
    pre_search_results: list[dict[str, Any]] = []
    if intent.official_domains:
        try:
            pre_search_results = await asyncio.wait_for(
                researcher.quick_search(
                    query=query,
                    query_domains=intent.official_domains,
                    aggregated_summary=False,
                ),
                timeout=SEARCH_TIMEOUT_SEC,
            )
            if pre_search_results:
                scraped_official = await _scrape_top_results_bounded(
                    pre_search_results,
                    max_pages=2,
                    timeout_sec=SCRAPE_TIMEOUT_SEC,
                )
                if scraped_official:
                    researcher.add_research_sources(scraped_official)
                    for s in scraped_official:
                        if s.get("raw_content"):
                            researcher.context.append(
                                f"Official Source ({s['url']}): {s['raw_content']}"
                            )
        except Exception as exc:
            diagnostics.record_error(f"official_presearch:{exc}")

    with diagnostics.stage("search_and_scrape"):
        await researcher.conduct_research()

    _ingest_retriever_telemetry(researcher, diagnostics)

    scraped_sources = list(researcher.get_research_sources() or [])
    visited_urls = list(researcher.get_source_urls() or [])
    ctx_items = researcher.get_research_context() or []
    ctx_str = "\n".join(str(x) for x in ctx_items) if isinstance(ctx_items, list) else str(ctx_items)
    diagnostics.llm_context_chars = len(ctx_str)

    # If conduct_research gathered 0 sources and 0 context, do not call write_report to hallucinate or write refusal
    if not visited_urls and not scraped_sources and not pre_search_results:
        diagnostics.classify_failure()
        return "", pre_search_results, scraped_sources

    await progress.set_state(
        RequestState.WRITING,
        "✍️ Написание структурированного отчёта по источникам..."
        if settings.language != "english"
        else "✍️ Writing structured report from sources...",
    )

    with diagnostics.stage("llm_write"):
        report_md = await asyncio.wait_for(
            researcher.write_report(),
            timeout=LLM_TIMEOUT_SEC * 2,
        )

    return report_md, pre_search_results, scraped_sources


async def _send_exported_files(
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    bundle: ResearchResultBundle,
    exported_files: dict[str, Any],
) -> None:
    """Send generated DOCX/MD/PDF files to the user chat."""
    if not exported_files:
        return
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_DOCUMENT)
    caption_labels = {
        "pdf": "📕 Отчёт в формате PDF",
        "docx": "📘 Отчёт в формате Word (DOCX)",
        "md": "📄 Исходный текст в формате Markdown (MD)",
    }
    for fmt, file_path in exported_files.items():
        try:
            with open(file_path, "rb") as doc_file:
                await context.bot.send_document(
                    chat_id=chat_id,
                    document=doc_file,
                    filename=file_path.name,
                    caption=f"{caption_labels.get(fmt, fmt.upper())} [#{bundle.request_id}]",
                )
        except Exception as exc:
            logger.error("Failed to send %s file to Telegram: %s", fmt, exc)


async def run_research(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    query: str,
    override_report_type: str | None = None,
) -> ResearchResultBundle | None:
    """Execute the full end-to-end research pipeline with isolated settings snapshot,
    diagnostics, cancellation support, and strict source verification.
    """
    user = update.effective_user
    chat_id = update.effective_chat.id
    msg_target = update.message or update.callback_query.message

    # Save last_query on mutable user store for retry buttons, then freeze a deep snapshot for this request!
    live_settings = user_store.get(user.id)
    live_settings.last_query = query
    settings = live_settings.snapshot()
    if override_report_type and override_report_type in REPORT_TYPES:
        settings.report_type = override_report_type

    request_id = uuid.uuid4().hex[:8]
    live_settings.last_request_id = request_id

    intent = extract_search_intent(query)
    report_type_cfg = REPORT_TYPES.get(settings.report_type, REPORT_TYPES["quick_search"])
    actual_mode = report_type_cfg["gptr_type"]
    tone_enum = TONE_ENUM_MAP.get(settings.tone, Tone.Objective)

    diagnostics = RequestDiagnostics(
        request_id=request_id,
        user_id=user.id,
        selected_mode=settings.report_type,
        actual_mode=actual_mode,
        language=settings.language,
        tone=settings.tone,
        raw_query_length=len(query),
        clean_search_query=intent.clean_query,
        required_official_domains=list(intent.official_domains),
    )

    status_msg = await msg_target.reply_text(
        f"⏳ *Запрос принят в очередь* `[#{request_id}]`\n"
        f"📌 *Тема:* `{query[:120]}`\n"
        f"📊 *Режим:* {report_type_cfg['title']}",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_cancel_keyboard(request_id, language=settings.language),
    )

    progress_handler = TelegramProgressHandler(
        status_message=status_msg,
        query=query,
        request_id=request_id,
        language=settings.language,
    )

    current_task = asyncio.current_task()
    active_ctx = ActiveRequestContext(
        request_id=request_id,
        user_id=user.id,
        query=query,
        settings=settings,
        diagnostics=diagnostics,
        progress=progress_handler,
        task=current_task,
    )
    ACTIVE_REQUESTS[request_id] = active_ctx
    USER_ACTIVE_REQUEST[user.id] = request_id

    total_timeout_sec = get_request_timeout_sec(actual_mode)
    start_ts = time.time()

    try:
        # 1. Acquire concurrency slot with bounded queue timeout
        with diagnostics.stage("queue_wait"):
            try:
                await asyncio.wait_for(_CONCURRENCY_SEMAPHORE.acquire(), timeout=QUEUE_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                diagnostics.record_timeout("queue", QUEUE_TIMEOUT_SEC)
                diagnostics.final_state = RequestState.FAILED.value
                diagnostics.emit_log("request_failed", level=logging.WARNING)
                await progress_handler.set_state(RequestState.FAILED)
                await status_msg.edit_text(
                    _localize_failure_message(FailureReason.TIMEOUT, settings.language),
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=get_failed_report_keyboard(request_id, settings.language),
                )
                return None

        try:
            if active_ctx.cancelled_event.is_set():
                raise asyncio.CancelledError()

            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

            # Configure GPTResearcher with isolated per-instance config (no global os.environ mutation)
            gptr_report_type = (
                "research_report" if actual_mode in ("quick_search", "outline_report") else actual_mode
            )
            researcher = GPTResearcher(
                query=query,
                report_type=gptr_report_type,
                tone=tone_enum,
                websocket=progress_handler,
                log_handler=progress_handler,
                query_domains=intent.explicit_site_domains or None,
                verbose=False,
            )
            researcher.cfg.language = settings.language
            if actual_mode == "deep":
                researcher.cfg.deep_research_depth = DEEP_MAX_DEPTH
                researcher.cfg.deep_research_breadth = DEEP_MAX_BREADTH
                if hasattr(researcher, "deep_researcher") and researcher.deep_researcher is not None:
                    researcher.deep_researcher.depth = DEEP_MAX_DEPTH
                    researcher.deep_researcher.breadth = DEEP_MAX_BREADTH

            # 2. Run mode-specific pipeline under total request timeout
            async def _run_mode() -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
                if actual_mode == "quick_search":
                    return await _execute_quick_mode(
                        researcher, query, intent, settings, diagnostics, progress_handler
                    )
                if actual_mode == "outline_report":
                    return await _execute_outline_mode(
                        researcher, query, intent, settings, diagnostics, progress_handler
                    )
                return await _execute_full_or_deep_mode(
                    researcher, query, intent, settings, diagnostics, progress_handler
                )

            raw_report, search_results, scraped_sources = await asyncio.wait_for(
                _run_mode(),
                timeout=total_timeout_sec,
            )

            if active_ctx.cancelled_event.is_set():
                raise asyncio.CancelledError()

            elapsed_sec = round(time.time() - start_ts, 2)
            visited_urls = list(researcher.get_source_urls() or [])
            cost_usd = float(researcher.get_costs() or 0.0)

            bundle = build_result_bundle(
                request_id=request_id,
                query=query,
                mode=actual_mode,
                raw_report=raw_report,
                search_results=search_results,
                scraped_sources=scraped_sources,
                visited_urls=visited_urls,
                required_official_domains=intent.official_domains,
                language=settings.language,
                duration_sec=elapsed_sec,
                cost_usd=cost_usd,
                failure_reason=diagnostics.failure_reason.value,
            )

            diagnostics.unique_urls_found = bundle.unique_urls_found
            diagnostics.pages_read_successfully = bundle.pages_read_successfully
            diagnostics.snippets_only_count = max(
                0, bundle.unique_urls_found - bundle.pages_read_successfully
            )
            diagnostics.sources_cited = bundle.sources_cited
            diagnostics.official_domains_found = bundle.official_domains_found

            # 3. Check if bundle failed (0 sources, refusal, or filtered out)
            if bundle.state == RequestState.FAILED:
                reason = diagnostics.classify_failure()
                if reason == FailureReason.NONE:
                    reason = (
                        FailureReason.RESULTS_FILTERED_OUT
                        if intent.explicit_site_domains and diagnostics.results_before_filter > 0
                        else FailureReason.EMPTY_SEARCH_RESULTS
                    )
                diagnostics.failure_reason = reason
                diagnostics.final_state = RequestState.FAILED.value
                diagnostics.emit_log("request_failed", level=logging.WARNING)

                await progress_handler.set_state(RequestState.FAILED)
                err_text = _localize_failure_message(
                    reason,
                    settings.language,
                    required_official_domains=intent.official_domains,
                )
                await status_msg.edit_text(
                    err_text,
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=get_failed_report_keyboard(request_id, settings.language),
                )
                return bundle

            # 4. Save bundle for button-triggered export and update status
            _store_bundle(bundle)
            diagnostics.final_state = bundle.state.value

            # Export files automatically ONLY in non-Quick modes when formats are selected
            exported_files: dict[str, Any] = {}
            if actual_mode != "quick_search" and settings.export_formats:
                await progress_handler.set_state(
                    RequestState.EXPORTING,
                    "📁 Экспорт проверенного отчёта в файлы..."
                    if settings.language != "english"
                    else "📁 Exporting verified report to files...",
                )
                with diagnostics.stage("export"):
                    exported_files = await export_bundle(bundle, settings.export_formats)

            if active_ctx.cancelled_event.is_set():
                raise asyncio.CancelledError()

            await progress_handler.set_state(bundle.state)
            diagnostics.emit_log("request_completed", level=logging.INFO)

            # 5. Format final status header (only show «Готовый отчёт» when state == COMPLETED!)
            header_title = (
                "✅ *Готовый отчёт*"
                if bundle.state == RequestState.COMPLETED
                else "⚠️ *Частичный отчёт (официальный домен не найден)*"
            )
            if settings.language == "english":
                header_title = (
                    "✅ *Completed Report*"
                    if bundle.state == RequestState.COMPLETED
                    else "⚠️ *Partial Report (official domain not found)*"
                )

            summary_header = (
                f"{header_title} `[#{request_id}]`\n\n"
                f"📌 *Тема:* `{query[:120]}`\n"
                f"📊 *Режим:* {report_type_cfg['title']}\n"
                f"⏱ *Время выполнения:* {int(elapsed_sec)} сек.\n"
                f"{format_metrics_block(bundle)}"
            )
            if cost_usd > 0:
                summary_header += f"\n💰 *Расход токенов:* \\${cost_usd:.4f}"

            await status_msg.edit_text(summary_header, parse_mode=ParseMode.MARKDOWN)

            # 6. Send unified report text with verified [1]..[n] URL list in chat
            include_quick_export_btn = actual_mode == "quick_search"
            await _send_long_report(
                chat_id=chat_id,
                context=context,
                report_md=bundle.full_markdown,
                request_id=request_id,
                include_export_button=include_quick_export_btn,
                language=settings.language,
            )

            # 7. Send exported files in non-Quick modes
            if exported_files:
                await _send_exported_files(chat_id, context, bundle, exported_files)

            return bundle

        finally:
            _CONCURRENCY_SEMAPHORE.release()

    except asyncio.CancelledError:
        diagnostics.failure_reason = FailureReason.CANCELLED
        diagnostics.final_state = RequestState.CANCELLED.value
        diagnostics.emit_log("request_cancelled", level=logging.INFO)
        try:
            await status_msg.edit_text(
                _localize_failure_message(FailureReason.CANCELLED, settings.language),
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            pass
        return None

    except asyncio.TimeoutError:
        diagnostics.record_timeout("request_total", total_timeout_sec)
        diagnostics.final_state = RequestState.FAILED.value
        diagnostics.emit_log("request_timeout", level=logging.WARNING)
        try:
            await status_msg.edit_text(
                _localize_failure_message(FailureReason.TIMEOUT, settings.language),
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=get_failed_report_keyboard(request_id, settings.language),
            )
        except Exception:
            pass
        return None

    except Exception as exc:
        logger.exception("Error during research execution for request_id=%s", request_id)
        diagnostics.record_error(f"{type(exc).__name__}: {exc}")
        diagnostics.failure_reason = FailureReason.INTERNAL_ERROR
        diagnostics.final_state = RequestState.FAILED.value
        diagnostics.emit_log("request_error", level=logging.ERROR)
        try:
            await status_msg.edit_text(
                _localize_failure_message(FailureReason.INTERNAL_ERROR, settings.language),
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=get_failed_report_keyboard(request_id, settings.language),
            )
        except Exception:
            pass
        return None

    finally:
        ACTIVE_REQUESTS.pop(request_id, None)
        if USER_ACTIVE_REQUEST.get(user.id) == request_id:
            USER_ACTIVE_REQUEST.pop(user.id, None)


async def _send_long_report(
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    report_md: str,
    request_id: str = "",
    include_export_button: bool = False,
    language: str = "russian",
) -> None:
    """Split a report into Telegram-safe chunks (<= 4000 chars) and send with post-report buttons."""
    max_len = 3800
    chunks: list[str] = []
    current = ""

    for paragraph in report_md.split("\n\n"):
        if len(current) + len(paragraph) + 2 <= max_len:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current)
            if len(paragraph) <= max_len:
                current = paragraph
            else:
                for i in range(0, len(paragraph), max_len):
                    chunks.append(paragraph[i : i + max_len])
                current = ""
    if current:
        chunks.append(current)

    max_chunks = 3
    total_chunks = len(chunks)
    chunks_to_send = chunks[:max_chunks]

    for idx, chunk in enumerate(chunks_to_send):
        is_last = idx == len(chunks_to_send) - 1
        footer = ""
        if is_last and total_chunks > max_chunks:
            footer = (
                "\n\n_... (Полная версия отчёта доступна в прикреплённых файлах ниже)_"
                if language != "english"
                else "\n\n_... (Full report is available in the exported files)_"
            )

        text_to_send = chunk + footer
        reply_markup = (
            get_post_report_keyboard(
                request_id=request_id,
                include_export_button=include_export_button,
                language=language,
            )
            if is_last
            else None
        )

        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=text_to_send,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup,
                disable_web_page_preview=True,
            )
        except BadRequest:
            plain_text = re.sub(r"[*_`#]", "", text_to_send)
            await context.bot.send_message(
                chat_id=chat_id,
                text=plain_text,
                reply_markup=reply_markup,
                disable_web_page_preview=True,
            )


def main() -> None:
    """Entry point to start the Telegram bot."""
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set in .env! Please add TELEGRAM_BOT_TOKEN=your_token to .env"
        )

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).concurrent_updates(True).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CommandHandler("cancel", cancel_command))
    app.add_handler(CommandHandler("quick", quick_command))
    app.add_handler(CommandHandler("deep", deep_command))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    logger.info("🤖 GPT Researcher Telegram Bot (NERMA AI) is starting polling...")
    app.run_polling(drop_pending_updates=True)


run_telegram_bot = main


if __name__ == "__main__":
    main()
