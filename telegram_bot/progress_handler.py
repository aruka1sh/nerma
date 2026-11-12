"""Real-time progress handler that streams GPT Researcher logs and RequestState updates to Telegram."""

from __future__ import annotations

import logging
import time
from typing import Any

from telegram import Message
from telegram.constants import ParseMode
from telegram.error import BadRequest, RetryAfter, TimedOut

from telegram_bot.config import RequestState
from telegram_bot.keyboards import get_cancel_keyboard

logger = logging.getLogger(__name__)


STATE_LABELS_RU: dict[RequestState, str] = {
    RequestState.QUEUED: "⏳ В очереди (`queued`)",
    RequestState.SEARCHING: "🌐 Поиск и проверка источников (`searching`)",
    RequestState.WRITING: "✍️ Синтез ответа по источникам (`writing`)",
    RequestState.EXPORTING: "📁 Подготовка документов (`exporting`)",
    RequestState.COMPLETED: "✅ Завершено (`completed`)",
    RequestState.PARTIAL: "⚠️ Частичный результат (`partial`)",
    RequestState.FAILED: "❌ Ошибка (`failed`)",
    RequestState.CANCELLED: "🚫 Отменено (`cancelled`)",
}

STATE_LABELS_EN: dict[RequestState, str] = {
    RequestState.QUEUED: "⏳ Queued (`queued`)",
    RequestState.SEARCHING: "🌐 Searching & verifying sources (`searching`)",
    RequestState.WRITING: "✍️ Synthesizing answer from sources (`writing`)",
    RequestState.EXPORTING: "📁 Exporting documents (`exporting`)",
    RequestState.COMPLETED: "✅ Completed (`completed`)",
    RequestState.PARTIAL: "⚠️ Partial result (`partial`)",
    RequestState.FAILED: "❌ Failed (`failed`)",
    RequestState.CANCELLED: "🚫 Cancelled (`cancelled`)",
}


class TelegramProgressHandler:
    """Custom websocket/log handler compatible with GPTResearcher that updates a Telegram message."""

    def __init__(
        self,
        status_message: Message,
        query: str,
        request_id: str = "",
        language: str = "russian",
        min_update_interval: float = 2.0,
    ) -> None:
        self.status_message = status_message
        self.query = query
        self.request_id = request_id
        self.language = language
        self.min_update_interval = min_update_interval
        self.last_update_time: float = 0.0
        self.last_text: str = ""
        self.steps: list[str] = []
        self.sources_count: int = 0
        self.state: RequestState = RequestState.QUEUED
        self.is_cancelled: bool = False
        self.start_time: float = time.time()

    async def set_state(self, state: RequestState, detail: str = "", force: bool = True) -> None:
        """Transition to an explicit `RequestState` and update the Telegram status message."""
        if self.is_cancelled and state != RequestState.CANCELLED:
            return
        self.state = state
        if state == RequestState.CANCELLED:
            self.is_cancelled = True
        if detail:
            self._append_step(detail)
        await self._refresh_message(force=force)

    async def send_json(self, data: dict[str, Any]) -> None:
        """Method called by GPTResearcher stream_output(websocket.send_json)."""
        if self.is_cancelled:
            return
        msg_type = data.get("type", "")
        content = data.get(" content") or data.get("content", "")
        output = data.get("output", "")

        text_to_process = str(output or content).strip()
        if not text_to_process:
            return

        if content == "added_source_url" or "🌐" in text_to_process or "Scraping" in text_to_process:
            self.sources_count += 1

        step_label = self._translate_step(content, text_to_process)
        if step_label:
            self._append_step(step_label)

        now = time.time()
        if now - self.last_update_time >= self.min_update_interval:
            await self._refresh_message()

    async def on_event(self, event_type: str, step: str, details: dict[str, Any]) -> None:
        """Optional callback for structured events."""
        if self.is_cancelled:
            return
        if step == "start":
            await self.set_state(
                RequestState.SEARCHING,
                "🚀 Инициализация поискового агента..."
                if self.language != "english"
                else "🚀 Initializing search agent...",
                force=False,
            )
        elif step == "conducting_research":
            await self.set_state(
                RequestState.SEARCHING,
                "🔍 Сбор данных из поисковых систем..."
                if self.language != "english"
                else "🔍 Querying search providers...",
                force=False,
            )
        elif step == "writing_report":
            await self.set_state(
                RequestState.WRITING,
                "✍️ Написание структурированного ответа..."
                if self.language != "english"
                else "✍️ Writing structured response...",
                force=True,
            )
        elif step == "report_completed":
            self._append_step(
                "✅ Синтез текста завершён!"
                if self.language != "english"
                else "✅ Text synthesis complete!"
            )
            await self._refresh_message(force=True)

    def _append_step(self, step_label: str) -> None:
        if not self.steps or self.steps[-1] != step_label:
            self.steps.append(step_label)
            if len(self.steps) > 6:
                self.steps = self.steps[-6:]

    def _translate_step(self, content_key: str, raw_text: str) -> str:
        """Convert raw English logs from GPTResearcher into clean Russian status lines."""
        mapping = {
            "starting_research": "🚀 Запуск исследования по теме...",
            "agent_generated": "🤖 Выбран специализированный ИИ-агент",
            "planning_research": "🧠 Формирование плана и поисковых запросов...",
            "running_subquery_research": "🔎 Поиск информации по подзапросам...",
            "added_source_url": f"🌐 Найдено источников: {self.sources_count}",
            "research_step_finalized": "📚 Анализ и фильтрация собранных материалов...",
            "generating_report": "✍️ Генерация итогового текста...",
            "report_written": "✅ Текст отчёта сформирован!",
        }
        if content_key in mapping:
            return mapping[content_key]

        clean = raw_text.replace("\n", " ").strip()
        if len(clean) > 90:
            clean = clean[:87] + "..."
        return clean

    async def _refresh_message(self, force: bool = False) -> None:
        if self.is_cancelled and self.state != RequestState.CANCELLED:
            return
        elapsed = int(time.time() - self.start_time)
        steps_block = "\n".join(f"• {s}" for s in self.steps[-5:]) if self.steps else "• Подготовка..."
        labels = STATE_LABELS_EN if self.language == "english" else STATE_LABELS_RU
        state_line = labels.get(self.state, self.state.value)

        short_q = self.query[:120] + ("..." if len(self.query) > 120 else "")
        req_tag = f" `[#{self.request_id}]`" if self.request_id else ""

        if self.language == "english":
            text = (
                f"🔬 *Research in progress*{req_tag}\n"
                f"📌 *Topic:* `{short_q}`\n"
                f"📡 *Status:* {state_line}\n"
                f"⏱ *Elapsed:* {elapsed} sec.\n\n"
                f"*Current steps:*\n{steps_block}"
            )
        else:
            text = (
                f"🔬 *Выполняется исследование*{req_tag}\n"
                f"📌 *Тема:* `{short_q}`\n"
                f"📡 *Статус:* {state_line}\n"
                f"⏱ *Прошло времени:* {elapsed} сек.\n\n"
                f"*Текущие этапы:*\n{steps_block}"
            )

        if text == self.last_text and not force:
            return

        reply_markup = None
        if self.request_id and self.state in (
            RequestState.QUEUED,
            RequestState.SEARCHING,
            RequestState.WRITING,
            RequestState.EXPORTING,
        ):
            reply_markup = get_cancel_keyboard(self.request_id, language=self.language)

        try:
            await self.status_message.edit_text(
                text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup,
            )
            self.last_text = text
            self.last_update_time = time.time()
        except RetryAfter as exc:
            self.last_update_time = time.time() + exc.retry_after
        except (BadRequest, TimedOut):
            try:
                await self.status_message.edit_text(
                    text.replace("*", "").replace("`", ""),
                    reply_markup=reply_markup,
                )
                self.last_text = text
                self.last_update_time = time.time()
            except Exception:
                pass
        except Exception as exc:
            logger.debug("Progress update skipped: %s", exc)
