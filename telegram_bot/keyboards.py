"""Inline and Reply keyboards for the GPT Researcher Telegram Bot."""

from __future__ import annotations

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)

from telegram_bot.config import (
    EXPORT_FORMATS,
    LANGUAGES,
    REPORT_TYPES,
    TONES,
    UserSettings,
)


def get_main_reply_keyboard() -> ReplyKeyboardMarkup:
    """Persistent bottom reply keyboard for quick navigation."""
    keyboard = [
        [
            KeyboardButton("🔍 Новое исследование"),
            KeyboardButton("⚙️ Настройки"),
        ],
        [
            KeyboardButton("📊 Режим отчёта"),
            KeyboardButton("🌐 Язык ответа"),
        ],
        [
            KeyboardButton("🎭 Тон изложения"),
            KeyboardButton("ℹ️ О боте / Помощь"),
        ],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, is_persistent=True)


def get_settings_keyboard(settings: UserSettings) -> InlineKeyboardMarkup:
    """Main settings overview inline keyboard."""
    rt_title = REPORT_TYPES.get(settings.report_type, {}).get("title", settings.report_type)
    tone_title = TONES.get(settings.tone, {}).get("title", settings.tone)
    lang_title = LANGUAGES.get(settings.language, {}).get("title", settings.language)
    fmt_list = ", ".join(settings.export_formats).upper() or "Нет"

    keyboard = [
        [
            InlineKeyboardButton(
                f"📊 Режим: {rt_title}", callback_data="menu:report_type"
            )
        ],
        [
            InlineKeyboardButton(
                f"🎭 Тон: {tone_title}", callback_data="menu:tone"
            )
        ],
        [
            InlineKeyboardButton(
                f"🌐 Язык: {lang_title}", callback_data="menu:language"
            )
        ],
        [
            InlineKeyboardButton(
                f"📁 Форматы файлов: {fmt_list}", callback_data="menu:formats"
            )
        ],
        [
            InlineKeyboardButton(
                "🔄 Сбросить по умолчанию", callback_data="settings:reset"
            )
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def get_report_type_keyboard(current: str) -> InlineKeyboardMarkup:
    """Inline keyboard for choosing report type."""
    keyboard = []
    for key, info in REPORT_TYPES.items():
        prefix = "✅ " if key == current else ""
        keyboard.append(
            [
                InlineKeyboardButton(
                    f"{prefix}{info['title']}",
                    callback_data=f"set_rt:{key}",
                )
            ]
        )
    keyboard.append([InlineKeyboardButton("⬅️ Назад в настройки", callback_data="menu:main")])
    return InlineKeyboardMarkup(keyboard)


def get_tone_keyboard(current: str) -> InlineKeyboardMarkup:
    """Inline keyboard for choosing research tone."""
    keyboard = []
    for key, info in TONES.items():
        prefix = "✅ " if key == current else ""
        keyboard.append(
            [
                InlineKeyboardButton(
                    f"{prefix}{info['title']}",
                    callback_data=f"set_tone:{key}",
                )
            ]
        )
    keyboard.append([InlineKeyboardButton("⬅️ Назад в настройки", callback_data="menu:main")])
    return InlineKeyboardMarkup(keyboard)


def get_language_keyboard(current: str) -> InlineKeyboardMarkup:
    """Inline keyboard for choosing output language."""
    keyboard = []
    for key, info in LANGUAGES.items():
        prefix = "✅ " if key == current else ""
        keyboard.append(
            [
                InlineKeyboardButton(
                    f"{prefix}{info['title']}",
                    callback_data=f"set_lang:{key}",
                )
            ]
        )
    keyboard.append([InlineKeyboardButton("⬅️ Назад в настройки", callback_data="menu:main")])
    return InlineKeyboardMarkup(keyboard)


def get_formats_keyboard(selected: list[str]) -> InlineKeyboardMarkup:
    """Inline keyboard for toggling export file formats."""
    keyboard = []
    for key, label in EXPORT_FORMATS.items():
        prefix = "✅ " if key in selected else "⬜ "
        keyboard.append(
            [
                InlineKeyboardButton(
                    f"{prefix}{label}",
                    callback_data=f"toggle_fmt:{key}",
                )
            ]
        )
    keyboard.append([InlineKeyboardButton("⬅️ Назад в настройки", callback_data="menu:main")])
    return InlineKeyboardMarkup(keyboard)


def get_cancel_keyboard(request_id: str, language: str = "russian") -> InlineKeyboardMarkup:
    """Inline keyboard with Cancel button attached to the active status message."""
    label = "🛑 Отменить запрос" if language != "english" else "🛑 Cancel Request"
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(label, callback_data=f"cancel:{request_id}")]]
    )


def get_post_report_keyboard(
    request_id: str = "",
    include_export_button: bool = False,
    language: str = "russian",
) -> InlineKeyboardMarkup:
    """Inline actions attached under a completed/partial report."""
    keyboard: list[list[InlineKeyboardButton]] = []
    if include_export_button and request_id:
        export_label = (
            "📥 Скачать файлы (DOCX / MD)"
            if language != "english"
            else "📥 Download Files (DOCX / MD)"
        )
        keyboard.append(
            [InlineKeyboardButton(export_label, callback_data=f"export:{request_id}")]
        )
    keyboard.extend(
        [
            [
                InlineKeyboardButton(
                    "🔄 Повторить в глубоком режиме" if language != "english" else "🔄 Retry in Deep Mode",
                    callback_data="action:rerun_deep",
                ),
            ],
            [
                InlineKeyboardButton(
                    "📝 Повторить как план (Outline)" if language != "english" else "📝 Retry as Outline",
                    callback_data="action:rerun_outline",
                ),
                InlineKeyboardButton(
                    "⚙️ Настройки" if language != "english" else "⚙️ Settings",
                    callback_data="menu:main",
                ),
            ],
        ]
    )
    return InlineKeyboardMarkup(keyboard)


def get_failed_report_keyboard(
    request_id: str = "",
    language: str = "russian",
) -> InlineKeyboardMarkup:
    """Inline actions attached under a failed request message."""
    retry_label = "🔁 Повторить запрос" if language != "english" else "🔁 Retry Query"
    std_label = (
        "📊 Попробовать Стандартный режим"
        if language != "english"
        else "📊 Try Standard Mode"
    )
    settings_label = "⚙️ Изменить настройки" if language != "english" else "⚙️ Change Settings"
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(retry_label, callback_data="action:retry_same")],
            [InlineKeyboardButton(std_label, callback_data="action:rerun_standard")],
            [InlineKeyboardButton(settings_label, callback_data="menu:main")],
        ]
    )
