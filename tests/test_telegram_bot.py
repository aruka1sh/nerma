"""Comprehensive unit and integration tests for the NERMA AI Telegram bot and search pipeline.

Compatible with both `python -m unittest tests/test_telegram_bot.py -v` and `pytest`.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from docx import Document

from gpt_researcher.retrievers.duckduckgo.duckduckgo import Duckduckgo
from gpt_researcher.skills.deep_research import DeepResearchSkill
import telegram_bot.bot as bot_module
import telegram_bot.exporter as exporter_module
from telegram_bot.bot import (
    ACTIVE_REQUESTS,
    SAVED_BUNDLES,
    callback_handler,
    run_research,
)
from telegram_bot.config import (
    RequestState,
    UserStore,
)
from telegram_bot.diagnostics import redact_secrets
from telegram_bot.exporter import (
    build_result_bundle,
    export_bundle,
    format_metrics_block,
    is_refusal_or_empty_report,
)
from telegram_bot.query_utils import (
    extract_search_intent,
    is_matching_domain,
)


def _make_fake_update_and_context(user_id: int = 101, chat_id: int = 202, text: str = ""):
    """Helper to build mocked Telegram Update and Context objects."""
    status_msg = MagicMock()
    status_msg.edit_text = AsyncMock()
    status_msg.chat_id = chat_id

    message = MagicMock()
    message.text = text
    message.chat_id = chat_id
    message.reply_text = AsyncMock(return_value=status_msg)

    user = SimpleNamespace(id=user_id, first_name="TestUser")
    chat = SimpleNamespace(id=chat_id)

    update = SimpleNamespace(
        effective_user=user,
        effective_chat=chat,
        message=message,
        callback_query=None,
    )

    bot = MagicMock()
    bot.send_chat_action = AsyncMock()
    bot.send_message = AsyncMock()
    bot.send_document = AsyncMock()

    context = SimpleNamespace(bot=bot, args=[])
    return update, context, status_msg


class TestNermaTelegramBotPipeline(unittest.IsolatedAsyncioTestCase):
    """Automated test suite covering all 12 required diagnostic, search, state, and export scenarios."""

    def setUp(self) -> None:
        ACTIVE_REQUESTS.clear()
        SAVED_BUNDLES.clear()

    # 1. Short search query extraction from a long conversational prompt
    def test_01_short_query_extraction_from_long_prompt(self) -> None:
        long_prompt = (
            "Кратко сравни TF-IDF и Word2Vec для классификации текста. "
            "Для каждого метода укажи принцип работы, одно преимущество и одно ограничение. "
            "Добавь две ссылки на первоисточники или официальную документацию и поясни, "
            "какое утверждение подтверждает каждый источник. Ответ дай на русском языке."
        )
        intent = extract_search_intent(long_prompt)

        self.assertLessEqual(len(intent.clean_query.split()), 12)
        self.assertIn("TF-IDF", intent.clean_query)
        self.assertIn("Word2Vec", intent.clean_query)
        self.assertNotIn("Ответ дай на русском языке", intent.clean_query)
        self.assertNotIn("Добавь две ссылки", intent.clean_query)
        self.assertTrue(intent.requires_official_source)
        self.assertIn("scikit-learn.org", intent.official_domains)
        self.assertIn("arxiv.org", intent.official_domains)
        self.assertEqual(intent.region, "ru-ru")

    # 2. Explicit site:domain and subdomain validation via urlparse
    def test_02_explicit_site_domain_and_subdomains(self) -> None:
        prompt = (
            "Search using this exact query: site:microsoft.com Microsoft founded 1975 Bill Gates Paul Allen. "
            "Who founded Microsoft and in what year? Cite a direct official source URL from microsoft.com "
            "or its subdomains. Answer in two sentences."
        )
        intent = extract_search_intent(prompt)
        self.assertEqual(intent.explicit_site_domains, ["microsoft.com"])
        self.assertIn("microsoft.com", intent.official_domains)
        self.assertNotIn("Answer in two sentences", intent.clean_query)
        self.assertNotIn("Search using this exact query", intent.clean_query)

        # Strict hostname validation via urlparse (not substring search!)
        self.assertTrue(is_matching_domain("https://www.microsoft.com/en-us/about", "microsoft.com"))
        self.assertTrue(is_matching_domain("https://news.microsoft.com/facts-about-microsoft/", "microsoft.com"))
        self.assertFalse(is_matching_domain("https://evil-microsoft.com/fake", "microsoft.com"))
        self.assertFalse(is_matching_domain("https://example.com/redirect?url=https://microsoft.com", "microsoft.com"))

        retriever = Duckduckgo(prompt, query_domains=["microsoft.com"])
        fake_ddgs_results = [
            {
                "title": "Fake Substring Site",
                "href": "https://notmicrosoft.com/history",
                "body": "Should be filtered out because hostname != microsoft.com",
            },
            {
                "title": "Facts About Microsoft",
                "href": "https://news.microsoft.com/facts-about-microsoft/",
                "body": "Microsoft was founded on April 4, 1975 by Bill Gates and Paul Allen.",
            },
        ]
        with patch.object(retriever, "_run_ddgs_text", return_value=fake_ddgs_results):
            results = retriever.search(max_results=5)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["href"], "https://news.microsoft.com/facts-about-microsoft/")
            self.assertEqual(retriever.last_search_meta["before_filter"], 2)
            self.assertEqual(retriever.last_search_meta["after_filter"], 1)

    # 3. Official source request when official domain IS found
    def test_03_official_source_request_when_domain_found(self) -> None:
        query = (
            "В каком году вышла первая версия Python и кто создал этот язык? "
            "Укажи ссылку на официальный источник python.org, подтверждающий ответ."
        )
        intent = extract_search_intent(query)
        self.assertIn("python.org", intent.official_domains)

        search_results = [
            {
                "title": "General Python FAQ — Python 3 documentation",
                "href": "https://docs.python.org/3/faq/general.html",
                "body": "Python was created by Guido van Rossum and first released in February 1991 (0.9.0).",
            },
            {
                "title": "Python (programming language) — Wikipedia",
                "href": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                "body": "Python was conceived in the late 1980s by Guido van Rossum.",
            },
        ]
        scraped_sources = [
            {
                "url": "https://docs.python.org/3/faq/general.html",
                "title": "General Python FAQ — Python 3 documentation",
                "raw_content": "Guido van Rossum created Python in 1989 and published version 0.9.0 in February 1991.",
            }
        ]
        raw_report = (
            "Язык программирования Python был создан Гвидо ван Россумом (Guido van Rossum), "
            "а первая публичная версия (0.9.0) вышла в феврале 1991 года [1]."
        )

        bundle = build_result_bundle(
            request_id="req003",
            query=query,
            mode="quick_search",
            raw_report=raw_report,
            search_results=search_results,
            scraped_sources=scraped_sources,
            required_official_domains=intent.official_domains,
            language="russian",
        )

        self.assertEqual(bundle.state, RequestState.COMPLETED)
        self.assertIn("python.org", bundle.official_domains_found)
        self.assertEqual(bundle.official_source_warning, "")
        self.assertTrue(bundle.sources[0].is_official)
        self.assertIn("https://docs.python.org/3/faq/general.html", bundle.full_markdown)

    # 4. Official source request when official domain is NOT found
    def test_04_official_source_request_when_domain_not_found(self) -> None:
        query = "Who founded Microsoft, and in what year? Cite an official microsoft.com source."
        intent = extract_search_intent(query)
        self.assertIn("microsoft.com", intent.official_domains)

        search_results = [
            {
                "title": "Microsoft — Wikipedia",
                "href": "https://en.wikipedia.org/wiki/Microsoft",
                "body": "Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975.",
            }
        ]
        raw_report = "Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975 [1]."

        bundle = build_result_bundle(
            request_id="req004",
            query=query,
            mode="quick_search",
            raw_report=raw_report,
            search_results=search_results,
            scraped_sources=[],
            required_official_domains=intent.official_domains,
            language="english",
        )

        self.assertEqual(bundle.state, RequestState.PARTIAL)
        self.assertEqual(bundle.official_domains_found, [])
        self.assertIn("microsoft.com", bundle.official_source_warning)
        self.assertIn("microsoft.com", bundle.full_markdown)
        self.assertEqual(bundle.failure_reason, "official_source_missing")

    # 5. Empty search / refusal is NOT marked completed and NOT exported
    async def test_05_empty_search_refusal_not_marked_completed_and_no_export(self) -> None:
        refusal_ru = (
            "Основываясь на предоставленных результатах поиска, ответить на ваш вопрос невозможно, "
            "так как результаты поиска отсутствуют (список пуст)."
        )
        self.assertTrue(is_refusal_or_empty_report(refusal_ru))

        bundle = build_result_bundle(
            request_id="req005",
            query="В каком году вышла первая версия Python?",
            mode="quick_search",
            raw_report=refusal_ru,
            search_results=[],
            scraped_sources=[],
            visited_urls=[],
            language="russian",
        )

        self.assertEqual(bundle.state, RequestState.FAILED)
        self.assertFalse(bundle.is_exportable)
        exported = await export_bundle(bundle, ["docx", "md", "pdf"])
        self.assertEqual(exported, {})

        update, context, status_msg = _make_fake_update_and_context(
            user_id=505, text="Вопрос с пустой выдачей"
        )
        with patch("telegram_bot.bot.GPTResearcher") as mock_gptr_cls:
            instance = MagicMock()
            instance.cfg = SimpleNamespace(
                smart_llm_model="gemini",
                smart_llm_provider="google_genai",
                smart_token_limit=1000,
                llm_kwargs={},
                language="russian",
            )
            instance.quick_search = AsyncMock(return_value=[])
            instance.get_source_urls = MagicMock(return_value=[])
            instance.get_costs = MagicMock(return_value=0.0)
            instance.search_telemetry = [
                {
                    "query_clean": "Вопрос с пустой выдачей",
                    "providers_used": ["ddgs"],
                    "before_filter": 0,
                    "after_filter": 0,
                    "retries": 1,
                    "fallback_used": True,
                    "errors": [],
                    "failure_reason": "empty_search_results",
                }
            ]
            mock_gptr_cls.return_value = instance

            res_bundle = await run_research(
                update, context, "Вопрос с пустой выдачей", override_report_type="quick_search"
            )
            self.assertIsNotNone(res_bundle)
            self.assertEqual(res_bundle.state, RequestState.FAILED)
            context.bot.send_document.assert_not_called()
            context.bot.send_message.assert_not_called()
            last_edit_call = status_msg.edit_text.call_args_list[-1]
            edited_text = last_edit_call.args[0]
            self.assertNotIn("Готовый отчёт", edited_text)
            self.assertTrue("0 результатов" in edited_text or "не вернул источников" in edited_text)

    # 6. Citations [1], [2] match bibliography in both chat and DOCX/MD
    async def test_06_citations_match_bibliography_in_chat_and_docx_md(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch.object(exporter_module, "OUTPUTS_DIR", tmp_path):
                search_results = [
                    {
                        "title": "Facts About Microsoft",
                        "href": "https://news.microsoft.com/facts-about-microsoft/",
                        "body": "Founded on April 4, 1975 by Bill Gates and Paul Allen.",
                    },
                    {
                        "title": "About Microsoft",
                        "href": "https://www.microsoft.com/en-us/about",
                        "body": "Microsoft enables digital transformation.",
                    },
                ]
                raw_report = (
                    "Компания Microsoft была основана 4 апреля 1975 года Биллом Гейтсом и Полом Алленом [1] [99]. "
                    "Штаб-квартира расположена в Редмонде [2]. "
                    "См. также [выдуманный сайт](https://hallucinated.example.com/fake)."
                )

                bundle = build_result_bundle(
                    request_id="req006",
                    query="Кто основал Microsoft?",
                    mode="quick_search",
                    raw_report=raw_report,
                    search_results=search_results,
                    scraped_sources=[
                        {
                            "url": "https://news.microsoft.com/facts-about-microsoft/",
                            "title": "Facts About Microsoft",
                            "raw_content": "Microsoft was founded on April 4, 1975 by Bill Gates and Paul Allen.",
                        }
                    ],
                    required_official_domains=["microsoft.com"],
                    language="russian",
                )

                self.assertEqual(bundle.state, RequestState.COMPLETED)
                self.assertNotIn("[99]", bundle.body_markdown)
                self.assertNotIn("hallucinated.example.com", bundle.full_markdown)
                self.assertIn("[1]", bundle.body_markdown)
                self.assertIn("[2]", bundle.body_markdown)
                self.assertIn(
                    "[1] [Facts About Microsoft](https://news.microsoft.com/facts-about-microsoft/)",
                    bundle.full_markdown,
                )
                self.assertIn(
                    "[2] [About Microsoft](https://www.microsoft.com/en-us/about)",
                    bundle.full_markdown,
                )

                exported = await export_bundle(bundle, ["md", "docx"])
                self.assertIn("md", exported)
                self.assertTrue(exported["md"].exists())
                self.assertIn("docx", exported)
                self.assertTrue(exported["docx"].exists())

                md_text = exported["md"].read_text(encoding="utf-8")
                self.assertIn("https://news.microsoft.com/facts-about-microsoft/", md_text)
                self.assertIn("https://www.microsoft.com/en-us/about", md_text)

                doc = Document(str(exported["docx"]))
                docx_full_text = "\n".join(p.text for p in doc.paragraphs)
                self.assertIn("https://news.microsoft.com/facts-about-microsoft/", docx_full_text)
                self.assertIn("https://www.microsoft.com/en-us/about", docx_full_text)

    # 7. Accuracy of the 3 source counters (found, read, cited)
    def test_07_source_counters_found_read_cited_accuracy(self) -> None:
        search_results = [
            {"title": "Source 1", "href": "https://docs.python.org/3/faq/general.html", "body": "Snippet 1"},
            {"title": "Source 2", "href": "https://www.python.org/doc/essays/blurb/", "body": "Snippet 2"},
            {"title": "Source 3", "href": "https://en.wikipedia.org/wiki/Python", "body": "Snippet 3"},
            {"title": "Source 4", "href": "https://realpython.com/python-history/", "body": "Snippet 4"},
        ]
        scraped_sources = [
            {
                "url": "https://docs.python.org/3/faq/general.html",
                "title": "Source 1",
                "raw_content": "Full page text 1 with detailed history of Python.",
            },
            {
                "url": "https://www.python.org/doc/essays/blurb/",
                "title": "Source 2",
                "raw_content": "Full page text 2 executive summary.",
            },
        ]
        raw_report = "Python создал Гвидо ван Россум в 1991 году [1]."

        bundle = build_result_bundle(
            request_id="req007",
            query="Кто создал Python?",
            mode="research_report",
            raw_report=raw_report,
            search_results=search_results,
            scraped_sources=scraped_sources,
            language="russian",
        )

        self.assertEqual(bundle.unique_urls_found, 4)
        self.assertEqual(bundle.pages_read_successfully, 2)
        self.assertEqual(bundle.sources_cited, 1)

        metrics_text = format_metrics_block(bundle)
        self.assertIn("Найдено уникальных URL: 4", metrics_text)
        self.assertIn("Успешно прочитано страниц: 2", metrics_text)
        self.assertIn("Процитировано источников: 1", metrics_text)

    # 8. Mode switching, user isolation, and in-flight settings snapshot
    def test_08_mode_switching_and_multi_user_settings_isolation(self) -> None:
        store = UserStore()
        u1 = store.get(1001)
        u2 = store.get(2002)

        u1.report_type = "deep"
        u1.language = "english"
        u1.tone = "Analytical"

        self.assertEqual(u2.report_type, "quick_search")
        self.assertEqual(u2.language, "russian")
        self.assertEqual(u2.tone, "Objective")

        in_flight_snapshot = store.snapshot(1001)
        u1.report_type = "outline_report"
        u1.language = "kazakh"
        u1.export_formats.append("pdf")

        self.assertEqual(in_flight_snapshot.report_type, "deep")
        self.assertEqual(in_flight_snapshot.language, "english")
        self.assertNotIn("pdf", in_flight_snapshot.export_formats)

    # 9. Button export uses saved ResearchResultBundle without re-running
    async def test_09_quick_export_button_uses_saved_bundle_without_rerunning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch.object(exporter_module, "OUTPUTS_DIR", tmp_path):
                bundle = build_result_bundle(
                    request_id="quick999",
                    query="Кто создал Python?",
                    mode="quick_search",
                    raw_report="Python был создан Гвидо ван Россумом в 1991 году [1].",
                    search_results=[
                        {
                            "title": "Python FAQ",
                            "href": "https://docs.python.org/3/faq/general.html",
                            "body": "Guido van Rossum created Python.",
                        }
                    ],
                    scraped_sources=[
                        {
                            "url": "https://docs.python.org/3/faq/general.html",
                            "title": "Python FAQ",
                            "raw_content": "Guido van Rossum created Python in 1991.",
                        }
                    ],
                    language="russian",
                )
                SAVED_BUNDLES["quick999"] = bundle

                query_cb = MagicMock()
                query_cb.data = "export:quick999"
                query_cb.from_user = SimpleNamespace(id=777)
                query_cb.answer = AsyncMock()
                query_cb.message = MagicMock()
                query_cb.message.chat_id = 888
                query_cb.message.reply_text = AsyncMock()

                update = SimpleNamespace(callback_query=query_cb)
                bot = MagicMock()
                bot.send_chat_action = AsyncMock()
                bot.send_document = AsyncMock()
                context = SimpleNamespace(bot=bot)

                with patch("telegram_bot.bot.GPTResearcher") as mock_gptr_cls:
                    await callback_handler(update, context)
                    mock_gptr_cls.assert_not_called()
                    self.assertEqual(bot.send_document.call_count, 2)

    # 10. Cancelling an in-flight request stops execution & prevents report
    async def test_10_cancel_running_request_stops_delivery(self) -> None:
        update, context, status_msg = _make_fake_update_and_context(
            user_id=901, text="Длинный запрос для отмены"
        )

        search_started = asyncio.Event()

        async def _slow_quick_search(*args, **kwargs):
            search_started.set()
            await asyncio.sleep(10.0)
            return [{"title": "Late", "href": "https://example.com", "body": "Late body"}]

        with patch("telegram_bot.bot.GPTResearcher") as mock_gptr_cls:
            instance = MagicMock()
            instance.cfg = SimpleNamespace(
                smart_llm_model="gemini",
                smart_llm_provider="google_genai",
                smart_token_limit=1000,
                llm_kwargs={},
                language="russian",
            )
            instance.quick_search = AsyncMock(side_effect=_slow_quick_search)
            instance.get_source_urls = MagicMock(return_value=[])
            instance.get_costs = MagicMock(return_value=0.0)
            mock_gptr_cls.return_value = instance

            task = asyncio.create_task(
                run_research(update, context, "Длинный запрос для отмены", override_report_type="quick_search")
            )
            await asyncio.wait_for(search_started.wait(), timeout=2.0)

            self.assertEqual(len(ACTIVE_REQUESTS), 1)
            req_id = next(iter(ACTIVE_REQUESTS.keys()))

            cancel_cb = MagicMock()
            cancel_cb.data = f"cancel:{req_id}"
            cancel_cb.from_user = SimpleNamespace(id=901)
            cancel_cb.answer = AsyncMock()
            cancel_cb.edit_message_text = AsyncMock()
            cancel_update = SimpleNamespace(callback_query=cancel_cb)

            await callback_handler(cancel_update, context)
            result = await task

            self.assertIsNone(result)
            self.assertNotIn(req_id, ACTIVE_REQUESTS)
            context.bot.send_message.assert_not_called()
            context.bot.send_document.assert_not_called()

    # 11. Search or scrape timeout handled cleanly within bounded time
    async def test_11_search_or_scrape_timeout_handled_cleanly(self) -> None:
        update, context, status_msg = _make_fake_update_and_context(
            user_id=911, text="Запрос с зависшим поиском"
        )

        async def _hung_search(*args, **kwargs):
            await asyncio.sleep(5.0)
            return []

        with (
            patch.object(bot_module, "SEARCH_TIMEOUT_SEC", 0.2),
            patch.object(bot_module, "get_request_timeout_sec", return_value=0.5),
            patch("telegram_bot.bot.GPTResearcher") as mock_gptr_cls,
        ):
            instance = MagicMock()
            instance.cfg = SimpleNamespace(
                smart_llm_model="gemini",
                smart_llm_provider="google_genai",
                smart_token_limit=1000,
                llm_kwargs={},
                language="russian",
            )
            instance.quick_search = AsyncMock(side_effect=_hung_search)
            instance.get_source_urls = MagicMock(return_value=[])
            instance.get_costs = MagicMock(return_value=0.0)
            mock_gptr_cls.return_value = instance

            res = await run_research(
                update, context, "Запрос с зависшим поиском", override_report_type="quick_search"
            )
            self.assertIsNone(res)
            context.bot.send_document.assert_not_called()
            last_edit = status_msg.edit_text.call_args_list[-1].args[0]
            self.assertIn("Превышено допустимое время ожидания", last_edit)

    # 12. Deep Research sub-queries stay anchored (no drift into 3.15-3.16)
    async def test_12_deep_research_subqueries_stay_anchored_no_version_drift(self) -> None:
        fake_researcher = MagicMock()
        fake_researcher.query = (
            "История создания первой версии языка Python (0.9.0 и 1.0): "
            "предпосылки проекта ABC в CWI, ключевые архитектурные решения Гвидо ван Россума."
        )
        fake_researcher.cfg = SimpleNamespace(
            strategic_llm_provider="google_genai",
            strategic_llm_model="gemini-2.5-flash",
            reasoning_effort="low",
            llm_kwargs={},
        )
        fake_researcher.retrievers = []
        fake_researcher.visited_urls = set()

        skill = DeepResearchSkill(fake_researcher)

        self.assertTrue(skill._is_drifted_query("Python release cycle PEP end of life schedule Python 3.15 3.16"))
        self.assertTrue(skill._is_drifted_query("Python 3.16 release schedule PEP 826"))
        self.assertFalse(skill._is_drifted_query("Guido van Rossum CWI ABC language Python 0.9.0 1991"))

        mock_llm_json = (
            '[{"query": "Guido van Rossum Python 0.9.0 1991 CWI ABC history", "researchGoal": "Origin of Python 0.9.0"}, '
            '{"query": "Python release cycle PEP end of life schedule Python 3.15 3.16", "researchGoal": "Modern versions"}]'
        )
        with patch(
            "gpt_researcher.skills.deep_research.create_chat_completion",
            new=AsyncMock(return_value=mock_llm_json),
        ):
            queries = await skill.generate_search_queries("Subcontext", num_queries=2)
            self.assertEqual(len(queries), 1)
            self.assertIn("0.9.0", queries[0]["query"])
            self.assertNotIn("3.15", queries[0]["query"])

    def test_13_secret_redaction_in_diagnostics(self) -> None:
        raw_log = (
            "Error calling https://api.telegram.org/bot1234567890:AAFakeTestTokenForUnitTestingOnly123456789/getUpdates "
            "with key AIzaFakeTestKeyForUnitTestingOnly1234567890"
        )
        redacted = redact_secrets(raw_log)
        self.assertNotIn("1234567890:AAF", redacted)
        self.assertNotIn("AIzaFake", redacted)
        self.assertIn("[REDACTED_SECRET]", redacted)


if __name__ == "__main__":
    unittest.main()
