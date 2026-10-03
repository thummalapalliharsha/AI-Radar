import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch
from urllib.parse import urlencode

from backend.app.api.server import app, evening_briefing, morning_briefing


async def get_response(path: str, params: dict[str, Any] | None = None) -> tuple[int, Any]:
    query_string = urlencode(params or {}).encode("ascii")
    response_messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        response_messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": query_string,
        "root_path": "",
        "headers": [(b"host", b"localhost")],
        "client": ("testclient", 12345),
        "server": ("localhost", 80),
    }
    await app(scope, receive, send)

    status = next(
        message["status"]
        for message in response_messages
        if message["type"] == "http.response.start"
    )
    body = b"".join(
        message.get("body", b"")
        for message in response_messages
        if message["type"] == "http.response.body"
    )
    return status, json.loads(body)


class TestReadOnlyApi(unittest.IsolatedAsyncioTestCase):
    async def test_health_returns_exact_response(self):
        status, body = await get_response("/api/health")

        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ok"})

    async def test_stories_24h_returns_only_the_required_shape(self):
        status, body = await get_response(
            "/api/stories",
            {"window": "24h"},
        )

        self.assertEqual(status, 200)
        self.assertIsInstance(body, list)
        if not body:
            self.assertEqual(body, [])
            return

        expected_fields = {
            "headline",
            "summary",
            "category",
            "source",
            "published_at",
            "url",
            "why_it_matters",
        }
        for story in body:
            self.assertEqual(set(story), expected_fields)
            for field in expected_fields - {"why_it_matters"}:
                self.assertIsInstance(story[field], str)
            self.assertTrue(
                isinstance(story["why_it_matters"], str)
                or story["why_it_matters"] is None
            )
            parsed_timestamp = datetime.fromisoformat(story["published_at"])
            self.assertIsNotNone(parsed_timestamp.tzinfo)

    async def test_stale_evening_briefing_is_available_and_marked_historical(self):
        stale_generated_at = (
            datetime.now(timezone.utc) - timedelta(days=2)
        ).isoformat()
        briefing = {
            "briefing_type": "evening",
            "generated_at": stale_generated_at,
            "status": {
                "message": "Historical briefing summary.",
                "comparison_period": {
                    "from": stale_generated_at,
                    "baseline_local": stale_generated_at,
                },
            },
            "stories": [],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "latest.json"
            path.write_text(json.dumps(briefing), encoding="utf-8")
            with patch("backend.app.api.server.LATEST_BRIEFING_FILE", path):
                response = evening_briefing()

        self.assertTrue(response["available"])
        self.assertFalse(response["is_current_day"])
        self.assertFalse(response["is_current_comparison"])
        self.assertEqual(response["summary"], "Historical briefing summary.")

    async def test_current_evening_comparison_is_marked_current(self):
        generated_at = datetime.now(timezone.utc).isoformat()
        briefing = {
            "briefing_type": "evening",
            "generated_at": generated_at,
            "status": {
                "message": "Current briefing summary.",
                "comparison_period": {
                    "from": generated_at,
                    "baseline_local": generated_at,
                },
            },
            "stories": [],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "latest.json"
            path.write_text(json.dumps(briefing), encoding="utf-8")
            with patch("backend.app.api.server.LATEST_BRIEFING_FILE", path):
                response = evening_briefing()

        self.assertTrue(response["available"])
        self.assertTrue(response["is_current_day"])
        self.assertTrue(response["is_current_comparison"])

    async def test_evening_with_missing_baseline_is_not_misrepresented_as_historical(self):
        generated_at = datetime.now(timezone.utc).isoformat()
        briefing = {
            "briefing_type": "evening",
            "generated_at": generated_at,
            "status": {
                "message": "No current-day morning baseline exists.",
                "baseline_missing": True,
            },
            "stories": [],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "latest.json"
            path.write_text(json.dumps(briefing), encoding="utf-8")
            with patch("backend.app.api.server.LATEST_BRIEFING_FILE", path):
                response = evening_briefing()

        self.assertTrue(response["available"])
        self.assertTrue(response["baseline_missing"])
        self.assertTrue(response["is_current_day"])
        self.assertFalse(response["is_current_comparison"])
        self.assertIsNone(response["baseline_local_date"])

    async def test_current_day_briefing_state_matrix(self):
        fixed_now = datetime(2026, 9, 30, 12, tzinfo=timezone.utc)

        class FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)

        def snapshot(kind, generated_at, baseline_at=None):
            comparison_period = None
            if baseline_at is not None:
                comparison_period = {
                    "from": baseline_at.isoformat(),
                    "baseline_local": baseline_at.isoformat(),
                }
            return {
                "briefing_type": kind,
                "generated_at": generated_at.isoformat(),
                "status": {
                    "message": f"{kind} snapshot",
                    "comparison_period": comparison_period,
                },
                "stories": [],
            }

        current_morning_at = fixed_now - timedelta(hours=5)
        current_evening_at = fixed_now - timedelta(hours=1)
        yesterday = fixed_now - timedelta(days=1)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            morning_path = root / "morning.json"
            evening_path = root / "evening.json"
            with (
                patch("backend.app.api.server.MORNING_BRIEFING_FILE", morning_path),
                patch("backend.app.api.server.LATEST_BRIEFING_FILE", evening_path),
                patch("backend.app.api.server.datetime", FixedDateTime),
            ):
                morning_path.write_text(
                    json.dumps(snapshot("morning", current_morning_at)),
                    encoding="utf-8",
                )
                evening_path.write_text(
                    json.dumps(
                        snapshot("evening", current_evening_at, current_morning_at)
                    ),
                    encoding="utf-8",
                )
                state_a_morning = morning_briefing()
                state_a_evening = evening_briefing()
                self.assertTrue(state_a_morning["is_current_day"])
                self.assertTrue(state_a_evening["is_current_day"])
                self.assertTrue(state_a_evening["is_current_comparison"])

                evening_path.unlink()
                state_b_morning = morning_briefing()
                state_b_evening = evening_briefing()
                self.assertTrue(state_b_morning["is_current_day"])
                self.assertEqual(state_b_evening, {"available": False})

                morning_path.write_text(
                    json.dumps(snapshot("morning", yesterday - timedelta(hours=2))),
                    encoding="utf-8",
                )
                evening_path.write_text(
                    json.dumps(
                        snapshot(
                            "evening",
                            yesterday,
                            yesterday - timedelta(hours=2),
                        )
                    ),
                    encoding="utf-8",
                )
                state_c_morning = morning_briefing()
                state_c_evening = evening_briefing()
                self.assertFalse(state_c_morning["is_current_day"])
                self.assertFalse(state_c_evening["is_current_day"])
                self.assertFalse(state_c_evening["is_current_comparison"])

                morning_path.write_text(
                    json.dumps(snapshot("morning", current_morning_at)),
                    encoding="utf-8",
                )
                evening_path.write_text(
                    json.dumps(snapshot("evening", current_evening_at, yesterday)),
                    encoding="utf-8",
                )
                state_d_morning = morning_briefing()
                state_d_evening = evening_briefing()
                self.assertTrue(state_d_morning["is_current_day"])
                self.assertTrue(state_d_evening["is_current_day"])
                self.assertFalse(state_d_evening["is_current_comparison"])

    async def test_search_returns_grounded_answer_and_sanitized_sources(self):
        article = {
            "title": "AI agent release",
            "source": "AI Radar Source",
            "category": "AI Agents",
            "published": "2026-09-30T08:00:00+00:00",
            "url": "https://example.com/agent",
            "summary": "A verified summary.",
            "why_it_matters": "A verified impact.",
            "similarity": 0.91,
            "distance": 0.09,
            "id": 123,
        }
        with (
            patch(
                "backend.app.api.server.check_ollama",
                return_value={
                    "online": True,
                    "has_embedding": True,
                    "has_llm": True,
                },
            ),
            patch("backend.app.api.server.classify_search_intent", return_value=True),
            patch("backend.app.api.server.search_vector_store", return_value=[article]),
            patch(
                "backend.app.api.server.validate_retrieved_articles",
                return_value=[article],
            ),
            patch(
                "backend.app.api.server.generate_grounded_answer",
                return_value="A grounded answer. [1]",
            ),
        ):
            status, body = await get_response(
                "/api/search",
                {"q": "latest AI agents"},
            )

        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "answered")
        self.assertEqual(body["answer"], "A grounded answer. [1]")
        self.assertEqual(
            body["sources"],
            [
                {
                    "title": "AI agent release",
                    "source": "AI Radar Source",
                    "category": "AI Agents",
                    "published": "2026-09-30T08:00:00+00:00",
                    "url": "https://example.com/agent",
                    "summary": "A verified summary.",
                    "why_it_matters": "A verified impact.",
                }
            ],
        )

    async def test_search_rejects_unrelated_query_before_retrieval(self):
        with (
            patch(
                "backend.app.api.server.check_ollama",
                return_value={
                    "online": True,
                    "has_embedding": True,
                    "has_llm": True,
                },
            ),
            patch("backend.app.api.server.classify_search_intent", return_value=False),
            patch("backend.app.api.server.search_vector_store") as search_vectors,
        ):
            status, body = await get_response(
                "/api/search",
                {"q": "capital of France"},
            )

        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "unrelated")
        search_vectors.assert_not_called()

    async def test_search_reports_no_relevant_information_without_an_answer(self):
        with (
            patch(
                "backend.app.api.server.check_ollama",
                return_value={
                    "online": True,
                    "has_embedding": True,
                    "has_llm": True,
                },
            ),
            patch("backend.app.api.server.classify_search_intent", return_value=True),
            patch("backend.app.api.server.search_vector_store", return_value=[]),
        ):
            status, body = await get_response(
                "/api/search",
                {"q": "AI underwater agriculture"},
            )

        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "no_relevant_information")
        self.assertNotIn("answer", body)
        self.assertNotIn("sources", body)

    async def test_search_rejects_whitespace_only_query(self):
        status, body = await get_response(
            "/api/search",
            {"q": "   "},
        )

        self.assertEqual(status, 422)
        self.assertIn("detail", body)


if __name__ == "__main__":
    unittest.main()
