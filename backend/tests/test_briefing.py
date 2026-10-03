import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from backend.app.services import briefing_generator as briefing
from backend.app.services.briefing_generator import build_sections


def story(article_id, published_timestamp, importance=8, category="AI Models"):
    return {
        "id": article_id,
        "title": f"AI story topic{article_id}",
        "source": "OpenAI",
        "url": f"https://example.com/{article_id}",
        "published": datetime.fromtimestamp(
            published_timestamp, timezone.utc
        ).isoformat(),
        "published_timestamp": published_timestamp,
        "category": category,
        "importance": importance,
        "career_relevance": 6,
        "ranking_score": float(importance),
        "ai_summary": f"Summary {article_id}",
        "why_it_matters": f"Impact {article_id}",
    }


class TestBriefingPersistence(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.paths = {
            "morning": root / "morning.json",
            "latest": root / "latest.json",
            "history": root / "history.json",
            "archive": root / "evening_briefings",
        }
        self.patches = [
            patch.object(briefing, "MORNING_BRIEFING_FILE", self.paths["morning"]),
            patch.object(briefing, "LATEST_BRIEFING_FILE", self.paths["latest"]),
            patch.object(briefing, "HISTORY_FILE", self.paths["history"]),
            patch.object(briefing, "EVENING_ARCHIVE_DIR", self.paths["archive"]),
            patch.object(briefing, "utc_now", return_value=datetime(2026, 9, 12, 9, tzinfo=timezone.utc)),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.tempdir.cleanup()

    def test_morning_persists_baseline_without_replacing_latest_evening(self):
        latest_evening = {
            "briefing_type": "evening",
            "generated_at": "2026-09-11T11:00:00+00:00",
            "stories": [],
        }
        self.paths["latest"].write_text(json.dumps(latest_evening), encoding="utf-8")
        candidate = story(1, 1_000, category="Developers & Tools")
        with patch.object(briefing, "get_candidates", return_value=[candidate]), \
            patch.object(briefing, "remove_duplicates", side_effect=lambda rows: rows), \
            patch.object(briefing, "rank_articles", side_effect=lambda rows: rows), \
            patch.object(briefing, "select_top_stories", side_effect=lambda rows, limit: rows[:limit]):
            result = briefing.build_morning_briefing(limit=1)

        morning = json.loads(self.paths["morning"].read_text(encoding="utf-8"))
        latest = json.loads(self.paths["latest"].read_text(encoding="utf-8"))
        self.assertEqual(result["briefing_type"], "morning")
        self.assertEqual(morning["generated_at"], "2026-09-12T09:00:00+00:00")
        self.assertEqual(latest, latest_evening)
        self.assertEqual(morning["stories"][0]["category"], "AI Tools")

    def test_morning_does_not_create_latest_evening_snapshot(self):
        candidate = story(1, 1_000)
        with patch.object(briefing, "get_candidates", return_value=[candidate]), \
            patch.object(briefing, "remove_duplicates", side_effect=lambda rows: rows), \
            patch.object(briefing, "rank_articles", side_effect=lambda rows: rows), \
            patch.object(briefing, "select_top_stories", side_effect=lambda rows, limit: rows[:limit]):
            briefing.build_morning_briefing(limit=1)

        self.assertTrue(self.paths["morning"].is_file())
        self.assertFalse(self.paths["latest"].exists())

    def test_history_appends_and_retains_last_thirty(self):
        self.paths["history"].write_text(
            json.dumps([{"type": "old", "n": index} for index in range(30)]),
            encoding="utf-8",
        )
        candidate = story(1, 1_000)
        with patch.object(briefing, "get_candidates", return_value=[candidate]), \
            patch.object(briefing, "remove_duplicates", side_effect=lambda rows: rows), \
            patch.object(briefing, "rank_articles", side_effect=lambda rows: rows), \
            patch.object(briefing, "select_top_stories", side_effect=lambda rows, limit: rows[:limit]):
            briefing.build_morning_briefing(limit=1)
        history = json.loads(self.paths["history"].read_text(encoding="utf-8"))
        self.assertEqual(len(history), 30)
        self.assertEqual(history[-1]["type"], "morning")

    def test_missing_baseline_produces_evening_status_without_morning_file(self):
        result = briefing.build_evening_briefing()
        self.assertEqual(result["briefing_type"], "evening")
        self.assertFalse(result["status"]["has_new_developments"])
        self.assertTrue(result["status"]["baseline_missing"])
        self.assertEqual(result["stories"], [])
        self.assertNotIn("morning.json", [path.name for path in Path(self.tempdir.name).iterdir()])

    def test_evening_selects_only_post_baseline_significant_nonbaseline_stories(self):
        baseline_timestamp = datetime(
            2026, 9, 12, 9, tzinfo=timezone.utc
        ).timestamp()
        morning = {
            "generated_at": "2026-09-12T09:00:00+00:00",
            "stories": [story(1, baseline_timestamp - 100)],
        }
        self.paths["morning"].write_text(json.dumps(morning), encoding="utf-8")
        candidates = [
            story(1, baseline_timestamp + 1, importance=10),
            story(2, baseline_timestamp + 100, importance=4),
            story(3, baseline_timestamp + 200, importance=8),
            story(4, baseline_timestamp - 100, importance=9),
        ]
        candidates[-1]["analyzed_at"] = "2026-09-12T09:05:00+00:00"
        with patch.object(briefing, "get_candidates", return_value=candidates), \
            patch.object(briefing, "remove_duplicates", side_effect=lambda rows: rows), \
            patch.object(briefing, "rank_articles", side_effect=lambda rows: rows), \
            patch.object(briefing, "select_top_stories", side_effect=lambda rows, limit: rows[:limit]):
            result = briefing.build_evening_briefing(limit=8)
        self.assertEqual([item["id"] for item in result["stories"]], [3])
        self.assertEqual(result["status"]["candidates_since_baseline"], 2)
        self.assertTrue(
            result["status"]["comparison_period"]["baseline_local"].startswith(
                "2026-09-12"
            )
        )

    def test_evening_rejects_a_stale_morning_baseline(self):
        old_evening = {
            "briefing_type": "evening",
            "generated_at": "2026-09-11T11:00:00+00:00",
            "stories": [story(7, 1_000)],
        }
        self.paths["morning"].write_text(
            json.dumps({
                "generated_at": "2026-09-11T10:31:30+00:00",
                "stories": [story(1, 1_000)],
            }),
            encoding="utf-8",
        )
        self.paths["latest"].write_text(json.dumps(old_evening), encoding="utf-8")
        with patch.object(briefing, "get_candidates") as candidates:
            result = briefing.build_evening_briefing()

        candidates.assert_not_called()
        self.assertFalse(result["status"]["has_new_developments"])
        self.assertTrue(result["status"]["baseline_missing"])
        self.assertIn("no current-day morning baseline", result["status"]["message"])
        self.assertEqual(result["stories"], [])
        self.assertEqual(json.loads(self.paths["latest"].read_text(encoding="utf-8")), old_evening)
        self.assertFalse(self.paths["archive"].exists())

    def test_evening_archives_the_previous_full_snapshot_before_replacement(self):
        baseline_timestamp = datetime(2026, 9, 12, 9, tzinfo=timezone.utc).timestamp()
        morning = {
            "generated_at": "2026-09-12T09:00:00+00:00",
            "stories": [story(1, baseline_timestamp - 100)],
        }
        previous_evening = {
            "briefing_type": "evening",
            "generated_at": "2026-09-11T11:00:00+00:00",
            "stories": [story(7, 1_000)],
        }
        self.paths["morning"].write_text(json.dumps(morning), encoding="utf-8")
        self.paths["latest"].write_text(json.dumps(previous_evening), encoding="utf-8")
        with patch.object(briefing, "get_candidates", return_value=[
            story(2, baseline_timestamp + 100, importance=3)
        ]), patch.object(briefing, "remove_duplicates", side_effect=lambda rows: rows), \
            patch.object(briefing, "rank_articles", side_effect=lambda rows: rows), \
            patch.object(briefing, "cluster_articles", side_effect=lambda rows: rows):
            result = briefing.build_evening_briefing()

        archives = list(self.paths["archive"].glob("*.json"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(json.loads(archives[0].read_text(encoding="utf-8")), previous_evening)
        self.assertEqual(result["briefing_type"], "evening")
        self.assertTrue(result["status"]["comparison_period"]["baseline_local"].startswith("2026-09-12"))
        self.assertNotEqual(
            json.loads(self.paths["latest"].read_text(encoding="utf-8")),
            previous_evening,
        )

    def test_evening_no_developments_repeats_baseline_with_explicit_message(self):
        baseline_timestamp = datetime(
            2026, 9, 12, 9, tzinfo=timezone.utc
        ).timestamp()
        morning = {
            "generated_at": "2026-09-12T09:00:00+00:00",
            "stories": [story(1, baseline_timestamp - 100)],
        }
        self.paths["morning"].write_text(json.dumps(morning), encoding="utf-8")
        with patch.object(briefing, "get_candidates", return_value=[story(2, baseline_timestamp + 100, importance=3)]), \
            patch.object(briefing, "remove_duplicates", side_effect=lambda rows: rows), \
            patch.object(briefing, "rank_articles", side_effect=lambda rows: rows):
            result = briefing.build_evening_briefing()
        self.assertFalse(result["status"]["has_new_developments"])
        self.assertIn("No major new AI developments since the morning briefing.", result["status"]["message"])
        self.assertEqual([item["id"] for item in result["stories"]], [1])

    def test_morning_selects_a_cluster_once(self):
        candidate = story(1, 1_000)
        clustered = dict(candidate)
        clustered.update({"cluster_id": "story-1", "member_ids": [1, 2], "member_count": 2})
        with patch.object(briefing, "get_candidates", return_value=[candidate]), \
            patch.object(briefing, "remove_duplicates", side_effect=lambda rows: rows), \
            patch.object(briefing, "rank_articles", side_effect=lambda rows: rows), \
            patch.object(briefing, "cluster_articles", return_value=[clustered]), \
            patch.object(briefing, "select_top_stories", side_effect=lambda rows, limit: rows[:limit]):
            result = briefing.build_morning_briefing(limit=8)
        self.assertEqual(len(result["stories"]), 1)
        self.assertEqual(result["stories"][0]["member_count"], 2)

    def test_robotics_has_dedicated_section_and_canonical_categories(self):
        sections = build_sections([
            {"id": 1, "category": "Robotics"},
            {"id": 2, "category": "Developers & Tools"},
        ])
        self.assertEqual([item["id"] for item in sections["robotics"]], [1])
        self.assertEqual([item["id"] for item in sections["developers_tools"]], [2])


class TestBriefingCliDispatch(unittest.TestCase):

    def run_cli(self, flag):
        import backend.app.main as main_module
        with patch.object(main_module, "initialize_database"), \
            patch.object(main_module, "build_briefing", return_value={}) as build, \
            patch.object(sys, "argv", ["python -m backend.app.main", flag]):
            main_module.main()
        return build

    def test_morning_dispatch(self):
        self.assertEqual(self.run_cli("--morning").call_args.args, ("morning",))

    def test_evening_dispatch(self):
        self.assertEqual(self.run_cli("--evening").call_args.args, ("evening",))


if __name__ == "__main__":
    unittest.main()
