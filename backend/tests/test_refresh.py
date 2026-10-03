import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.app.services.news_collector import (
    _collect_news_with_status,
    _save_news_with_status,
)
from backend.app.services.refresh_radar import (
    get_last_successful_refresh,
    refresh_radar,
)


class TestRadarRefresh(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.state_path = Path(self.temp_dir.name) / "refresh_state.json"
        self.state_path_patcher = patch(
            "backend.app.services.refresh_radar.REFRESH_STATE_FILE",
            self.state_path,
        )
        self.state_path_patcher.start()
        self.addCleanup(self.state_path_patcher.stop)

    def stats(self, total=100, ai=20, last="collected"):
        return {
            "total_collected": total,
            "ai_news_count": ai,
            "last_collected": last,
        }

    def collection(
        self,
        new_inserted=0,
        collected_total=0,
        feeds_succeeded=20,
        feed_failures=None,
        save_failures=None,
    ):
        return {
            "new_inserted": new_inserted,
            "collected_total": collected_total,
            "feeds_total": 20,
            "feeds_succeeded": feeds_succeeded,
            "feed_failures": feed_failures or [],
            "save_failures": save_failures or [],
        }

    def processing(self, processed=0, failed=0, remaining=0):
        return {
            "processed": processed,
            "successful": processed - failed,
            "failed": failed,
            "remaining": remaining,
        }

    def indexing(self, pending=0, failed=0):
        return {
            "indexed": 20,
            "added": 0,
            "updated": 0,
            "removed": 0,
            "unchanged": 20,
            "failed": failed,
            "pending": pending,
        }

    @patch("backend.app.services.refresh_radar.get_news_stats")
    @patch("backend.app.services.refresh_radar.process_articles_batch")
    @patch("backend.app.services.refresh_radar.run_collection")
    @patch("backend.app.services.refresh_radar.sync_embeddings")
    def test_refresh_processes_only_one_bounded_newest_batch(
        self, sync_embeddings, run_collection, process_articles_batch, get_news_stats
    ):
        get_news_stats.side_effect = [
            self.stats(),
            self.stats(total=102, ai=21),
        ]
        run_collection.return_value = self.collection(
            new_inserted=2, collected_total=5
        )
        process_articles_batch.return_value = self.processing(
            processed=10, remaining=746
        )
        sync_embeddings.return_value = self.indexing()

        result = refresh_radar(recent_days=7, batch_size=10)

        run_collection.assert_called_once_with(recent_days=7)
        process_articles_batch.assert_called_once_with(
            limit=10,
            recent_days=7,
            newest_first=True,
        )
        sync_embeddings.assert_called_once_with(verbose=False)
        self.assertTrue(result["success"])
        self.assertEqual(result["new_articles"], 2)
        self.assertEqual(result["new_verified_ai_stories"], 1)
        self.assertEqual(result["remaining"], 746)
        self.assertIsNotNone(result["last_successful_refresh"])
        self.assertEqual(
            get_last_successful_refresh(),
            result["last_successful_refresh"],
        )

    @patch("backend.app.services.refresh_radar.get_news_stats")
    @patch("backend.app.services.refresh_radar.process_articles_batch")
    @patch("backend.app.services.refresh_radar.run_collection")
    @patch("backend.app.services.refresh_radar.sync_embeddings")
    def test_no_new_articles_still_completes_refresh(
        self, sync_embeddings, run_collection, process_articles_batch, get_news_stats
    ):
        get_news_stats.side_effect = [self.stats(), self.stats()]
        run_collection.return_value = self.collection()
        process_articles_batch.return_value = self.processing(remaining=756)
        sync_embeddings.return_value = self.indexing()

        result = refresh_radar()

        self.assertTrue(result["success"])
        self.assertEqual(result["new_articles"], 0)
        self.assertEqual(result["new_verified_ai_stories"], 0)
        self.assertEqual(result["remaining"], 756)
        process_articles_batch.assert_called_once()

    @patch("backend.app.services.refresh_radar.get_news_stats")
    @patch("backend.app.services.refresh_radar.process_articles_batch")
    @patch("backend.app.services.refresh_radar.run_collection")
    @patch("backend.app.services.refresh_radar.sync_embeddings")
    def test_analysis_failure_preserves_previous_success_timestamp(
        self, sync_embeddings, run_collection, process_articles_batch, get_news_stats
    ):
        self.state_path.write_text(
            json.dumps({"last_successful_refresh": "previous-success"}),
            encoding="utf-8",
        )
        get_news_stats.side_effect = [self.stats(), self.stats(total=101)]
        run_collection.return_value = self.collection(new_inserted=1)
        process_articles_batch.return_value = self.processing(processed=1, failed=1)
        sync_embeddings.return_value = self.indexing()

        result = refresh_radar()

        self.assertFalse(result["success"])
        self.assertIsNone(result["last_successful_refresh"])
        self.assertEqual(get_last_successful_refresh(), "previous-success")
        self.assertEqual(result["failed"], 1)

    @patch("backend.app.services.refresh_radar.get_news_stats")
    @patch("backend.app.services.refresh_radar.process_articles_batch")
    @patch("backend.app.services.refresh_radar.run_collection")
    @patch("backend.app.services.refresh_radar.sync_embeddings")
    def test_partial_feed_failure_is_reported_without_discarding_success(
        self, sync_embeddings, run_collection, process_articles_batch, get_news_stats
    ):
        get_news_stats.side_effect = [self.stats(), self.stats()]
        run_collection.return_value = self.collection(
            feeds_succeeded=19,
            feed_failures=[{"source": "OpenAI", "error": "timeout"}],
        )
        process_articles_batch.return_value = self.processing()
        sync_embeddings.return_value = self.indexing()

        result = refresh_radar()

        self.assertTrue(result["success"])
        self.assertTrue(result["partial"])
        self.assertEqual(len(result["feed_failures"]), 1)
        self.assertIsNotNone(get_last_successful_refresh())

    @patch("backend.app.services.refresh_radar.get_news_stats")
    @patch("backend.app.services.refresh_radar.process_articles_batch")
    @patch("backend.app.services.refresh_radar.run_collection")
    @patch("backend.app.services.refresh_radar.sync_embeddings")
    def test_no_successful_feeds_does_not_advance_timestamp(
        self, sync_embeddings, run_collection, process_articles_batch, get_news_stats
    ):
        get_news_stats.side_effect = [self.stats(), self.stats()]
        run_collection.return_value = self.collection(
            feeds_succeeded=0,
            feed_failures=[{"source": "OpenAI", "error": "timeout"}],
        )
        process_articles_batch.return_value = self.processing()
        sync_embeddings.return_value = self.indexing()

        result = refresh_radar()

        self.assertFalse(result["success"])
        self.assertIn("No RSS feeds", result["error"])
        self.assertIsNone(get_last_successful_refresh())

    @patch("backend.app.services.refresh_radar.get_news_stats")
    @patch("backend.app.services.refresh_radar.process_articles_batch")
    @patch("backend.app.services.refresh_radar.run_collection")
    @patch("backend.app.services.refresh_radar.sync_embeddings")
    def test_save_failure_does_not_advance_timestamp(
        self, sync_embeddings, run_collection, process_articles_batch, get_news_stats
    ):
        get_news_stats.side_effect = [self.stats(), self.stats()]
        run_collection.return_value = self.collection(
            save_failures=[{"title": "story", "error": "disk full"}]
        )
        process_articles_batch.return_value = self.processing()
        sync_embeddings.return_value = self.indexing()

        result = refresh_radar()

        self.assertFalse(result["success"])
        self.assertIn("failed to save", result["error"])
        self.assertIsNone(get_last_successful_refresh())

    @patch("backend.app.services.refresh_radar.get_news_stats")
    @patch("backend.app.services.refresh_radar.process_articles_batch")
    @patch("backend.app.services.refresh_radar.run_collection")
    @patch("backend.app.services.refresh_radar.sync_embeddings")
    def test_index_pending_does_not_advance_timestamp(
        self, sync_embeddings, run_collection, process_articles_batch, get_news_stats
    ):
        self.state_path.write_text(
            json.dumps({"last_successful_refresh": "previous-success"}),
            encoding="utf-8",
        )
        get_news_stats.side_effect = [self.stats(), self.stats()]
        run_collection.return_value = self.collection()
        process_articles_batch.return_value = self.processing()
        sync_embeddings.return_value = self.indexing(pending=1)

        result = refresh_radar()

        self.assertFalse(result["success"])
        self.assertEqual(result["index_pending"], 1)
        self.assertIn("remain unindexed", result["error"])
        self.assertEqual(get_last_successful_refresh(), "previous-success")


class TestCollectionDiagnostics(unittest.TestCase):

    @patch(
        "backend.app.services.news_collector.RSS_SOURCES",
        [
            {"name": "Feed A", "url": "https://a.example/rss", "category": "AI Research"},
            {"name": "Feed B", "url": "https://b.example/rss", "category": "AI Research"},
        ],
    )
    @patch("backend.app.services.news_collector.fetch_feed")
    def test_collection_retains_partial_feed_failures(self, fetch_feed):
        fetch_feed.side_effect = [
            SimpleNamespace(entries=[]),
            RuntimeError("connection timed out"),
        ]

        articles, feeds_succeeded, failures = _collect_news_with_status()

        self.assertEqual(articles, [])
        self.assertEqual(feeds_succeeded, 1)
        self.assertEqual(
            failures,
            [{"source": "Feed B", "error": "connection timed out"}],
        )

    @patch("backend.app.services.news_collector.save_article")
    def test_save_failure_is_returned_instead_of_only_logged(self, save_article):
        save_article.side_effect = [True, RuntimeError("database locked")]
        articles = [
            {"title": "Saved story"},
            {"title": "Failed story"},
        ]

        inserted, failures = _save_news_with_status(articles)

        self.assertEqual(inserted, 1)
        self.assertEqual(
            failures,
            [{"title": "Failed story", "error": "database locked"}],
        )


if __name__ == "__main__":
    unittest.main()
