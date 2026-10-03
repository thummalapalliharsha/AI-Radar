import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from backend.app.main import print_system_status


class TestSystemStatusTimestamps(unittest.TestCase):

    @patch("backend.app.main.initialize_database", return_value={
        "database": "isolated.db",
        "timestamps_repaired": 0,
    })
    @patch("backend.app.main.get_news_stats", return_value={
        "total_collected": 0,
        "ai_news_count": 0,
        "stories_selected": 0,
        "sources_count": 0,
        "stories_today": 0,
        "last_collected": "2026-09-30T00:00:00+00:00",
    })
    @patch("backend.app.main.count_unprocessed_articles", return_value=0)
    @patch("backend.app.main.get_unindexed_articles", return_value=[])
    def test_collection_timestamp_is_not_labeled_as_a_run_timestamp(
        self,
        get_unindexed_articles,
        count_unprocessed_articles,
        get_news_stats,
        initialize_database,
    ):
        output = StringIO()
        with redirect_stdout(output):
            print_system_status()

        self.assertIn("Latest stored article collected at", output.getvalue())
        self.assertNotIn("Last collection run", output.getvalue())
        initialize_database.assert_called_once_with()
        get_news_stats.assert_called_once_with()
        self.assertEqual(count_unprocessed_articles.call_count, 2)
        get_unindexed_articles.assert_called_once_with(limit=100000)


if __name__ == "__main__":
    unittest.main()
