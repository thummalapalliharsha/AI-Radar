"""
Persistence-layer tests.

These run against the real data/ai_radar.db. Everything here is read-only
apart from initialize_database(), which only creates missing schema objects and
fills NULL timestamps, so collected rows are never modified or removed.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from backend.app.config import DATABASE_PATH
from backend.app.utils import database
from backend.app.utils.database import (
    get_category_counts,
    get_connection,
    get_latest_news,
    get_news_stats,
    initialize_database,
    parse_published_value,
    save_article,
    table_exists,
)


def row_count() -> int:
    connection = get_connection()
    try:
        return connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    finally:
        connection.close()


class TestSchemaBootstrap(unittest.TestCase):
    """Requirements 1 and 9: the schema is ready, and no data is lost."""

    def test_initialize_is_idempotent_and_non_destructive(self):
        before = row_count()
        initialize_database()
        initialize_database()
        self.assertEqual(row_count(), before)

    def test_database_lives_at_the_configured_path(self):
        self.assertTrue(DATABASE_PATH.exists())
        self.assertEqual(DATABASE_PATH.name, "ai_radar.db")

    def test_articles_table_exists_with_every_required_column(self):
        initialize_database()
        connection = get_connection()
        try:
            self.assertTrue(table_exists(connection, "articles"))
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(articles)").fetchall()
            }
        finally:
            connection.close()

        for column in [
            "id", "title", "url", "summary", "source", "category",
            "published", "published_timestamp", "collected_at",
            "is_ai_news", "importance", "career_relevance", "ai_summary",
            "why_it_matters", "should_show", "embedding_indexed", "analyzed_at",
        ]:
            with self.subTest(column=column):
                self.assertIn(column, columns)

    def test_every_dated_row_has_a_usable_timestamp(self):
        initialize_database()
        connection = get_connection()
        try:
            orphans = connection.execute(
                """
                SELECT COUNT(*) FROM articles
                WHERE published_timestamp IS NULL
                  AND published IS NOT NULL
                  AND TRIM(published) <> ''
                """
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(
            orphans, 0, "rows with a date but no timestamp are invisible to the pipeline"
        )


class TestPublishedValueParsing(unittest.TestCase):

    def test_iso_8601(self):
        self.assertIsNotNone(parse_published_value("2026-09-01T17:35:37+00:00"))

    def test_rfc_2822(self):
        # Written by earlier builds; the ISO parser cannot read it.
        self.assertIsNotNone(parse_published_value("Wed, 26 Aug 2026 10:00:00 GMT"))

    def test_zulu_suffix(self):
        self.assertIsNotNone(parse_published_value("2026-09-01T17:35:37Z"))

    def test_numeric_passthrough(self):
        self.assertEqual(parse_published_value(1788284137.0), 1788284137.0)

    def test_unparseable_values(self):
        for value in [None, "", "   ", "not a date at all"]:
            with self.subTest(value=value):
                self.assertIsNone(parse_published_value(value))


class TestCanonicalArticleIdentity(unittest.TestCase):

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "isolated.db"
        self.path_patcher = patch.object(database, "DATABASE_PATH", self.database_path)
        self.path_patcher.start()
        self.addCleanup(self.path_patcher.stop)
        database.create_tables()

    def test_existing_trailing_slash_identity_is_not_inserted_twice(self):
        connection = database.get_connection()
        connection.execute(
            """
            INSERT INTO articles (title, url, source, collected_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                "Existing article",
                "https://example.com/story/",
                "Example",
                "2026-09-30T00:00:00+00:00",
            ),
        )
        connection.commit()
        connection.close()

        inserted = save_article(
            {
                "title": "Existing article",
                "url": "https://example.com/story",
                "source": "Example",
                "collected_at": "2026-09-30T00:01:00+00:00",
            }
        )

        connection = database.get_connection()
        count = connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        connection.close()
        self.assertFalse(inserted)
        self.assertEqual(count, 1)

    def test_new_url_is_normalized_and_repeat_ingestion_is_ignored(self):
        article = {
            "title": "New article",
            "url": "https://example.com/new-story/?utm_source=feed#section",
            "source": "Example",
            "collected_at": "2026-09-30T00:00:00+00:00",
        }

        self.assertTrue(save_article(article))
        self.assertFalse(
            save_article(
                {
                    **article,
                    "url": "https://example.com/new-story",
                }
            )
        )

        connection = database.get_connection()
        rows = connection.execute("SELECT url FROM articles").fetchall()
        connection.close()
        self.assertEqual([row["url"] for row in rows], ["https://example.com/new-story"])


class TestFeedQueries(unittest.TestCase):

    def test_stats_have_the_expected_shape(self):
        stats = get_news_stats()
        for key in [
            "total_collected",
            "ai_news_count",
            "stories_selected",
            "sources_count",
            "stories_today",
            "last_collected",
        ]:
            self.assertIn(key, stats)
        self.assertGreaterEqual(stats["total_collected"], stats["ai_news_count"])

    def test_category_counts_are_canonical(self):
        from backend.app.config import ALLOWED_CATEGORIES

        for name in get_category_counts():
            with self.subTest(category=name):
                self.assertIn(name, ALLOWED_CATEGORIES)

    def test_legacy_and_canonical_category_names_agree(self):
        # Requirement 13: asking for either spelling returns the same feed.
        canonical = get_latest_news(limit=50, category="AI Tools")
        legacy = get_latest_news(limit=50, category="Developers & Tools")
        self.assertEqual(
            [row["id"] for row in canonical], [row["id"] for row in legacy]
        )

    def test_feed_only_returns_verified_stories(self):
        for row in get_latest_news(limit=25):
            with self.subTest(article=row["id"]):
                self.assertEqual(row["is_ai_news"], 1)
                self.assertEqual(row["should_show"], 1)

    def test_feed_respects_the_minimum_importance(self):
        for row in get_latest_news(limit=25, min_importance=8):
            self.assertGreaterEqual(row["importance"], 8)

    def test_feed_is_ordered_newest_first(self):
        stamps = [
            row["published_timestamp"] or 0
            for row in get_latest_news(limit=25)
        ]
        self.assertEqual(stamps, sorted(stamps, reverse=True))


if __name__ == "__main__":
    unittest.main()


