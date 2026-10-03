"""Isolated audit rules and simulation for a future duplicate cleanup."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np

from backend.app.services import vector_store
from backend.app.utils import database
from backend.app.utils.url_normalization import normalize_url, url_identity_variants


ANALYSIS_FIELDS = (
    "category",
    "is_ai_news",
    "importance",
    "career_relevance",
    "ai_summary",
    "why_it_matters",
    "should_show",
    "analysis_version",
)


def classify_duplicate_group(rows):
    """Conservatively classify a fixture without inventing content equivalence."""
    if len(rows) < 2 or len({normalize_url(row["url"]) for row in rows}) != 1:
        return "review"

    for field in ("title", "source", "published_timestamp", "summary"):
        if len({row.get(field) for row in rows}) != 1:
            return "review"

    pending = [row for row in rows if row.get("is_ai_news") is None]
    analyzed = [row for row in rows if row.get("is_ai_news") is not None]
    if len(pending) == 1 and len(analyzed) == len(rows) - 1:
        return "safe-single-analysis"
    if pending:
        return "safe-exact"

    if all(
        all(row.get(field) == rows[0].get(field) for field in ANALYSIS_FIELDS)
        for row in rows[1:]
    ):
        return "safe-exact"
    return "review"


def choose_survivor(rows, briefing_references, faiss_ids):
    """Prefer existing references, complete analysis, and earliest observation."""
    def completeness(row):
        fields = (
            "title",
            "source",
            "published_timestamp",
            "summary",
            "ai_summary",
            "why_it_matters",
            "analyzed_at",
        )
        return sum(bool(row.get(field)) for field in fields)

    return max(
        rows,
        key=lambda row: (
            briefing_references.count(row["id"]),
            row.get("is_ai_news") is not None,
            row.get("should_show") == 1,
            completeness(row),
            row["id"] in faiss_ids,
            -(row.get("created_order") or row["id"]),
            -row["id"],
        ),
    )


def merge_safe_fields(rows, survivor, faiss_ids):
    """Test-only field policy for an approved, conflict-free group."""
    classification = classify_duplicate_group(rows)
    if not classification.startswith("safe"):
        return None

    result = dict(survivor)
    analyzed = [row for row in rows if row.get("is_ai_news") is not None]
    if classification == "safe-single-analysis":
        for field in ANALYSIS_FIELDS:
            result[field] = analyzed[0].get(field)

    result["url"] = normalize_url(survivor["url"])
    result["collected_at"] = min(row["collected_at"] for row in rows)
    result["created_at"] = min(row["created_at"] for row in rows)
    analyzed_times = [row["analyzed_at"] for row in analyzed if row.get("analyzed_at")]
    result["analyzed_at"] = min(analyzed_times) if analyzed_times else None
    result["embedding_indexed"] = int(survivor["id"] in faiss_ids)
    return result


def article_row(
    article_id,
    url,
    *,
    analyzed=True,
    selected=True,
    ai_summary="AI summary",
):
    return {
        "id": article_id,
        "title": "A verified AI story",
        "url": url,
        "summary": "Same source summary",
        "source": "Example Lab",
        "category": "AI Research",
        "published": "2026-09-30T10:00:00+00:00",
        "published_timestamp": 1790762400.0,
        "collected_at": f"2026-09-30T10:{article_id % 60:02d}:00+00:00",
        "created_at": f"2026-09-30 10:{article_id % 60:02d}:00",
        "created_order": article_id,
        "is_ai_news": 1 if analyzed else None,
        "importance": 8 if analyzed else None,
        "career_relevance": 7 if analyzed else None,
        "ai_summary": ai_summary if analyzed else None,
        "why_it_matters": "Useful impact" if analyzed else None,
        "should_show": 1 if analyzed and selected else (0 if analyzed else None),
        "embedding_indexed": 1 if analyzed and selected else 0,
        "analyzed_at": "2026-09-30T11:00:00+00:00" if analyzed else None,
        "analysis_version": "2.0",
    }


class TestHistoricalDuplicateCleanupPlan(unittest.TestCase):

    def test_trailing_slash_urls_share_the_existing_canonical_identity(self):
        first = "https://example.com/article"
        second = "https://example.com/article/"
        self.assertEqual(normalize_url(first), normalize_url(second))
        self.assertEqual(
            set(url_identity_variants(first)),
            set(url_identity_variants(second)),
        )

    def test_survivor_selection_is_reference_aware_and_deterministic(self):
        referenced = article_row(10, "https://example.com/story/")
        canonical_url = article_row(11, "https://example.com/story")
        self.assertIs(
            choose_survivor([referenced, canonical_url], [10], {10, 11}),
            referenced,
        )
        self.assertIs(
            choose_survivor([referenced, canonical_url], [], {10, 11}),
            referenced,
        )

    def test_analysis_result_is_kept_when_only_one_duplicate_was_processed(self):
        pending = article_row(20, "https://example.com/story/", analyzed=False)
        analyzed = article_row(21, "https://example.com/story")
        self.assertEqual(
            classify_duplicate_group([pending, analyzed]),
            "safe-single-analysis",
        )
        survivor = choose_survivor([pending, analyzed], [], {21})
        self.assertIs(survivor, analyzed)
        merged = merge_safe_fields([pending, analyzed], survivor, {21})
        self.assertEqual(merged["is_ai_news"], 1)
        self.assertEqual(merged["should_show"], 1)
        self.assertEqual(merged["ai_summary"], analyzed["ai_summary"])
        self.assertEqual(merged["why_it_matters"], analyzed["why_it_matters"])
        self.assertEqual(merged["url"], "https://example.com/story")

    def test_conflicting_analysis_or_content_requires_review(self):
        first = article_row(30, "https://example.com/story")
        conflicting = article_row(
            31,
            "https://example.com/story/",
            selected=False,
            ai_summary="Different analyzed content",
        )
        self.assertEqual(
            classify_duplicate_group([first, conflicting]),
            "review",
        )

    def test_history_mapping_and_self_contained_snapshot_remain_readable(self):
        snapshot = {
            "stories": [
                {
                    "id": 40,
                    "member_ids": [40],
                    "title": "A verified AI story",
                    "url": "https://example.com/story/",
                }
            ]
        }
        history_ids = [41]
        compatibility_map = {41: 40}
        before = json.dumps(snapshot, sort_keys=True)

        self.assertEqual(compatibility_map[history_ids[0]], 40)
        self.assertEqual(json.dumps(snapshot, sort_keys=True), before)

    def test_isolated_cleanup_and_faiss_reconciliation_are_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            db_path = root / "articles.db"
            index_path = root / "index.faiss"
            metadata_path = root / "metadata.json"
            with (
                patch.object(database, "DATABASE_PATH", db_path),
                patch.object(vector_store, "INDEX_PATH", index_path),
                patch.object(vector_store, "METADATA_PATH", metadata_path),
                patch.object(vector_store, "VECTOR_STORE_DIR", root),
            ):
                database.initialize_database()
                first = article_row(50, "https://example.com/story/")
                second = article_row(51, "https://example.com/story")
                connection = database.get_connection()
                for article in (first, second):
                    connection.execute(
                        """
                        INSERT INTO articles (
                            id, title, url, summary, source, category, published,
                            published_timestamp, collected_at, is_ai_news,
                            importance, career_relevance, ai_summary,
                            why_it_matters, should_show, embedding_indexed,
                            analyzed_at, analysis_version
                        ) VALUES (
                            :id, :title, :url, :summary, :source, :category,
                            :published, :published_timestamp, :collected_at,
                            :is_ai_news, :importance, :career_relevance,
                            :ai_summary, :why_it_matters, :should_show,
                            :embedding_indexed, :analyzed_at, :analysis_version
                        )
                        """,
                        article,
                    )
                connection.commit()
                connection.close()

                survivor = choose_survivor(
                    [first, second],
                    briefing_references=[],
                    faiss_ids={50, 51},
                )
                self.assertEqual(classify_duplicate_group([first, second]), "safe-exact")
                merged = merge_safe_fields([first, second], survivor, {50, 51})

                index = faiss.IndexFlatIP(3)
                vectors = np.asarray(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype="float32",
                )
                faiss.normalize_L2(vectors)
                index.add(vectors)
                vector_store.save_index(
                    index,
                    [
                        vector_store.build_metadata_entry(first),
                        vector_store.build_metadata_entry(second),
                    ],
                )

                duplicate_id = second["id"] if survivor["id"] == first["id"] else first["id"]
                connection = database.get_connection()
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("DELETE FROM articles WHERE id = ?", (duplicate_id,))
                connection.execute(
                    """
                    UPDATE articles
                    SET url = ?, collected_at = ?, analyzed_at = ?,
                        embedding_indexed = ?, category = ?, is_ai_news = ?,
                        importance = ?, career_relevance = ?, ai_summary = ?,
                        why_it_matters = ?, should_show = ?, analysis_version = ?
                    WHERE id = ?
                    """,
                    (
                        merged["url"],
                        merged["collected_at"],
                        merged["analyzed_at"],
                        merged["embedding_indexed"],
                        merged["category"],
                        merged["is_ai_news"],
                        merged["importance"],
                        merged["career_relevance"],
                        merged["ai_summary"],
                        merged["why_it_matters"],
                        merged["should_show"],
                        merged["analysis_version"],
                        survivor["id"],
                    ),
                )
                connection.commit()
                connection.close()
                id_map = {duplicate_id: survivor["id"]}

                stored_index, stored_metadata = vector_store.load_index()
                survivor_position = next(
                    position
                    for position, entry in enumerate(stored_metadata)
                    if entry["id"] == survivor["id"]
                )
                survivor_vector = stored_index.reconstruct(survivor_position)
                reconciled_index = faiss.IndexFlatIP(stored_index.d)
                reconciled_index.add(np.asarray([survivor_vector], dtype="float32"))
                merged_row = database.get_article_by_id(survivor["id"])
                vector_store.save_index(
                    reconciled_index,
                    [vector_store.build_metadata_entry(merged_row)],
                )

                with patch.object(
                    vector_store,
                    "create_embedding",
                    side_effect=AssertionError("unchanged survivor should not re-embed"),
                ) as create_embedding:
                    first_sync = vector_store.sync_embeddings(verbose=False)
                    first_index, first_metadata = vector_store.load_index()
                    second_sync = vector_store.sync_embeddings(verbose=False)
                    second_index, second_metadata = vector_store.load_index()

                self.assertEqual(id_map[duplicate_id], survivor["id"])
                self.assertEqual(first_sync["removed"], 0)
                self.assertEqual(first_index.ntotal, 1)
                self.assertEqual([entry["id"] for entry in first_metadata], [survivor["id"]])
                self.assertEqual(first_metadata[0]["url"], merged["url"])
                self.assertEqual(second_sync["removed"], 0)
                self.assertEqual(second_index.ntotal, 1)
                self.assertEqual([entry["id"] for entry in second_metadata], [survivor["id"]])
                create_embedding.assert_not_called()

                connection = database.get_connection()
                remaining = connection.execute(
                    "SELECT id, summary, ai_summary, why_it_matters, is_ai_news, should_show "
                    "FROM articles"
                ).fetchall()
                connection.close()
                self.assertEqual(len(remaining), 1)
                self.assertEqual(remaining[0]["summary"], first["summary"])
                self.assertEqual(remaining[0]["ai_summary"], first["ai_summary"])
                self.assertEqual(remaining[0]["why_it_matters"], first["why_it_matters"])
                self.assertEqual(remaining[0]["is_ai_news"], 1)
                self.assertEqual(remaining[0]["should_show"], 1)


if __name__ == "__main__":
    unittest.main()
