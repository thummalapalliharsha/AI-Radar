import sqlite3
import tempfile
import unittest
from pathlib import Path
import re
from unittest.mock import patch

import numpy as np

from backend.app.services import vector_store
from backend.app.utils import database


def article(article_id, title="Story", category="AI Research"):
    title = title if title != "Story" else f"Story {article_id}"
    return {
        "id": article_id,
        "title": title,
        "url": f"https://example.com/{article_id}",
        "source": "Test Source",
        "category": category,
        "summary": "Research summary",
        "ai_summary": "Analysed summary",
        "why_it_matters": "Useful to builders",
        "published": "2026-09-12T00:00:00+00:00",
        "published_timestamp": float(article_id),
        "importance": 7,
        "career_relevance": 7,
    }


class TestVectorReconciliation(unittest.TestCase):

    def run_sync(self, articles, embeddings, existing=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path = root / "ai_radar.index"
            metadata_path = root / "metadata.json"
            with patch.object(vector_store, "INDEX_PATH", index_path), \
                patch.object(vector_store, "METADATA_PATH", metadata_path), \
                patch.object(vector_store, "VECTOR_STORE_DIR", root), \
                patch.object(vector_store, "get_indexable_articles", lambda: articles), \
                patch.object(vector_store, "mark_as_indexed"), \
                patch.object(
                    vector_store,
                    "create_embedding",
                    side_effect=lambda text: embeddings.get(
                        int(match.group(1)) if (match := re.search(r"Story (\d+)", text)) else 1,
                        embeddings[1],
                    ),
                ) as embed:
                if existing:
                    index = vector_store.faiss.IndexFlatIP(2)
                    matrix = np.vstack([embeddings[item["id"]] for item in existing]).astype("float32")
                    vector_store.faiss.normalize_L2(matrix)
                    index.add(matrix)
                    vector_store.save_index(
                        index,
                        [vector_store.build_metadata_entry(item) for item in existing],
                    )
                result = vector_store.sync_embeddings(verbose=False)
                final_index, metadata = vector_store.load_index()
                return result, final_index, metadata, embed.call_count

    def test_stale_removed_and_missing_added(self):
        old = [article(1), article(2)]
        current = [article(2), article(3)]
        embeddings = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0]), 3: np.array([1.0, 1.0])}
        result, index, metadata, calls = self.run_sync(current, embeddings, existing=old)
        self.assertEqual(result["removed"], 1)
        self.assertEqual(result["added"], 1)
        self.assertEqual(calls, 1)
        self.assertEqual(index.ntotal, 2)
        self.assertEqual({entry["id"] for entry in metadata}, {2, 3})

    def test_changed_article_is_reembedded(self):
        old = [article(1, title="Old title")]
        current = [article(1, title="New title")]
        embeddings = {1: np.array([0.0, 1.0])}
        result, index, metadata, calls = self.run_sync(current, embeddings, existing=old)
        self.assertEqual(result["updated"], 1)
        self.assertEqual(result["unchanged"], 0)
        self.assertEqual(calls, 1)
        self.assertEqual(index.ntotal, len(metadata))
        self.assertEqual(metadata[0]["title"], "New title")

    def test_zero_change_does_not_reembed(self):
        current = [article(1), article(2)]
        embeddings = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
        result, _, _, first_calls = self.run_sync(current, embeddings)
        self.assertEqual(first_calls, 2)

        result, _, _, second_calls = self.run_sync(
            current, embeddings, existing=current
        )
        self.assertEqual(result["added"], 0)
        self.assertEqual(result["removed"], 0)
        self.assertEqual(result["updated"], 0)
        self.assertEqual(result["unchanged"], 2)
        self.assertEqual(second_calls, 0)

    def test_metadata_and_index_stay_aligned(self):
        current = [article(1), article(2), article(3)]
        embeddings = {key: np.array([float(key), 1.0]) for key in [1, 2, 3]}
        result, index, metadata, _ = self.run_sync(current, embeddings)
        self.assertTrue(index is not None)
        self.assertEqual(index.ntotal, len(metadata))
        self.assertEqual(result["indexed"], 3)

    def test_failed_rebuild_persistence_does_not_change_embedding_flags(self):
        article_row = article(10)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(vector_store, "INDEX_PATH", root / "index.faiss"),
                patch.object(vector_store, "METADATA_PATH", root / "metadata.json"),
                patch.object(vector_store, "VECTOR_STORE_DIR", root),
                patch.object(vector_store, "FAISS_GENERATION_MANIFEST_PATH", None),
                patch.object(
                    vector_store, "get_indexable_articles", return_value=[article_row]
                ),
                patch.object(
                    vector_store,
                    "create_embedding",
                    return_value=np.array([1.0, 0.0], dtype=np.float32),
                ),
                patch.object(
                    vector_store,
                    "save_index",
                    side_effect=OSError("simulated persistence failure"),
                ),
                patch.object(vector_store, "replace_embedding_flags") as replace_flags,
                patch.object(vector_store, "mark_as_indexed") as mark_flags,
            ):
                with self.assertRaisesRegex(OSError, "persistence failure"):
                    vector_store.sync_embeddings(rebuild=True, verbose=False)

        replace_flags.assert_not_called()
        mark_flags.assert_not_called()

    def test_rebuild_updates_flags_only_after_successful_persistence(self):
        article_row = article(11)
        calls = []

        def persist(index, metadata, **kwargs):
            self.assertEqual(index.ntotal, len(metadata))
            self.assertIsNone(kwargs["manifest_path"])
            calls.append("persist")

        def replace_flags(article_ids):
            self.assertEqual(article_ids, [11])
            calls.append("flags")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(vector_store, "INDEX_PATH", root / "index.faiss"),
                patch.object(vector_store, "METADATA_PATH", root / "metadata.json"),
                patch.object(vector_store, "VECTOR_STORE_DIR", root),
                patch.object(vector_store, "FAISS_GENERATION_MANIFEST_PATH", None),
                patch.object(
                    vector_store, "get_indexable_articles", return_value=[article_row]
                ),
                patch.object(
                    vector_store,
                    "create_embedding",
                    return_value=np.array([1.0, 0.0], dtype=np.float32),
                ),
                patch.object(vector_store, "save_index", side_effect=persist),
                patch.object(
                    vector_store, "replace_embedding_flags", side_effect=replace_flags
                ),
            ):
                result = vector_store.sync_embeddings(rebuild=True, verbose=False)

        self.assertEqual(result["indexed"], 1)
        self.assertEqual(calls, ["persist", "flags"])

    def test_embedding_flag_replacement_rolls_back_as_one_transaction(self):
        with tempfile.TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "flags.db"
            connection = sqlite3.connect(database_path)
            connection.execute(
                "CREATE TABLE articles (id INTEGER PRIMARY KEY, embedding_indexed INTEGER)"
            )
            connection.executemany(
                "INSERT INTO articles (id, embedding_indexed) VALUES (?, ?)",
                [(1, 1), (2, 0)],
            )
            connection.execute(
                """
                CREATE TRIGGER reject_flag_update
                BEFORE UPDATE OF embedding_indexed ON articles
                WHEN NEW.id = 2 AND NEW.embedding_indexed = 1
                BEGIN
                    SELECT RAISE(ABORT, 'simulated flag update failure');
                END
                """
            )
            connection.commit()
            connection.close()

            with patch.object(
                database,
                "get_connection",
                side_effect=lambda: sqlite3.connect(database_path),
            ):
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError, "simulated flag update failure"
                ):
                    database.replace_embedding_flags([2])

            connection = sqlite3.connect(database_path)
            flags = connection.execute(
                "SELECT id, embedding_indexed FROM articles ORDER BY id"
            ).fetchall()
            connection.close()

        self.assertEqual(flags, [(1, 1), (2, 0)])


if __name__ == "__main__":
    unittest.main()
