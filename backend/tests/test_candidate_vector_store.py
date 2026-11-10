import json
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from backend.app.services.candidate_vector_store import build_candidate_index


def metadata(article_id):
    return {
        "id": article_id,
        "title": f"Story {article_id}",
        "url": f"https://example.com/{article_id}",
        "source": "Synthetic source",
        "category": "AI Research",
    }


def flat_index(vectors):
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype(np.float32))
    return index


class TestCandidateVectorStore(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.existing_vectors = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        self.existing_index = flat_index(self.existing_vectors)
        self.existing_metadata = [metadata(10), metadata(20)]
        self.index_path = self.root / "candidate" / "candidate.index"
        self.metadata_path = self.root / "candidate" / "metadata.json"

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, embeddings=None, new_metadata=None):
        return build_candidate_index(
            self.existing_index,
            self.existing_metadata,
            embeddings if embeddings is not None else {30: np.array([1.0, 1.0, 0.0])},
            new_metadata if new_metadata is not None else {30: metadata(30)},
            self.index_path,
            self.metadata_path,
        )

    def test_builds_candidate_preserving_vectors_and_appending_new_ids(self):
        candidate, records = self.build(
            {
                30: np.array([1.0, 1.0, 0.0]),
                40: np.array([0.0, 1.0, 1.0]),
            },
            {30: metadata(30), 40: metadata(40)},
        )
        self.assertEqual(candidate.ntotal, 4)
        self.assertEqual([record["id"] for record in records], [10, 20, 30, 40])
        np.testing.assert_array_equal(candidate.reconstruct(0), self.existing_vectors[0])
        np.testing.assert_array_equal(candidate.reconstruct(1), self.existing_vectors[1])
        np.testing.assert_allclose(candidate.reconstruct(2), [2**-0.5, 2**-0.5, 0.0])
        np.testing.assert_allclose(candidate.reconstruct(3), [0.0, 2**-0.5, 2**-0.5])
        self.assertTrue(self.index_path.is_file())
        self.assertTrue(self.metadata_path.is_file())

    def test_rejects_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate IDs"):
            self.build(
                {10: np.array([1.0, 0.0, 0.0])},
                {10: metadata(10)},
            )
        with self.assertRaisesRegex(ValueError, "duplicate article IDs"):
            build_candidate_index(
                self.existing_index,
                [metadata(10), metadata(10)],
                {},
                {},
                self.index_path,
                self.metadata_path,
            )

    def test_rejects_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "must have dimension"):
            self.build({30: np.array([1.0, 0.0])}, {30: metadata(30)})

    def test_rejects_nan_and_infinite_embeddings(self):
        for invalid in (
            np.array([np.nan, 0.0, 1.0]),
            np.array([np.inf, 0.0, 1.0]),
            np.array([0.0, -np.inf, 1.0]),
        ):
            with self.subTest(vector=invalid):
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    self.build({30: invalid}, {30: metadata(30)})

    def test_rejects_metadata_vector_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "does not match vector count"):
            build_candidate_index(
                self.existing_index,
                [metadata(10)],
                {},
                {},
                self.index_path,
                self.metadata_path,
            )

    def test_rejects_malformed_metadata_and_zero_vectors(self):
        with self.assertRaisesRegex(ValueError, "integer 'id'"):
            self.build({30: np.ones(3)}, {30: {"title": "Missing ID"}})
        with self.assertRaisesRegex(ValueError, "zero magnitude"):
            self.build({30: np.zeros(3)}, {30: metadata(30)})

    def test_rejects_existing_output_paths_without_overwriting(self):
        self.index_path.parent.mkdir(parents=True)
        self.index_path.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(self.index_path.read_bytes(), b"keep")
        self.assertFalse(self.metadata_path.exists())

    def test_candidate_paths_are_isolated_from_production_inputs(self):
        production_index = self.root / "production.index"
        production_metadata = self.root / "production.json"
        faiss.write_index(self.existing_index, str(production_index))
        production_metadata.write_text(
            json.dumps(self.existing_metadata), encoding="utf-8"
        )
        original_index_bytes = production_index.read_bytes()
        original_metadata_bytes = production_metadata.read_bytes()

        self.build()

        self.assertEqual(production_index.read_bytes(), original_index_bytes)
        self.assertEqual(production_metadata.read_bytes(), original_metadata_bytes)
        self.assertNotEqual(self.index_path.resolve(), production_index.resolve())
        self.assertNotEqual(self.metadata_path.resolve(), production_metadata.resolve())


if __name__ == "__main__":
    unittest.main()
