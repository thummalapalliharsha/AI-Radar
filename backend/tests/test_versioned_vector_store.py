import hashlib
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np

from backend.app.services import vector_store
from backend.app.services.versioned_vector_store import (
    GenerationError,
    initialize_generation_manifest,
    load_generation,
    switch_active_generation,
    write_and_activate_generation,
)


def write_generation(root, generation_id, article_ids, dimension=3, metric="inner_product"):
    generation_dir = root / generation_id
    generation_dir.mkdir()
    index = (
        faiss.IndexFlatIP(dimension)
        if metric == "inner_product"
        else faiss.IndexFlatL2(dimension)
    )
    vectors = np.zeros((len(article_ids), dimension), dtype=np.float32)
    for position in range(len(article_ids)):
        vectors[position, position % dimension] = 1.0
    if len(vectors):
        index.add(vectors)

    index_path = generation_dir / "index.faiss"
    metadata_path = generation_dir / "metadata.json"
    faiss.write_index(index, str(index_path))
    metadata = [
        {"id": article_id, "title": f"Story {article_id}"}
        for article_id in article_ids
    ]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return {
        "index": f"{generation_id}/index.faiss",
        "metadata": f"{generation_id}/metadata.json",
        "count": len(article_ids),
        "dimension": dimension,
        "metric": metric,
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    }


class TestVersionedVectorStore(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest_path = self.root / "active.json"
        self.generation_a = write_generation(self.root, "generation-a", [1, 2])
        self.generation_b = write_generation(self.root, "generation-b", [3, 4, 5])
        self.write_manifest("generation-a")

    def tearDown(self):
        self.temporary.cleanup()

    def write_manifest(self, active, generations=None):
        manifest = {
            "version": 1,
            "active_generation": active,
            "generations": generations
            or {
                "generation-a": self.generation_a,
                "generation-b": self.generation_b,
            },
        }
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def test_existing_default_loader_paths_remain_unchanged(self):
        index_path = self.root / "production.index"
        metadata_path = self.root / "production.json"
        index = faiss.IndexFlatIP(3)
        index.add(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        faiss.write_index(index, str(index_path))
        metadata_path.write_text(json.dumps([{"id": 99}]), encoding="utf-8")

        with (
            patch.object(vector_store, "INDEX_PATH", index_path),
            patch.object(vector_store, "METADATA_PATH", metadata_path),
            patch.object(vector_store, "FAISS_GENERATION_MANIFEST_PATH", None),
            patch.object(vector_store, "ensure_vector_dir"),
        ):
            loaded_index, metadata = vector_store.load_index()

        self.assertEqual(loaded_index.ntotal, 1)
        self.assertEqual(metadata, [{"id": 99}])

    def test_fixed_path_reads_do_not_create_locks_in_vector_store_directory(self):
        fixed_store = self.root / "fixed-store"
        fixed_store.mkdir()
        index_path = fixed_store / "production.index"
        metadata_path = fixed_store / "production.json"
        index = faiss.IndexFlatIP(3)
        index.add(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        faiss.write_index(index, str(index_path))
        metadata_path.write_text(json.dumps([{"id": 99}]), encoding="utf-8")

        with (
            patch.object(vector_store, "INDEX_PATH", index_path),
            patch.object(vector_store, "METADATA_PATH", metadata_path),
            patch.object(vector_store, "FAISS_GENERATION_MANIFEST_PATH", None),
        ):
            loaded_index, metadata = vector_store.load_index()
            loaded_metadata = vector_store.load_metadata()
            lock_path = vector_store._fixed_store_lock_path()

        self.assertEqual(loaded_index.ntotal, 1)
        self.assertEqual(metadata, [{"id": 99}])
        self.assertEqual(loaded_metadata, [{"id": 99}])
        self.assertNotEqual(lock_path.parent, fixed_store)
        self.assertEqual(
            {path.name for path in fixed_store.iterdir()},
            {"production.index", "production.json"},
        )

    def test_default_save_uses_fixed_paths_and_reopens_a_matching_pair(self):
        index_path = self.root / "fixed.index"
        metadata_path = self.root / "fixed.json"
        index = faiss.IndexFlatIP(3)
        index.add(np.array([[0.0, 1.0, 0.0]], dtype=np.float32))
        with (
            patch.object(vector_store, "INDEX_PATH", index_path),
            patch.object(vector_store, "METADATA_PATH", metadata_path),
            patch.object(vector_store, "FAISS_GENERATION_MANIFEST_PATH", None),
        ):
            generation_id = vector_store.save_index(index, [{"id": 77}])
            loaded_index, metadata = vector_store.load_index()

        self.assertIsNone(generation_id)
        self.assertEqual(loaded_index.ntotal, 1)
        self.assertEqual(metadata, [{"id": 77}])
        self.assertTrue(index_path.is_file())
        self.assertTrue(metadata_path.is_file())

    def test_interrupted_fixed_pair_write_restores_original_files(self):
        index_path = self.root / "production.index"
        metadata_path = self.root / "production.json"
        original_index = faiss.IndexFlatIP(3)
        original_index.add(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        faiss.write_index(original_index, str(index_path))
        metadata_path.write_text(json.dumps([{"id": 66}]), encoding="utf-8")
        original_bytes = (index_path.read_bytes(), metadata_path.read_bytes())

        replacement_index = faiss.IndexFlatIP(3)
        replacement_index.add(np.array([[0.0, 1.0, 0.0]], dtype=np.float32))
        replace = vector_store.os.replace

        def interrupted_replace(source, destination):
            if Path(destination) == metadata_path and Path(source).name == "metadata.json":
                raise OSError("simulated metadata replacement failure")
            return replace(source, destination)

        with (
            patch.object(vector_store, "INDEX_PATH", index_path),
            patch.object(vector_store, "METADATA_PATH", metadata_path),
            patch.object(vector_store, "FAISS_GENERATION_MANIFEST_PATH", None),
            patch.object(vector_store, "ensure_vector_dir"),
            patch.object(vector_store.os, "replace", side_effect=interrupted_replace),
        ):
            with self.assertRaisesRegex(OSError, "metadata replacement"):
                vector_store.save_index(replacement_index, [{"id": 67}])

        self.assertEqual(index_path.read_bytes(), original_bytes[0])
        self.assertEqual(metadata_path.read_bytes(), original_bytes[1])
        loaded_index, loaded_metadata = (
            faiss.read_index(str(index_path)),
            json.loads(metadata_path.read_text(encoding="utf-8")),
        )
        self.assertEqual(loaded_index.ntotal, len(loaded_metadata))
        self.assertEqual(loaded_metadata, [{"id": 66}])

    def test_loads_valid_active_and_explicit_generations(self):
        active_index, active_metadata = load_generation(self.manifest_path)
        selected_index, selected_metadata = vector_store.load_index(
            manifest_path=str(self.manifest_path), generation_id="generation-b"
        )
        selected_metadata_only = vector_store.load_metadata(
            manifest_path=self.manifest_path, generation_id="generation-b"
        )
        self.assertEqual(active_index.ntotal, 2)
        self.assertEqual([item["id"] for item in active_metadata], [1, 2])
        self.assertEqual(selected_index.ntotal, 3)
        self.assertEqual([item["id"] for item in selected_metadata], [3, 4, 5])
        self.assertEqual(selected_metadata_only, selected_metadata)

    def test_configured_manifest_is_used_without_falling_back(self):
        with patch.object(
            vector_store, "FAISS_GENERATION_MANIFEST_PATH", self.manifest_path
        ):
            loaded_index, metadata = vector_store.load_index()
        self.assertEqual(loaded_index.ntotal, 2)
        self.assertEqual([item["id"] for item in metadata], [1, 2])

        missing_manifest = self.root / "missing-active.json"
        with (
            patch.object(
                vector_store, "FAISS_GENERATION_MANIFEST_PATH", missing_manifest
            ),
            patch.object(vector_store, "INDEX_PATH", self.root / "production.index"),
            patch.object(
                vector_store, "METADATA_PATH", self.root / "production.json"
            ),
        ):
            with self.assertRaisesRegex(GenerationError, "does not exist"):
                vector_store.load_index()

    def test_index_stats_identify_one_manifest_generation(self):
        with (
            patch.object(
                vector_store, "FAISS_GENERATION_MANIFEST_PATH", self.manifest_path
            ),
            patch.object(vector_store, "INDEX_PATH", self.root / "fixed.index"),
            patch.object(vector_store, "METADATA_PATH", self.root / "fixed.json"),
        ):
            stats = vector_store.get_index_stats()

        self.assertEqual(stats["generation_id"], "generation-a")
        self.assertEqual(stats["indexed"], 2)
        self.assertEqual(stats["metadata"], 2)
        self.assertEqual(stats["metric"], "inner_product")
        self.assertEqual(
            Path(stats["index_path"]),
            self.root / "generation-a" / "index.faiss",
        )

    def test_missing_manifest_is_explicit_error(self):
        with self.assertRaisesRegex(GenerationError, "does not exist"):
            load_generation(self.root / "missing.json")

    def test_invalid_manifest_is_rejected(self):
        self.manifest_path.write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(GenerationError, "Could not read"):
            load_generation(self.manifest_path)

        self.write_manifest("generation-a")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["version"] = 2
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(GenerationError, "version 1"):
            load_generation(self.manifest_path)

    def test_missing_index_or_metadata_is_rejected(self):
        original_entry = self.generation_a.copy()
        for key in ("index", "metadata"):
            with self.subTest(key=key):
                entry = original_entry.copy()
                entry[key] = "generation-a/missing.file"
                self.write_manifest("generation-a", {"generation-a": entry})
                with self.assertRaisesRegex(GenerationError, "artifact is missing"):
                    load_generation(self.manifest_path)
        self.write_manifest("generation-a")

    def test_count_mismatch_is_rejected(self):
        entry = self.generation_a.copy()
        entry["count"] = 3
        self.write_manifest("generation-a", {"generation-a": entry})
        with self.assertRaisesRegex(GenerationError, "vector count"):
            load_generation(self.manifest_path)

    def test_duplicate_metadata_ids_are_rejected(self):
        metadata_path = self.root / "generation-a" / "metadata.json"
        metadata_path.write_text(
            json.dumps([{"id": 1}, {"id": 1}]), encoding="utf-8"
        )
        entry = self.generation_a.copy()
        entry["metadata_sha256"] = hashlib.sha256(
            metadata_path.read_bytes()
        ).hexdigest()
        self.write_manifest("generation-a", {"generation-a": entry})
        with self.assertRaisesRegex(GenerationError, "duplicate article IDs"):
            load_generation(self.manifest_path)

    def test_invalid_dimension_and_metric_are_rejected(self):
        entry = self.generation_a.copy()
        entry["dimension"] = 4
        self.write_manifest("generation-a", {"generation-a": entry})
        with self.assertRaisesRegex(GenerationError, "dimension"):
            load_generation(self.manifest_path)

        entry = self.generation_a.copy()
        entry["metric"] = "cosine"
        self.write_manifest("generation-a", {"generation-a": entry})
        with self.assertRaisesRegex(GenerationError, "unsupported metric"):
            load_generation(self.manifest_path)

        entry = self.generation_a.copy()
        entry["metric"] = "l2"
        self.write_manifest("generation-a", {"generation-a": entry})
        with self.assertRaisesRegex(GenerationError, "unsupported metric"):
            load_generation(self.manifest_path)

    def test_hash_mismatch_detects_incomplete_or_mutated_generation(self):
        index_path = self.root / "generation-b" / "index.faiss"
        index_path.write_bytes(index_path.read_bytes() + b"incomplete")
        with self.assertRaisesRegex(GenerationError, "index hash"):
            switch_active_generation(self.manifest_path, "generation-b")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["active_generation"], "generation-a")

    def test_switch_is_atomic_and_rollback_selects_previous_generation(self):
        before = self.manifest_path.read_bytes()
        switch_active_generation(self.manifest_path, "generation-b")
        after = self.manifest_path.read_bytes()
        self.assertNotEqual(before, after)
        self.assertEqual(
            json.loads(after)["active_generation"], "generation-b"
        )
        loaded, metadata = load_generation(self.manifest_path)
        self.assertEqual(loaded.ntotal, len(metadata))
        self.assertEqual([item["id"] for item in metadata], [3, 4, 5])

        switch_active_generation(self.manifest_path, "generation-a")
        rolled_back, rollback_metadata = load_generation(self.manifest_path)
        self.assertEqual(rolled_back.ntotal, 2)
        self.assertEqual([item["id"] for item in rollback_metadata], [1, 2])

    def test_manifest_can_represent_production_and_candidate_and_roll_back(self):
        production_index = self.root / "generation-a" / "index.faiss"
        production_metadata = self.root / "generation-a" / "metadata.json"
        candidate_index = self.root / "generation-b" / "index.faiss"
        candidate_metadata = self.root / "generation-b" / "metadata.json"
        production_hash = hashlib.sha256(production_index.read_bytes()).hexdigest()
        candidate_hash = hashlib.sha256(candidate_index.read_bytes()).hexdigest()
        manifest_path = self.root / "isolated" / "active.json"

        initialize_generation_manifest(
            manifest_path,
            {
                "production": (production_index, production_metadata),
                "candidate": (candidate_index, candidate_metadata),
            },
            active_generation="production",
        )
        self.assertEqual(
            [item["id"] for item in load_generation(manifest_path)[1]], [1, 2]
        )
        switch_active_generation(manifest_path, "candidate")
        self.assertEqual(
            [item["id"] for item in load_generation(manifest_path)[1]], [3, 4, 5]
        )
        switch_active_generation(manifest_path, "production")
        self.assertEqual(
            [item["id"] for item in load_generation(manifest_path)[1]], [1, 2]
        )
        self.assertEqual(
            hashlib.sha256(production_index.read_bytes()).hexdigest(),
            production_hash,
        )
        self.assertEqual(
            hashlib.sha256(candidate_index.read_bytes()).hexdigest(), candidate_hash
        )

    def test_manifest_selected_save_publishes_new_pair_and_keeps_old_generations(self):
        original_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        index = faiss.IndexFlatIP(3)
        index.add(np.array([[0.0, 0.0, 1.0]], dtype=np.float32))
        fixed_index = self.root / "fixed-production.index"
        fixed_metadata = self.root / "fixed-production.json"
        fixed_index.write_bytes(b"fixed index sentinel")
        fixed_metadata.write_bytes(b"fixed metadata sentinel")

        with (
            patch.object(vector_store, "INDEX_PATH", fixed_index),
            patch.object(vector_store, "METADATA_PATH", fixed_metadata),
        ):
            generation_id = vector_store.save_index(
                index,
                [{"id": 88}],
                manifest_path=self.manifest_path,
                expected_active_generation="generation-a",
            )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        loaded_index, metadata = load_generation(self.manifest_path)

        self.assertNotEqual(generation_id, "generation-a")
        self.assertEqual(manifest["active_generation"], generation_id)
        self.assertTrue({"generation-a", "generation-b"}.issubset(manifest["generations"]))
        self.assertEqual(loaded_index.ntotal, 1)
        self.assertEqual(metadata, [{"id": 88}])
        self.assertEqual(original_manifest["active_generation"], "generation-a")
        self.assertEqual(fixed_index.read_bytes(), b"fixed index sentinel")
        self.assertEqual(fixed_metadata.read_bytes(), b"fixed metadata sentinel")

    def test_stale_writer_cannot_overwrite_a_new_active_generation(self):
        switch_active_generation(self.manifest_path, "generation-b")
        index = faiss.IndexFlatIP(3)
        index.add(np.array([[0.0, 0.0, 1.0]], dtype=np.float32))

        with self.assertRaisesRegex(GenerationError, "changed during indexing"):
            write_and_activate_generation(
                self.manifest_path,
                index,
                [{"id": 88}],
                expected_active_generation="generation-a",
            )

        self.assertEqual(
            json.loads(self.manifest_path.read_text(encoding="utf-8"))[
                "active_generation"
            ],
            "generation-b",
        )

    def test_failed_generation_write_leaves_manifest_and_active_pair_unchanged(self):
        original_manifest = self.manifest_path.read_bytes()
        index = faiss.IndexFlatIP(3)
        index.add(np.array([[0.0, 0.0, 1.0]], dtype=np.float32))

        with patch(
            "backend.app.services.versioned_vector_store.faiss.write_index",
            side_effect=OSError("simulated interrupted generation write"),
        ):
            with self.assertRaisesRegex(OSError, "interrupted"):
                write_and_activate_generation(
                    self.manifest_path,
                    index,
                    [{"id": 88}],
                    expected_active_generation="generation-a",
                )

        self.assertEqual(self.manifest_path.read_bytes(), original_manifest)
        self.assertEqual(
            [item["id"] for item in load_generation(self.manifest_path)[1]], [1, 2]
        )

    def test_failed_manifest_replacement_preserves_active_manifest(self):
        original = self.manifest_path.read_bytes()
        with patch(
            "backend.app.services.versioned_vector_store.os.replace",
            side_effect=OSError("simulated interrupted replace"),
        ):
            with self.assertRaisesRegex(GenerationError, "atomically update"):
                switch_active_generation(self.manifest_path, "generation-b")
        self.assertEqual(self.manifest_path.read_bytes(), original)
        self.assertEqual(
            list(self.root.glob(".active.json.*.tmp")), []
        )

    def test_concurrent_readers_never_observe_a_mismatched_pair(self):
        errors = []
        start = threading.Event()

        def reader():
            start.wait()
            for _ in range(60):
                try:
                    index, metadata = load_generation(self.manifest_path)
                    ids = [item["id"] for item in metadata]
                    if index.ntotal != len(ids):
                        errors.append("count mismatch")
                    if ids not in ([1, 2], [3, 4, 5]):
                        errors.append(f"unexpected generation IDs: {ids}")
                except Exception as error:
                    errors.append(str(error))

        def writer():
            start.wait()
            for position in range(40):
                selected = "generation-a" if position % 2 else "generation-b"
                switch_active_generation(self.manifest_path, selected)

        with ThreadPoolExecutor(max_workers=7) as executor:
            readers = [executor.submit(reader) for _ in range(6)]
            writer_future = executor.submit(writer)
            start.set()
            writer_future.result()
            for future in readers:
                future.result()
        self.assertEqual(errors, [])

    def test_concurrent_readers_and_generation_writers_stay_consistent(self):
        errors = []
        start = threading.Event()

        def reader():
            start.wait()
            for _ in range(60):
                try:
                    index, metadata = load_generation(self.manifest_path)
                    ids = [item["id"] for item in metadata]
                    if index.ntotal != len(ids) or ids not in (
                        [1, 2],
                        [3, 4, 5],
                        [6],
                    ):
                        errors.append(f"mismatched generation: {ids}")
                except Exception as error:
                    errors.append(str(error))

        def writer():
            start.wait()
            for _ in range(20):
                index = faiss.IndexFlatIP(3)
                index.add(np.array([[0.0, 0.0, 1.0]], dtype=np.float32))
                write_and_activate_generation(
                    self.manifest_path, index, [{"id": 6}]
                )
                switch_active_generation(self.manifest_path, "generation-a")

        with ThreadPoolExecutor(max_workers=5) as executor:
            readers = [executor.submit(reader) for _ in range(4)]
            writer_future = executor.submit(writer)
            start.set()
            writer_future.result()
            for future in readers:
                future.result()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
