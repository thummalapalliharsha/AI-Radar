"""Build and validate isolated FAISS candidate artifacts.

This module deliberately has no application configuration, database, or
embedding-service dependencies. Callers must provide all data and output paths.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import faiss
import numpy as np

ArticleId = int
OutputPath = Union[str, os.PathLike]


def _article_id(metadata: Mapping[str, Any], label: str) -> ArticleId:
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{label} metadata must be an object.")
    article_id = metadata.get("id")
    if isinstance(article_id, bool) or not isinstance(article_id, int):
        raise ValueError(f"{label} metadata must contain an integer 'id'.")
    return article_id


def _copy_metadata(
    records: Sequence[Mapping[str, Any]], expected_count: int, label: str
) -> Tuple[List[Dict[str, Any]], List[ArticleId]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError(f"{label} metadata must be a sequence of objects.")
    if len(records) != expected_count:
        raise ValueError(
            f"{label} metadata count ({len(records)}) does not match "
            f"vector count ({expected_count})."
        )

    copied_records: List[Dict[str, Any]] = []
    ids: List[ArticleId] = []
    for position, record in enumerate(records):
        article_id = _article_id(record, f"{label} record {position}")
        try:
            copied = json.loads(json.dumps(dict(record), allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{label} record {position} is not valid JSON metadata."
            ) from error
        copied_records.append(copied)
        ids.append(article_id)

    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} metadata contains duplicate article IDs.")
    return copied_records, ids


def _output_paths(
    output_index_path: OutputPath, output_metadata_path: OutputPath
) -> Tuple[Path, Path]:
    index_path = Path(output_index_path)
    metadata_path = Path(output_metadata_path)
    if index_path.resolve() == metadata_path.resolve():
        raise ValueError("Candidate index and metadata paths must be different.")
    if index_path.is_symlink() or metadata_path.is_symlink():
        raise ValueError("Candidate output paths must not be symbolic links.")
    if index_path.exists():
        raise FileExistsError(f"Candidate index path already exists: {index_path}")
    if metadata_path.exists():
        raise FileExistsError(
            f"Candidate metadata path already exists: {metadata_path}"
        )
    return index_path, metadata_path


def _validate_candidate(
    candidate_index: faiss.Index,
    candidate_metadata: Sequence[Mapping[str, Any]],
    original_ids: Sequence[ArticleId],
    original_vectors: np.ndarray,
    new_ids: Sequence[ArticleId],
    normalized_new_vectors: np.ndarray,
) -> None:
    if candidate_index.ntotal != len(candidate_metadata):
        raise ValueError("Candidate vector and metadata counts do not match.")
    if candidate_index.d != original_vectors.shape[1]:
        raise ValueError("Candidate vector dimension changed.")

    candidate_ids = [
        _article_id(record, "Candidate") for record in candidate_metadata
    ]
    expected_ids = list(original_ids) + list(new_ids)
    if candidate_ids != expected_ids:
        raise ValueError("Candidate metadata order does not match vector order.")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Candidate metadata contains duplicate article IDs.")

    for position in range(candidate_index.ntotal):
        vector = np.asarray(candidate_index.reconstruct(position), dtype=np.float32)
        if vector.shape != (candidate_index.d,) or not np.isfinite(vector).all():
            raise ValueError(f"Candidate vector {position} is invalid.")

    if original_ids:
        persisted_originals = np.vstack(
            [
                candidate_index.reconstruct(position)
                for position in range(len(original_ids))
            ]
        ).astype(np.float32)
        if not np.array_equal(persisted_originals, original_vectors):
            raise ValueError("Candidate did not preserve all original vectors.")

    if new_ids:
        persisted_new = np.vstack(
            [
                candidate_index.reconstruct(len(original_ids) + position)
                for position in range(len(new_ids))
            ]
        ).astype(np.float32)
        if not np.array_equal(persisted_new, normalized_new_vectors):
            raise ValueError(
                "Candidate new vectors do not match their metadata order."
            )


def build_candidate_index(
    existing_index: faiss.Index,
    existing_metadata: Sequence[Mapping[str, Any]],
    new_embeddings: Mapping[ArticleId, np.ndarray],
    new_metadata: Mapping[ArticleId, Mapping[str, Any]],
    output_index_path: OutputPath,
    output_metadata_path: OutputPath,
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """Write and independently validate a candidate index at explicit paths.

    Existing vectors and metadata remain in their original order. New vectors
    are appended in the insertion order of ``new_embeddings`` and normalized
    to match the production inner-product index convention. Existing output
    files are never overwritten.
    """
    if not isinstance(existing_index, faiss.Index):
        raise ValueError("An existing FAISS index is required.")
    if not existing_index.is_trained or existing_index.d <= 0:
        raise ValueError(
            "The existing FAISS index is not trained or has no dimension."
        )
    metric_type = getattr(existing_index, "metric_type", None)
    if metric_type not in (faiss.METRIC_INNER_PRODUCT, faiss.METRIC_L2):
        raise ValueError(f"Unsupported FAISS metric type: {metric_type!r}.")

    existing_records, original_ids = _copy_metadata(
        existing_metadata, int(existing_index.ntotal), "Existing"
    )
    if not isinstance(new_embeddings, Mapping) or not isinstance(new_metadata, Mapping):
        raise ValueError(
            "New embeddings and metadata must be mappings keyed by article ID."
        )

    new_ids = list(new_embeddings.keys())
    for article_id in new_ids:
        if isinstance(article_id, bool) or not isinstance(article_id, int):
            raise ValueError("New embedding keys must be integer article IDs.")
    for article_id in new_metadata:
        if isinstance(article_id, bool) or not isinstance(article_id, int):
            raise ValueError("New metadata keys must be integer article IDs.")
    if set(new_ids) != set(new_metadata.keys()):
        raise ValueError("New embedding IDs and new metadata IDs must match exactly.")
    if set(original_ids).intersection(new_ids):
        raise ValueError("New article IDs duplicate IDs in the existing index.")

    new_records: List[Dict[str, Any]] = []
    normalized_vectors: List[np.ndarray] = []
    for article_id in new_ids:
        record = new_metadata[article_id]
        if _article_id(record, "New") != article_id:
            raise ValueError(
                f"New metadata ID does not match supplied article ID {article_id}."
            )
        copied, _ = _copy_metadata([record], 1, "New")
        vector = np.asarray(new_embeddings[article_id], dtype=np.float32)
        if vector.shape != (existing_index.d,):
            raise ValueError(
                f"Embedding for article {article_id} must have dimension "
                f"{existing_index.d}; got shape {vector.shape}."
            )
        if not np.isfinite(vector).all():
            raise ValueError(f"Embedding for article {article_id} is non-finite.")
        if not np.any(vector):
            raise ValueError(f"Embedding for article {article_id} has zero magnitude.")
        normalized_vectors.append(vector.copy())
        new_records.append(copied[0])

    if existing_index.ntotal:
        original_vectors = np.vstack(
            [
                existing_index.reconstruct(position)
                for position in range(existing_index.ntotal)
            ]
        ).astype(np.float32)
    else:
        original_vectors = np.empty((0, existing_index.d), dtype=np.float32)
    if not np.isfinite(original_vectors).all():
        raise ValueError("Existing index contains non-finite vectors.")
    if len(original_vectors) and np.any(np.all(original_vectors == 0, axis=1)):
        raise ValueError("Existing index contains a zero-magnitude vector.")

    normalized_new_vectors = (
        np.vstack(normalized_vectors).astype(np.float32)
        if normalized_vectors
        else np.empty((0, existing_index.d), dtype=np.float32)
    )
    if len(normalized_new_vectors):
        faiss.normalize_L2(normalized_new_vectors)
        if not np.isfinite(normalized_new_vectors).all() or np.any(
            np.all(normalized_new_vectors == 0, axis=1)
        ):
            raise ValueError("A new embedding could not be normalized safely.")

    index_path, metadata_path = _output_paths(
        output_index_path, output_metadata_path
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    candidate_index = (
        faiss.IndexFlatIP(existing_index.d)
        if metric_type == faiss.METRIC_INNER_PRODUCT
        else faiss.IndexFlatL2(existing_index.d)
    )
    vectors = np.vstack((original_vectors, normalized_new_vectors))
    if len(vectors):
        candidate_index.add(vectors)
    candidate_metadata = existing_records + new_records
    reserved_paths: List[Path] = []
    try:
        for path in (index_path, metadata_path):
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            reserved_paths.append(path)
            os.close(descriptor)

        faiss.write_index(candidate_index, str(index_path))
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(
                candidate_metadata,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        reopened_index = faiss.read_index(str(index_path))
        with metadata_path.open("r", encoding="utf-8") as handle:
            reopened_metadata = json.load(handle)
        reopened_metadata, _ = _copy_metadata(
            reopened_metadata, int(reopened_index.ntotal), "Candidate"
        )
        _validate_candidate(
            reopened_index,
            reopened_metadata,
            original_ids,
            original_vectors,
            new_ids,
            normalized_new_vectors,
        )
    except Exception:
        for path in reserved_paths:
            path.unlink(missing_ok=True)
        raise

    return reopened_index, reopened_metadata
