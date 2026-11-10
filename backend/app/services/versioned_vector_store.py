"""Read and atomically switch validated, immutable FAISS generations."""

import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Generator, List, Mapping, Optional, Sequence, Tuple, Union

import faiss
import numpy as np


class GenerationError(ValueError):
    """Raised when a generation manifest or its artifacts are invalid."""


_METRICS = {
    "inner_product": faiss.METRIC_INNER_PRODUCT,
}
_GENERATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_LOCKS: Dict[str, RLock] = {}
_LOCKS_GUARD = RLock()


@contextmanager
def exclusive_file_lock(
    lock_path: Union[os.PathLike, str]
) -> Generator[None, None, None]:
    """Serialize local readers/writers across threads and processes."""
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = str(path.resolve())
    with _LOCKS_GUARD:
        thread_lock = _LOCKS.setdefault(key, RLock())

    with thread_lock:
        with path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                while True:
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as error:
                        if error.errno not in (errno.EACCES, errno.EDEADLK):
                            raise
                        time.sleep(0.01)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(manifest_path: Path) -> Dict[str, Any]:
    for attempt in range(25):
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            break
        except PermissionError as error:
            if attempt == 24:
                raise GenerationError(
                    f"Could not read generation manifest {manifest_path}: {error}"
                ) from error
            time.sleep(0.01)
        except FileNotFoundError as error:
            raise GenerationError(
                f"Generation manifest does not exist: {manifest_path}"
            ) from error
        except (OSError, json.JSONDecodeError) as error:
            raise GenerationError(
                f"Could not read generation manifest {manifest_path}: {error}"
            ) from error

    version = manifest.get("version") if isinstance(manifest, dict) else None
    if isinstance(version, bool) or version != 1:
        raise GenerationError("Generation manifest must be an object with version 1.")
    active = manifest.get("active_generation")
    generations = manifest.get("generations")
    if not isinstance(active, str) or not active:
        raise GenerationError("Manifest active_generation must be a non-empty string.")
    if not isinstance(generations, dict) or active not in generations:
        raise GenerationError(
            "Manifest active_generation must identify an entry in generations."
        )
    for generation_id, entry in generations.items():
        if not isinstance(generation_id, str) or not generation_id:
            raise GenerationError("Manifest generation IDs must be non-empty strings.")
        _validate_entry_shape(entry, generation_id)
    return manifest


def _validate_entry_shape(entry: Any, generation_id: str) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        raise GenerationError(f"Generation {generation_id!r} must be an object.")
    for key in ("index", "metadata", "metric", "index_sha256", "metadata_sha256"):
        if not isinstance(entry.get(key), str) or not entry[key]:
            raise GenerationError(
                f"Generation {generation_id!r} has an invalid {key!r} field."
            )
    for key in ("count", "dimension"):
        value = entry.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise GenerationError(
                f"Generation {generation_id!r} has an invalid {key!r} field."
            )
    if entry["count"] < 0 or entry["dimension"] <= 0:
        raise GenerationError(
            f"Generation {generation_id!r} has invalid count or dimension values."
        )
    if entry["metric"] not in _METRICS:
        raise GenerationError(
            f"Generation {generation_id!r} has unsupported metric "
            f"{entry['metric']!r}."
        )
    for key in ("index_sha256", "metadata_sha256"):
        if (
            len(entry[key]) != 64
            or any(character not in "0123456789abcdef" for character in entry[key])
        ):
            raise GenerationError(
                f"Generation {generation_id!r} has an invalid {key!r} value."
            )
    return entry


def _generation_path(root: Path, relative_path: str, generation_id: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise GenerationError(
            f"Generation {generation_id!r} artifact paths must be relative."
        )
    resolved_root = root.resolve()
    resolved_path = (root / path).resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise GenerationError(
            f"Generation {generation_id!r} artifact path escapes the manifest directory."
        )
    return resolved_path


def _validate_artifacts(
    index: faiss.Index, metadata: Any, generation_id: str
) -> Tuple[List[Dict[str, Any]], str]:
    if not index.is_trained or index.d <= 0:
        raise GenerationError(
            f"Generation {generation_id!r} index is not trained or has no dimension."
        )
    if index.metric_type != _METRICS["inner_product"]:
        raise GenerationError(
            f"Generation {generation_id!r} index metric is unsupported."
        )
    if not isinstance(metadata, list) or len(metadata) != index.ntotal:
        raise GenerationError(
            f"Generation {generation_id!r} metadata count does not match its index."
        )

    ids: List[int] = []
    records: List[Dict[str, Any]] = []
    for position, item in enumerate(metadata):
        if not isinstance(item, dict):
            raise GenerationError(
                f"Generation {generation_id!r} metadata record {position} is not an object."
            )
        article_id = item.get("id")
        if isinstance(article_id, bool) or not isinstance(article_id, int):
            raise GenerationError(
                f"Generation {generation_id!r} metadata record {position} "
                "has no integer article ID."
            )
        ids.append(article_id)
        records.append(item)
    if len(set(ids)) != len(ids):
        raise GenerationError(
            f"Generation {generation_id!r} metadata contains duplicate article IDs."
        )

    if index.ntotal:
        vectors = np.asarray(index.reconstruct_n(0, index.ntotal), dtype=np.float32)
        if vectors.shape != (index.ntotal, index.d) or not np.isfinite(vectors).all():
            raise GenerationError(
                f"Generation {generation_id!r} contains invalid vector values."
            )
    return records, "inner_product"


def _entry_for_files(
    root: Path, generation_id: str, index_path: Path, metadata_path: Path
) -> Dict[str, Any]:
    try:
        index = faiss.read_index(str(index_path))
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError, RuntimeError) as error:
        raise GenerationError(
            f"Could not validate generation {generation_id!r}: {error}"
        ) from error
    _, metric = _validate_artifacts(index, metadata, generation_id)
    return {
        "index": index_path.relative_to(root).as_posix(),
        "metadata": metadata_path.relative_to(root).as_posix(),
        "count": int(index.ntotal),
        "dimension": int(index.d),
        "metric": metric,
        "index_sha256": _sha256(index_path),
        "metadata_sha256": _sha256(metadata_path),
    }


def _write_manifest_atomic(path: Path, manifest: Dict[str, Any]) -> None:
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(25):
            try:
                os.replace(temporary_path, path)
                break
            except PermissionError:
                if attempt == 24:
                    raise
                time.sleep(0.01)
        temporary_path = None
    except OSError as error:
        raise GenerationError(
            f"Could not atomically update generation manifest {path}: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_generation_id(generation_id: str) -> None:
    if (
        not isinstance(generation_id, str)
        or not _GENERATION_ID_PATTERN.fullmatch(generation_id)
        or generation_id in {".", ".."}
    ):
        raise GenerationError(f"Invalid generation ID: {generation_id!r}.")


def _manifest_lock_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(f".{manifest_path.name}.lock")


def _sync_lock_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(f".{manifest_path.name}.sync.lock")


def _stage_generation(
    root: Path,
    generation_id: str,
    index: faiss.Index,
    metadata: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    _validate_generation_id(generation_id)
    if not isinstance(index, faiss.Index):
        raise GenerationError("A FAISS index is required.")
    try:
        serialized_metadata = json.dumps(
            list(metadata), indent=2, ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise GenerationError(f"Generation metadata is not valid JSON: {error}") from error

    normalized_metadata = json.loads(serialized_metadata)
    _validate_artifacts(index, normalized_metadata, generation_id)
    generations_dir = root / "generations"
    generations_dir.mkdir(parents=True, exist_ok=True)
    generation_dir = generations_dir / generation_id
    if generation_dir.exists():
        raise GenerationError(f"Generation already exists: {generation_id!r}.")

    staging_dir = generations_dir / f".staging-{uuid.uuid4().hex}"
    staging_dir.mkdir()
    try:
        index_path = staging_dir / "index.faiss"
        metadata_path = staging_dir / "metadata.json"
        faiss.write_index(index, str(index_path))
        metadata_path.write_text(serialized_metadata, encoding="utf-8")
        staged_entry = _entry_for_files(
            root, generation_id, index_path, metadata_path
        )
        _load_entry(root, generation_id, {
            **staged_entry,
            "index": index_path.relative_to(root).as_posix(),
            "metadata": metadata_path.relative_to(root).as_posix(),
        })
        os.replace(staging_dir, generation_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    entry = dict(staged_entry)
    entry["index"] = (generation_dir / "index.faiss").relative_to(root).as_posix()
    entry["metadata"] = (generation_dir / "metadata.json").relative_to(root).as_posix()
    return entry


def initialize_generation_manifest(
    manifest_path: Union[os.PathLike, str],
    generations: Mapping[str, Tuple[Union[os.PathLike, str], Union[os.PathLike, str]]],
    active_generation: str,
) -> None:
    """Copy and validate named source pairs into a new manifest root."""
    path = Path(manifest_path)
    if path.exists():
        raise GenerationError(f"Generation manifest already exists: {path}")
    if not generations or active_generation not in generations:
        raise GenerationError("The active generation must identify a supplied pair.")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _manifest_lock_path(path)

    with exclusive_file_lock(lock_path):
        if path.exists():
            raise GenerationError(f"Generation manifest already exists: {path}")
        generation_entries: Dict[str, Dict[str, Any]] = {}
        generations_dir = path.parent / "generations"
        generations_dir.mkdir(parents=True, exist_ok=True)
        for generation_id, source_pair in generations.items():
            _validate_generation_id(generation_id)
            if len(source_pair) != 2:
                raise GenerationError(
                    f"Generation {generation_id!r} must have index and metadata paths."
                )
            generation_dir = generations_dir / generation_id
            if generation_dir.exists():
                raise GenerationError(f"Generation already exists: {generation_id!r}.")
            staging_dir = generations_dir / f".staging-{uuid.uuid4().hex}"
            staging_dir.mkdir()
            try:
                index_source, metadata_source = map(Path, source_pair)
                shutil.copyfile(index_source, staging_dir / "index.faiss")
                shutil.copyfile(metadata_source, staging_dir / "metadata.json")
                entry = _entry_for_files(
                    path.parent,
                    generation_id,
                    staging_dir / "index.faiss",
                    staging_dir / "metadata.json",
                )
                _load_entry(
                    path.parent,
                    generation_id,
                    {
                        **entry,
                        "index": (
                            staging_dir / "index.faiss"
                        ).relative_to(path.parent).as_posix(),
                        "metadata": (
                            staging_dir / "metadata.json"
                        ).relative_to(path.parent).as_posix(),
                    },
                )
                os.replace(staging_dir, generation_dir)
                entry["index"] = (
                    generation_dir / "index.faiss"
                ).relative_to(path.parent).as_posix()
                entry["metadata"] = (
                    generation_dir / "metadata.json"
                ).relative_to(path.parent).as_posix()
                generation_entries[generation_id] = entry
            except Exception:
                shutil.rmtree(staging_dir, ignore_errors=True)
                raise

        _write_manifest_atomic(
            path,
            {
                "version": 1,
                "active_generation": active_generation,
                "generations": generation_entries,
            },
        )


def active_generation_id(manifest_path: Union[os.PathLike, str]) -> str:
    """Return the manifest's currently active generation ID."""
    return _read_manifest(Path(manifest_path))["active_generation"]


def generation_details(
    manifest_path: Union[os.PathLike, str], generation_id: Optional[str] = None
) -> Dict[str, Any]:
    """Return validated, non-sensitive details for the selected generation."""
    path = Path(manifest_path)
    manifest = _read_manifest(path)
    selected_id = generation_id or manifest["active_generation"]
    entry = manifest["generations"].get(selected_id)
    if entry is None:
        raise GenerationError(f"Generation {selected_id!r} is not present in the manifest.")
    _load_entry(path.parent, selected_id, entry)
    return {
        "generation_id": selected_id,
        "index_path": str(_generation_path(path.parent, entry["index"], selected_id)),
        "metadata_path": str(
            _generation_path(path.parent, entry["metadata"], selected_id)
        ),
        "indexed": entry["count"],
        "metadata": entry["count"],
        "dimension": entry["dimension"],
        "metric": entry["metric"],
    }


def write_and_activate_generation(
    manifest_path: Union[os.PathLike, str],
    index: faiss.Index,
    metadata: Sequence[Mapping[str, Any]],
    expected_active_generation: Optional[str] = None,
) -> str:
    """Persist an immutable validated generation, then atomically activate it."""
    path = Path(manifest_path)
    lock_path = _manifest_lock_path(path)
    with exclusive_file_lock(lock_path):
        manifest = _read_manifest(path)
        if (
            expected_active_generation is not None
            and manifest["active_generation"] != expected_active_generation
        ):
            raise GenerationError(
                "The active generation changed during indexing; refusing to publish "
                "a generation based on stale data."
            )

        generation_id = f"generation-{uuid.uuid4().hex}"
        entry = _stage_generation(path.parent, generation_id, index, metadata)
        manifest["generations"][generation_id] = entry
        manifest["active_generation"] = generation_id
        _write_manifest_atomic(path, manifest)
        return generation_id


def _load_entry(
    root: Path, generation_id: str, entry: Dict[str, Any]
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    index_path = _generation_path(root, entry["index"], generation_id)
    metadata_path = _generation_path(root, entry["metadata"], generation_id)
    if index_path == metadata_path:
        raise GenerationError(
            f"Generation {generation_id!r} uses the same path for index and metadata."
        )
    for path in (index_path, metadata_path):
        if not path.is_file():
            raise GenerationError(
                f"Generation {generation_id!r} is incomplete; artifact is missing: {path}"
            )
    try:
        index_hash = _sha256(index_path)
        metadata_hash = _sha256(metadata_path)
    except OSError as error:
        raise GenerationError(
            f"Could not verify generation {generation_id!r} artifacts: {error}"
        ) from error
    if index_hash != entry["index_sha256"]:
        raise GenerationError(
            f"Generation {generation_id!r} index hash does not match the manifest."
        )
    if metadata_hash != entry["metadata_sha256"]:
        raise GenerationError(
            f"Generation {generation_id!r} metadata hash does not match the manifest."
        )

    try:
        index = faiss.read_index(str(index_path))
    except Exception as error:
        raise GenerationError(
            f"Could not read generation {generation_id!r} FAISS index: {error}"
        ) from error
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise GenerationError(
            f"Could not read generation {generation_id!r} metadata: {error}"
        ) from error

    if not index.is_trained or index.d != entry["dimension"]:
        raise GenerationError(
            f"Generation {generation_id!r} index dimension/training state "
            "does not match the manifest."
        )
    if index.ntotal != entry["count"]:
        raise GenerationError(
            f"Generation {generation_id!r} vector count does not match the manifest."
        )
    if index.metric_type != _METRICS[entry["metric"]]:
        raise GenerationError(
            f"Generation {generation_id!r} index metric does not match the manifest."
        )
    _validate_artifacts(index, metadata, generation_id)
    return index, metadata


def load_generation(
    manifest_path: Union[os.PathLike, str], generation_id: Optional[str] = None
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """Load one validated generation, defaulting to the manifest's active ID."""
    path = Path(manifest_path)
    manifest = _read_manifest(path)
    selected_id = (
        generation_id
        if generation_id is not None
        else manifest["active_generation"]
    )
    entry = manifest["generations"].get(selected_id)
    if entry is None:
        raise GenerationError(
            f"Generation {selected_id!r} is not present in the manifest."
        )
    return _load_entry(path.parent, selected_id, entry)


def switch_active_generation(
    manifest_path: Union[os.PathLike, str], generation_id: str
) -> None:
    """Validate a complete generation, then atomically publish it as active."""
    path = Path(manifest_path)
    with exclusive_file_lock(_sync_lock_path(path)):
        with exclusive_file_lock(_manifest_lock_path(path)):
            manifest = _read_manifest(path)
            entry = manifest["generations"].get(generation_id)
            if entry is None:
                raise GenerationError(
                    f"Generation {generation_id!r} is not present in the manifest."
                )
            _load_entry(path.parent, generation_id, entry)
            manifest["active_generation"] = generation_id
            _write_manifest_atomic(path, manifest)
