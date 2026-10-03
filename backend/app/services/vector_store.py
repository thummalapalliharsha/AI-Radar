"""
AI Radar - FAISS vector store (local, CPU-only, free).

    SQLite  ->  Ollama nomic-embed-text  ->  FAISS IndexFlatIP  ->  search

The index is IndexFlatIP over L2-normalised vectors, so an inner-product score
is a cosine similarity in [-1, 1]. Row i of the index always corresponds to
entry i of metadata.json.

Synchronisation reconciles SQLite eligibility with the local index: stale IDs
are removed, missing IDs are embedded, and only changed searchable content is
re-embedded. Existing vectors are retained by ID while the local IndexFlatIP
structure is rewritten. Pass rebuild=True (or `python -m backend.app.main
--rebuild-index`) only to force a full re-embed.

No ChromaDB, no cloud embeddings, no API keys.
"""

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import requests

from backend.app.config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    MIN_SEMANTIC_SIMILARITY,
    OLLAMA_EMBED_TIMEOUT,
    OLLAMA_EMBED_URL,
    TOP_K_SEARCH_RESULTS,
    VECTOR_STORE_DIR,
    normalize_category,
)

TEMPORAL_QUERY_TERMS = (
    "latest",
    "recent",
    "today",
    "yesterday",
    "this week",
    "newly",
    "recently announced",
    "current",
)
from backend.app.utils.database import (
    get_indexable_articles,
    mark_as_indexed,
    reset_embedding_flags,
)

# Kept as module attributes so existing callers and tests keep working.
INDEX_PATH = FAISS_INDEX_PATH
METADATA_PATH = FAISS_META_PATH
OLLAMA_URL = OLLAMA_EMBED_URL


def ensure_vector_dir() -> None:
    """Create the vector-store directory on demand (never at import time)."""
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OLLAMA EMBEDDINGS
# ============================================================

def create_embedding(text: str) -> np.ndarray:
    """
    Embed text with the local Ollama embedding model.

    Returns a float32 vector. Raises RuntimeError when Ollama is unreachable so
    callers can surface a clear "start Ollama" message instead of a stack trace.
    """
    if not text or not text.strip():
        raise ValueError("Cannot create an embedding from empty text.")

    payload = {"model": EMBEDDING_MODEL, "prompt": text}

    try:
        response = requests.post(
            OLLAMA_EMBED_URL, json=payload, timeout=OLLAMA_EMBED_TIMEOUT
        )
        response.raise_for_status()
        embedding = response.json().get("embedding")

        if not embedding:
            raise ValueError("Ollama returned an empty embedding.")

        return np.asarray(embedding, dtype="float32")

    except requests.exceptions.ConnectionError as error:
        raise RuntimeError(
            "Ollama is not reachable. Start Ollama and try again "
            f"(expected at {OLLAMA_EMBED_URL})."
        ) from error
    except requests.exceptions.Timeout as error:
        raise RuntimeError("Ollama embedding request timed out.") from error
    except requests.exceptions.RequestException as error:
        raise RuntimeError(f"Ollama embedding request failed: {error}") from error


def build_article_text(article: Dict[str, Any]) -> str:
    """Flatten an article into the text that gets embedded."""
    title = article.get("title") or ""
    category = normalize_category(article.get("category"))
    source = article.get("source") or ""
    summary = article.get("ai_summary") or article.get("summary") or ""
    why_it_matters = article.get("why_it_matters") or ""

    return "\n".join(
        [
            f"Title: {title}",
            f"Category: {category}",
            f"Source: {source}",
            f"Summary: {summary}",
            f"Why it matters: {why_it_matters}",
        ]
    )


def build_metadata_entry(article: Dict[str, Any]) -> Dict[str, Any]:
    """Build the metadata record stored alongside a vector."""
    return {
        "id": article["id"],
        "title": article.get("title") or "Untitled",
        "url": article.get("url") or "",
        "source": article.get("source") or "",
        "category": normalize_category(article.get("category")),
        "published": article.get("published") or "",
        "published_timestamp": article.get("published_timestamp"),
        "importance": article.get("importance"),
        "career_relevance": article.get("career_relevance"),
        "summary": article.get("ai_summary") or article.get("summary") or "",
        "why_it_matters": article.get("why_it_matters") or "",
    }


def _metadata_signature(metadata: Dict[str, Any]) -> Tuple[Any, ...]:
    """Fields whose changes alter the searchable article representation."""
    return (
        metadata.get("title") or "",
        normalize_category(metadata.get("category")),
        metadata.get("source") or "",
        metadata.get("summary") or "",
        metadata.get("why_it_matters") or "",
    )


def has_temporal_intent(query: str) -> bool:
    """Return whether the user explicitly asks for time-sensitive coverage."""
    normalized = re.sub(r"\s+", " ", (query or "").strip().lower())
    return any(term in normalized for term in TEMPORAL_QUERY_TERMS)


def recency_score(published_timestamp: Any, now: Optional[datetime] = None) -> float:
    """Return a 0..1 freshness score, without affecting normal queries."""
    try:
        published = datetime.fromtimestamp(float(published_timestamp), timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return 0.0
    reference = now or datetime.now(timezone.utc)
    age_days = max(0.0, (reference - published).total_seconds() / 86400.0)
    return max(0.0, 1.0 - min(age_days, 30.0) / 30.0)


# ============================================================
# INDEX PERSISTENCE
# ============================================================

def load_metadata() -> List[Dict[str, Any]]:
    """Read metadata.json, returning [] when it is missing or unreadable."""
    if not METADATA_PATH.exists():
        return []
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError) as error:
        print(f"Could not read {METADATA_PATH.name}: {error}")
        return []


def load_index() -> Tuple[Optional[faiss.Index], List[Dict[str, Any]]]:
    """
    Load the persisted index and its metadata.

    Returns (None, []) when either artefact is missing, unreadable, or when the
    two have drifted out of alignment. Reconciliation then has no safely
    addressable retained vectors and must generate a replacement corpus.
    """
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        return None, []

    try:
        index = faiss.read_index(str(INDEX_PATH))
    except Exception as error:
        print(f"Could not read {INDEX_PATH.name}: {error}")
        return None, []

    metadata = load_metadata()

    if index.ntotal != len(metadata):
        print(
            f"FAISS index and metadata are out of sync "
            f"({index.ntotal} vectors vs {len(metadata)} records). "
            "A full rebuild is required."
        )
        return None, []

    return index, metadata


def save_index(index: faiss.Index, metadata: List[Dict[str, Any]]) -> None:
    """Persist the index and metadata atomically enough for a local tool."""
    ensure_vector_dir()
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def get_index_stats() -> Dict[str, Any]:
    """Health snapshot of the vector store."""
    index, metadata = load_index()

    if index is None:
        return {
            "indexed": 0,
            "metadata": len(load_metadata()),
            "index_exists": INDEX_PATH.exists(),
            "healthy": False,
            "dimension": None,
            "indexed_documents": 0,
            "embedding_model": EMBEDDING_MODEL,
            "index_path": str(INDEX_PATH),
            "metadata_path": str(METADATA_PATH),
        }

    return {
        "indexed": int(index.ntotal),
        "metadata": len(metadata),
        "index_exists": True,
        "healthy": True,
        "dimension": int(index.d),
        "indexed_documents": int(index.ntotal),
        "embedding_model": EMBEDDING_MODEL,
        "index_path": str(INDEX_PATH),
        "metadata_path": str(METADATA_PATH),
    }


# ============================================================
# INCREMENTAL SYNCHRONISATION
# ============================================================

def sync_embeddings(
    batch_size: Optional[int] = None,
    rebuild: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Reconcile FAISS with the currently eligible SQLite articles.

    Existing vectors are retained by article ID. Stale IDs are removed, missing
    IDs are embedded, and only rows whose searchable metadata changed are
    re-embedded. The local IndexFlatIP structure is rewritten from retained and
    newly generated vectors; this is not a full Ollama re-embedding.
    """
    articles = [dict(row) for row in get_indexable_articles()]

    if verbose:
        print("=" * 70)
        print("AI RADAR - FAISS VECTOR INDEX SYNC")
        print("=" * 70)
        print(f"Eligible AI articles : {len(articles)}")
        print(f"Embedding model      : {EMBEDDING_MODEL}")
        print(f"Index file           : {INDEX_PATH}")

    index, metadata = load_index()
    if rebuild:
        index, metadata = None, []
        reset_embedding_flags()
        if verbose:
            print("Mode                 : full rebuild (forced)")
    elif verbose:
        print(f"Mode                 : {'reconcile' if index is not None else 'initial build'}")

    current_by_id = {article["id"]: article for article in articles}
    existing_by_id = {entry.get("id"): (position, entry) for position, entry in enumerate(metadata)}
    eligible_ids = set(current_by_id)
    existing_ids = set(existing_by_id)
    stale_ids = existing_ids - eligible_ids
    missing_ids = eligible_ids - existing_ids
    changed_ids = {
        article_id
        for article_id in eligible_ids & existing_ids
        if _metadata_signature(build_metadata_entry(current_by_id[article_id]))
        != _metadata_signature(existing_by_id[article_id][1])
    }

    retained_ids = sorted(eligible_ids & existing_ids, key=lambda article_id: existing_by_id[article_id][0])
    pending_ids = set(missing_ids) | set(changed_ids)
    pending = [current_by_id[article_id] for article_id in current_by_id if article_id in pending_ids]
    total_pending = len(pending)

    if batch_size is not None and batch_size > 0:
        pending = pending[:batch_size]

    if verbose:
        print(f"Unchanged retained   : {len(retained_ids) - len(changed_ids)}")
        print(f"Stale removed        : {len(stale_ids)}")
        print(f"Missing to embed     : {len(missing_ids)}")
        print(f"Changed to re-embed  : {len(changed_ids)}")
        print(f"Queued for embedding : {total_pending}")
        if len(pending) != total_pending:
            print(f"This run             : {len(pending)} (batch limit)")

    vectors_by_id: Dict[int, np.ndarray] = {}
    metadata_by_id: Dict[int, Dict[str, Any]] = {}
    for article_id in retained_ids:
        position, old_metadata = existing_by_id[article_id]
        if article_id not in changed_ids and index is not None:
            vectors_by_id[article_id] = index.reconstruct(position)
            metadata_by_id[article_id] = old_metadata

    embedded_ids: List[int] = []
    failed = 0

    for position, article in enumerate(pending, start=1):
        title = article.get("title") or "Untitled"
        if verbose:
            print(f"\n[{position}/{len(pending)}] {title[:100]}")

        try:
            vector = create_embedding(build_article_text(article))
        except RuntimeError as error:
            # Ollama is down or timing out: stop instead of failing every row.
            print(f"   Aborting sync: {error}")
            break
        except Exception as error:
            failed += 1
            print(f"   Failed: {error}")
            continue

        vectors_by_id[article["id"]] = vector
        metadata_by_id[article["id"]] = build_metadata_entry(article)
        embedded_ids.append(article["id"])
        if verbose:
            print("   Embedded")

    # Keep an old vector as a temporary, retryable fallback when a changed row
    # could not be embedded. This preserves ID/count alignment without marking
    # the stale representation as current in SQLite.
    if index is not None:
        for article_id in changed_ids - set(embedded_ids):
            position, old_metadata = existing_by_id[article_id]
            vectors_by_id[article_id] = index.reconstruct(position)
            metadata_by_id[article_id] = old_metadata

    if vectors_by_id:
        ordered_ids = [article["id"] for article in articles if article["id"] in vectors_by_id]
        matrix = np.vstack([vectors_by_id[article_id] for article_id in ordered_ids]).astype("float32")
        faiss.normalize_L2(matrix)
        dimension = int(matrix.shape[1])
        if index is not None and index.d != dimension:
            raise RuntimeError(f"Embedding dimension changed ({index.d} -> {dimension}); use --rebuild-index.")
    elif index is not None:
        ordered_ids = []
        dimension = int(index.d)
        matrix = np.empty((0, dimension), dtype="float32")
    else:
        ordered_ids = []
        dimension = None
        matrix = np.empty((0, 0), dtype="float32")

    if dimension is not None:
        reconciled_index = faiss.IndexFlatIP(dimension)
        if len(matrix):
            reconciled_index.add(matrix)
        reconciled_metadata = [metadata_by_id[article_id] for article_id in ordered_ids]
        save_index(reconciled_index, reconciled_metadata)
        unchanged_ids = [article_id for article_id in retained_ids if article_id not in changed_ids]
        mark_as_indexed(embedded_ids + unchanged_ids)
        indexed_count = int(reconciled_index.ntotal)
    else:
        indexed_count = 0

    remaining = len(pending_ids - set(embedded_ids))

    if verbose:
        print("\n" + "=" * 70)
        print("FAISS VECTOR INDEX SYNC COMPLETE")
        print("=" * 70)
        print(f"Added/missing      : {len(missing_ids & set(embedded_ids))}")
        print(f"Updated/re-embedded: {len(changed_ids & set(embedded_ids))}")
        print(f"Removed/stale      : {len(stale_ids)}")
        print(f"Unchanged retained : {len(retained_ids) - len(changed_ids)}")
        print(f"Total vectors      : {indexed_count}")
        print(f"Vector dimension   : {dimension}")
        print(f"Failed            : {failed}")
        print(f"Still queued      : {max(0, remaining)}")
        if remaining > 0:
            print(
                f"NOTE: {remaining} analysed article(s) have no embedding yet, so "
                "semantic search cannot retrieve them."
            )
            print("      Finish with: python -m backend.app.main --index")
        print(f"Index file        : {INDEX_PATH}")
        print(f"Metadata file     : {METADATA_PATH}")

    return {
        "indexed": indexed_count,
        "added": len(missing_ids & set(embedded_ids)),
        "updated": len(changed_ids & set(embedded_ids)),
        "removed": len(stale_ids),
        "unchanged": len(retained_ids) - len(changed_ids),
        "failed": failed,
        "pending": remaining,
    }


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def search_articles(
    query: str,
    top_k: int = TOP_K_SEARCH_RESULTS,
    min_similarity: Optional[float] = MIN_SEMANTIC_SIMILARITY,
) -> List[Dict[str, Any]]:
    """
    Cosine-similarity search over the AI Radar corpus.

        query -> Ollama embedding -> FAISS IndexFlatIP -> threshold filter

    Every hit carries `similarity` (1.0 = identical) and `distance`
    (1 - similarity). Hits below `min_similarity` are dropped, which is what
    stops the dashboard from grounding an answer in unrelated articles; pass
    min_similarity=None to inspect raw neighbours.
    """
    if not query or not query.strip():
        return []

    index, metadata = load_index()
    if index is None or index.ntotal == 0:
        return []

    try:
        query_vector = create_embedding(query).reshape(1, -1).astype("float32")
    except Exception as error:
        print(f"Could not embed the query: {error}")
        return []

    if query_vector.shape[1] != index.d:
        print(
            f"Query dimension {query_vector.shape[1]} does not match index "
            f"dimension {index.d}. Re-run `python -m backend.app.main --rebuild-index`."
        )
        return []

    faiss.normalize_L2(query_vector)

    try:
        requested = max(1, int(top_k))
    except (TypeError, ValueError):
        requested = TOP_K_SEARCH_RESULTS

    temporal = has_temporal_intent(query)
    candidate_count = min(requested * 3 if temporal else requested, index.ntotal)
    scores, indices = index.search(query_vector, candidate_count)

    results: List[Dict[str, Any]] = []
    for score, position in zip(scores[0], indices[0]):
        position = int(position)
        if position < 0 or position >= len(metadata):
            continue

        similarity = float(score)
        if min_similarity is not None and similarity < min_similarity:
            continue

        article = dict(metadata[position])
        article["category"] = normalize_category(article.get("category"))
        article["similarity"] = similarity
        article["distance"] = round(1.0 - similarity, 6)
        article["temporal_score"] = recency_score(article.get("published_timestamp"))
        results.append(article)

    if temporal:
        results.sort(
            key=lambda article: (
                article["similarity"] + 0.12 * article["temporal_score"],
                article["similarity"],
            ),
            reverse=True,
        )
        results = results[:requested]
    else:
        results = results[:requested]

    return results


def search_vector_store(
    query: str,
    top_k: int = TOP_K_SEARCH_RESULTS,
    min_similarity: Optional[float] = MIN_SEMANTIC_SIMILARITY,
) -> List[Dict[str, Any]]:
    """Backward-compatible alias used by the dashboard and the tests."""
    return search_articles(query, top_k=top_k, min_similarity=min_similarity)


if __name__ == "__main__":
    stats = get_index_stats()
    print("AI Radar - FAISS Vector Store")
    print("-" * 70)
    print(f"Index path       : {stats['index_path']}")
    print(f"Index exists     : {stats['index_exists']}")
    print(f"Healthy          : {stats['healthy']}")
    print(f"Indexed vectors  : {stats['indexed_documents']}")
    print(f"Metadata records : {stats['metadata']}")
    print(f"Dimension        : {stats['dimension']}")
    print(f"Embedding model  : {stats['embedding_model']}")
    print(f"Min similarity   : {MIN_SEMANTIC_SIMILARITY}")

    print("\nSynchronising...")
    print(sync_embeddings())



