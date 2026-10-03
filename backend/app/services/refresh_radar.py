"""Dashboard-triggered incremental refresh orchestration."""

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from backend.app.config import (
    DEFAULT_BATCH_SIZE,
    RECENT_DAYS,
    REFRESH_STATE_FILE,
)
from backend.app.services.news_collector import run_collection
from backend.app.services.process_articles import process_articles_batch
from backend.app.services.vector_store import sync_embeddings
from backend.app.utils.database import get_news_stats


def get_last_successful_refresh() -> Optional[str]:
    """Return the persisted completion time, never an article collection time."""
    try:
        state = json.loads(REFRESH_STATE_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None

    if not isinstance(state, dict):
        raise ValueError("Refresh state must contain a JSON object")
    value = state.get("last_successful_refresh")
    return value if isinstance(value, str) else None


def _save_successful_refresh(timestamp: str) -> None:
    REFRESH_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=REFRESH_STATE_FILE.parent,
        prefix=f"{REFRESH_STATE_FILE.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = temporary_file.name
        json.dump({"last_successful_refresh": timestamp}, temporary_file, indent=2)
        temporary_file.write("\n")
    try:
        os.replace(temporary_path, REFRESH_STATE_FILE)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def refresh_radar(
    recent_days: int = RECENT_DAYS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """Collect, analyze one bounded newest batch, and reconcile eligible index rows."""
    before = get_news_stats()
    collection = run_collection(recent_days=recent_days)
    processing = process_articles_batch(
        limit=batch_size,
        recent_days=recent_days,
        newest_first=True,
    )
    indexing = sync_embeddings(verbose=False)
    after = get_news_stats()

    failures = int(processing.get("failed", 0))
    index_failures = int(indexing.get("failed", 0))
    index_pending = int(indexing.get("pending", 0))
    feeds_succeeded = int(collection.get("feeds_succeeded", 0))
    feed_failures = collection.get("feed_failures", [])
    save_failures = collection.get("save_failures", [])

    errors = []
    if feeds_succeeded == 0:
        errors.append("No RSS feeds could be collected")
    if save_failures:
        first_save_failure = save_failures[0]
        errors.append(
            f"{len(save_failures)} article(s) failed to save: "
            f"{first_save_failure.get('title', '')}: "
            f"{first_save_failure.get('error', '')}"
        )
    if failures:
        errors.append(f"{failures} article(s) failed analysis")
    if index_failures:
        errors.append(f"{index_failures} article(s) failed indexing")
    if index_pending:
        errors.append(f"{index_pending} eligible article(s) remain unindexed")
    success = not errors

    last_successful_refresh = None
    if success:
        last_successful_refresh = datetime.now(timezone.utc).isoformat()
        _save_successful_refresh(last_successful_refresh)

    return {
        "success": success,
        "partial": success and bool(feed_failures),
        "new_articles": int(collection.get("new_inserted", 0)),
        "collected_articles": int(collection.get("collected_total", 0)),
        "feeds_succeeded": feeds_succeeded,
        "feeds_total": int(collection.get("feeds_total", feeds_succeeded)),
        "feed_failures": feed_failures,
        "save_failures": save_failures,
        "new_verified_ai_stories": max(
            0, int(after.get("ai_news_count", 0)) - int(before.get("ai_news_count", 0))
        ),
        "processed": int(processing.get("processed", 0)),
        "failed": failures,
        "remaining": int(processing.get("remaining", 0)),
        "indexed_articles": int(indexing.get("indexed", 0)),
        "index_added": int(indexing.get("added", 0)),
        "index_updated": int(indexing.get("updated", 0)),
        "index_removed": int(indexing.get("removed", 0)),
        "index_unchanged": int(indexing.get("unchanged", 0)),
        "index_failed": index_failures,
        "index_pending": index_pending,
        "error": "; ".join(errors) if errors else None,
        "last_successful_refresh": last_successful_refresh,
    }
