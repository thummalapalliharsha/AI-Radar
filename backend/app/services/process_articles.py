import sys
import argparse
from typing import Dict

"""
AI Radar - batch article analysis through local Ollama.

Each run drains a bounded batch so a long backlog can be processed across
several invocations without holding a single long-running process open.
"""

import re
from typing import Any, List, Optional, Sequence

from backend.app.config import DEFAULT_BATCH_SIZE, RECENT_DAYS
from backend.app.services.ai_analyzer import analyze_article
from backend.app.services.news_collector import RSS_SOURCES
from backend.app.services.ranking import STRONG_AI_KEYWORDS, normalize_text
from backend.app.utils.database import (
    count_unprocessed_articles,
    get_articles_by_ids,
    get_unprocessed_candidates,
    initialize_database,
    update_ai_analysis,
)

# ============================================================
# ANALYSIS QUEUE PRIORITY
# ============================================================
# Ollama analysis is the scarce resource: a run analyses `limit` articles while
# far more may be pending. Ordering the queue newest-first starves it - each new
# collection pushes yesterday's announcements further down, so a flagship post
# from a primary lab can wait behind an unbounded stream of low-value items and
# never be analysed, indexed, or made searchable.
#
# The queue is therefore ordered by how likely an article is to matter (primary
# source tier, then AI signal in its own text), and oldest-first within the same
# band so nothing can be starved indefinitely.

SOURCE_TIERS = {source["name"]: source.get("tier", 3) for source in RSS_SOURCES}
TIER_PRIORITY = {1: 2, 2: 1, 3: 0}
UNKNOWN_SOURCE_PRIORITY = 0

_STRONG_AI_PATTERNS = [
    re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)") for keyword in STRONG_AI_KEYWORDS
]


def count_strong_ai_terms(text: Any) -> int:
    """How many distinct strong AI terms appear in the text, on word boundaries."""
    normalised = normalize_text(text)
    if not normalised:
        return 0
    return sum(1 for pattern in _STRONG_AI_PATTERNS if pattern.search(normalised))


def source_priority(source: Any) -> int:
    """2 for a primary lab feed, 1 for authoritative media, 0 otherwise."""
    tier = SOURCE_TIERS.get(str(source or ""), 3)
    return TIER_PRIORITY.get(tier, UNKNOWN_SOURCE_PRIORITY)


def ai_signal(title: Any, summary: Any) -> int:
    """0-3, from how densely the article's own text reads as AI coverage."""
    title_hits = count_strong_ai_terms(title)
    summary_hits = count_strong_ai_terms(summary)

    signal = 0
    if title_hits >= 1:
        signal += 1
    if title_hits >= 3:
        signal += 1
    if summary_hits >= 2:
        signal += 1
    return signal


def analysis_priority(article: Any) -> int:
    """0-5; a higher value is analysed sooner."""
    return source_priority(article["source"]) + ai_signal(
        article["title"], article["summary"]
    )


def prioritise_queue(candidates: Sequence[Any], limit: int) -> List[Any]:
    """
    Order the pending set for analysis and return the first `limit` IDs.

    Sort key: priority descending, then publication time ascending. The
    ascending tiebreak is what guarantees the backlog drains - the oldest
    unanalysed article in a band is always taken before a newer one.
    """
    ordered = sorted(
        candidates,
        key=lambda row: (
            -analysis_priority(row),
            row["published_timestamp"] or 0.0,
        ),
    )
    return [row["id"] for row in ordered[: max(0, limit)]]


def newest_first_queue(candidates: Sequence[Any], limit: int) -> List[Any]:
    """Select a bounded, deterministic batch of the newest pending articles."""
    ordered = sorted(
        candidates,
        key=lambda row: (
            -(row["published_timestamp"] or 0.0),
            row["id"],
        ),
    )
    return [row["id"] for row in ordered[: max(0, limit)]]


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def next_analysis_preview(
    recent_days: Optional[int] = RECENT_DAYS,
    limit: int = 5,
) -> List[Any]:
    """The articles the next analysis run would pick, in the order it picks them."""
    candidates = get_unprocessed_candidates(recent_days=recent_days)
    if not candidates:
        return []
    return get_articles_by_ids(prioritise_queue(candidates, limit))


def process_articles_batch(
    limit: int = DEFAULT_BATCH_SIZE,
    recent_days: Optional[int] = RECENT_DAYS,
    newest_first: bool = False,
) -> Dict[str, int]:
    """
    Analyse a batch of unprocessed articles with local Ollama.

    Pass recent_days=None to work through the entire backlog, including archive
    entries older than the collection window.
    """
    print("=" * 70)
    print("AI RADAR â€” ARTICLE ANALYSIS PIPELINE")
    print("=" * 70)

    initialize_database()

    window = "entire backlog" if recent_days is None else f"last {recent_days} days"
    print(f"Window: {window}")

    # Rank the whole pending set, then take the batch off the top. Slicing in
    # SQL first is what let a major announcement stay unanalysed run after run.
    candidates = get_unprocessed_candidates(recent_days=recent_days)
    pending_total = len(candidates)

    if pending_total == 0:
        print("âœ… Backlog clear: No unprocessed articles found in the collection window.")
        return {"processed": 0, "successful": 0, "failed": 0, "remaining": 0}

    selected_ids = (
        newest_first_queue(candidates, limit)
        if newest_first
        else prioritise_queue(candidates, limit)
    )
    unprocessed = get_articles_by_ids(selected_ids)
    total_found = len(unprocessed)

    print(f"Pending in window: {pending_total}")

    queue_description = "newest pending" if newest_first else "highest-priority"
    print(f"ðŸ”„ Analysing the {total_found} {queue_description} articles...\n")

    successful = 0
    failed = 0

    for idx, row in enumerate(unprocessed, start=1):
        article = dict(row)
        title = article.get("title", "")
        source = article.get("source", "")
        article_id = article["id"]
        priority = analysis_priority(row)

        print(
            f"[{idx}/{total_found}] (p{priority}) {title[:68]}... ({source})"
        )

        try:
            analysis = analyze_article(article)
            if analysis:
                update_ai_analysis(article_id, analysis)
                successful += 1
                status_icon = "â­ AI News" if analysis["is_ai_news"] else "ðŸš« Non-AI (Filtered)"
                print(f"    â””â”€ {status_icon} | Cat: {analysis['category']} | Imp: {analysis['importance']}/10")
            else:
                failed += 1
                print("    â””â”€ âŒ Analysis failed or timed out.")
        except Exception as error:
            failed += 1
            print(f"    â””â”€ âŒ Error: {error}")

    remaining_count = count_unprocessed_articles(recent_days=recent_days)

    print("\n" + "=" * 70)
    print(f"BATCH COMPLETE: {successful} successful, {failed} failed.")
    print(f"Remaining unprocessed ({window}): {remaining_count}")
    if remaining_count:
        # Say this loudly: an unanalysed article is invisible to the feed, the
        # briefings and semantic search, so a silent shortfall looks like loss.
        print(
            f"NOTE: {remaining_count} collected article(s) are still unanalysed,"
            " so they are not searchable yet."
        )
        print("      Finish the window with: python -m backend.app.main --process --drain")
    print("=" * 70)

    return {
        "processed": total_found,
        "successful": successful,
        "failed": failed,
        "remaining": remaining_count,
        "pending_before": pending_total,
    }


def drain_articles(
    limit: int = DEFAULT_BATCH_SIZE,
    recent_days: Optional[int] = RECENT_DAYS,
    max_batches: int = 200,
) -> Dict[str, int]:
    """
    Keep analysing in `limit`-sized batches until the window has nothing left.

    One bounded batch cannot keep up with a collection run that adds dozens of
    articles, so the pending set only ever grew and anything below the batch
    size stayed invisible to search. Draining gives the pipeline a coherent end
    state: everything collected in the window is classified, and therefore
    indexable and searchable.
    """
    totals = {
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "remaining": 0,
        "batches": 0,
    }

    for batch in range(1, max_batches + 1):
        print()
        print(f"--- drain batch {batch} ---")
        result = process_articles_batch(limit=limit, recent_days=recent_days)

        totals["processed"] += result["processed"]
        totals["successful"] += result["successful"]
        totals["failed"] += result["failed"]
        totals["remaining"] = result["remaining"]
        totals["batches"] = batch

        if result["remaining"] == 0:
            break
        if result["processed"] == 0:
            # Rows remain but none are selectable: stop instead of spinning.
            break
        if result["successful"] == 0:
            print("Every article in this batch failed; stopping so Ollama can be checked.")
            break

    print()
    print("=" * 70)
    print(
        f"DRAIN COMPLETE: {totals['successful']} analysed across "
        f"{totals['batches']} batch(es), {totals['failed']} failed, "
        f"{totals['remaining']} still pending."
    )
    print("=" * 70)
    return totals


def main():
    parser = argparse.ArgumentParser(description="AI Radar Batch Article Processor")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of articles to analyze in this run (default: 10)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=RECENT_DAYS,
        help=f"Lookback window in days (default: {RECENT_DAYS})",
    )
    parser.add_argument(
        "--backlog",
        action="store_true",
        help="Ignore the lookback window and analyse the whole backlog",
    )
    parser.add_argument(
        "--drain",
        action="store_true",
        help="Repeat batches until nothing is left unanalysed in the window",
    )
    args = parser.parse_args()
    window_days = None if args.backlog else args.days
    if args.drain:
        drain_articles(limit=args.limit, recent_days=window_days)
    else:
        process_articles_batch(limit=args.limit, recent_days=window_days)


if __name__ == "__main__":
    main()

