"""
AI Radar - pipeline orchestrator CLI.

    python -m backend.app.main --status
    python -m backend.app.main --collect --process --index
    python -m backend.app.main --morning
    python -m backend.app.main --evening
    python -m backend.app.main --full-pipeline --limit 10

Every subcommand initialises the SQLite schema first, and no subcommand ever
deletes collected articles, briefing snapshots or the FAISS index.
"""

import argparse
import sys
from typing import Any, Dict

import requests

from backend.app.config import (
    DEFAULT_BATCH_SIZE,
    EMBEDDING_MODEL,
    MIN_SEMANTIC_SIMILARITY,
    OLLAMA_HEALTH_TIMEOUT,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_TAGS_URL,
    RECENT_DAYS,
)
from backend.app.services.briefing_generator import build_briefing, determine_mode
from backend.app.services.news_collector import run_collection
from backend.app.services.process_articles import (
    analysis_priority,
    drain_articles,
    next_analysis_preview,
    process_articles_batch,
)
from backend.app.services.vector_store import get_index_stats, sync_embeddings
from backend.app.utils.database import (
    count_unprocessed_articles,
    get_category_counts,
    get_news_stats,
    get_unindexed_articles,
    initialize_database,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def check_ollama_health() -> Dict[str, Any]:
    """Report whether the local Ollama daemon exposes both required models."""
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=OLLAMA_HEALTH_TIMEOUT)
        response.raise_for_status()
        models = [m.get("name", "") for m in response.json().get("models", [])]
        return {
            "status": "online",
            "models": models,
            "has_llm": any(OLLAMA_MODEL.split(":")[0] in m for m in models),
            "has_embedding": any(EMBEDDING_MODEL.split(":")[0] in m for m in models),
        }
    except requests.exceptions.RequestException:
        return {
            "status": "offline",
            "models": [],
            "has_llm": False,
            "has_embedding": False,
        }


def print_system_status() -> None:
    """Print database, vector-store and Ollama telemetry."""
    print("\n" + "=" * 70)
    print("AI RADAR - SYSTEM TELEMETRY & STATUS")
    print("=" * 70)

    info = initialize_database()
    stats = get_news_stats()
    backlog_window = count_unprocessed_articles(recent_days=RECENT_DAYS)
    backlog_total = count_unprocessed_articles(recent_days=None)
    unindexed = len(get_unindexed_articles(limit=100000))

    print("\nDATABASE & REPOSITORY")
    print(f"  File                        : {info['database']}")
    print(f"  Total raw articles ingested : {stats['total_collected']}")
    print(f"  Verified AI stories         : {stats['ai_news_count']}")
    print(f"  Selected for the feed       : {stats['stories_selected']}")
    print(f"  Active monitored sources    : {stats['sources_count']}")
    print(f"  Published in last 24 hours  : {stats['stories_today']}")
    print(f"  Unprocessed (last {RECENT_DAYS} days)   : {backlog_window}")
    print(f"  Unprocessed (all time)      : {backlog_total}")
    print(f"  Awaiting embedding          : {unindexed}")
    print(
        f"  Latest stored article collected at: "
        f"{stats.get('last_collected') or 'never'}"
    )
    if info["timestamps_repaired"]:
        print(f"  Timestamps repaired now     : {info['timestamps_repaired']}")

    if backlog_window:
        preview = next_analysis_preview(recent_days=RECENT_DAYS, limit=5)
        print("")
        print("NEXT UP FOR ANALYSIS (highest priority first)")
        for position, row in enumerate(preview, start=1):
            print(
                f"  {position}. p{analysis_priority(row)}  "
                f"{str(row['published'])[:10]}  "
                f"{str(row['source'])[:20]:<20}  "
                f"{str(row['title'])[:52]}"
            )
        print("  Drain them with: python -m backend.app.main --process --drain")

    categories = get_category_counts()
    if categories:
        print("\nVERIFIED STORIES BY CATEGORY")
        for name, total in sorted(categories.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<22}: {total}")

    vector_stats = get_index_stats()
    print("\nVECTOR STORE (FAISS, CPU)")
    print(f"  Index file                  : {vector_stats['index_path']}")
    print(f"  Metadata file               : {vector_stats['metadata_path']}")
    print(f"  Index healthy               : {vector_stats['healthy']}")
    print(f"  Indexed vectors             : {vector_stats['indexed_documents']}")
    print(f"  Metadata records            : {vector_stats['metadata']}")
    print(f"  Vector dimension            : {vector_stats['dimension']}")
    print(f"  Embedding model             : {vector_stats['embedding_model']}")
    print(f"  Min cosine similarity       : {MIN_SEMANTIC_SIMILARITY}")

    ollama = check_ollama_health()
    llm_state = "ready" if ollama["has_llm"] else "MISSING"
    embed_state = "ready" if ollama["has_embedding"] else "MISSING"
    print("\nLOCAL OLLAMA SERVICE")
    print(f"  Host                        : {OLLAMA_HOST}")
    print(f"  Service status              : {ollama['status'].upper()}")
    print(f"  Classification LLM          : {OLLAMA_MODEL} ({llm_state})")
    print(f"  Embedding model             : {EMBEDDING_MODEL} ({embed_state})")
    if ollama["status"] == "offline":
        print("  Start Ollama, then re-run:     ollama serve")
    if ollama["status"] == "online" and not ollama["has_embedding"]:
        print(f"  Pull the embedding model:      ollama pull {EMBEDDING_MODEL}")
    if ollama["status"] == "online" and not ollama["has_llm"]:
        print(f"  Pull the LLM:                  ollama pull {OLLAMA_MODEL}")

    print("\n" + "=" * 70)


def run_full_pipeline(
    limit: int = DEFAULT_BATCH_SIZE,
    days: int = RECENT_DAYS,
    backlog: bool = False,
) -> None:
    """Collect, analyse, embed and brief in one pass."""
    print("\n" + "#" * 70)
    print("STARTING FULL AI RADAR PIPELINE")
    print("#" * 70)

    print("\n[STEP 1/4] Collecting multi-tier RSS feeds...")
    run_collection(recent_days=days)

    print("\n[STEP 2/4] Analysing every collected article in the window...")
    # Drain rather than nibble: an unanalysed article never reaches the FAISS
    # index, so a partial pass leaves the search corpus behind the live feed.
    drain_articles(limit=limit, recent_days=None if backlog else days)

    print("\n[STEP 3/4] Synchronising FAISS embeddings...")
    sync_embeddings()

    mode = determine_mode()
    print(f"\n[STEP 4/4] Generating the {mode.upper()} briefing snapshot...")
    build_briefing(mode)

    print("\n" + "#" * 70)
    print("AI RADAR PIPELINE RUN COMPLETE")
    print("#" * 70 + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backend.app.main",
        description="AI Radar - local AI intelligence pipeline orchestrator",
    )
    parser.add_argument("--collect", action="store_true", help="Run the RSS collector")
    parser.add_argument("--process", action="store_true", help="Run the Ollama analyzer")
    parser.add_argument("--index", action="store_true", help="Sync FAISS embeddings")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Re-embed every verified article and rewrite the FAISS index",
    )
    parser.add_argument("--morning", action="store_true", help="Morning baseline briefing")
    parser.add_argument("--evening", action="store_true", help="Evening differential briefing")
    parser.add_argument("--full-pipeline", action="store_true", help="Run every stage in order")
    parser.add_argument("--status", action="store_true", help="Show telemetry and exit")
    parser.add_argument(
        "--backlog",
        action="store_true",
        help="Analyse the whole backlog, ignoring the lookback window",
    )
    parser.add_argument(
        "--drain",
        action="store_true",
        help="With --process, repeat batches until the window is fully analysed",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Articles analysed per run (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=RECENT_DAYS,
        help=f"Lookback window in days (default: {RECENT_DAYS})",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    actions = (
        args.collect,
        args.process,
        args.index,
        args.rebuild_index,
        args.morning,
        args.evening,
        args.full_pipeline,
    )

    # No action requested (or an explicit --status): report and stop.
    if args.status or not any(actions):
        print_system_status()
        return

    # Every path below queries the database, so make sure it is ready first.
    initialize_database()

    if args.collect:
        run_collection(recent_days=args.days)

    if args.process:
        window_days = None if args.backlog else args.days
        if args.drain:
            drain_articles(limit=args.limit, recent_days=window_days)
        else:
            process_articles_batch(limit=args.limit, recent_days=window_days)

    if args.rebuild_index:
        sync_embeddings(rebuild=True)
    elif args.index:
        # No batch cap: an article that is analysed but not embedded cannot be
        # retrieved by semantic search, so the index must not lag the database.
        sync_embeddings()

    if args.morning:
        build_briefing("morning")

    if args.evening:
        build_briefing("evening")

    if args.full_pipeline:
        run_full_pipeline(limit=args.limit, days=args.days, backlog=args.backlog)


if __name__ == "__main__":
    main()


