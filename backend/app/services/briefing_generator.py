import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

"""
AI Radar - morning baseline and evening differential briefings.

The 9:00 AM IST run writes data/morning_briefing.json (the baseline). The
7:00 PM IST run reads that baseline back and reports only what was published
after it, stating plainly when nothing significant has happened rather than
inventing an update.
"""

from backend.app.config import (
    BRIEFING_SIZE,
    EVENING_ARCHIVE_DIR,
    LATEST_BRIEFING_FILE,
    MORNING_BRIEFING_FILE,
    HISTORY_FILE,
    LOCAL_TIMEZONE,
    MORNING_HOUR,
    EVENING_HOUR,
    SIGNIFICANT_IMPORTANCE,
    SIGNIFICANT_CAREER_RELEVANCE,
    SIGNIFICANT_RANKING_SCORE,
    normalize_category,
)
from backend.app.services.ranking import (
    get_candidates,
    remove_duplicates,
    rank_articles,
    select_top_stories,
)
from backend.app.services.story_clustering import cluster_articles

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def local_now() -> datetime:
    """Return current India Standard Time datetime."""
    return datetime.now(LOCAL_TIMEZONE)


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse integer/float unix timestamp or ISO string into timezone-aware UTC datetime."""
    if value is None:
        return None

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None

    value_str = str(value).strip()
    if not value_str:
        return None

    try:
        dt = datetime.fromisoformat(value_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    return None


def ensure_data_directory() -> None:
    LATEST_BRIEFING_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_json_file(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as error:
        print(f"âš ï¸ Could not read {path}: {error}")
        return default


def save_json_file(path: Path, data: Any) -> None:
    ensure_data_directory()
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def archive_existing_evening_snapshot() -> None:
    """Preserve the previous full evening snapshot before replacing the latest one."""
    if not LATEST_BRIEFING_FILE.exists():
        return

    previous = json.loads(LATEST_BRIEFING_FILE.read_text(encoding="utf-8"))
    if not isinstance(previous, dict) or previous.get("briefing_type") != "evening":
        return

    previous_generated_at = parse_timestamp(previous.get("generated_at"))
    if previous_generated_at is None:
        raise ValueError("The existing evening briefing has an invalid generated_at.")

    archive_name = previous_generated_at.strftime("%Y%m%dT%H%M%S.%fZ.json")
    archive_path = EVENING_ARCHIVE_DIR / archive_name
    if archive_path.exists():
        archived = json.loads(archive_path.read_text(encoding="utf-8"))
        if archived == previous:
            return
        raise FileExistsError(f"An evening briefing archive already exists: {archive_path}")

    EVENING_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    save_json_file(archive_path, previous)


def article_to_story(article: Dict[str, Any]) -> Dict[str, Any]:
    """Format an analyzed article dict into a standardized briefing story object."""
    newest_timestamp = article.get(
        "newest_published_timestamp", article.get("published_timestamp")
    )
    return {
        "id": article["id"],
        "title": article["title"],
        "source": article["source"],
        "url": article["url"],
        "published": article.get("published", ""),
        "published_timestamp": newest_timestamp,
        "category": normalize_category(
            article.get("special_category") or article.get("category")
        ),
        "importance": article.get("importance", 1),
        "career_relevance": article.get("career_relevance", 1),
        "ranking_score": article.get("ranking_score", 5.0),
        "summary": article.get("ai_summary") or article.get("summary", ""),
        "why_it_matters": article.get("why_it_matters", ""),
        "cluster_id": article.get("cluster_id"),
        "member_ids": article.get("member_ids", [article["id"]]),
        "member_count": article.get("member_count", 1),
        "sources": article.get("sources", [article.get("source", "")]),
        "source_count": article.get("source_count", 1),
        "source_urls": article.get("source_urls", []),
    }


def build_sections(stories: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group stories into specialized curated sections."""
    sections: Dict[str, List[Dict[str, Any]]] = {
        "top_stories": stories[:3],
        "research": [],
        "developers_tools": [],
        "career": [],
        "india_ai": [],
        "robotics": [],
        "other": [],
    }

    for story in stories:
        category = normalize_category(story.get("category"))
        if category == "AI Research":
            sections["research"].append(story)
        elif category == "AI Tools":
            sections["developers_tools"].append(story)
        elif category == "AI Jobs & Careers":
            sections["career"].append(story)
        elif category == "India AI":
            sections["india_ai"].append(story)
        elif category == "Robotics":
            sections["robotics"].append(story)
        else:
            sections["other"].append(story)

    return sections


def build_morning_briefing(limit: int = BRIEFING_SIZE) -> Dict[str, Any]:
    """
    Generate the 9:00 AM IST morning intelligence baseline.
    """
    print("\n" + "=" * 70)
    print("AI RADAR â€” MORNING INTELLIGENCE BRIEFING")
    print("=" * 70)

    candidates = get_candidates()
    unique = remove_duplicates(candidates)
    ranked = rank_articles(unique)
    clustered = cluster_articles(ranked)
    top_articles = select_top_stories(clustered, limit=limit)

    stories = [article_to_story(art) for art in top_articles]
    generated_at = utc_now()
    generated_at_local = generated_at.astimezone(LOCAL_TIMEZONE)

    briefing = {
        "version": "2.0",
        "briefing_type": "morning",
        "generated_at": generated_at.isoformat(),
        "generated_at_local": generated_at_local.isoformat(),
        "article_count": len(stories),
        "status": {
            "has_new_developments": True,
            "message": (
                f"Morning baseline created at "
                f"{generated_at_local.strftime('%d %b %Y, %I:%M %p IST')} "
                f"from verified AI developments."
            ),
            "comparison_period": None,
            "baseline_local_date": generated_at_local.date().isoformat(),
        },
        "stories": stories,
        "sections": build_sections(stories),
    }

    save_json_file(MORNING_BRIEFING_FILE, briefing)

    # Append to audit history
    history = load_json_file(HISTORY_FILE, [])
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "type": "morning",
            "generated_at": generated_at.isoformat(),
            "story_ids": [s["id"] for s in stories],
        }
    )
    save_json_file(HISTORY_FILE, history[-30:])

    print(f"âœ… Morning baseline generated ({len(stories)} stories).")
    return briefing


def is_significant(article: Dict[str, Any]) -> bool:
    """Determine if a newly published article constitutes a significant development."""
    importance = article.get("importance") or 0
    career = article.get("career_relevance") or 0
    score = article.get("ranking_score") or 0.0

    return (
        importance >= SIGNIFICANT_IMPORTANCE
        or career >= SIGNIFICANT_CAREER_RELEVANCE
        or score >= SIGNIFICANT_RANKING_SCORE
    )


def build_missing_baseline_evening_briefing() -> Dict[str, Any]:
    """Persist an explicit evening status when no baseline can be compared."""
    generated_at = utc_now()
    generated_at_local = generated_at.astimezone(LOCAL_TIMEZONE)
    message = (
        "Evening differential briefing cannot be computed because no current-day "
        "morning baseline exists. Generate a morning briefing for today first."
    )
    briefing = {
        "version": "2.0",
        "briefing_type": "evening",
        "generated_at": generated_at.isoformat(),
        "generated_at_local": generated_at_local.isoformat(),
        "article_count": 0,
        "status": {
            "has_new_developments": False,
            "baseline_missing": True,
            "message": message,
            "comparison_period": None,
            "candidates_since_baseline": 0,
        },
        "stories": [],
        "sections": build_sections([]),
    }
    latest_briefing = load_json_file(LATEST_BRIEFING_FILE, None)
    if not (
        isinstance(latest_briefing, dict)
        and latest_briefing.get("briefing_type") == "evening"
    ):
        save_json_file(LATEST_BRIEFING_FILE, briefing)

    history = load_json_file(HISTORY_FILE, [])
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "type": "evening",
            "generated_at": generated_at.isoformat(),
            "has_new_developments": False,
            "baseline_missing": True,
            "story_count": 0,
        }
    )
    save_json_file(HISTORY_FILE, history[-30:])
    print(f"âš ï¸ {message}")
    return briefing


def build_evening_briefing(limit: int = BRIEFING_SIZE) -> Dict[str, Any]:
    """
    Generate the 7:00 PM IST evening update by evaluating developments since the morning baseline.
    """
    print("\n" + "=" * 70)
    print("AI RADAR â€” EVENING INTELLIGENCE UPDATE")
    print("=" * 70)

    morning = load_json_file(MORNING_BRIEFING_FILE, None)
    if not morning or not morning.get("generated_at"):
        return build_missing_baseline_evening_briefing()

    morning_dt = parse_timestamp(morning.get("generated_at"))
    if not morning_dt:
        return build_missing_baseline_evening_briefing()

    generated_at = utc_now()
    generated_at_local = generated_at.astimezone(LOCAL_TIMEZONE)
    baseline_local = morning_dt.astimezone(LOCAL_TIMEZONE)
    if baseline_local.date() != generated_at_local.date():
        return build_missing_baseline_evening_briefing()

    morning_story_ids = {
        story.get("id")
        for story in morning.get("stories", [])
        if story.get("id") is not None
    }

    candidates = get_candidates()
    unique = remove_duplicates(candidates)
    ranked = rank_articles(unique)
    clustered = cluster_articles(ranked)

    # Only articles published strictly after the baseline was written, and never
    # a story the baseline already carried.
    new_articles = []
    for article in clustered:
        published = parse_timestamp(article.get("published_timestamp"))
        if published is None or published <= morning_dt:
            continue
        if set(article.get("member_ids", [article.get("id")])) & morning_story_ids:
            continue
        new_articles.append(article)

    significant_articles = [art for art in new_articles if is_significant(art)]

    baseline_age_hours = (generated_at - morning_dt).total_seconds() / 3600

    print(f"Morning baseline: {baseline_local.strftime('%d %b %Y, %I:%M %p IST')}")
    print(f"Baseline age: {baseline_age_hours:.1f} hours")
    print(f"Articles published since baseline: {len(new_articles)}")
    print(f"Significant new developments: {len(significant_articles)}")

    if significant_articles:
        significant_articles.sort(
            key=lambda a: (
                a.get("ranking_score") or 0.0,
                a.get("published_timestamp") or 0.0,
            ),
            reverse=True,
        )
        selected = select_top_stories(significant_articles, limit=limit)
        stories = [article_to_story(art) for art in selected]

        briefing = {
            "version": "2.0",
            "briefing_type": "evening",
            "generated_at": generated_at.isoformat(),
            "generated_at_local": generated_at_local.isoformat(),
            "article_count": len(stories),
            "status": {
                "has_new_developments": True,
                "message": (
                    f"{len(stories)} significant AI development(s) published "
                    f"since the morning baseline of "
                    f"{baseline_local.strftime('%d %b %Y, %I:%M %p IST')}."
                ),
                "comparison_period": {
                    "from": morning_dt.isoformat(),
                    "to": generated_at.isoformat(),
                    "baseline_local": baseline_local.isoformat(),
                    "baseline_age_hours": round(baseline_age_hours, 1),
                },
                "candidates_since_baseline": len(new_articles),
            },
            "stories": stories,
            "sections": build_sections(stories),
        }
    else:
        morning_stories = morning.get("stories", [])
        briefing = {
            "version": "2.0",
            "briefing_type": "evening",
            "generated_at": generated_at.isoformat(),
            "generated_at_local": generated_at_local.isoformat(),
            "article_count": len(morning_stories),
            "status": {
                "has_new_developments": False,
                "message": (
                    "No major new AI developments since the morning briefing. "
                    f"Compared against the morning baseline of "
                    f"{baseline_local.strftime('%d %b %Y, %I:%M %p IST')}. "
                    "The morning baseline stories are repeated unchanged."
                ),
                "comparison_period": {
                    "from": morning_dt.isoformat(),
                    "to": generated_at.isoformat(),
                    "baseline_local": baseline_local.isoformat(),
                    "baseline_age_hours": round(baseline_age_hours, 1),
                },
                "candidates_since_baseline": len(new_articles),
            },
            "stories": morning_stories,
            "sections": build_sections(morning_stories),
        }

    archive_existing_evening_snapshot()
    save_json_file(LATEST_BRIEFING_FILE, briefing)

    history = load_json_file(HISTORY_FILE, [])
    if isinstance(history, list):
        history.append(
            {
                "type": "evening",
                "generated_at": generated_at.isoformat(),
                "has_new_developments": briefing["status"]["has_new_developments"],
                "story_count": len(briefing["stories"]),
            }
        )
        save_json_file(HISTORY_FILE, history[-30:])

    print(f"âœ… Evening update generated: {briefing['status']['message']}")
    return briefing


def determine_mode() -> str:
    """Determine whether to run morning or evening briefing based on local IST hour."""
    curr_hour = local_now().hour
    if MORNING_HOUR <= curr_hour < EVENING_HOUR:
        return "morning"
    return "evening"


def build_briefing(mode: Optional[str] = None) -> Dict[str, Any]:
    if mode is None:
        mode = determine_mode()
    mode = mode.lower().strip()
    if mode == "morning":
        return build_morning_briefing()
    elif mode == "evening":
        return build_evening_briefing()
    else:
        raise ValueError(f"Invalid briefing mode '{mode}'. Use 'morning' or 'evening'.")


if __name__ == "__main__":
    mode_arg = sys.argv[1].lower() if len(sys.argv) > 1 else determine_mode()
    build_briefing(mode_arg)
