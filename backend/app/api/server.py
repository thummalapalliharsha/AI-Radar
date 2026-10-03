"""Read-only API endpoints backed by the existing AI Radar data."""

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import (
    DATABASE_PATH,
    LATEST_BRIEFING_FILE,
    LOCAL_TIMEZONE,
    MORNING_BRIEFING_FILE,
    TOP_K_SEARCH_RESULTS,
)
from backend.app.services.vector_store import search_vector_store
from backend.app.ui.components import (
    NO_GROUNDED_FACTS_MESSAGE,
    check_ollama,
    classify_search_intent,
    generate_grounded_answer,
    validate_retrieved_articles,
)
from backend.app.utils.database import get_latest_news, parse_published_value

logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=[],
)


def _required_text(value: Any, field: str, story_id: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(
            status_code=500,
            detail=f"Verified story {story_id} is missing required field '{field}'.",
        )
    return value


def _briefing_local_date(value: Any, field: str, expected_type: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise HTTPException(
            status_code=500,
            detail=f"The stored {expected_type} briefing has an invalid {field}.",
        )

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=f"The stored {expected_type} briefing has an invalid {field}.",
        ) from error

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(LOCAL_TIMEZONE).date().isoformat()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/stories")
def stories(
    window: Literal["24h", "7d"] = Query(...),
    limit: int = Query(20, ge=1, le=50),
) -> list[dict[str, str | None]]:
    if not DATABASE_PATH.is_file():
        raise HTTPException(status_code=503, detail="The story database is unavailable.")

    now = datetime.now(timezone.utc)
    now_timestamp = now.timestamp()
    duration = timedelta(hours=24) if window == "24h" else timedelta(days=7)
    cutoff = (now - duration).timestamp()

    rows = get_latest_news(limit=50, min_importance=0)
    matching: list[tuple[float, dict[str, str | None]]] = []

    for row in rows:
        timestamp = parse_published_value(row.get("published_timestamp"))
        if timestamp is None or not math.isfinite(timestamp):
            continue
        if timestamp < cutoff or timestamp > now_timestamp:
            continue

        story_id = row.get("id")
        summary = row.get("ai_summary") or row.get("summary")
        story = {
            "headline": _required_text(row.get("title"), "title", story_id),
            "summary": _required_text(summary, "summary", story_id),
            "category": _required_text(row.get("category"), "category", story_id),
            "source": _required_text(row.get("source"), "source", story_id),
            "published_at": datetime.fromtimestamp(
                timestamp, tz=timezone.utc
            ).isoformat(),
            "url": _required_text(row.get("url"), "url", story_id),
            "why_it_matters": row.get("why_it_matters"),
        }
        matching.append((timestamp, story))

    matching.sort(key=lambda item: item[0], reverse=True)
    return [story for _, story in matching[:limit]]


def _read_briefing(path: Path, expected_type: Literal["morning", "evening"]) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False}

    try:
        with path.open("r", encoding="utf-8") as briefing_file:
            briefing = json.load(briefing_file)
    except JSONDecodeError as error:
        raise HTTPException(
            status_code=500,
            detail=f"The stored {expected_type} briefing is not valid JSON.",
        ) from error

    if not isinstance(briefing, dict):
        raise HTTPException(
            status_code=500,
            detail=f"The stored {expected_type} briefing has an invalid structure.",
        )

    briefing_type = briefing.get("briefing_type")
    if expected_type == "evening" and briefing_type == "morning":
        return {"available": False}
    if briefing_type != expected_type:
        raise HTTPException(
            status_code=500,
            detail=f"The stored briefing is not a valid {expected_type} briefing.",
        )

    generated_at = briefing.get("generated_at")
    status = briefing.get("status")
    summary = status.get("message") if isinstance(status, dict) else None
    briefing_stories = briefing.get("stories")
    if (
        not isinstance(generated_at, str)
        or not generated_at
        or not isinstance(summary, str)
        or not isinstance(briefing_stories, list)
    ):
        raise HTTPException(
            status_code=500,
            detail=f"The stored {expected_type} briefing is missing required fields.",
        )

    generated_local_date = _briefing_local_date(
        generated_at, "generated_at", expected_type
    )
    current_local_date = datetime.now(LOCAL_TIMEZONE).date().isoformat()
    status_data = status if isinstance(status, dict) else {}
    comparison_period = status_data.get("comparison_period")
    baseline_value = None
    if isinstance(comparison_period, dict):
        baseline_value = comparison_period.get("baseline_local") or comparison_period.get("from")
    baseline_local_date = _briefing_local_date(
        baseline_value, "comparison baseline", expected_type
    )
    is_current_day = generated_local_date == current_local_date

    return {
        "available": True,
        "baseline_missing": status_data.get("baseline_missing") is True,
        "generated_at": generated_at,
        "summary": summary,
        "stories": briefing_stories,
        "current_local_date": current_local_date,
        "generated_local_date": generated_local_date,
        "baseline_local_date": baseline_local_date,
        "is_current_day": is_current_day,
        "is_current_comparison": (
            expected_type == "evening"
            and is_current_day
            and baseline_local_date == current_local_date
        ),
    }


@app.get("/api/briefing/morning")
def morning_briefing() -> dict[str, Any]:
    return _read_briefing(MORNING_BRIEFING_FILE, "morning")


@app.get("/api/briefing/evening")
def evening_briefing() -> dict[str, Any]:
    return _read_briefing(LATEST_BRIEFING_FILE, "evening")


def _search_source(article: dict[str, Any]) -> dict[str, str | None]:
    url = article.get("url")
    return {
        "title": str(article.get("title") or ""),
        "source": str(article.get("source") or ""),
        "category": str(article.get("category") or ""),
        "published": str(article.get("published") or ""),
        "url": url if isinstance(url, str) and url else None,
        "summary": str(article.get("summary") or ""),
        "why_it_matters": str(article.get("why_it_matters") or ""),
    }


@app.get("/api/search")
def search(q: str = Query(..., min_length=1, max_length=1000)) -> dict[str, Any]:
    query = q.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Enter an AI question or topic to search.")

    ollama = check_ollama()
    if not ollama["online"] or not ollama["has_embedding"] or not ollama["has_llm"]:
        raise HTTPException(
            status_code=503,
            detail="Local AI search is currently unavailable.",
        )

    try:
        if classify_search_intent(query) is not True:
            return {
                "status": "unrelated",
                "message": "This search is limited to AI and machine learning.",
            }

        matches = search_vector_store(query, top_k=TOP_K_SEARCH_RESULTS)
        matches = validate_retrieved_articles(query, matches)
        if not matches:
            return {
                "status": "no_relevant_information",
                "message": "No sufficiently relevant AI information was found in AI Radar sources.",
            }

        sources = [_search_source(article) for article in matches]
        answer = generate_grounded_answer(query, matches)
        if answer == NO_GROUNDED_FACTS_MESSAGE:
            return {
                "status": "no_relevant_information",
                "message": answer,
            }
        if not answer:
            return {
                "status": "sources_only",
                "message": (
                    "A grounded answer is unavailable right now. "
                    "Verified source articles are shown below."
                ),
                "sources": sources,
            }

        return {"status": "answered", "answer": answer, "sources": sources}
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Local AI search failed.")
        raise HTTPException(
            status_code=500,
            detail="Local AI search failed. Please try again.",
        ) from error
