"""Top current AI stories built from the existing verified ranking pipeline."""

from typing import Any, Dict, List, Optional

from backend.app.services.ranking import (
    get_candidates,
    rank_articles,
    remove_duplicates,
    select_top_stories,
)
from backend.app.services.story_clustering import cluster_articles


def _top_story_diversity_score(
    story: Dict[str, Any],
    category_counts: Dict[str, int],
    source_counts: Dict[str, int],
) -> float:
    """Apply soft diminishing returns without changing global ranking."""
    category = str(story.get("category") or "Other")
    source = str(story.get("source") or "Unknown")
    score = float(story.get("ranking_score") or 0.0)
    category_penalty = 0.45 * category_counts.get(category, 0)
    research_penalty = 0.55 * category_counts.get("AI Research", 0) if category == "AI Research" else 0.0
    source_penalty = 0.20 * source_counts.get(source, 0)
    return score - category_penalty - research_penalty - source_penalty


def _select_diverse_top_stories(
    ranked: List[Dict[str, Any]], limit: int
) -> List[Dict[str, Any]]:
    """Select high-scoring stories while softly widening category/source mix."""
    selected: List[Dict[str, Any]] = []
    remaining = list(ranked)
    category_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    while remaining and len(selected) < max(0, limit):
        best = max(
            remaining,
            key=lambda story: (
                _top_story_diversity_score(story, category_counts, source_counts),
                float(story.get("ranking_score") or 0.0),
                float(story.get("published_timestamp") or 0.0),
            ),
        )
        remaining.remove(best)
        selected.append(best)
        category = str(best.get("category") or "Other")
        source = str(best.get("source") or "Unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
    return selected


def get_top_stories(
    limit: int = 10,
    articles: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return ranked, verified, de-duplicated representative stories."""
    candidates = articles if articles is not None else get_candidates()
    eligible = [
        dict(article)
        for article in candidates
        if article.get("is_ai_news", 1) == 1
        and article.get("should_show", 1) == 1
    ]
    unique = remove_duplicates(eligible)
    ranked = rank_articles(unique)
    clustered = cluster_articles(ranked)
    clustered_ranked = rank_articles(clustered)
    return _select_diverse_top_stories(clustered_ranked, limit=max(0, limit))



