"""Deterministic same-event clustering for verified AI stories."""

import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np

from backend.app.config import (
    STORY_CLUSTER_MIN_SHARED_TERMS,
    STORY_CLUSTER_SIMILARITY_THRESHOLD,
    normalize_category,
)
from backend.app.services.ranking import SOURCE_AUTHORITY, normalize_text
from backend.app.services.vector_store import load_index

_STOPWORDS = {
    "about", "after", "ai", "and", "are", "from", "for", "how", "into",
    "its", "new", "news", "of", "on", "the", "their", "this", "to", "with",
}
_RESEARCH_SOURCE_MARKERS = ("arxiv", "research", "laboratory", "lab ")
_EVENT_TERMS = {
    "announces", "announced", "launches", "launched", "releases", "released",
    "unveils", "unveiled", "introduces", "introduced", "acquires", "acquired",
    "raises", "raised", "funding", "partnership", "deal", "joins", "hiring",
    "hires", "appoints", "appointed", "rolls", "deployed", "deployment",
}
_GENERIC_EVENT_TERMS = _EVENT_TERMS | {
    "model", "models", "developers", "development", "access", "system",
    "systems", "paper", "research", "study", "technology",
}


def title_terms(article: Dict[str, Any]) -> Set[str]:
    """Extract meaningful title terms used as a conservative event guard."""
    words = re.findall(r"[a-z0-9]+", normalize_text(article.get("title")))
    return {word for word in words if len(word) >= 3 and word not in _STOPWORDS}


def normalized_title(article: Dict[str, Any]) -> str:
    return " ".join(sorted(title_terms(article)))


def _is_research_paper(article: Dict[str, Any]) -> bool:
    source = normalize_text(article.get("source"))
    category = normalize_category(article.get("category"))
    return category == "AI Research" or any(marker in source for marker in _RESEARCH_SOURCE_MARKERS)


def _exact_duplicate(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_url = normalize_text(left.get("url"))
    right_url = normalize_text(right.get("url"))
    if left_url and left_url == right_url:
        return True
    left_title = normalize_text(left.get("title"))
    right_title = normalize_text(right.get("title"))
    if left_title and left_title == right_title:
        return True
    if normalized_title(left) and normalized_title(left) == normalized_title(right):
        return True
    if not left_title or not right_title:
        return False
    return SequenceMatcher(None, left_title, right_title).ratio() >= 0.94


def _authority(article: Dict[str, Any]) -> float:
    return SOURCE_AUTHORITY.get(str(article.get("source", "")), 7.5)


def _published(article: Dict[str, Any]) -> float:
    try:
        return float(article.get("published_timestamp") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _representative_key(article: Dict[str, Any]):
    return (
        _authority(article),
        float(article.get("ranking_score") or 0.0),
        _published(article),
    )


def _vector_map(articles: Iterable[Dict[str, Any]]) -> Dict[int, np.ndarray]:
    index, metadata = load_index()
    if index is None:
        return {}
    positions = {entry.get("id"): position for position, entry in enumerate(metadata)}
    vectors = {}
    for article in articles:
        article_id = article.get("id")
        position = positions.get(article_id)
        if position is not None:
            vectors[article_id] = index.reconstruct(position)
    return vectors


def _same_event(
    left: Dict[str, Any],
    right: Dict[str, Any],
    vectors: Dict[int, np.ndarray],
) -> bool:
    if _exact_duplicate(left, right):
        return True
    # Related papers are not event duplicates. Only exact/near-exact records
    # collapse for research sources; semantic similarity is too broad here.
    if _is_research_paper(left) or _is_research_paper(right):
        return False
    left_vector = vectors.get(left.get("id"))
    right_vector = vectors.get(right.get("id"))
    if left_vector is None or right_vector is None:
        return False
    left_norm = left_vector / max(float(np.linalg.norm(left_vector)), 1e-12)
    right_norm = right_vector / max(float(np.linalg.norm(right_vector)), 1e-12)
    similarity = float(np.dot(left_norm, right_norm))
    left_terms = title_terms(left)
    right_terms = title_terms(right)
    shared_terms = len(left_terms & right_terms)
    event_terms = (left_terms | right_terms) & _EVENT_TERMS
    concrete_terms = (left_terms & right_terms) - _GENERIC_EVENT_TERMS
    title_union = left_terms | right_terms
    title_overlap = shared_terms / max(len(title_union), 1)
    return (
        similarity >= STORY_CLUSTER_SIMILARITY_THRESHOLD
        and shared_terms >= STORY_CLUSTER_MIN_SHARED_TERMS
        and len(event_terms) >= 1
        and len(concrete_terms) >= 1
        and title_overlap >= 0.25
    )


def _cluster_story(cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
    representative = max(cluster, key=_representative_key)
    members = sorted(cluster, key=_published, reverse=True)
    member_ids = [member["id"] for member in members]
    sources = sorted({str(member.get("source", "")) for member in members if member.get("source")})
    source_urls = [
        {"source": member.get("source", ""), "url": member.get("url", "")}
        for member in members
        if member.get("url")
    ]
    story = dict(representative)
    story.update(
        {
            "representative_article": dict(representative),
            "category": normalize_category(
                representative.get("special_category") or representative.get("category")
            ),
            "cluster_id": f"story-{min(member_ids)}",
            "member_ids": member_ids,
            "member_count": len(members),
            "sources": sources,
            "source_count": len(sources),
            "source_urls": source_urls,
            "cluster_articles": members,
            "newest_published_timestamp": _published(members[0]),
        }
    )
    return story


def cluster_articles(
    articles: List[Dict[str, Any]],
    vectors: Optional[Dict[int, np.ndarray]] = None,
) -> List[Dict[str, Any]]:
    """Collapse verified same-event articles into representative stories."""
    if not articles:
        return []
    eligible = [
        article
        for article in articles
        if article.get("is_ai_news", 1) == 1
        and article.get("should_show", 1) == 1
    ]
    if not eligible:
        return []
    vectors = vectors if vectors is not None else _vector_map(eligible)
    clusters: List[List[Dict[str, Any]]] = []
    for article in sorted(eligible, key=_published, reverse=True):
        target = next(
            (cluster for cluster in clusters if _same_event(article, cluster[0], vectors)),
            None,
        )
        if target is None:
            clusters.append([article])
        else:
            target.append(article)
    result = [_cluster_story(cluster) for cluster in clusters]
    collapsed = len(eligible) - len(result)
    largest = max((story["member_count"] for story in result), default=0)
    cross_source_clusters = sum(1 for story in result if story["source_count"] > 1)
    same_source_duplicate_clusters = sum(
        1 for story in result if story["member_count"] > 1 and story["source_count"] == 1
    )
    print("STORY CLUSTERING")
    print(f"Articles considered: {len(eligible)}")
    print(f"Story clusters: {len(result)}")
    print(f"Articles collapsed: {collapsed}")
    print(f"Cross-source clusters: {cross_source_clusters}")
    print(f"Same-source duplicate clusters: {same_source_duplicate_clusters}")
    print(f"Largest cluster: {largest} articles")
    return result


