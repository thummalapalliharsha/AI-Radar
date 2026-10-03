"""
AI Radar - multi-factor ranking engine.

Composite score = importance (30%) + recency (30%) + career relevance (15%)
+ AI depth (15%) + source authority (10%), with diversity caps applied when the
top stories are selected for a briefing.
"""

import re
import sys
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

from backend.app.config import RECENT_DAYS, normalize_category
from backend.app.utils.database import get_connection

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# AI RADAR â€” RANKING ENGINE & SOURCE AUTHORITY
# ============================================================

MAX_CANDIDATES = 100
DEFAULT_TOP_STORIES = 10

# Source Authority Tiers (0 to 10 scale)
SOURCE_AUTHORITY = {
    "OpenAI": 10.0,
    "Google AI Blog": 10.0,
    "Microsoft AI": 10.0,
    "Microsoft Research": 10.0,
    "AWS Machine Learning": 10.0,
    "NVIDIA Newsroom": 10.0,
    "NVIDIA Blog": 10.0,
    "NVIDIA Developer Blog": 10.0,
    "Hugging Face": 10.0,
    "MIT Technology Review AI": 9.0,
    "TechCrunch AI": 8.5,
    "VentureBeat AI": 8.5,
    "Google India Blog": 8.5,
    "YourStory Technology": 8.0,
}

STRONG_AI_KEYWORDS = [
    "ai",
    "artificial intelligence",
    "machine learning",
    "llm",
    "large language model",
    "generative ai",
    "agent",
    "ai agent",
    "agentic",
    "foundation model",
    "reasoning model",
    "inference",
    "neural network",
    "deep learning",
    "robotics",
    "humanoid",
    "computer vision",
    "natural language",
    "openai",
    "nvidia",
    "anthropic",
    "google deepmind",
    "gemini",
    "claude",
    "gpt",
    "nemotron",
    "meta ai",
    "llama",
]

WEAK_AI_KEYWORDS = [
    "gaming",
    "gamescom",
    "financial results",
    "quarter results",
    "stock",
    "investor",
    "revenue",
    "real estate",
    "quotes",
]

DEVELOPER_KEYWORDS = [
    "developer",
    "developers",
    "coding",
    "code",
    "api",
    "sdk",
    "programming",
    "software development",
    "inference",
    "developer tools",
    "open source",
    "github",
]

RESEARCH_KEYWORDS = [
    "benchmark",
    "dataset",
    "evaluation",
    "reinforcement learning",
    "paper",
    "technical report",
    "arxiv",
    "research lab",
    "machine learning research",
]

JOB_KEYWORDS = [
    "hiring",
    "layoffs",
    "workforce",
    "labor market",
    "talent program",
    "recruitment",
    "fellowship",
    "research fellowship",
]

INDIA_KEYWORDS = [
    "indiaai",
    "meity",
    "indian ai",
    "india ai mission",
    "iit",
    "indian institute of technology",
    "indian startup",
    "indian ai company",
    "india's ai",
]

ROBOTICS_KEYWORDS = [
    "robotics",
    "robot",
    "humanoid",
    "robotaxi",
    "embodied ai",
    "embodied intelligence",
    "autonomous machine",
    "vision-language-action",
]

AI_ROBOTICS_SIGNALS = [
    "ai",
    "artificial intelligence",
    "machine learning",
    "model",
    "neural",
    "autonomous",
    "perception",
    "planning",
    "navigation",
    "control policy",
]

INDIA_CONTEXT_KEYWORDS = [
    "india",
    "indian",
    "bengaluru",
    "bangalore",
    "hyderabad",
    "chennai",
    "mumbai",
    "delhi",
    "gurugram",
    "pune",
    "telangana",
    "tamil nadu",
    "karnataka",
    "meity",
    "indiaai",
]


def normalize_text(text: Any) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def contains_keyword(text: str, keywords: List[str]) -> bool:
    norm = normalize_text(text)
    return any(re.search(r"\b" + re.escape(kw) + r"\b", norm) for kw in keywords)


def calculate_recency(published_timestamp: Any) -> float:
    """
    Convert article age into a 0-10 recency score with dynamic decay.
    """
    try:
        published = datetime.fromtimestamp(float(published_timestamp), tz=timezone.utc)
        now = datetime.now(timezone.utc)
        age_hours = (now - published).total_seconds() / 3600

        if age_hours <= 6:
            return 10.0
        if age_hours <= 12:
            return 9.5
        if age_hours <= 24:
            return 9.0
        if age_hours <= 48:
            return 7.5
        if age_hours <= 72:
            return 6.0
        if age_hours <= 120:
            return 4.5
        if age_hours <= 168:
            return 3.0
        return 1.0
    except Exception:
        return 0.0


def calculate_ai_depth(article: Dict[str, Any]) -> float:
    """
    Estimate the depth and density of AI/ML concepts in the story.
    """
    text = " ".join(
        [
            normalize_text(article.get("title")),
            normalize_text(article.get("summary")),
            normalize_text(article.get("ai_summary")),
            normalize_text(article.get("why_it_matters")),
        ]
    )

    strong_matches = sum(1 for kw in STRONG_AI_KEYWORDS if kw in text)
    weak_matches = sum(1 for kw in WEAK_AI_KEYWORDS if kw in text)

    score = 5.0
    if strong_matches >= 1:
        score += 1.5
    if strong_matches >= 3:
        score += 1.5
    if strong_matches >= 6:
        score += 1.0

    if weak_matches >= 2 and strong_matches <= 2:
        score -= 2.0

    return max(0.0, min(10.0, score))


def detect_special_category(article: Dict[str, Any]) -> str:
    """
    Refine an article's category for the specialised sub-feeds.

    Always returns a canonical ALLOWED_CATEGORIES value.
    """
    text = " ".join(
        [
            normalize_text(article.get("title")),
            normalize_text(article.get("summary")),
            normalize_text(article.get("ai_summary")),
            normalize_text(article.get("why_it_matters")),
        ]
    )

    if contains_keyword(text, ROBOTICS_KEYWORDS) and contains_keyword(
        text, AI_ROBOTICS_SIGNALS
    ):
        return "Robotics"

    india_anchor = contains_keyword(text, INDIA_KEYWORDS)
    india_context = contains_keyword(text, INDIA_CONTEXT_KEYWORDS)
    india_ai = contains_keyword(text, STRONG_AI_KEYWORDS)
    if india_anchor or (india_context and india_ai):
        return "India AI"

    if contains_keyword(text, JOB_KEYWORDS) and contains_keyword(
        text, ["ai", "artificial intelligence", "machine learning", "research"]
    ):
        return "AI Jobs & Careers"
    if contains_keyword(text, DEVELOPER_KEYWORDS):
        # Canonical name for what earlier builds called "Developers & Tools".
        return "AI Tools"
    if contains_keyword(text, RESEARCH_KEYWORDS):
        return "AI Research"

    return normalize_category(article.get("category"))


def calculate_score(article: Dict[str, Any]) -> float:
    """
    Calculate final composite ranking score:
    Importance (30%) + Recency (30%) + Career Relevance (15%) + AI Depth (15%) + Source Authority (10%)
    """
    importance = float(article.get("importance") or 1)
    career_relevance = float(article.get("career_relevance") or 1)
    recency = calculate_recency(article.get("published_timestamp"))
    ai_depth = calculate_ai_depth(article)
    source_name = str(article.get("source", ""))
    source_authority = SOURCE_AUTHORITY.get(source_name, 7.5)

    composite = (
        importance * 0.30
        + recency * 0.30
        + career_relevance * 0.15
        + ai_depth * 0.15
        + source_authority * 0.10
    )
    return round(max(1.0, min(10.0, composite)), 2)


def get_candidates(recent_days: int = RECENT_DAYS, max_candidates: int = MAX_CANDIDATES) -> List[Dict[str, Any]]:
    """Fetch recent analyzed AI articles from SQLite."""
    connection = get_connection()

    cutoff = (datetime.now(timezone.utc) - timedelta(days=recent_days)).timestamp()

    query = """
        SELECT
            id,
            title,
            url,
            summary,
            source,
            category,
            published,
            published_timestamp,
            is_ai_news,
            importance,
            career_relevance,
            ai_summary,
            why_it_matters,
            should_show
        FROM articles
        WHERE
            is_ai_news = 1
            AND should_show = 1
            AND published_timestamp IS NOT NULL
            AND published_timestamp >= ?
        ORDER BY published_timestamp DESC
        LIMIT ?
    """

    rows = connection.execute(query, (cutoff, max_candidates)).fetchall()
    connection.close()
    return [dict(row) for row in rows]


def remove_duplicates(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate articles by unique URL and normalized title."""
    unique = []
    seen_urls = set()
    seen_titles = set()

    for article in articles:
        url = article.get("url", "").strip()
        title_norm = normalize_text(article.get("title", ""))

        if not url or url in seen_urls:
            continue
        if title_norm and title_norm in seen_titles:
            continue

        seen_urls.add(url)
        if title_norm:
            seen_titles.add(title_norm)
        unique.append(article)

    return unique


def rank_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score and sort articles by composite ranking score."""
    scored = []
    for article in articles:
        item = dict(article)
        item["special_category"] = detect_special_category(item)
        item["recency_score"] = calculate_recency(item.get("published_timestamp"))
        item["ai_depth_score"] = calculate_ai_depth(item)
        item["ranking_score"] = calculate_score(item)
        scored.append(item)

    scored.sort(
        key=lambda a: (a["ranking_score"], a.get("published_timestamp") or 0.0),
        reverse=True,
    )
    return scored


def select_top_stories(
    ranked_articles: List[Dict[str, Any]],
    limit: int = DEFAULT_TOP_STORIES,
    max_per_source: int = 3,
    max_per_category: int = 3,
) -> List[Dict[str, Any]]:
    """
    Select diverse top stories enforcing source and category distribution caps.
    """
    selected = []
    source_counts: Dict[str, int] = {}
    category_counts: Dict[str, int] = {}

    for article in ranked_articles:
        if len(selected) >= limit:
            break

        source = article.get("source", "Unknown")
        category = normalize_category(article.get("special_category"))

        if source_counts.get(source, 0) >= max_per_source:
            continue
        if category_counts.get(category, 0) >= max_per_category:
            continue

        selected.append(article)
        source_counts[source] = source_counts.get(source, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1

    return selected


if __name__ == "__main__":
    candidates = get_candidates()
    unique = remove_duplicates(candidates)
    ranked = rank_articles(unique)
    top = select_top_stories(ranked, limit=5)
    print("Top Ranked AI Stories:")
    for idx, story in enumerate(top, start=1):
        print(f"[{idx}] {story['title']} (Score: {story['ranking_score']}, Source: {story['source']})")


