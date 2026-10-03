import re
import sys
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from typing import List, Dict, Any, Optional

import feedparser
import requests

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backend.app.config import ALLOWED_CATEGORIES, RECENT_DAYS, normalize_category
from backend.app.utils.database import initialize_database, save_article
from backend.app.utils.url_normalization import normalize_url

FEED_TIMEOUT = 20
USER_AGENT = "AI-Radar/2.0 (local, personal news aggregator)"

RSS_SOURCES = [
    # Tier 1 â€” Primary AI Labs & Major Platforms (Trust: 1.0)
    {
        "name": "OpenAI",
        "url": "https://openai.com/news/rss.xml",
        "category": "AI Companies",
        "tier": 1,
        "trust_score": 1.0,
    },
    {
        "name": "Google AI Blog",
        "url": "https://blog.google/technology/ai/rss/",
        "category": "AI Companies",
        "tier": 1,
        "trust_score": 1.0,
    },
    {
        # blogs.microsoft.com/ai/feed/ now returns 410 Gone; the main Microsoft
        # blog is the live replacement and the analyzer filters its non-AI posts.
        "name": "Microsoft AI",
        "url": "https://blogs.microsoft.com/feed/",
        "category": "AI Companies",
        "tier": 1,
        "trust_score": 1.0,
    },
    {
        "name": "Microsoft Research",
        "url": "https://www.microsoft.com/en-us/research/feed/",
        "category": "AI Research",
        "tier": 1,
        "trust_score": 1.0,
    },
    {
        "name": "AWS Machine Learning",
        "url": "https://aws.amazon.com/blogs/machine-learning/feed/",
        "category": "AI Companies",
        "tier": 1,
        "trust_score": 1.0,
    },
    {
        "name": "NVIDIA Newsroom",
        "url": "https://nvidianews.nvidia.com/releases.xml",
        "category": "AI Companies",
        "tier": 1,
        "trust_score": 1.0,
    },
    {
        "name": "NVIDIA Blog",
        "url": "https://feeds.feedburner.com/nvidiablog",
        "category": "AI Companies",
        "tier": 1,
        "trust_score": 1.0,
    },
    {
        "name": "NVIDIA Developer Blog",
        "url": "https://developer.nvidia.com/blog/feed",
        "category": "AI Tools",
        "tier": 1,
        "trust_score": 1.0,
    },
    {
        "name": "Hugging Face",
        "url": "https://huggingface.co/blog/feed.xml",
        "category": "AI Research",
        "tier": 1,
        "trust_score": 1.0,
    },
    {
        "name": "Google DeepMind",
        "url": "https://deepmind.google/blog/rss.xml",
        "category": "AI Research",
        "tier": 1,
        "trust_score": 1.0,
    },
    {
        "name": "Google Research",
        "url": "https://research.google/blog/rss/",
        "category": "AI Research",
        "tier": 1,
        "trust_score": 1.0,
    },
    {
        "name": "Apple Machine Learning Research",
        "url": "https://machinelearning.apple.com/rss.xml",
        "category": "AI Research",
        "tier": 1,
        "trust_score": 1.0,
    },
    {
        "name": "arXiv Artificial Intelligence",
        "url": "https://export.arxiv.org/rss/cs.AI",
        "category": "AI Research",
        "tier": 1,
        "trust_score": 1.0,
    },
    # Tier 2 â€” Authoritative Technology Journalism (Trust: 0.85 - 0.90)
    {
        "name": "MIT Technology Review AI",
        "url": "https://www.technologyreview.com/topic/artificial-intelligence/feed/",
        "category": "AI Research",
        "tier": 2,
        "trust_score": 0.90,
    },
    {
        "name": "TechCrunch AI",
        "url": "https://techcrunch.com/category/artificial-intelligence/feed/",
        "category": "AI Companies",
        "tier": 2,
        "trust_score": 0.85,
    },
    {
        "name": "The Verge AI",
        "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
        "category": "AI Companies",
        "tier": 2,
        "trust_score": 0.85,
    },
    {
        "name": "Economic Times Technology",
        "url": "https://economictimes.indiatimes.com/tech/rssfeeds/13357270.cms",
        "category": "India AI",
        "tier": 2,
        "trust_score": 0.85,
    },
    {
        "name": "Inc42",
        "url": "https://inc42.com/buzz/feed/",
        "category": "India AI",
        "tier": 2,
        "trust_score": 0.80,
    },
    # Tier 3 â€” Regional AI & Ecosystem Developments (Trust: 0.80 - 0.85)
    {
        "name": "Google India Blog",
        "url": "https://blog.google/intl/en-in/rss/",
        "category": "India AI",
        "tier": 3,
        "trust_score": 0.85,
    },
    {
        "name": "YourStory Technology",
        "url": "https://yourstory.com/feed",
        "category": "India AI",
        "tier": 3,
        "trust_score": 0.80,
    },
]

def clean_text(text: Any) -> str:
    """Remove HTML tags, character entities, and redundant whitespace."""
    if not text:
        return ""

    text = str(text)
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Replace common HTML entities
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#39;", "'", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    # Remove space before punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    # Consolidate whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_published_date(entry: Dict[str, Any]) -> Optional[datetime]:
    """Convert various RSS publication date formats to UTC datetime."""
    date_value = (
        entry.get("published")
        or entry.get("updated")
        or entry.get("created")
        or entry.get("pubDate")
    )

    if not date_value:
        return None

    try:
        parsed_date = parsedate_to_datetime(date_value)
        if parsed_date.tzinfo is None:
            parsed_date = parsed_date.replace(tzinfo=timezone.utc)
        return parsed_date.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        pass

    # Fallback to standard string formats
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(str(date_value).strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue

    return None


def extract_summary(entry: Dict[str, Any]) -> str:
    """Extract and clean the best available summary or description from the RSS entry."""
    summary = (
        entry.get("summary")
        or entry.get("description")
        or (entry.get("content", [{}])[0].get("value") if entry.get("content") else "")
        or ""
    )
    return clean_text(summary)


def extract_url(entry: Dict[str, Any]) -> str:
    """Extract and normalize the article link."""
    url = entry.get("link", "")
    if not url:
        links = entry.get("links", [])
        for link in links:
            if link.get("rel") == "alternate":
                url = link.get("href", "")
                break
    return normalize_url(url)


def fetch_feed(url: str):
    """
    Fetch and parse one RSS feed with an explicit timeout.

    feedparser.parse(url) does its own network call with no timeout, so a slow
    or hung server would stall the whole pipeline run. Fetching the bytes with
    requests first bounds every feed to FEED_TIMEOUT seconds.
    """
    response = requests.get(
        url,
        timeout=FEED_TIMEOUT,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
        },
    )
    response.raise_for_status()
    return feedparser.parse(response.content)


def _collect_news_with_status(
    recent_days: int = RECENT_DAYS,
) -> tuple[List[Dict[str, Any]], int, List[Dict[str, str]]]:
    """
    Collect recent articles and retain feed-level success and failure details.
    """
    articles = []
    cutoff_time = datetime.now(timezone.utc) - timedelta(days=recent_days)

    seen_urls = set()
    seen_titles = set()
    feeds_succeeded = 0
    feed_failures = []

    for source in RSS_SOURCES:
        try:
            feed = fetch_feed(source["url"])
            feeds_succeeded += 1

            for entry in feed.entries:
                title = clean_text(entry.get("title", ""))
                url = extract_url(entry)
                summary = extract_summary(entry)
                published_datetime = parse_published_date(entry)

                if not title or not url or published_datetime is None:
                    continue

                if published_datetime < cutoff_time:
                    continue

                if url in seen_urls:
                    continue

                normalized_title = title.lower().strip()
                if normalized_title in seen_titles:
                    continue

                seen_urls.add(url)
                seen_titles.add(normalized_title)

                articles.append(
                    {
                        "title": title,
                        "url": url,
                        "summary": summary,
                        "source": source["name"],
                        "category": normalize_category(source["category"]),
                        "published": published_datetime.isoformat(),
                        "published_timestamp": published_datetime.timestamp(),
                        "collected_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
        except Exception as error:
            print(f"âš ï¸ Error reading {source['name']}: {error}")
            feed_failures.append(
                {"source": source["name"], "error": str(error)}
            )

    # Sort newest first
    articles.sort(key=lambda a: a["published_timestamp"], reverse=True)
    return articles, feeds_succeeded, feed_failures


def collect_news(recent_days: int = RECENT_DAYS) -> List[Dict[str, Any]]:
    """Collect recent unique articles from all configured high-authority feeds."""
    articles, _, _ = _collect_news_with_status(recent_days=recent_days)
    return articles


def _save_news_with_status(
    news: List[Dict[str, Any]],
) -> tuple[int, List[Dict[str, str]]]:
    new_articles = 0
    save_failures = []
    for article in news:
        try:
            inserted = save_article(article)
            if inserted:
                new_articles += 1
        except Exception as error:
            print(f"âŒ Database error saving '{article.get('title', '')}': {error}")
            save_failures.append(
                {"title": str(article.get("title", "")), "error": str(error)}
            )
    return new_articles, save_failures


def save_news_to_database(news: List[Dict[str, Any]]) -> int:
    """Save collected articles into SQLite. Returns newly inserted rows."""
    new_articles, _ = _save_news_with_status(news)
    return new_articles


def run_collection(recent_days: int = RECENT_DAYS) -> Dict[str, Any]:
    """Execute full collection pipeline and return run summary."""
    print("=" * 70)
    print("AI RADAR â€” GLOBAL NEWS COLLECTOR")
    print("=" * 70)
    print(f"Collection Window: Last {recent_days} days")
    print(f"Monitored Sources: {len(RSS_SOURCES)}")

    initialize_database()

    news, feeds_succeeded, feed_failures = _collect_news_with_status(
        recent_days=recent_days
    )
    new_count, save_failures = _save_news_with_status(news)

    print(f"\nâœ… Collected {len(news)} unique articles.")
    print(f"ðŸ“¥ Newly inserted into database: {new_count}")
    return {
        "collected_total": len(news),
        "new_inserted": new_count,
        "feeds_total": len(RSS_SOURCES),
        "feeds_succeeded": feeds_succeeded,
        "feed_failures": feed_failures,
        "save_failures": save_failures,
    }


def verify_source_categories() -> None:
    """Fail fast if a feed is seeded with a category outside the vocabulary."""
    unknown = [
        source["name"]
        for source in RSS_SOURCES
        if source["category"] not in ALLOWED_CATEGORIES
    ]
    if unknown:
        raise ValueError(
            "RSS sources use categories outside ALLOWED_CATEGORIES: "
            + ", ".join(unknown)
        )


verify_source_categories()


if __name__ == "__main__":
    run_collection()
