"""
AI Radar - SQLite persistence layer.

Every helper here is safe to call against an existing database: the schema is
created only when missing, columns are added in place, and no statement ever
drops or rewrites collected rows.
"""

import sqlite3
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from typing import List, Dict, Any, Optional

from backend.app.config import DATABASE_PATH, normalize_category, category_match_values
from backend.app.utils.url_normalization import url_identity_variants


def get_connection() -> sqlite3.Connection:
    """Open a SQLite connection with row access by column name."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def create_tables() -> None:
    """Create the articles table if it does not exist. Never drops data."""
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            url TEXT UNIQUE NOT NULL,
            summary TEXT,
            source TEXT NOT NULL,
            category TEXT,
            published TEXT,
            published_timestamp REAL,
            collected_at TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            is_ai_news INTEGER,
            importance INTEGER,
            career_relevance INTEGER,
            ai_summary TEXT,
            why_it_matters TEXT,
            should_show INTEGER,

            embedding_indexed INTEGER DEFAULT 0,
            analyzed_at TEXT,
            analysis_version TEXT DEFAULT '2.0'
        )
        """
    )

    connection.commit()
    connection.close()


REQUIRED_COLUMNS: Dict[str, str] = {
    "summary": "TEXT",
    "category": "TEXT",
    "published": "TEXT",
    "published_timestamp": "REAL",
    "collected_at": "TEXT",
    "is_ai_news": "INTEGER",
    "importance": "INTEGER",
    "career_relevance": "INTEGER",
    "ai_summary": "TEXT",
    "why_it_matters": "TEXT",
    "should_show": "INTEGER",
    "embedding_indexed": "INTEGER DEFAULT 0",
    "analyzed_at": "TEXT",
    "analysis_version": "TEXT DEFAULT '2.0'",
}


def ensure_columns() -> None:
    """Add any missing column to an existing articles table without data loss."""
    connection = get_connection()
    cursor = connection.cursor()

    if not table_exists(connection, "articles"):
        connection.close()
        return

    existing_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(articles)").fetchall()
    }

    for column_name, column_type in REQUIRED_COLUMNS.items():
        if column_name not in existing_columns:
            cursor.execute(
                f"ALTER TABLE articles ADD COLUMN {column_name} {column_type}"
            )

    connection.commit()
    connection.close()


def ensure_indexes() -> None:
    """Create the indexes the dashboard and pipeline queries rely on."""
    connection = get_connection()
    cursor = connection.cursor()

    if not table_exists(connection, "articles"):
        connection.close()
        return

    statements = (
        "CREATE INDEX IF NOT EXISTS idx_articles_published_ts "
        "ON articles (published_timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_articles_feed "
        "ON articles (is_ai_news, should_show, published_timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_articles_category "
        "ON articles (category)",
        "CREATE INDEX IF NOT EXISTS idx_articles_unprocessed "
        "ON articles (is_ai_news, published_timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_articles_embedding "
        "ON articles (embedding_indexed)",
    )
    for statement in statements:
        cursor.execute(statement)

    connection.commit()
    connection.close()


def parse_published_value(value: Any) -> Optional[float]:
    """
    Convert a stored `published` string into a UTC unix timestamp.

    Understands ISO-8601 ("2026-09-01T17:35:37+00:00") and RFC-2822
    ("Wed, 26 Aug 2026 10:00:00 GMT"), which are both present in databases
    written by earlier builds. Returns None when nothing can be parsed.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).timestamp()
    except (TypeError, ValueError):
        pass

    try:
        parsed = parsedate_to_datetime(text)
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).timestamp()
    except (TypeError, ValueError, OverflowError):
        pass

    return None


def backfill_published_timestamps() -> int:
    """
    Fill published_timestamp for rows that only stored a date string.

    Rows without a timestamp are invisible to the analyzer, the ranking engine
    and the dashboard feed, so this repairs them in place. Only NULL values are
    written; existing timestamps are left untouched.
    """
    connection = get_connection()
    cursor = connection.cursor()

    if not table_exists(connection, "articles"):
        connection.close()
        return 0

    rows = cursor.execute(
        """
        SELECT id, published
        FROM articles
        WHERE published_timestamp IS NULL
          AND published IS NOT NULL
          AND TRIM(published) <> ''
        """
    ).fetchall()

    repaired = 0
    for row in rows:
        timestamp = parse_published_value(row["published"])
        if timestamp is None:
            continue
        cursor.execute(
            "UPDATE articles SET published_timestamp = ? WHERE id = ?",
            (timestamp, row["id"]),
        )
        repaired += 1

    connection.commit()
    connection.close()
    return repaired


def initialize_database(verbose: bool = False) -> Dict[str, Any]:
    """
    Bring the database to a queryable state. Safe to call on every start-up.

    The Streamlit dashboard and every CLI entry point call this before their
    first query so a missing table or column can never surface as an error.
    """
    create_tables()
    ensure_columns()
    ensure_indexes()
    repaired = backfill_published_timestamps()

    if verbose and repaired:
        print(f"Repaired published_timestamp on {repaired} row(s).")

    return {"database": str(DATABASE_PATH), "timestamps_repaired": repaired}


def save_article(article: Dict[str, Any]) -> bool:
    """
    Insert an article unless its canonical source URL is already stored.

    Returns True when a new row was created, False when it already existed.
    """
    connection = get_connection()
    cursor = connection.cursor()
    url_variants = url_identity_variants(article["url"])
    canonical_url = url_variants[0]
    placeholders = ",".join("?" for _ in url_variants)
    cursor.execute("BEGIN IMMEDIATE")
    existing = cursor.execute(
        f"SELECT 1 FROM articles WHERE url IN ({placeholders}) LIMIT 1",
        url_variants,
    ).fetchone()
    if existing:
        connection.rollback()
        connection.close()
        return False

    cursor.execute(
        """
        INSERT OR IGNORE INTO articles
        (
            title,
            url,
            summary,
            source,
            category,
            published,
            published_timestamp,
            collected_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            article["title"],
            canonical_url,
            article.get("summary", ""),
            article["source"],
            normalize_category(article.get("category")),
            article.get("published", ""),
            article.get("published_timestamp"),
            article.get("collected_at", datetime.now(timezone.utc).isoformat()),
        ),
    )

    connection.commit()
    inserted = cursor.rowcount == 1
    connection.close()
    return inserted


def get_articles(limit: int = 20) -> List[sqlite3.Row]:
    """Return the most recently published raw articles."""
    connection = get_connection()
    rows = connection.execute(
        """
        SELECT *
        FROM articles
        ORDER BY published_timestamp DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    connection.close()
    return rows


def get_unprocessed_articles(
    limit: int = 10,
    recent_days: Optional[int] = 7,
) -> List[sqlite3.Row]:
    """
    Return articles the analyzer has not seen yet (is_ai_news IS NULL).

    Ordered oldest-first on purpose. A newest-first queue starves: while new
    articles keep arriving, anything that falls past the batch size is pushed
    further down on every run and is never analysed. Oldest-first drains
    monotonically, so every collected article eventually gets a turn.

    backend.app.services.process_articles reorders this queue by source tier and AI
    signal before spending Ollama time on it; this function is the plain
    fallback ordering.

    Pass recent_days=None to ignore the freshness window and work through the
    whole backlog, including archive entries older than the collection window.
    """
    connection = get_connection()

    if recent_days is None:
        rows = connection.execute(
            """
            SELECT *
            FROM articles
            WHERE is_ai_news IS NULL
              AND published_timestamp IS NOT NULL
            ORDER BY published_timestamp ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    else:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=recent_days)
        ).timestamp()
        rows = connection.execute(
            """
            SELECT *
            FROM articles
            WHERE is_ai_news IS NULL
              AND published_timestamp IS NOT NULL
              AND published_timestamp >= ?
            ORDER BY published_timestamp ASC
            LIMIT ?
            """,
            (cutoff, limit),
        ).fetchall()

    connection.close()
    return rows


def get_unprocessed_candidates(
    recent_days: Optional[int] = 7,
) -> List[sqlite3.Row]:
    """
    Every unanalysed article in scope, with only the columns needed to rank it.

    The caller prioritises the whole pending set rather than a SQL-truncated
    slice of it: truncating first is what let an important announcement sit
    below the batch size run after run.
    """
    connection = get_connection()

    if recent_days is None:
        rows = connection.execute(
            """
            SELECT id, title, summary, source, published, published_timestamp
            FROM articles
            WHERE is_ai_news IS NULL
              AND published_timestamp IS NOT NULL
            """
        ).fetchall()
    else:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=recent_days)
        ).timestamp()
        rows = connection.execute(
            """
            SELECT id, title, summary, source, published, published_timestamp
            FROM articles
            WHERE is_ai_news IS NULL
              AND published_timestamp IS NOT NULL
              AND published_timestamp >= ?
            """,
            (cutoff,),
        ).fetchall()

    connection.close()
    return rows


def get_articles_by_ids(article_ids: List[int]) -> List[sqlite3.Row]:
    """Full rows for the given IDs, returned in the order the caller asked for."""
    if not article_ids:
        return []

    connection = get_connection()
    placeholders = ",".join("?" for _ in article_ids)
    rows = connection.execute(
        f"SELECT * FROM articles WHERE id IN ({placeholders})",
        list(article_ids),
    ).fetchall()
    connection.close()

    by_id = {row["id"]: row for row in rows}
    return [by_id[article_id] for article_id in article_ids if article_id in by_id]


def count_unprocessed_articles(recent_days: Optional[int] = 7) -> int:
    """Count unanalyzed articles, optionally limited to the freshness window."""
    connection = get_connection()

    if recent_days is None:
        row = connection.execute(
            """
            SELECT COUNT(*) FROM articles
            WHERE is_ai_news IS NULL AND published_timestamp IS NOT NULL
            """
        ).fetchone()
    else:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=recent_days)
        ).timestamp()
        row = connection.execute(
            """
            SELECT COUNT(*) FROM articles
            WHERE is_ai_news IS NULL
              AND published_timestamp IS NOT NULL
              AND published_timestamp >= ?
            """,
            (cutoff,),
        ).fetchone()

    connection.close()
    return int(row[0]) if row else 0


def update_ai_analysis(article_id: int, analysis: Dict[str, Any]) -> None:
    """Persist an Ollama analysis result and queue the row for re-embedding."""
    connection = get_connection()
    cursor = connection.cursor()
    now_iso = datetime.now(timezone.utc).isoformat()

    cursor.execute(
        """
        UPDATE articles
        SET
            is_ai_news = ?,
            category = ?,
            importance = ?,
            career_relevance = ?,
            ai_summary = ?,
            why_it_matters = ?,
            should_show = ?,
            analyzed_at = ?,
            embedding_indexed = 0
        WHERE id = ?
        """,
        (
            int(bool(analysis["is_ai_news"])),
            normalize_category(analysis.get("category")),
            analysis["importance"],
            analysis["career_relevance"],
            analysis.get("summary", ""),
            analysis.get("why_it_matters", ""),
            int(bool(analysis["should_show"])),
            now_iso,
            article_id,
        ),
    )

    connection.commit()
    connection.close()


def get_unindexed_articles(limit: int = 50) -> List[sqlite3.Row]:
    """Return verified AI articles that still need a FAISS embedding."""
    connection = get_connection()
    rows = connection.execute(
        """
        SELECT *
        FROM articles
        WHERE is_ai_news = 1
          AND should_show = 1
          AND (embedding_indexed IS NULL OR embedding_indexed = 0)
        ORDER BY published_timestamp DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    connection.close()
    return rows


def get_indexable_articles() -> List[sqlite3.Row]:
    """Every article eligible for the FAISS index, newest first."""
    connection = get_connection()
    rows = connection.execute(
        """
        SELECT
            id, title, url, summary, source, category,
            published, published_timestamp, importance, career_relevance,
            ai_summary, why_it_matters
        FROM articles
        WHERE is_ai_news = 1
          AND should_show = 1
        ORDER BY published_timestamp DESC
        """
    ).fetchall()
    connection.close()
    return rows


def mark_as_indexed(article_ids: List[int]) -> None:
    """Flag article IDs as present in the FAISS index."""
    if not article_ids:
        return

    connection = get_connection()
    placeholders = ",".join("?" for _ in article_ids)
    connection.execute(
        f"UPDATE articles SET embedding_indexed = 1 WHERE id IN ({placeholders})",
        list(article_ids),
    )
    connection.commit()
    connection.close()


def reset_embedding_flags() -> None:
    """Clear every embedding flag so the next sync rebuilds the index."""
    connection = get_connection()
    connection.execute("UPDATE articles SET embedding_indexed = 0")
    connection.commit()
    connection.close()


def get_latest_news(
    limit: int = 30,
    category: Optional[str] = None,
    min_importance: int = 1,
) -> List[Dict[str, Any]]:
    """
    Verified AI articles for the dashboard feed, newest first.

    `category` accepts any spelling: it is resolved to a canonical value and
    matched against every historical alias, so a row stored years ago as
    "Developers & Tools" is still returned for "AI Tools".
    """
    connection = get_connection()

    if category and category not in {"All", "All Categories"}:
        aliases = category_match_values(category)
        placeholders = ",".join("?" for _ in aliases)
        query = f"""
            SELECT *
            FROM articles
            WHERE is_ai_news = 1
              AND should_show = 1
              AND LOWER(TRIM(COALESCE(category, ''))) IN ({placeholders})
              AND COALESCE(importance, 0) >= ?
            ORDER BY published_timestamp DESC, importance DESC
            LIMIT ?
        """
        params: List[Any] = [*aliases, min_importance, limit]
    else:
        query = """
            SELECT *
            FROM articles
            WHERE is_ai_news = 1
              AND should_show = 1
              AND COALESCE(importance, 0) >= ?
            ORDER BY published_timestamp DESC, importance DESC
            LIMIT ?
        """
        params = [min_importance, limit]

    rows = connection.execute(query, params).fetchall()
    connection.close()
    return [dict(row) for row in rows]


def get_article_by_id(article_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single article by primary key."""
    connection = get_connection()
    row = connection.execute(
        "SELECT * FROM articles WHERE id = ?", (article_id,)
    ).fetchone()
    connection.close()
    return dict(row) if row else None


def get_category_counts() -> Dict[str, int]:
    """Verified story counts per canonical category."""
    connection = get_connection()
    rows = connection.execute(
        """
        SELECT category, COUNT(*) AS total
        FROM articles
        WHERE is_ai_news = 1 AND should_show = 1
        GROUP BY category
        """
    ).fetchall()
    connection.close()

    counts: Dict[str, int] = {}
    for row in rows:
        canonical = normalize_category(row["category"])
        counts[canonical] = counts.get(canonical, 0) + int(row["total"])
    return counts


EMPTY_STATS: Dict[str, Any] = {
    "total_collected": 0,
    "ai_news_count": 0,
    "stories_selected": 0,
    "sources_count": 0,
    "stories_today": 0,
    "last_collected": None,
}


def get_news_stats() -> Dict[str, Any]:
    """Aggregate counters for the dashboard header and the CLI status report."""
    connection = None
    try:
        connection = get_connection()
        if not table_exists(connection, "articles"):
            return dict(EMPTY_STATS)

        cursor = connection.cursor()
        total = cursor.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        ai_news = cursor.execute(
            "SELECT COUNT(*) FROM articles WHERE is_ai_news = 1"
        ).fetchone()[0]
        shown = cursor.execute(
            "SELECT COUNT(*) FROM articles WHERE is_ai_news = 1 AND should_show = 1"
        ).fetchone()[0]
        sources = cursor.execute(
            "SELECT COUNT(DISTINCT source) FROM articles WHERE is_ai_news = 1"
        ).fetchone()[0]

        one_day_ago = (
            datetime.now(timezone.utc) - timedelta(hours=24)
        ).timestamp()
        stories_today = cursor.execute(
            """
            SELECT COUNT(*) FROM articles
            WHERE is_ai_news = 1 AND published_timestamp >= ?
            """,
            (one_day_ago,),
        ).fetchone()[0]

        last_row = cursor.execute(
            "SELECT collected_at FROM articles ORDER BY id DESC LIMIT 1"
        ).fetchone()

        return {
            "total_collected": total,
            "ai_news_count": ai_news,
            "stories_selected": shown,
            "sources_count": sources,
            "stories_today": stories_today,
            "last_collected": last_row[0] if last_row else None,
        }
    except sqlite3.Error as error:
        print(f"Database statistics unavailable: {error}")
        return dict(EMPTY_STATS)
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    info = initialize_database(verbose=True)
    print(f"Database ready at {info['database']}")
    print("Current stats:", get_news_stats())
    print("Categories:", get_category_counts())
