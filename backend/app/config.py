"""
AI Radar - Centralised configuration.

Single source of truth for:
  * filesystem paths (always absolute, so any working directory works)
  * local Ollama endpoints / models
  * pipeline + ranking + semantic-search thresholds
  * the canonical category vocabulary (and legacy aliases)

The project is deliberately local-first and free: SQLite + FAISS-CPU + Ollama.
No cloud services, no API keys, no ChromaDB.
"""

import os
from pathlib import Path
from typing import Dict, List
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# ============================================================
# BASE DIRECTORIES  (absolute - never depend on the CWD)
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BASE_DIR

DATA_DIR = BASE_DIR / "data"
DATABASE_DIR = DATA_DIR
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Load .env explicitly from the project root so the dashboard and the CLI
# behave identically regardless of where the process was started from.
load_dotenv(BASE_DIR / ".env")

# ============================================================
# SQLITE DATABASE
# ============================================================

DATABASE_PATH = DATABASE_DIR / "ai_radar.db"

# ============================================================
# FAISS VECTOR STORE (persistent, local, CPU-only)
# ============================================================

# The live artefacts the pipeline reads and writes.
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "ai_radar.index"
FAISS_META_PATH = VECTOR_STORE_DIR / "metadata.json"

# Older builds wrote a flat index directly into data/. Those files are kept on
# disk for reference but are no longer read or written by the pipeline.
LEGACY_FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
LEGACY_FAISS_META_PATH = DATA_DIR / "faiss_meta.json"

# ============================================================
# BRIEFING SNAPSHOT FILES
# ============================================================

LATEST_BRIEFING_FILE = DATA_DIR / "latest_briefing.json"
MORNING_BRIEFING_FILE = DATA_DIR / "morning_briefing.json"
HISTORY_FILE = DATA_DIR / "briefing_history.json"
EVENING_ARCHIVE_DIR = DATA_DIR / "evening_briefings"
REFRESH_STATE_FILE = DATA_DIR / "refresh_state.json"

# ============================================================
# LOCAL OLLAMA CONFIGURATION
# ============================================================

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_GENERATE_URL = f"{OLLAMA_HOST}/api/generate"
OLLAMA_EMBED_URL = f"{OLLAMA_HOST}/api/embeddings"
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# Request timeouts (seconds)
OLLAMA_HEALTH_TIMEOUT = 5
OLLAMA_EMBED_TIMEOUT = 120
OLLAMA_ANALYZE_TIMEOUT = 120
OLLAMA_INTENT_TIMEOUT = 20
OLLAMA_RELEVANCE_TIMEOUT = 30
OLLAMA_ANSWER_TIMEOUT = 90

# ============================================================
# PIPELINE CONFIGURATION
# ============================================================

LOCAL_TIMEZONE = ZoneInfo("Asia/Kolkata")
MORNING_HOUR = 9      # 9 AM IST baseline briefing
EVENING_HOUR = 19     # 7 PM IST differential briefing
RECENT_DAYS = 7
BRIEFING_SIZE = 8
DEFAULT_BATCH_SIZE = 10

# ============================================================
# RANKING THRESHOLDS
# ============================================================

SIGNIFICANT_IMPORTANCE = 7
SIGNIFICANT_CAREER_RELEVANCE = 7
SIGNIFICANT_RANKING_SCORE = 7.0

# ============================================================
# SEMANTIC SEARCH THRESHOLDS
# ============================================================
# The FAISS index is IndexFlatIP over L2-normalised vectors, so a raw score is
# a cosine similarity in [-1, 1] where 1.0 is an exact match:
#     distance = 1 - similarity
# Measured on nomic-embed-text against this corpus:
#     on-topic AI queries with real coverage ....... 0.60 - 0.75
#     off-topic or uncovered queries ............... 0.42 - 0.50
# 0.55 therefore separates "grounded" from "nothing relevant" cleanly.
MAX_SEMANTIC_DISTANCE = 0.45
MIN_SEMANTIC_SIMILARITY = round(1.0 - MAX_SEMANTIC_DISTANCE, 4)   # 0.55
TOP_K_SEARCH_RESULTS = 6

# Story clustering uses the existing normalized FAISS vectors. The title-term
# guard prevents broad AI vocabulary from merging unrelated events.
STORY_CLUSTER_SIMILARITY_THRESHOLD = 0.82
STORY_CLUSTER_MIN_SHARED_TERMS = 2

# ============================================================
# CANONICAL CATEGORY VOCABULARY
# ============================================================
# Exactly one spelling per concept. Everything that touches a category - the
# LLM classifier, the ranking engine, the RSS seed categories, the dashboard
# filters and the briefing sections - resolves through normalize_category().

ALLOWED_CATEGORIES: List[str] = [
    "AI Models",
    "AI Agents",
    "AI Tools",
    "AI Research",
    "AI Companies",
    "AI Jobs & Careers",
    "AI Policy",
    "India AI",
    "Robotics",
    "Open Source AI",
    "Other",
]

DEFAULT_CATEGORY = "Other"

# Legacy / free-form spellings mapped onto a canonical value. Keys are lowercase.
CATEGORY_ALIASES: Dict[str, str] = {
    # "Developers & Tools" was the historical name for "AI Tools".
    "developers & tools": "AI Tools",
    "developers and tools": "AI Tools",
    "developer tools": "AI Tools",
    "developer & tools": "AI Tools",
    "dev tools": "AI Tools",
    "devtools": "AI Tools",
    "ai developer tools": "AI Tools",
    "ai tooling": "AI Tools",
    "tools": "AI Tools",
    "platforms": "AI Tools",
    # Industry and hardware coverage lives under AI Companies.
    "ai industry": "AI Companies",
    "industry": "AI Companies",
    "ai business": "AI Companies",
    "business": "AI Companies",
    "companies": "AI Companies",
    "ai company": "AI Companies",
    "ai hardware": "AI Companies",
    "hardware": "AI Companies",
    "ai chips": "AI Companies",
    "chips": "AI Companies",
    "ai infrastructure": "AI Companies",
    "funding": "AI Companies",
    # Models and agents
    "models": "AI Models",
    "ai model": "AI Models",
    "foundation models": "AI Models",
    "llms": "AI Models",
    "llm": "AI Models",
    "ai models & agents": "AI Models",
    "agents": "AI Agents",
    "ai agent": "AI Agents",
    "agentic ai": "AI Agents",
    # Research
    "research": "AI Research",
    "ai papers": "AI Research",
    "papers": "AI Research",
    "benchmarks": "AI Research",
    # Careers
    "ai jobs": "AI Jobs & Careers",
    "jobs": "AI Jobs & Careers",
    "jobs & careers": "AI Jobs & Careers",
    "careers": "AI Jobs & Careers",
    "ai careers": "AI Jobs & Careers",
    "hiring": "AI Jobs & Careers",
    # Policy
    "policy": "AI Policy",
    "ai regulation": "AI Policy",
    "regulation": "AI Policy",
    "ai governance": "AI Policy",
    "ai safety": "AI Policy",
    # Regional
    "india": "India AI",
    "indian ai": "India AI",
    "india ai ecosystem": "India AI",
    # Robotics and open source
    "robotics & automation": "Robotics",
    "robots": "Robotics",
    "open source": "Open Source AI",
    "opensource ai": "Open Source AI",
    "open-source ai": "Open Source AI",
    # Generic buckets
    "ai news": "Other",
    "general": "Other",
    "general ai": "Other",
    "uncategorized": "Other",
    "uncategorised": "Other",
    "misc": "Other",
}

# Categories a user can browse in the dashboard (the catch-all is excluded).
BROWSABLE_CATEGORIES: List[str] = [c for c in ALLOWED_CATEGORIES if c != DEFAULT_CATEGORY]

_CANONICAL_LOOKUP: Dict[str, str] = {c.lower(): c for c in ALLOWED_CATEGORIES}


def normalize_category(value: object) -> str:
    """
    Resolve any category spelling to a single canonical ALLOWED_CATEGORIES value.

    normalize_category("Developers & Tools") -> "AI Tools"
    normalize_category("ai models")          -> "AI Models"
    normalize_category(None)                 -> "Other"
    """
    if value is None:
        return DEFAULT_CATEGORY

    key = str(value).strip()
    if not key:
        return DEFAULT_CATEGORY

    lowered = key.lower()
    if lowered in _CANONICAL_LOOKUP:
        return _CANONICAL_LOOKUP[lowered]
    if lowered in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[lowered]

    # Tolerate "and"/"&" and hyphen/spacing differences before giving up.
    squashed = lowered.replace(" and ", " & ").replace("-", " ")
    squashed = " ".join(squashed.split())
    if squashed in _CANONICAL_LOOKUP:
        return _CANONICAL_LOOKUP[squashed]
    if squashed in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[squashed]

    return DEFAULT_CATEGORY


def category_match_values(canonical: str) -> List[str]:
    """
    Every lowercase spelling that should match `canonical` in SQL.

    Rows written by earlier builds keep their original spelling because the
    database is never rewritten, so queries compare LOWER(category) against
    this list instead of a single literal.
    """
    target = normalize_category(canonical)
    values = {target.lower()}
    values.update(
        alias
        for alias, mapped in CATEGORY_ALIASES.items()
        if mapped == target and alias
    )
    return sorted(values)

