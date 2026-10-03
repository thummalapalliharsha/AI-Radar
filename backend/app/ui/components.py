"""
AI Radar - Streamlit UI components and grounded-search helpers.

Everything user-visible is escaped before it reaches unsafe_allow_html, and the
retrieval-augmented answer is generated only from articles that already passed
the semantic-similarity threshold in backend.app.services.vector_store.
"""

import html
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

from backend.app.config import (
    LOCAL_TIMEZONE,
    MIN_SEMANTIC_SIMILARITY,
    OLLAMA_ANSWER_TIMEOUT,
    OLLAMA_GENERATE_URL,
    OLLAMA_HEALTH_TIMEOUT,
    OLLAMA_INTENT_TIMEOUT,
    OLLAMA_RELEVANCE_TIMEOUT,
    OLLAMA_MODEL,
    OLLAMA_TAGS_URL,
    EMBEDDING_MODEL,
    normalize_category,
)

NO_GROUNDED_FACTS_MESSAGE = (
    "Based on the current AI Radar database, specific details regarding this "
    "query are not documented."
)


def esc(value: Any) -> str:
    """HTML-escape a value for safe interpolation into a markup block."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def format_time_ago(published_timestamp: Any) -> str:
    """Human-readable relative age, falling back to an IST date."""
    if not published_timestamp:
        return ""
    try:
        published = datetime.fromtimestamp(
            float(published_timestamp), tz=timezone.utc
        )
    except (TypeError, ValueError, OSError, OverflowError):
        return ""

    seconds = (datetime.now(timezone.utc) - published).total_seconds()
    if seconds < 0:
        return "just now"
    if seconds < 3600:
        return f"{max(1, int(seconds // 60))}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    if seconds < 604800:
        return f"{int(seconds // 86400)}d ago"
    return published.astimezone(LOCAL_TIMEZONE).strftime("%d %b %Y")


# ============================================================
# MINIMAL MARKDOWN -> HTML
# ============================================================
# The grounded answer is rendered inside a styled card, and Streamlit treats a
# whole unsafe_allow_html block as HTML - raw markdown from the LLM would show
# up as literal asterisks. This converts the few constructs the model actually
# produces, after escaping, so nothing the model returns can inject markup.

_BULLET_RE = re.compile(r"^\s*(?:[-*â€¢]|\d+[.)])\s+(.*)$")
_ORDERED_RE = re.compile(r"^\s*\d+[.)]\s+")


def _inline_markdown(text: str) -> str:
    """Apply bold/italic/code/citation styling to already-escaped text."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\w)_(?!_)(.+?)(?<!_)_(?!\w)", r"<em>\1</em>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"`([^`]+?)`", r"<code>\1</code>", text)
    text = re.sub(r"\[(\d+)\]", r'<span class="citation">[\1]</span>', text)
    return text


def markdown_to_html(text: Any) -> str:
    """Convert a short markdown answer into safe HTML for a styled card."""
    if not text:
        return ""

    blocks: List[str] = []
    list_items: List[str] = []
    list_tag = "ul"
    paragraph: List[str] = []

    def flush_list() -> None:
        if list_items:
            items = "".join(f"<li>{item}</li>" for item in list_items)
            blocks.append(f"<{list_tag}>{items}</{list_tag}>")
            list_items.clear()

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(f"<p>{' '.join(paragraph)}</p>")
            paragraph.clear()

    for raw_line in str(text).splitlines():
        line = raw_line.rstrip()
        escaped = _inline_markdown(esc(line.strip()))

        if not line.strip():
            flush_paragraph()
            flush_list()
            continue

        bullet = _BULLET_RE.match(line)
        if bullet:
            flush_paragraph()
            list_tag = "ol" if _ORDERED_RE.match(line) else "ul"
            list_items.append(_inline_markdown(esc(bullet.group(1).strip())))
            continue

        flush_list()
        paragraph.append(escaped)

    flush_paragraph()
    flush_list()
    return "".join(blocks)


# ============================================================
# STORY CARD
# ============================================================

def story_summary(story: Dict[str, Any]) -> str:
    """
    Best available description for a story.

    Database rows keep the raw RSS text in `summary` and the LLM text in
    `ai_summary`; briefing snapshots and FAISS metadata already store the LLM
    text in `summary`. Preferring `ai_summary` keeps the analysed text on top
    whichever shape the caller passes in.
    """
    return story.get("ai_summary") or story.get("summary") or ""


def render_story_card(story: Dict[str, Any], index: Optional[int] = None) -> None:
    """Render one verified story with its impact analysis and source link."""
    title = story.get("title") or "Untitled Story"
    source = story.get("source") or "Verified Source"
    category = normalize_category(
        story.get("category") or story.get("special_category")
    )
    importance = story.get("importance") or 5
    career = story.get("career_relevance") or 5
    summary = story_summary(story)
    why = story.get("why_it_matters") or ""
    url = story.get("url") or ""
    time_str = format_time_ago(story.get("published_timestamp"))
    similarity = story.get("similarity")

    prefix = f"{index}. " if index is not None else ""

    meta_pills = [
        f'<span class="pill pill-source">{esc(source)}</span>',
        f'<span class="pill pill-cat">{esc(category)}</span>',
        f'<span class="pill pill-importance">Impact {esc(importance)}/10</span>',
        f'<span class="pill pill-cat">Career {esc(career)}/10</span>',
    ]
    if similarity is not None:
        meta_pills.append(
            f'<span class="pill pill-match">Match {float(similarity) * 100:.0f}%</span>'
        )
    if time_str:
        meta_pills.append(f'<span class="pill-time">{esc(time_str)}</span>')

    why_block = ""
    if why:
        why_block = (
            '<div class="why-box">'
            '<div class="why-label">Why it matters</div>'
            f"{esc(why)}"
            "</div>"
        )

    st.markdown(
        f"""
        <div class="story-card">
            <div class="story-header">
                <div class="story-title">{esc(prefix)}{esc(title)}</div>
            </div>
            <div class="story-meta">{''.join(meta_pills)}</div>
            <div class="story-summary">{esc(summary)}</div>
            {why_block}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if url:
        col1, _ = st.columns([2, 8])
        with col1:
            st.link_button(
                "Read original source", url, use_container_width=True
            )

    source_urls = story.get("source_urls") or []
    if len(source_urls) > 1:
        with st.expander(f"View {len(source_urls)} covered sources"):
            for source in source_urls:
                label = source.get("source") or "Source"
                link = source.get("url") or ""
                if link:
                    st.markdown(f"- [{esc(label)}]({esc(link)})")


# ============================================================
# OLLAMA HEALTH
# ============================================================

def check_ollama() -> Dict[str, Any]:
    """Report whether the local Ollama daemon and both models are available."""
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=OLLAMA_HEALTH_TIMEOUT)
        response.raise_for_status()
        models = [
            model.get("name", "") for model in response.json().get("models", [])
        ]
        return {
            "online": True,
            "models": models,
            "has_llm": any(OLLAMA_MODEL.split(":")[0] in m for m in models),
            "has_embedding": any(
                EMBEDDING_MODEL.split(":")[0] in m for m in models
            ),
        }
    except requests.exceptions.RequestException:
        return {
            "online": False,
            "models": [],
            "has_llm": False,
            "has_embedding": False,
        }


# ============================================================
# QUERY DOMAIN INTENT
# ============================================================
# A coarse gate: obvious AI vocabulary is accepted, obvious off-topic
# vocabulary is rejected, and anything ambiguous is sent to the local LLM.
# The similarity threshold in vector_store is the real guard against
# answering something the corpus does not cover.

AI_QUERY_TERMS = [
    "ai", "a.i.", "artificial intelligence", "agi", "machine learning", "ml",
    "deep learning", "neural network", "neural networks", "llm", "llms",
    "large language model", "language model", "foundation model", "frontier model",
    "reasoning model", "transformer", "embedding", "embeddings", "fine tuning",
    "fine-tuning", "rag", "inference", "training run", "gpu", "gpus", "tpu",
    "npu", "accelerator", "datacenter", "data center", "openai", "chatgpt",
    "gpt", "anthropic", "claude", "gemini", "deepmind", "llama", "mistral",
    "nvidia", "hugging face", "huggingface", "copilot", "midjourney",
    "stable diffusion", "diffusion model", "computer vision", "nlp",
    "natural language", "speech recognition", "multimodal", "agent", "agents",
    "agentic", "autonomous agent", "chatbot", "prompt", "prompting",
    "alignment", "guardrail", "guardrails", "red teaming", "hallucination",
    "robotics", "robot", "humanoid", "self driving", "autonomous vehicle",
    "open weights", "open source model", "benchmark", "mlops", "vector database",
]

NON_AI_QUERY_TERMS = [
    "cricket", "ipl", "football", "soccer", "basketball", "tennis", "hockey",
    "match score", "scorecard", "world cup", "olympics", "recipe", "recipes",
    "biryani", "cooking", "restaurant", "weather", "forecast", "monsoon",
    "horoscope", "astrology", "bollywood", "hollywood", "movie", "movies",
    "film", "song", "songs", "lyrics", "celebrity", "actor", "actress",
    "flight booking", "hotel booking", "holiday", "vacation", "tourism",
    "share price", "stock price", "sensex", "nifty", "mutual fund", "crypto",
    "bitcoin", "ethereum", "lottery", "election result", "exam result",
    "medical advice", "symptom", "symptoms", "dating", "joke", "jokes",
    "plumber", "leaking tap", "car repair",
]


def _mentions(text: str, terms: List[str]) -> bool:
    """Word-boundary containment test.

    A substring test would match "ai" inside "Mumbai" and classify a cricket
    question as AI-related, so every term is matched on word boundaries.
    """
    for term in terms:
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text):
            return True
    return False


def keyword_search_intent(query: str) -> Optional[bool]:
    """Deterministic verdict from vocabulary alone, or None when unclear."""
    text = (query or "").lower()
    if not text.strip():
        return None

    has_ai = _mentions(text, AI_QUERY_TERMS)
    has_non_ai = _mentions(text, NON_AI_QUERY_TERMS)

    if has_ai and not has_non_ai:
        return True
    if has_non_ai and not has_ai:
        return False
    return None


def classify_search_intent(
    query: str, timeout: int = OLLAMA_INTENT_TIMEOUT
) -> Optional[bool]:
    """
    Decide whether a query belongs to the AI/ML domain at all.

    Returns True for AI questions, False for off-topic ones, and None only for
    an empty query. Unambiguous vocabulary short-circuits; ambiguous wording is
    sent to the local LLM, and if Ollama cannot answer, the query is rejected
    rather than guessed - AI Radar answers AI questions only.
    """
    clean_query = (query or "").strip()
    if not clean_query:
        return None

    verdict = keyword_search_intent(clean_query)
    if verdict is not None:
        return verdict

    prompt = f"""You classify search queries for AI Radar, a database that only
covers artificial intelligence news.

QUERY:
"{clean_query}"

Answer true when the query is about artificial intelligence, machine learning,
deep learning, LLMs, AI agents, AI research, AI tools, AI hardware such as
GPUs, AI companies, AI policy or AI careers.

Answer false for every other subject, including sports, cooking, travel,
weather, entertainment, share prices, crypto and general consumer topics.

Reply with JSON only, using exactly this shape and no extra keys:
{{"is_ai_related": true}}   or   {{"is_ai_related": false}}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }

    try:
        response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        parsed = json.loads(response.json().get("response", "{}") or "{}")
        return bool(parsed.get("is_ai_related", False))
    except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError):
        return False


# ============================================================
# GROUNDED ANSWER (RAG)
# ============================================================

def build_context_block(articles: List[Dict[str, Any]]) -> str:
    """Render retrieved articles as the only facts the model may use."""
    blocks = []
    for position, article in enumerate(articles, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{position}] TITLE: {article.get('title', '')}",
                    f"SOURCE: {article.get('source', '')}",
                    f"CATEGORY: {normalize_category(article.get('category'))}",
                    f"PUBLISHED: {article.get('published', '')}",
                    f"SUMMARY: {story_summary(article)}",
                    f"WHY IT MATTERS: {article.get('why_it_matters', '')}",
                    f"URL: {article.get('url', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def generate_grounded_answer(
    query: str,
    articles: List[Dict[str, Any]],
    timeout: int = OLLAMA_ANSWER_TIMEOUT,
) -> Optional[str]:
    """
    Summarise the retrieved articles, and only those, into a short answer.

    `articles` must already be threshold-filtered by the vector store. Returns
    None when Ollama is unavailable so the caller can show the verified source
    cards on their own instead of inventing prose.
    """
    if not articles:
        return None

    prompt = f"""You are the AI Radar research assistant.

Every article below has already been retrieved and verified as relevant to the
user's input. Your job is to brief the user on what those articles report.

The input may be a full question or just a topic or product name. If it is a
topic, summarise what the articles say about that topic.

RULES
1. Use nothing beyond these articles - no outside knowledge, no assumptions.
2. Never invent a company, number, date, model name or benchmark result.
3. Cite the article number inline like [1] or [2] after each claim.
4. Keep it tight: 2 to 4 bullet points, or two short paragraphs.
5. Only if not one of the articles mentions the subject at all, reply with
   exactly this sentence and nothing else: {NO_GROUNDED_FACTS_MESSAGE}
   Do not use that sentence merely because the articles are incomplete - report
   what they do say and leave the rest out.

RETRIEVED ARTICLES
{build_context_block(articles)}

USER INPUT
"{query}"

BRIEFING:
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }

    try:
        response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        answer = (response.json().get("response") or "").strip()
        return answer or None
    except requests.exceptions.RequestException as error:
        print(f"Grounded answer generation failed: {error}")
        return None


def validate_retrieved_articles(
    query: str,
    articles: List[Dict[str, Any]],
    timeout: int = OLLAMA_RELEVANCE_TIMEOUT,
) -> List[Dict[str, Any]]:
    """Use Ollama plus a narrow topic-anchor check to filter false positives."""
    if not articles:
        return []

    candidate_lines = []
    for position, article in enumerate(articles, start=1):
        candidate_lines.append(
            f"{position}. TITLE: {article.get('title', '')}\n"
            f"CATEGORY: {normalize_category(article.get('category'))}\n"
            f"SUMMARY: {story_summary(article)}\n"
            f"WHY IT MATTERS: {article.get('why_it_matters', '')}"
        )

    prompt = f"""You are a strict relevance checker for AI Radar.

Decide which candidate articles directly address the user's AI topic or question.
Semantic similarity alone is not evidence. Reject candidates that only share a
company, location, generic AI word, or broad technology term. Keep a candidate
only when its title or supplied summary materially answers the user's topic.

USER QUERY:
{query}

CANDIDATES:
{chr(10).join(candidate_lines)}

Return JSON only with this exact shape:
{{"relevant_numbers": [1, 3]}}
Return an empty list when none directly address the query.
"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    try:
        response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        parsed = json.loads(response.json().get("response", "{}") or "{}")
        numbers = parsed.get("relevant_numbers", [])
        if not isinstance(numbers, list):
            return []
        accepted = {
            int(number) for number in numbers if isinstance(number, (int, float))
        }
        validated = [
            article for position, article in enumerate(articles, start=1)
            if position in accepted
        ]
        # For a specific multi-term query, require at least one distinctive
        # query concept in the candidate text. This blocks generic semantic
        # neighbours such as a fluid-control paper for "underwater agriculture"
        # while leaving broad queries like "AI agents" to Ollama's judgment.
        ignored_terms = {
            "ai", "latest", "recent", "today", "yesterday", "current",
            "developments", "development", "news", "work", "what", "are",
            "the", "in", "for", "of", "and", "on", "this", "week",
        }
        query_terms = {
            term for term in re.findall(r"[a-z0-9]+", query.lower())
            if len(term) > 2 and term not in ignored_terms
        }
        if len(query_terms) >= 2:
            anchored = []
            for article in validated:
                article_text = " ".join(
                    [
                        str(article.get("title", "")),
                        story_summary(article),
                        str(article.get("why_it_matters", "")),
                    ]
                ).lower()
                if any(re.search(rf"\b{re.escape(term)}\b", article_text) for term in query_terms):
                    anchored.append(article)
            return anchored
        return validated
    except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError):
        return []


# ============================================================
# GUIDANCE CARDS
# ============================================================

def render_invalid_search_guidance(query: str = "") -> None:
    """Shown when the query is not an AI question at all."""
    asked = (
        f'<p class="guidance-query">You asked: <b>{esc(query)}</b></p>'
        if query
        else ""
    )
    st.markdown(
        f"""
        <div class="invalid-query-box">
            <div class="guidance-title">This search is limited to AI and machine learning</div>
            {asked}
            <p class="guidance-body">
                AI Radar only stores verified artificial-intelligence coverage:
                models, agents, research, developer tools, AI companies, AI
                hardware, policy and AI careers. It cannot answer questions
                outside that domain.
            </p>
            <div class="guidance-subtitle">Try one of these instead</div>
            <ul class="guidance-list">
                <li><b>Models</b> - What are the latest reasoning model releases?</li>
                <li><b>Agents</b> - How are autonomous agents used in cybersecurity?</li>
                <li><b>Tools</b> - Which AI developer tools launched recently?</li>
                <li><b>Research</b> - Latest benchmarks and reasoning papers</li>
                <li><b>India AI</b> - Recent AI initiatives in India</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_search_guidance(query: str) -> None:
    """Shown when nothing in the corpus clears the similarity threshold."""
    st.markdown(
        f"""
        <div class="invalid-query-box empty-result-box">
            <div class="guidance-title">No sufficiently relevant AI information found</div>
            <p class="guidance-body">
                Nothing in the AI Radar database matches
                <b>{esc(query)}</b> with enough semantic confidence
                (minimum cosine similarity {MIN_SEMANTIC_SIMILARITY:.2f}).
                No answer is generated, because an answer would not be grounded
                in verified articles.
            </p>
            <p class="guidance-body-dim">
                Broaden the wording, or collect and analyse more articles with
                <code>python -m backend.app.main --collect --process --index</code>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ungrounded_answer_notice() -> None:
    """Shown when the local LLM cannot be reached for synthesis."""
    st.warning(
        "Ollama did not return a synthesised answer, so only the verified "
        "source articles are shown below. Nothing has been generated from "
        "outside the AI Radar database.",
        icon=":material/info:",
    )



