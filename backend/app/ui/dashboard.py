"""
AI Radar - Streamlit dashboard (the application entry point).

    python -m streamlit run backend/app/ui/dashboard.py

Streamlit only puts the script's own folder on sys.path, so the repository root
is prepended below before anything from `backend.app` is imported.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime  # noqa: E402

import streamlit as st  # noqa: E402

from backend.app.config import (  # noqa: E402
    BROWSABLE_CATEGORIES,
    EMBEDDING_MODEL,
    HISTORY_FILE,
    LATEST_BRIEFING_FILE,
    LOCAL_TIMEZONE,
    MIN_SEMANTIC_SIMILARITY,
    MORNING_BRIEFING_FILE,
    OLLAMA_MODEL,
    TOP_K_SEARCH_RESULTS,
)
from backend.app.services.briefing_generator import load_json_file  # noqa: E402
from backend.app.services.vector_store import get_index_stats, search_vector_store  # noqa: E402
from backend.app.services.story_clustering import cluster_articles  # noqa: E402
from backend.app.services.top_stories import get_top_stories  # noqa: E402
from backend.app.services.refresh_radar import (  # noqa: E402
    get_last_successful_refresh,
    refresh_radar,
)
from backend.app.ui.top_story_components import render_top_story_card  # noqa: E402
from backend.app.ui.components import (  # noqa: E402
    check_ollama,
    classify_search_intent,
    esc,
    generate_grounded_answer,
    validate_retrieved_articles,
    markdown_to_html,
    render_empty_search_guidance,
    render_invalid_search_guidance,
    render_story_card,
    format_time_ago,
    render_ungrounded_answer_notice,
)
from backend.app.ui.styles import inject_custom_styles  # noqa: E402
from backend.app.utils.database import (  # noqa: E402
    get_latest_news,
    get_news_stats,
    initialize_database,
)

st.set_page_config(
    page_title="AI Radar - Global AI Intelligence",
    page_icon=":material/radar:",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_styles()


@st.cache_resource(show_spinner=False)
def bootstrap_database() -> dict:
    """
    Create/migrate the SQLite schema before the first query runs.

    cache_resource keeps this to one execution per server process rather than
    once per rerun. It never drops or rewrites collected rows.
    """
    return initialize_database()


DB_INFO = bootstrap_database()

if "refresh_notice" not in st.session_state:
    st.session_state.refresh_notice = None


# ============================================================
# NAVIGATION
# ============================================================

VIEW_LIVE = "Live Latest AI"
VIEW_TOP = "Top Stories"
VIEW_SEARCH = "AI Semantic Search"
VIEW_BRIEFINGS = "Intelligence Briefings"
VIEW_RESEARCH = "AI Research"
VIEW_TOOLS = "AI Tools & Platforms"
VIEW_MODELS = "Models & Agents"
VIEW_CAREERS = "AI Jobs & Careers"
VIEW_INDIA = "India AI"
VIEW_ROBOTICS = "AI Robotics"

VIEWS = [
    VIEW_LIVE,
    VIEW_TOP,
    VIEW_SEARCH,
    VIEW_BRIEFINGS,
    VIEW_RESEARCH,
    VIEW_TOOLS,
    VIEW_MODELS,
    VIEW_CAREERS,
    VIEW_INDIA,
    VIEW_ROBOTICS,
]

MONITORED_FEEDS = [
    "OpenAI Research & News",
    "Google AI & DeepMind",
    "Microsoft AI & Research",
    "AWS Machine Learning",
    "NVIDIA Newsroom, Blog & Developer",
    "Hugging Face Research",
    "MIT Technology Review AI",
    "TechCrunch AI & VentureBeat AI",
    "India AI ecosystem updates",
]

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div>
                <div class="sidebar-brand-name">AI RADAR</div>
                <div class="sidebar-brand-tag">Global AI Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("100% local - Ollama - FAISS - SQLite")
    st.divider()

    selected_view = st.radio("Navigation", VIEWS, index=0)

    st.divider()
    ollama = check_ollama()
    index_stats = get_index_stats()

    st.markdown("### System health")
    if ollama["online"]:
        st.success(f"Ollama online - {OLLAMA_MODEL}", icon=":material/check_circle:")
        if not ollama["has_embedding"]:
            st.warning(
                f"Embedding model {EMBEDDING_MODEL} not found. "
                f"Run: ollama pull {EMBEDDING_MODEL}",
                icon=":material/warning:",
            )
        if not ollama["has_llm"]:
            st.warning(
                f"LLM {OLLAMA_MODEL} not found. Run: ollama pull {OLLAMA_MODEL}",
                icon=":material/warning:",
            )
    else:
        st.error(
            "Ollama is offline. Start it to enable search and analysis.",
            icon=":material/error:",
        )

    if index_stats["healthy"]:
        st.caption(
            f"FAISS index: {index_stats['indexed_documents']} vectors "
            f"({index_stats['dimension']}d)"
        )
    else:
        st.caption("FAISS index: not built yet - run `python -m backend.app.main --index`")

    if DB_INFO.get("timestamps_repaired"):
        st.caption(
            f"Repaired {DB_INFO['timestamps_repaired']} missing publication "
            "timestamp(s) on start-up"
        )

    st.divider()
    st.markdown("### Monitored labs & feeds")
    for feed in MONITORED_FEEDS:
        st.caption(feed)

    st.divider()
    if st.button("Refresh radar feed", use_container_width=True):
        with st.spinner("Refreshing AI Radar..."):
            try:
                refresh_result = refresh_radar()
            except Exception as error:
                refresh_result = {"success": False, "error": str(error)}

        if refresh_result.get("success"):
            partial_note = ""
            if refresh_result.get("partial"):
                failed_feeds = ", ".join(
                    failure["source"] for failure in refresh_result["feed_failures"]
                )
                partial_note = (
                    f" Partial collection: {len(refresh_result['feed_failures'])} "
                    f"feed(s) failed ({failed_feeds})."
                )
            backlog_note = ""
            if refresh_result.get("remaining", 0):
                backlog_note = (
                    f" {refresh_result['remaining']} pending article(s) remain "
                    "in the recent window."
                )
            if refresh_result.get("new_articles", 0):
                st.session_state.refresh_notice = (
                    "success",
                    "Radar refreshed successfully - "
                    f"{refresh_result['new_articles']} new articles, "
                    f"{refresh_result['new_verified_ai_stories']} new verified AI stories."
                    f"{partial_note}{backlog_note}",
                )
            else:
                st.session_state.refresh_notice = (
                    "info",
                    "Refresh completed with no new articles. "
                    f"Analyzed {refresh_result['processed']} pending article(s)."
                    f"{partial_note}{backlog_note}",
                )
        else:
            st.session_state.refresh_notice = (
                "error",
                "Radar refresh did not complete; the previous successful sync timestamp was retained. "
                f"{refresh_result.get('error') or 'Article processing or vector indexing did not finish.'}",
            )
        st.cache_data.clear()
        st.rerun()


# ============================================================
# HERO + METRICS
# ============================================================

stats = get_news_stats()

last_sync = "No successful refresh yet"
sync_value = get_last_successful_refresh()
if sync_value:
    try:
        refreshed_at = datetime.fromisoformat(
            str(sync_value).replace("Z", "+00:00")
        )
        last_sync = refreshed_at.astimezone(LOCAL_TIMEZONE).strftime(
            "%d %b %Y, %I:%M %p IST"
        )
    except ValueError:
        last_sync = str(sync_value)

if st.session_state.refresh_notice:
    notice_type, notice_text = st.session_state.refresh_notice
    getattr(st, notice_type)(notice_text)
    st.session_state.refresh_notice = None

st.markdown(
    f"""
    <div class="radar-hero">
        <div class="radar-title">AI RADAR</div>
        <div class="radar-subtitle">
            Continuous intelligence on verified global artificial-intelligence
            developments: frontier models, agentic systems, developer tooling,
            research and policy. Classified, ranked and searched entirely on
            this machine.
        </div>
        <div class="radar-live-badge">
            <div class="radar-live-dot"></div>
            <span>LIVE INTELLIGENCE FEED &middot; LAST SYNC: {esc(last_sync)}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def render_metric_row() -> None:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-label">Verified AI stories</div>
                <div class="metric-value">{stats['ai_news_count']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Monitored sources</div>
                <div class="metric-value">{stats['sources_count']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Published last 24h</div>
                <div class="metric-value">{stats['stories_today']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Raw articles ingested</div>
                <div class="metric-value">{stats['total_collected']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feed(
    articles: list,
    empty_message: str,
    caption: str = "",
) -> None:
    """Render a list of stories, or a friendly note when the list is empty."""
    if not articles:
        st.info(empty_message)
        return
    if caption:
        st.caption(caption)
    for position, story in enumerate(articles, start=1):
        render_story_card(story, index=position)


def published_key(story: dict) -> float:
    """Sort key that tolerates a missing or NULL published_timestamp."""
    try:
        return float(story.get("published_timestamp") or 0.0)
    except (TypeError, ValueError):
        return 0.0


# ============================================================
# VIEW: LIVE LATEST AI
# ============================================================

if selected_view == VIEW_LIVE:
    render_metric_row()

    st.markdown(
        '<div class="section-header">Continuous AI news stream</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Verified, AI-analysed developments ordered by publication recency and "
        "impact score."
    )

    filter_column, impact_column = st.columns([3, 1])
    with filter_column:
        category_filter = st.selectbox(
            "Category filter",
            ["All Categories", *BROWSABLE_CATEGORIES],
            index=0,
            label_visibility="collapsed",
        )
    with impact_column:
        min_impact = st.slider(
            "Minimum impact", 1, 10, 5, label_visibility="collapsed"
        )

    selected_category = (
        None if category_filter == "All Categories" else category_filter
    )
    live_articles = get_latest_news(
        limit=40, category=selected_category, min_importance=min_impact
    )
    live_articles = cluster_articles(live_articles)

    render_feed(
        live_articles,
        "No verified AI articles match the selected filters. Lower the minimum "
        "impact, pick another category, or run "
        "`python -m backend.app.main --collect --process --index`.",
        caption=f"Showing {len(live_articles)} verified intelligence stories",
    )


# ============================================================
# VIEW: TOP STORIES
# ============================================================

elif selected_view == VIEW_TOP:
    st.markdown(
        '<div class="section-header">Top Stories</div>',
        unsafe_allow_html=True,
    )
    top_stories = get_top_stories(limit=10)
    if not top_stories:
        st.info("No top AI stories are available right now.")
    else:
        source_count = sum(int(story.get("source_count") or 1) for story in top_stories)
        latest_update = max(
            (published_key(story) for story in top_stories),
            default=0.0,
        )
        latest_label = format_time_ago(latest_update) if latest_update else ""
        st.markdown(
            '<div class="top-story-subtitle">The most important AI developments right now.</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"{len(top_stories)} stories Â· {source_count} covered sources"
            + (f" Â· updated {latest_label}" if latest_label else "")
        )
        for rank, story in enumerate(top_stories, start=1):
            render_top_story_card(story, rank=rank)


# ============================================================
# VIEW: AI SEMANTIC SEARCH (RAG)
# ============================================================

elif selected_view == VIEW_SEARCH:
    st.markdown(
        '<div class="section-header">Ask AI Radar</div>',
        unsafe_allow_html=True,
    )
    st.write(
        f"Semantic search across the verified AI Radar repository, grounded in "
        f"local **{EMBEDDING_MODEL}** embeddings and answered by local "
        f"**{OLLAMA_MODEL}**. Only AI questions are answered, and only from "
        f"retrieved articles scoring at least "
        f"{MIN_SEMANTIC_SIMILARITY:.2f} cosine similarity."
    )

    if not index_stats["healthy"]:
        st.warning(
            "The FAISS index is empty or out of sync, so semantic search has "
            "nothing to retrieve. Run `python -m backend.app.main --index` "
            "(or `--rebuild-index`) first.",
            icon=":material/warning:",
        )

    with st.form("semantic_search", clear_on_submit=False):
        query = st.text_input(
            "Search query",
            placeholder="e.g. What are the latest developments in AI agents for cybersecurity?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button(
            "Run semantic search", type="primary", use_container_width=True
        )

    if submitted:
        clean_query = (query or "").strip()

        if not clean_query:
            st.warning("Enter an AI question or topic to search.")
        elif not ollama["online"]:
            st.error(
                "Ollama is offline, so the query cannot be embedded. Start "
                "Ollama and search again.",
                icon=":material/error:",
            )
        else:
            with st.spinner("Checking that the question is about AI..."):
                is_ai_question = classify_search_intent(clean_query)

            if is_ai_question is not True:
                # Requirement: never attempt an answer outside the AI domain.
                render_invalid_search_guidance(clean_query)
            else:
                with st.spinner("Retrieving from the FAISS index..."):
                    # search_vector_store enforces MIN_SEMANTIC_SIMILARITY, so
                    # anything returned here is genuinely relevant.
                    matches = search_vector_store(
                        clean_query, top_k=TOP_K_SEARCH_RESULTS
                    )
                    matches = validate_retrieved_articles(clean_query, matches)

                if not matches:
                    render_empty_search_guidance(clean_query)
                else:
                    with st.spinner("Synthesising a grounded answer..."):
                        answer = generate_grounded_answer(clean_query, matches)

                    if answer:
                        best = max(match["similarity"] for match in matches)
                        st.markdown(
                            f"""
                            <div class="search-answer-card">
                                <div class="search-answer-title">AI Radar intelligence briefing</div>
                                <div class="search-answer-body">{markdown_to_html(answer)}</div>
                                <div class="search-grounding-note">
                                    Grounded in {len(matches)} retrieved article(s) &middot;
                                    best match {best * 100:.0f}% &middot;
                                    threshold {MIN_SEMANTIC_SIMILARITY * 100:.0f}%
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        render_ungrounded_answer_notice()

                    st.markdown(
                        '<div class="section-header" style="font-size:1.2rem;">'
                        "Supporting verified sources</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"{len(matches)} source document(s) above the "
                        f"{MIN_SEMANTIC_SIMILARITY:.2f} similarity threshold"
                    )
                    for position, match in enumerate(matches, start=1):
                        render_story_card(match, index=position)


# ============================================================
# VIEW: INTELLIGENCE BRIEFINGS
# ============================================================

elif selected_view == VIEW_BRIEFINGS:
    st.markdown(
        '<div class="section-header">Scheduled intelligence snapshots</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "The morning baseline captures the 9:00 AM IST landscape. The evening "
        "update compares against that saved baseline and says so plainly when "
        "nothing major has happened since."
    )

    morning_tab, evening_tab, history_tab = st.tabs(
        ["Morning baseline (9 AM IST)", "Evening update (7 PM IST)", "History"]
    )

    with morning_tab:
        morning = load_json_file(MORNING_BRIEFING_FILE, None)
        if not morning:
            st.info(
                "No morning baseline yet. Generate one with "
                "`python -m backend.app.main --morning`."
            )
        else:
            generated = morning.get(
                "generated_at_local", morning.get("generated_at", "")
            )
            st.markdown(
                f"""
                <div class="briefing-banner">
                    <div class="briefing-banner-title">Morning intelligence baseline</div>
                    <div class="briefing-banner-desc">
                        Generated {esc(generated)} &middot;
                        {morning.get('article_count', 0)} curated stories
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_feed(morning.get("stories", []), "This baseline is empty.")

    with evening_tab:
        latest = load_json_file(LATEST_BRIEFING_FILE, None)
        if not latest or latest.get("briefing_type") != "evening":
            st.info(
                "No evening update yet. Generate one with "
                "`python -m backend.app.main --evening`."
            )
        else:
            status = latest.get("status", {})
            has_new = bool(status.get("has_new_developments"))
            message = status.get("message", "Evening comparison completed.")
            generated = latest.get(
                "generated_at_local", latest.get("generated_at", "")
            )
            title_colour = "#34d399" if has_new else "#94a3b8"
            comparison = status.get("comparison_period") or {}
            baseline = comparison.get("from", "")

            st.markdown(
                f"""
                <div class="briefing-banner">
                    <div class="briefing-banner-title" style="color:{title_colour};">
                        Evening intelligence update
                    </div>
                    <div class="briefing-banner-desc">
                        <b>{esc(message)}</b><br>
                        Generated {esc(generated)}<br>
                        Compared against the morning baseline of {esc(baseline)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if not has_new:
                st.info(
                    "No major new AI developments since the morning briefing. "
                    "The morning baseline stories are shown below unchanged.",
                    icon=":material/info:",
                )

            render_feed(latest.get("stories", []), "This update is empty.")

    with history_tab:
        history = load_json_file(HISTORY_FILE, [])
        if not isinstance(history, list) or not history:
            st.info("No briefing runs recorded yet.")
        else:
            for entry in reversed(history[-15:]):
                kind = str(entry.get("type", "unknown")).title()
                when = entry.get("generated_at", "")
                if entry.get("type") == "evening":
                    flag = (
                        "new developments"
                        if entry.get("has_new_developments")
                        else "no major new developments"
                    )
                    st.caption(
                        f"{kind} - {when} - {entry.get('story_count', 0)} "
                        f"stories - {flag}"
                    )
                else:
                    st.caption(
                        f"{kind} - {when} - "
                        f"{len(entry.get('story_ids', []))} stories"
                    )


# ============================================================
# VIEWS: CATEGORY SUB-FEEDS
# ============================================================
# Each view asks for a canonical category name. get_latest_news() also matches
# every historical alias, so rows stored as "Developers & Tools" still show up
# under "AI Tools".

elif selected_view == VIEW_RESEARCH:
    st.markdown(
        '<div class="section-header">AI research, benchmarks & papers</div>',
        unsafe_allow_html=True,
    )
    render_feed(
        get_latest_news(limit=30, category="AI Research"),
        "No research stories in the current window.",
    )

elif selected_view == VIEW_TOOLS:
    st.markdown(
        '<div class="section-header">AI developer tools, SDKs & platforms</div>',
        unsafe_allow_html=True,
    )
    render_feed(
        get_latest_news(limit=30, category="AI Tools"),
        "No developer tooling stories in the current window.",
    )

elif selected_view == VIEW_MODELS:
    st.markdown(
        '<div class="section-header">Foundation models & autonomous agents</div>',
        unsafe_allow_html=True,
    )
    models = get_latest_news(limit=25, category="AI Models")
    agents = get_latest_news(limit=25, category="AI Agents")
    seen_ids = {story["id"] for story in models}
    combined = models + [
        story for story in agents if story["id"] not in seen_ids
    ]
    combined.sort(key=published_key, reverse=True)
    render_feed(
        combined,
        "No model or agent stories in the current window.",
    )

elif selected_view == VIEW_CAREERS:
    st.markdown(
        '<div class="section-header">AI careers, talent & hiring</div>',
        unsafe_allow_html=True,
    )
    render_feed(
        get_latest_news(limit=30, category="AI Jobs & Careers"),
        "No career or hiring stories in the current window.",
    )

elif selected_view == VIEW_INDIA:
    st.markdown(
        '<div class="section-header">India AI ecosystem & initiatives</div>',
        unsafe_allow_html=True,
    )
    render_feed(
        get_latest_news(limit=30, category="India AI"),
        "No India AI stories in the current window.",
    )

elif selected_view == VIEW_ROBOTICS:
    st.markdown(
        '<div class="section-header">AI-powered robotics & embodied systems</div>',
        unsafe_allow_html=True,
    )
    render_feed(
        get_latest_news(limit=30, category="Robotics"),
        "No AI robotics stories in the current window.",
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    f"AI Radar - local AI intelligence - Ollama {OLLAMA_MODEL} - "
    f"{EMBEDDING_MODEL} - FAISS - SQLite - Streamlit"
)
