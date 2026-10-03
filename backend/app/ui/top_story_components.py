"""Focused presentation helpers for the Top Stories view."""

from typing import Any, Dict

import streamlit as st

from backend.app.ui.components import esc, format_time_ago, story_summary
from backend.app.config import normalize_category


def render_top_story_card(story: Dict[str, Any], rank: int) -> None:
    """Render a compact intelligence-style Top Stories card."""
    title = story.get("title") or "Untitled story"
    source = story.get("source") or "Verified source"
    category = normalize_category(story.get("category"))
    summary = story_summary(story)
    why = story.get("why_it_matters") or ""
    impact = story.get("importance") or 0
    time_str = format_time_ago(story.get("published_timestamp"))
    url = story.get("url") or ""
    source_count = story.get("source_count") or 0
    sources = story.get("source_urls") or []

    impact_label = "HIGH IMPACT" if impact >= 8 else "SIGNIFICANT" if impact >= 7 else "WATCH"
    source_line = " | ".join(value for value in [category, source, time_str] if value)
    support_line = f"{source_count} sources covering this story" if source_count > 1 else ""

    st.markdown(
        f"""
        <article class="top-story-card {'top-story-lead' if rank == 1 else ''}">
            <div class="top-story-kicker"><span class="top-story-rank">#{rank}</span>
                <span class="top-story-impact">{esc(impact_label)}</span></div>
            <h3 class="top-story-title">{esc(title)}</h3>
            <p class="top-story-summary">{esc(summary)}</p>
            <div class="top-story-meta">{esc(source_line)}</div>
            {f'<div class="top-story-why"><span>WHY IT MATTERS</span>{esc(why)}</div>' if why else ''}
        </article>
        """,
        unsafe_allow_html=True,
    )

    actions = []
    if url:
        actions.append(("Read original source", url))
    if support_line:
        st.caption(support_line)
    if actions:
        columns = st.columns([2, 8])
        with columns[0]:
            st.link_button(actions[0][0], actions[0][1], width="stretch")
    if len(sources) > 1:
        with st.expander("Supporting sources"):
            for source_item in sources:
                label = source_item.get("source") or "Source"
                link = source_item.get("url") or ""
                if link:
                    st.markdown(f"- [{esc(label)}]({esc(link)})")



