"""
AI Radar - CSS design system.

Self-contained on purpose: no Google Fonts, no CDN, no external request of any
kind. The type stack resolves to fonts the operating system already ships, so
the dashboard renders identically offline.
"""

import streamlit as st

FONT_STACK = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI Variable Text", "Segoe UI", '
    'Roboto, "Helvetica Neue", Ubuntu, Cantarell, "Noto Sans", sans-serif'
)

MONO_STACK = (
    'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, '
    '"Liberation Mono", monospace'
)

_TYPOGRAPHY_CSS = """
html, body, [class*="css"], .stApp, button, input, textarea, select {
    font-family: __FONT__;
}

code, kbd, pre, .citation {
    font-family: __MONO__;
}
""".replace("__FONT__", FONT_STACK).replace("__MONO__", MONO_STACK)

_LAYOUT_CSS = """
.stApp {
    background-color: #080c16;
    color: #f1f5f9;
}

.block-container {
    max-width: 1360px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

.section-header {
    font-size: 1.45rem;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 1.5rem;
    margin-bottom: 1.2rem;
}

.radar-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #1e293b;
    border-radius: 20px;
    padding: 2rem 2.4rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    position: relative;
    overflow: hidden;
}

.radar-hero::before {
    content: "";
    position: absolute;
    top: 0; right: 0; width: 300px; height: 100%;
    background: radial-gradient(circle at 100% 0%, rgba(99, 102, 241, 0.15), transparent 70%);
    pointer-events: none;
}

.radar-title {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #f8fafc;
    margin-bottom: 0.4rem;
}

.radar-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    line-height: 1.5;
    max-width: 820px;
}

.radar-live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34d399;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 9999px;
    margin-top: 0.8rem;
}

.radar-live-dot {
    width: 7px;
    height: 7px;
    background-color: #10b981;
    border-radius: 50%;
    animation: radar-pulse 2s infinite;
}

@keyframes radar-pulse {
    0%   { opacity: 1;   transform: scale(1); }
    50%  { opacity: 0.4; transform: scale(1.2); }
    100% { opacity: 1;   transform: scale(1); }
}
"""

_METRIC_CSS = """
.metric-container {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}

.metric-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    transition: transform 0.2s, border-color 0.2s;
}

.metric-card:hover {
    border-color: #334155;
    transform: translateY(-2px);
}

.metric-label {
    color: #64748b;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}

.metric-value {
    color: #f8fafc;
    font-size: 1.8rem;
    font-weight: 700;
}

@media (max-width: 900px) {
    .metric-container { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
"""

_CARD_CSS = """
.story-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 0.9rem;
    transition: border-color 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
}

.story-card:hover {
    border-color: #3b82f6;
    box-shadow: 0 8px 25px -5px rgba(59, 130, 246, 0.12);
}

.story-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 0.75rem;
}

.story-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.4;
}

.story-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
}

.pill {
    display: inline-flex;
    align-items: center;
    padding: 3px 9px;
    border-radius: 6px;
    font-size: 0.76rem;
    font-weight: 600;
}

.pill-source { background: #1e293b; color: #93c5fd; border: 1px solid #334155; }
.pill-cat    { background: #1e1b4b; color: #c7d2fe; border: 1px solid #312e81; }
.pill-time   { color: #64748b; font-size: 0.8rem; }

.pill-importance {
    background: rgba(245, 158, 11, 0.12);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.pill-match {
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.story-summary {
    color: #cbd5e1;
    font-size: 0.95rem;
    line-height: 1.6;
    margin-bottom: 1rem;
}

.why-box {
    background: rgba(15, 23, 42, 0.8);
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.1rem;
    color: #93c5fd;
    font-size: 0.9rem;
    line-height: 1.5;
}

.why-label {
    font-weight: 700;
    color: #60a5fa;
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.2rem;
}

.top-story-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.15rem 1.35rem 1.05rem;
    margin: 0 0 0.35rem;
}

.top-story-lead {
    border-color: #334155;
    background: linear-gradient(135deg, #111c31 0%, #0f172a 78%);
}

.top-story-kicker {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin-bottom: 0.55rem;
}

.top-story-rank {
    color: #f8fafc;
    font-size: 1.15rem;
    font-weight: 800;
}

.top-story-impact {
    color: #fbbf24;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}

.top-story-title {
    color: #f8fafc;
    font-size: 1.18rem;
    line-height: 1.3;
    margin: 0 0 0.6rem;
}

.top-story-lead .top-story-title { font-size: 1.35rem; }

.top-story-summary {
    color: #cbd5e1;
    font-size: 0.92rem;
    line-height: 1.55;
    margin: 0 0 0.7rem;
}

.top-story-meta {
    color: #93c5fd;
    font-size: 0.78rem;
    font-weight: 600;
    line-height: 1.4;
}

.top-story-why {
    border-left: 2px solid #3b82f6;
    color: #cbd5e1;
    font-size: 0.82rem;
    line-height: 1.45;
    margin-top: 0.85rem;
    padding-left: 0.75rem;
}

.top-story-why span {
    color: #60a5fa;
    display: block;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-bottom: 0.18rem;
}
"""

_SEARCH_CSS = """
.search-answer-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    border: 1px solid #3b82f6;
    border-radius: 16px;
    padding: 1.8rem;
    margin: 1.5rem 0 2rem 0;
    box-shadow: 0 10px 30px -10px rgba(59, 130, 246, 0.2);
}

.search-answer-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #93c5fd;
    margin-bottom: 0.9rem;
}

.search-answer-body {
    color: #f1f5f9;
    font-size: 1rem;
    line-height: 1.7;
}

.search-answer-body p { margin: 0 0 0.8rem 0; }
.search-answer-body ul,
.search-answer-body ol { margin: 0 0 0.8rem 1.2rem; padding-left: 0.6rem; }
.search-answer-body li { margin-bottom: 0.45rem; }
.search-answer-body code {
    background: rgba(148, 163, 184, 0.14);
    padding: 1px 5px;
    border-radius: 4px;
    font-size: 0.88em;
}

.citation {
    color: #60a5fa;
    font-weight: 600;
    font-size: 0.85em;
}

.search-grounding-note {
    color: #64748b;
    font-size: 0.82rem;
    margin-top: 0.6rem;
}

.invalid-query-box {
    background: #1e1b2e;
    border: 1px solid #4338ca;
    border-radius: 16px;
    padding: 1.6rem;
    margin-top: 1.5rem;
    color: #e0e7ff;
}

.empty-result-box {
    border-color: #f59e0b;
    background: rgba(245, 158, 11, 0.06);
}

.guidance-title {
    font-size: 1.12rem;
    font-weight: 700;
    color: #a5b4fc;
    margin-bottom: 0.6rem;
}

.empty-result-box .guidance-title { color: #fbbf24; }

.guidance-subtitle {
    font-weight: 600;
    color: #93c5fd;
    margin: 1rem 0 0.4rem 0;
    font-size: 0.9rem;
}

.guidance-body   { color: #cbd5e1; font-size: 0.94rem; line-height: 1.6; margin: 0 0 0.5rem 0; }
.guidance-body-dim { color: #94a3b8; font-size: 0.86rem; line-height: 1.6; margin: 0; }
.guidance-query  { color: #94a3b8; font-size: 0.9rem; margin: 0 0 0.6rem 0; }
.guidance-list   { color: #94a3b8; font-size: 0.9rem; line-height: 1.8; margin: 0; }

.briefing-banner {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.8rem;
}

.briefing-banner-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #38bdf8;
    margin-bottom: 0.4rem;
}

.briefing-banner-desc { color: #94a3b8; font-size: 0.9rem; line-height: 1.6; }

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.5rem;
}

.sidebar-brand-name {
    font-size: 1.3rem;
    font-weight: 800;
    color: #f8fafc;
    line-height: 1.1;
}

.sidebar-brand-tag { font-size: 0.78rem; color: #818cf8; font-weight: 600; }
"""

_STYLESHEET = "".join(
    [_TYPOGRAPHY_CSS, _LAYOUT_CSS, _METRIC_CSS, _CARD_CSS, _SEARCH_CSS]
)


def inject_custom_styles() -> None:
    """Inject the AI Radar stylesheet once per page render."""
    st.markdown(f"<style>{_STYLESHEET}</style>", unsafe_allow_html=True)



