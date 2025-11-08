from __future__ import annotations

import html
from contextlib import contextmanager

import streamlit as st

THEME_BG = '#0b1221'
THEME_SURFACE = '#111a2e'
THEME_BORDER = 'rgba(93, 220, 255, 0.12)'
THEME_ACCENT = '#5ddcff'
THEME_ACCENT_ALT = '#a855f7'
THEME_TEXT = '#e5ecff'
THEME_MUTED = '#8f9acb'

PLOT_CONFIG = {
    'displayModeBar': False,
    'responsive': True,
}

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {{
    --sb-bg: {THEME_BG};
    --sb-surface: {THEME_SURFACE};
    --sb-border: {THEME_BORDER};
    --sb-accent: {THEME_ACCENT};
    --sb-accent-alt: {THEME_ACCENT_ALT};
    --sb-text: {THEME_TEXT};
    --sb-muted: {THEME_MUTED};
}}

html, body, [class*="css"]  {{
    background: var(--sb-bg) !important;
    color: var(--sb-text) !important;
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}}

section[data-testid="stSidebar"] {{
    background: rgba(11, 18, 33, 0.85);
    border-right: 1px solid var(--sb-border);
    backdrop-filter: blur(12px);
}}

.block-container {{
    padding: 2.5rem 1.8rem 3rem;
    max-width: 1200px;
}}

.hero-card {{
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1.5rem;
    padding: 2rem;
    background: linear-gradient(135deg, rgba(93,220,255,0.12), rgba(168,85,247,0.08));
    border: 1px solid var(--sb-border);
    border-radius: 24px;
    box-shadow: 0 25px 45px -30px rgba(0,0,0,0.6);
}}

.hero-meta {{
    flex: 1 1 260px;
    min-width: 220px;
}}

.hero-card h1 {{
    font-size: clamp(2.2rem, 4vw, 2.8rem);
    margin: 0;
    letter-spacing: -0.04em;
}}

.hero-card p {{
    color: var(--sb-muted);
    margin: 0.4rem 0 0;
    font-size: 0.95rem;
}}

.hero-badges {{
    flex: 2 1 420px;
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
}}

.hero-badge {{
    flex: 1 1 180px;
    min-width: 160px;
    padding: 0.85rem 1.1rem;
    background: rgba(17, 26, 46, 0.88);
    border: 1px solid var(--sb-border);
    border-radius: 14px;
    transition: border-color 0.25s ease, transform 0.25s ease;
}}

.hero-badge:hover {{
    transform: translateY(-3px);
    border-color: rgba(93, 220, 255, 0.35);
}}

.hero-badge span.label {{
    display: block;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.65rem;
    color: var(--sb-muted);
}}

.hero-badge span.value {{
    display: block;
    margin-top: 0.3rem;
    font-size: 1.1rem;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}}

.hero-badge span.value.positive {{ color: #4ade80; }}
.hero-badge span.value.negative {{ color: #f87171; }}
.hero-badge span.value.neutral {{ color: var(--sb-text); }}

.metric-label {{
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.7rem;
    color: var(--sb-muted);
}}

.metric-value {{
    display: block;
    margin-top: 0.4rem;
    font-size: 2.1rem;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}}

.tir-breakdown {{
    display: inline-flex;
    align-items: baseline;
    gap: 0.75rem;
    margin-left: 0.75rem;
    font-size: 1.05rem;
}}

.tir-chip {{
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    cursor: help;
}}

.metric-caption {{
    margin-top: 0.6rem;
    font-size: 0.8rem;
    color: var(--sb-muted);
}}
.metric-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 2rem 0 1rem;
}}

.metric-card {{
    background: rgba(17, 26, 46, 0.9);
    border: 1px solid var(--sb-border);
    border-radius: 18px;
    padding: 1.25rem 1.4rem;
    transition: transform 0.25s ease, border-color 0.25s ease;
    position: relative;
}}

.metric-card::after {{
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 18px;
    padding: 1px;
    background: linear-gradient(135deg, rgba(93,220,255,0.35), transparent);
    -webkit-mask:
        linear-gradient(#fff 0 0) content-box,
        linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
            mask-composite: exclude;
    pointer-events: none;
}}

.metric-card:hover {{
    transform: translateY(-4px);
    border-color: rgba(93, 220, 255, 0.35);
}}

.metric-card--form {{
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
}}

.metric-card--form .metric-label {{
    font-size: 0.68rem;
    margin-bottom: 0.1rem;
}}

.metric-card--form .stSelectbox, .metric-card--form .stDateInput {{
    padding-bottom: 0.2rem;
}}

.control-card {{
    background: rgba(17, 26, 46, 0.8);
    border: 1px solid var(--sb-border);
    border-radius: 20px;
    padding: 1.4rem 1.6rem 0.4rem;
    margin: 1.2rem 0 1.5rem;
}}

.panel-card {{
    background: rgba(17, 26, 46, 0.92);
    border: 1px solid var(--sb-border);
    border-radius: 24px;
    padding: 1.4rem;
    box-shadow: 0 20px 45px -40px rgba(0,0,0,0.75);
}}

.panel-title {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 1rem;
}}

.panel-title h3 {{
    margin: 0;
    font-size: 1.2rem;
    letter-spacing: 0.02em;
}}

.panel-title span {{
    font-size: 0.8rem;
    color: var(--sb-muted);
}}

.divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(93,220,255,0.35), transparent);
    margin: 2rem 0;
}}

.stSelectbox, .stDateInput {{
    padding-bottom: 1rem;
}}

.stPlotlyChart {{
    border-radius: 14px;
    overflow: hidden;
}}

.sidebar-card {{
    background: rgba(17, 26, 46, 0.85);
    border: 1px solid var(--sb-border);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
    font-size: 0.85rem;
    line-height: 1.5;
}}

.sidebar-card strong {{
    display: block;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.65rem;
    color: var(--sb-muted);
    margin-bottom: 0.3rem;
}}

@media (max-width: 768px) {{
    .block-container {{
        padding: 1.5rem 1.1rem 2.5rem;
    }}

    .hero-card {{
        padding: 1.5rem;
    }}

    .hero-badges {{
        flex-direction: column;
    }}

    .control-card {{
        padding: 1rem 1.2rem 0.2rem;
    }}
}}
</style>
"""

def inject_base_theme() -> None:
    """Render the global CSS theme into the Streamlit app."""
    st.markdown(_CSS, unsafe_allow_html=True)


@contextmanager
def panel_card(title: str, subtitle: str | None = None):
    """Render a stylized container that matches the dashboard theme."""
    safe_title = html.escape(title)
    subtitle_html = f'<span>{html.escape(subtitle)}</span>' if subtitle else ''
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-title">
                <h3>{safe_title}</h3>
                {subtitle_html}
            </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)
