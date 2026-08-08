"""
Custom styling for the Memory-Augmented Chatbot Streamlit UI.
.3
Usage (in ui/app.py):

    from ui.style import inject_custom_css, ROUTE_LABELS, route_badge_html, memory_card_html, stat_pills_html

    st.set_page_config(page_title="Memory-Augmented Chatbot", page_icon="🧠", layout="wide")
    inject_custom_css()

Pair this with .streamlit/config.toml (base theme colors) -- both use the
same palette so the native Streamlit chrome (buttons, inputs, sidebar) and
the custom HTML/CSS elements here look like one coherent design, not two
different apps stitched together.

Palette:
    Background   #0B1220  deep slate navy      -- calm, low-glare, easy on the eyes for long sessions
    Surface      #131C2E  panel/card background
    Border       #22304A  subtle dividers
    Text         #E6EAF2  primary text (high contrast on navy)
    Muted text   #93A0B8  secondary/caption text
    Accent       #4FD1C5  teal -- primary brand accent (buttons, links, title)
    RAG route    #4FD1C5  teal   -- "knowledge base" lookups
    KG route     #A78BFA  violet -- "knowledge graph" lookups
    Tool route   #F5A962  amber  -- "live tool" calls
    Success      #34D399  green
"""

from __future__ import annotations

import streamlit as st

ROUTE_LABELS = {"rag": "📚 Knowledge Base", "kg": "🕸️ Knowledge Graph", "tool": "🛠️ Tool"}

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0B1220;
        --surface: #131C2E;
        --border: #22304A;
        --text: #E6EAF2;
        --muted: #93A0B8;
        --accent: #4FD1C5;
        --accent-soft: rgba(79, 209, 197, 0.14);
        --kg: #A78BFA;
        --kg-soft: rgba(167, 139, 250, 0.16);
        --tool: #F5A962;
        --tool-soft: rgba(245, 169, 98, 0.16);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text);
    }

    /* App title */
    h1 {
        font-weight: 700 !important;
        background: linear-gradient(90deg, var(--accent) 0%, var(--kg) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.01em;
    }
    [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.6rem;
        background: var(--surface);
        border: 1px solid var(--border);
    }
    [data-testid="stChatMessage"]:has(img[alt="user"]) {
        background: var(--accent-soft);
        border-color: rgba(79, 209, 197, 0.35);
    }
    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: var(--kg) !important;
    }
    [data-testid="stChatMessageAvatarUser"] {
        background-color: var(--accent) !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: var(--text);
        font-weight: 600;
    }

    .memory-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: 8px;
        padding: 0.55rem 0.85rem;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
        color: var(--text);
    }

    .stat-pill {
        display: inline-block;
        background: var(--accent-soft);
        color: var(--accent);
        border: 1px solid rgba(79, 209, 197, 0.3);
        border-radius: 999px;
        padding: 0.22rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }

    /* Route badges -- distinct color per retrieval path so it's scannable
       at a glance which layer of the system answered */
    .route-badge {
        display: inline-block;
        border-radius: 6px;
        padding: 0.18rem 0.6rem;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        border: 1px solid transparent;
    }
    .route-rag  { background: var(--accent-soft); color: var(--accent); border-color: rgba(79, 209, 197, 0.35); }
    .route-kg   { background: var(--kg-soft);     color: var(--kg);     border-color: rgba(167, 139, 250, 0.35); }
    .route-tool { background: var(--tool-soft);   color: var(--tool);   border-color: rgba(245, 169, 98, 0.35); }

    /* Buttons */
    div.stButton > button {
        border-radius: 8px;
        border: 1px solid var(--border);
        background: transparent;
        color: var(--text);
        font-weight: 500;
        transition: all 0.15s ease;
    }
    div.stButton > button:hover {
        border-color: var(--accent);
        background: var(--accent-soft);
        color: var(--accent);
    }

    /* Text input / chat input */
    [data-testid="stChatInput"], input, textarea {
        border-radius: 12px !important;
    }
    [data-testid="stChatInput"] {
        border: 1px solid var(--border) !important;
    }

    /* Expander (debug panel) */
    [data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        background: var(--surface);
    }

    /* Code blocks inside debug panel */
    code, pre {
        background: rgba(255, 255, 255, 0.04) !important;
        border-radius: 6px;
    }

    /* Dividers */
    hr {
        border-color: var(--border) !important;
    }
</style>
"""


def inject_custom_css() -> None:
    """Call once, right after st.set_page_config(), to apply the theme."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def route_badge_html(route: str) -> str:
    """Return the HTML snippet for a colored route badge (rag/kg/tool)."""
    label = ROUTE_LABELS.get(route, route)
    return f'<span class="route-badge route-{route}">{label}</span>'


def memory_card_html(fact: str) -> str:
    """Return the HTML snippet for a single sidebar memory-fact card."""
    return f'<div class="memory-card">{fact}</div>'


def stat_pills_html(*pills: str) -> str:
    """Return the HTML snippet for a row of stat pills, e.g. '12 chunks'."""
    return "".join(f'<span class="stat-pill">{p}</span>' for p in pills)