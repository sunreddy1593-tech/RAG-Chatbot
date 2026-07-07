"""Streamlit chat interface for the Mutual Fund FAQ Assistant (Phase 7).

A compliance-first chat UI over the Phase 6 RAG pipeline, styled to the
**Stitch "Stewardship" design system** (see
``stitch_mutual_fund_faq_assistant/stewardship_interface/DESIGN.md``).

The five Stitch screens map 1:1 onto branches the pipeline already emits, so the
UI only *styles* each branch — it adds no new backend behaviour:

    1. welcome_state          -> first load, empty transcript
    2. factual_answer         -> grounded answer (green Source chip + footer)
    3. refusal_state          -> advisory refusal (amber left-border card)
    4. out_of_scope_state     -> out-of-scope / not-in-corpus (gray italic)
    5. configuration_warning  -> missing GROQ_API_KEY / vector index

Compliance/robustness notes:
- The UI collects **no** personal data — no login, no PII fields (edge 8.4 / 9.3);
  queries are kept only in this session's transient state, never persisted.
- Startup problems (missing ``GROQ_API_KEY`` / absent vector index) are surfaced
  as the Stitch configuration-warning banner rather than stack traces
  (edge 8.6 / 10.1 / 10.2).
- All dynamic text is HTML-escaped before rendering, so echoed input can't inject
  markup even though we render custom HTML (edge 8.7).
- All backend calls are wrapped so an unexpected error shows a safe message and
  is logged server-side, never dumped to the user (edge 8.6).

Run with:  streamlit run ui/app.py
"""

from __future__ import annotations

import html
import logging
import os
import sys
from pathlib import Path

import streamlit as st

# When launched via `streamlit run ui/app.py` (e.g. on Streamlit Community
# Cloud), Streamlit puts the script's own directory (ui/) on sys.path rather
# than the repo root, so first-party imports (`config`, `rag`, `scheduler`,
# `vectorstore`) fail with ModuleNotFoundError. Add the repo root explicitly.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Bridge Streamlit secrets -> environment BEFORE importing config: config reads
# keys like GROQ_API_KEY via os.getenv at import time, and Streamlit doesn't
# reliably expose secrets as env vars before that. Secrets set in the app's
# Secrets box (TOML, e.g. `GROQ_API_KEY = "gsk_..."`) are copied here.
try:
    for _key, _val in st.secrets.items():
        if isinstance(_val, (str, int, float, bool)) and _key not in os.environ:
            os.environ[_key] = str(_val)
except Exception:  # no secrets configured (e.g. local dev without secrets.toml)
    pass

import config  # noqa: E402  (import after sys.path + secrets bootstrap above)
from config import DISCLAIMER, EDUCATIONAL_LINK  # noqa: E402
from rag import prompts  # noqa: E402  (lightweight: only imports from config)

logger = logging.getLogger(__name__)

EXAMPLE_QUESTIONS = [
    "What is the expense ratio of HDFC Large Cap Fund?",
    "What is the exit load for HDFC Small Cap Fund?",
    "What is the minimum SIP amount for HDFC Mid Cap Fund?",
]

IN_SCOPE_SCHEMES = [
    "HDFC Large Cap Fund",
    "HDFC Mid Cap Fund",
    "HDFC Small Cap Fund",
    "HDFC Gold ETF FoF",
    "HDFC Silver ETF FoF",
]

WELCOME_HTML = (
    "I answer <strong>factual</strong> questions about five HDFC Mutual Fund "
    "schemes — <span class=\"mf-accent\">Large Cap, Mid Cap, Small Cap, Gold ETF "
    "FoF, and Silver ETF FoF</span> — using official sources (HDFC AMC, AMFI, "
    "SEBI, CAMS). Ask me about expense ratios, exit loads, minimum SIP amounts, "
    "lock-in, and similar facts."
)

_SAFE_ERROR = (
    "Sorry — something went wrong while answering that. Please try again in a "
    f"moment.\n\n{DISCLAIMER}"
)

# --------------------------------------------------------------------------- #
# Stitch design tokens (from DESIGN.md) used by the scoped CSS below.
# --------------------------------------------------------------------------- #
NAVY = "#1E2A54"         # primary / authority — user bubble
BLUE = "#2F6BFF"         # trustworthy blue — primary action / focus
CANVAS = "#F5F7FA"       # soft neutral gray — app canvas
WHITE = "#FFFFFF"        # card / surface
GREEN = "#1B8A5A"        # verified green — source chips
GREEN_TINT = "#E6F4EA"
AMBER = "#B7791F"        # warning amber — disclaimers / warnings
AMBER_TINT = "#FEF3C7"
BORDER = "#E2E8F0"
MUTED = "#64748B"
SCOPE_BG = "#F1F3F5"


def _inject_theme() -> None:
    """Inject the Stitch design system as a scoped CSS block (once per run)."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stSidebar"], .stChatInput textarea, .stButton button {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }}
        [data-testid="stAppViewContainer"] {{ background: {CANVAS}; }}
        [data-testid="stHeader"] {{ background: transparent; }}

        /* App title / caption -------------------------------------------------- */
        .mf-title {{
            font-size: 20px; font-weight: 700; color: {NAVY};
            letter-spacing: -0.01em; margin-bottom: 2px;
        }}
        .mf-subtitle {{ font-size: 13px; color: {MUTED}; margin-bottom: 8px; }}

        /* Persistent disclaimer banner (amber) --------------------------------- */
        .mf-disclaimer {{
            display: flex; align-items: center; justify-content: center; gap: 8px;
            background: {AMBER_TINT}; color: {AMBER}; font-size: 13px;
            font-weight: 600; padding: 8px 16px; border-radius: 8px;
            margin: 4px 0 16px 0; text-align: center;
        }}

        /* Configuration-warning banner (screen 5) ------------------------------ */
        .mf-config-warning {{
            display: flex; align-items: flex-start; gap: 10px;
            background: {AMBER_TINT}; border-left: 4px solid {AMBER};
            color: {AMBER}; font-size: 13.5px; line-height: 1.5;
            padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 0 0 12px 0;
        }}
        .mf-config-warning .mf-ico {{ font-size: 16px; line-height: 1.4; }}

        /* Welcome card (screen 1) ---------------------------------------------- */
        .mf-welcome {{
            background: {WHITE}; border: 1px solid {BORDER}; border-radius: 12px;
            padding: 24px 28px; box-shadow: 0 4px 12px rgba(30,42,84,0.05);
            margin-bottom: 8px;
        }}
        .mf-welcome h2 {{
            font-size: 22px; font-weight: 700; color: {NAVY}; margin: 0 0 12px 0;
        }}
        .mf-welcome p {{ font-size: 16px; line-height: 1.6; color: {NAVY}; margin: 0; }}
        .mf-accent {{ color: {BLUE}; font-weight: 600; }}
        .mf-scope-note {{
            font-size: 13px; color: {MUTED}; border-top: 1px solid {BORDER};
            padding-top: 14px; margin-top: 16px;
        }}
        .mf-scope-note strong {{ color: #BA1A1A; }}

        /* Chat rows + bubbles -------------------------------------------------- */
        .mf-row {{ display: flex; margin: 14px 0; }}
        .mf-row-user {{ justify-content: flex-end; }}
        .mf-row-assistant {{ justify-content: flex-start; }}

        .mf-bubble {{
            max-width: 88%; padding: 16px 20px; border-radius: 12px;
            font-size: 15.5px; line-height: 1.6;
        }}
        .mf-user {{
            background: {NAVY}; color: #FFFFFF; border-top-right-radius: 2px;
        }}
        .mf-assistant {{
            background: {WHITE}; border: 1px solid {BORDER}; color: {NAVY};
            border-top-left-radius: 2px;
            box-shadow: 0 4px 12px rgba(30,42,84,0.05);
        }}
        .mf-refusal {{ border-left: 4px solid {AMBER}; }}
        .mf-scope {{
            background: {SCOPE_BG}; border: 1px solid {BORDER};
            color: #45464F; font-style: italic; box-shadow: none;
        }}

        .mf-agent-label {{
            display: flex; align-items: center; gap: 8px; font-size: 12px;
            font-weight: 700; letter-spacing: 0.02em; margin-bottom: 10px;
        }}
        .mf-agent-label.assistant {{ color: {NAVY}; }}
        .mf-agent-label.refusal {{ color: {AMBER}; }}
        .mf-agent-label.scope {{ color: {MUTED}; font-style: normal; }}
        .mf-badge {{ font-size: 14px; }}

        .mf-body {{ margin: 0; }}
        .mf-body p {{ margin: 0 0 8px 0; }}

        /* Answer footer: source chip + last-updated ---------------------------- */
        .mf-meta {{
            display: flex; align-items: center; justify-content: space-between;
            gap: 12px; flex-wrap: wrap;
            border-top: 1px solid rgba(30,42,84,0.08);
            padding-top: 12px; margin-top: 16px;
        }}
        .mf-chip {{
            display: inline-flex; align-items: center; gap: 6px;
            background: {GREEN_TINT}; color: {GREEN}; font-size: 12px;
            font-weight: 600; padding: 3px 12px; border-radius: 9999px;
            text-decoration: none;
        }}
        .mf-chip:hover {{ opacity: 0.85; }}
        .mf-updated {{ font-size: 11px; font-style: italic; color: {MUTED}; }}

        /* Refusal action link -------------------------------------------------- */
        .mf-edu-link {{
            display: inline-flex; align-items: center; gap: 6px;
            background: {BLUE}; color: #FFFFFF; font-size: 13px; font-weight: 600;
            padding: 8px 16px; border-radius: 8px; text-decoration: none;
            margin-top: 14px;
        }}
        .mf-edu-link:hover {{ opacity: 0.9; }}

        /* Suggestion chips + buttons ------------------------------------------- */
        .stButton button {{
            border-radius: 9999px !important; border: 1px solid {BORDER} !important;
            background: {WHITE} !important; color: {NAVY} !important;
            font-size: 13.5px !important; font-weight: 500 !important;
            transition: all 0.15s ease;
        }}
        .stButton button:hover {{
            border-color: {BLUE} !important; color: {BLUE} !important;
        }}

        /* Chat input ----------------------------------------------------------- */
        .stChatInput textarea {{ font-size: 15px !important; }}

        /* Sidebar identity block ----------------------------------------------- */
        .mf-brand {{ display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }}
        .mf-brand-mark {{
            width: 40px; height: 40px; border-radius: 10px; background: {NAVY};
            color: #FFFFFF; display: flex; align-items: center; justify-content: center;
            font-size: 20px;
        }}
        .mf-brand-name {{ font-size: 18px; font-weight: 800; color: {NAVY}; line-height: 1; }}
        .mf-brand-tag {{
            font-size: 11px; letter-spacing: 0.04em; color: {MUTED};
            text-transform: uppercase;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Response -> Stitch screen-state classification.
# The pipeline returns a plain string; we key off the (stable) template leads
# defined in rag/prompts.py so styling stays in sync with the backend.
# --------------------------------------------------------------------------- #
_REFUSAL_LEAD = prompts.REFUSAL_TEMPLATE.split("\n", 1)[0][:60]
_SCOPE_LEAD = prompts.SCOPE_TEMPLATE.split("\n", 1)[0][:60]
_BUSY_LEAD = prompts.BUSY_TEMPLATE.split("\n", 1)[0][:40]
_ERROR_LEAD = prompts.SERVICE_ERROR_TEMPLATE.split("\n", 1)[0][:40]
_EMPTY_LEAD = prompts.EMPTY_QUERY_TEMPLATE[:40]
_NOT_IN_CORPUS_LEAD = "This information is not available in the current corpus"


def _classify_state(response: str) -> str:
    """Map a pipeline response string to one of the Stitch screen states."""
    text = response or ""
    if _REFUSAL_LEAD and _REFUSAL_LEAD in text:
        return "refusal"
    if any(p in text for p in prompts._REFUSAL_PREFACE.values()):
        return "refusal"
    if _SCOPE_LEAD and _SCOPE_LEAD in text:
        return "scope"
    if _NOT_IN_CORPUS_LEAD in text:
        return "scope"  # out-of-corpus shares the out-of-scope treatment
    if _BUSY_LEAD in text or _ERROR_LEAD in text:
        return "notice"
    if text.startswith(_EMPTY_LEAD):
        return "notice"
    if "Source:" in text and "Last updated from sources:" in text:
        return "factual"
    return "assistant"


def _split_factual(response: str) -> tuple[str, str | None, str | None]:
    """Split a formatted factual answer into (body, source_url, last_updated)."""
    body_lines: list[str] = []
    source: str | None = None
    updated: str | None = None
    for line in response.splitlines():
        stripped = line.strip()
        if stripped.startswith("Source:"):
            source = stripped[len("Source:"):].strip()
        elif stripped.startswith("Last updated from sources:"):
            updated = stripped[len("Last updated from sources:"):].strip()
        else:
            body_lines.append(line)
    body = "\n".join(body_lines).strip()
    return body, source, updated


def _domain(url: str | None) -> str:
    if not url:
        return "source"
    from urllib.parse import urlparse

    try:
        return (urlparse(url).hostname or url).replace("www.", "")
    except ValueError:
        return "source"


def _esc(text: str) -> str:
    """HTML-escape + convert newlines to <br> for safe custom rendering."""
    return html.escape(text or "").replace("\n", "<br>")


# --------------------------------------------------------------------------- #
# Styled renderers (one per Stitch screen state).
# --------------------------------------------------------------------------- #
def _render_user(text: str) -> None:
    st.markdown(
        f'<div class="mf-row mf-row-user">'
        f'<div class="mf-bubble mf-user">{_esc(text)}</div></div>',
        unsafe_allow_html=True,
    )


def _render_assistant(response: str) -> None:
    state = _classify_state(response)
    if state == "factual":
        _render_factual(response)
    elif state == "refusal":
        _render_refusal(response)
    elif state == "scope":
        _render_scope(response)
    else:
        _render_notice(response)


def _render_factual(response: str) -> None:
    body, source, updated = _split_factual(response)
    chip = (
        f'<a class="mf-chip" href="{html.escape(source, quote=True)}" '
        f'target="_blank" rel="noopener">&#128279; Source: '
        f'{html.escape(_domain(source))}</a>'
        if source
        else ""
    )
    updated_html = (
        f'<span class="mf-updated">Last updated from sources: '
        f"{html.escape(updated)}</span>"
        if updated
        else ""
    )
    st.markdown(
        f'<div class="mf-row mf-row-assistant">'
        f'<div class="mf-bubble mf-assistant">'
        f'<div class="mf-agent-label assistant">'
        f'<span class="mf-badge">&#129302;</span> Assistant</div>'
        f'<div class="mf-body">{_esc(body)}</div>'
        f'<div class="mf-meta">{chip}{updated_html}</div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )


def _render_refusal(response: str) -> None:
    # Strip the educational link out of the body; it becomes a styled button.
    body = "\n".join(
        line for line in response.splitlines() if EDUCATIONAL_LINK not in line
    ).strip()
    link = (
        f'<a class="mf-edu-link" href="{html.escape(EDUCATIONAL_LINK, quote=True)}" '
        f'target="_blank" rel="noopener">&#8599; AMFI Investor Corner</a>'
    )
    st.markdown(
        f'<div class="mf-row mf-row-assistant">'
        f'<div class="mf-bubble mf-assistant mf-refusal">'
        f'<div class="mf-agent-label refusal">'
        f'<span class="mf-badge">&#9888;</span> Regulatory Notice</div>'
        f'<div class="mf-body">{_esc(body)}</div>'
        f"{link}"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def _render_scope(response: str) -> None:
    st.markdown(
        f'<div class="mf-row mf-row-assistant">'
        f'<div class="mf-bubble mf-assistant mf-scope">'
        f'<div class="mf-agent-label scope">'
        f'<span class="mf-badge">&#8505;</span> System Response</div>'
        f'<div class="mf-body">{_esc(response)}</div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )


def _render_notice(response: str) -> None:
    st.markdown(
        f'<div class="mf-row mf-row-assistant">'
        f'<div class="mf-bubble mf-assistant">'
        f'<div class="mf-agent-label assistant">'
        f'<span class="mf-badge">&#129302;</span> Assistant</div>'
        f'<div class="mf-body">{_esc(response)}</div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Startup / backend helpers.
# --------------------------------------------------------------------------- #
def _startup_warnings() -> list[str]:
    """Non-fatal setup problems -> Stitch config-warning banner (edge 10.1/10.2)."""
    warnings: list[str] = []
    if not config.GROQ_API_KEY:
        warnings.append(
            "<code>GROQ_API_KEY</code> is not set, so factual answers can't be "
            "generated. Add it to your <code>.env</code> (see "
            "<code>.env.example</code>). Advisory questions are still refused "
            "correctly."
        )
    if not config.INDEX_DIR.exists():
        warnings.append(
            "The vector index was not found. Run ingestion and indexing first "
            "(<code>python -m vectorstore.indexer</code>) before asking factual "
            "questions."
        )
    return warnings


def _safe_answer(query: str) -> str:
    """Call the RAG pipeline, converting any failure into a safe message."""
    try:
        from rag.pipeline import answer

        return answer(query)
    except Exception:  # never leak a stack trace to the UI (edge 8.6)
        logger.exception("pipeline.answer failed for query")
        return _SAFE_ERROR


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="mf-brand">'
            '<div class="mf-brand-mark">&#128737;</div>'
            '<div><div class="mf-brand-name">MF Assistant</div>'
            '<div class="mf-brand-tag">Institutional Grade AI</div></div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.subheader("About")
        st.write(
            "A Retrieval-Augmented FAQ assistant for HDFC Mutual Fund schemes. "
            "Every answer is grounded in official documents and cites its source."
        )
        st.subheader("In scope")
        st.markdown("\n".join(f"- {s}" for s in IN_SCOPE_SCHEMES))
        st.markdown(
            f'<div class="mf-disclaimer">&#9888; {html.escape(DISCLAIMER)}</div>',
            unsafe_allow_html=True,
        )
        _render_freshness()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def _render_freshness() -> None:
    """Show when the corpus was last refreshed by the Phase 10 scheduler."""
    try:
        from scheduler.refresh import read_status

        status = read_status()
    except Exception:  # never let a status read break the UI
        status = None
    if not status:
        return
    when = (status.get("finished_at") or "")[:10]
    label = {"ok": "updated", "no-change": "checked (no change)"}.get(
        status.get("status"), status.get("status", "unknown")
    )
    st.caption(f"Corpus last {label}: {when or 'unknown'}")


def _handle_query(query: str) -> None:
    """Append the user turn + the assistant's answer to the transcript."""
    st.session_state.messages.append({"role": "user", "content": query})
    _render_user(query)
    with st.spinner("Looking it up…"):
        response = _safe_answer(query)
    _render_assistant(response)
    st.session_state.messages.append({"role": "assistant", "content": response})


def main() -> None:
    st.set_page_config(page_title="Mutual Fund FAQ Assistant", page_icon="💬")
    _inject_theme()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    _render_sidebar()

    st.markdown(
        '<div class="mf-title">Mutual Fund FAQ Assistant</div>'
        '<div class="mf-subtitle">Facts-only answers about HDFC Mutual Fund '
        "schemes, from official sources.</div>",
        unsafe_allow_html=True,
    )

    # Persistent disclaimer banner — rendered on every run so it's always visible.
    st.markdown(
        f'<div class="mf-disclaimer">&#9888; {html.escape(DISCLAIMER)}</div>',
        unsafe_allow_html=True,
    )

    # Configuration-warning banner (Stitch screen 5).
    for warning in _startup_warnings():
        st.markdown(
            f'<div class="mf-config-warning">'
            f'<span class="mf-ico">&#9888;</span><div>{warning}</div></div>',
            unsafe_allow_html=True,
        )

    # Welcome card + example chips only before the first message (screen 1).
    example_clicked: str | None = None
    if not st.session_state.messages:
        st.markdown(
            f'<div class="mf-welcome">'
            f"<h2>Hi!</h2>"
            f"<p>{WELCOME_HTML}</p>"
            f'<p class="mf-scope-note"><strong>Scope limitation:</strong> '
            f"I can't give investment advice, comparisons, or return "
            f"predictions.</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Try an example:**")
        cols = st.columns(len(EXAMPLE_QUESTIONS))
        for col, question in zip(cols, EXAMPLE_QUESTIONS):
            if col.button(question, use_container_width=True):
                example_clicked = question

    # Replay the transcript with the styled renderers.
    for message in st.session_state.messages:
        if message["role"] == "user":
            _render_user(message["content"])
        else:
            _render_assistant(message["content"])

    typed = st.chat_input("Ask a factual question about an HDFC mutual fund scheme…")
    query = example_clicked or typed
    if query and query.strip():
        _handle_query(query.strip())
        # Rerun so the welcome/examples block disappears after the first turn.
        if example_clicked:
            st.rerun()


if __name__ == "__main__":
    main()
