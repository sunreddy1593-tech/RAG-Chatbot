"""Streamlit chat interface for the Mutual Fund FAQ Assistant (Phase 7).

A minimal, compliance-first chat UI over the Phase 6 RAG pipeline:

- Welcome message + scope note (5 in-scope HDFC schemes only).
- 3 clickable example questions.
- Chat input with history; each answer is rendered exactly as the pipeline
  formats it (answer body + single ``Source:`` link + ``Last updated`` footer).
- A **persistent** "Facts-only. No investment advice." disclaimer banner.

Compliance/robustness notes:
- The UI collects **no** personal data — no login, no PII fields (edge 8.4 / 9.3);
  queries are kept only in this session's transient state, never persisted.
- Startup problems (missing ``GROQ_API_KEY`` / absent vector index) are surfaced
  as friendly warnings rather than stack traces (edge 8.6 / 10.1 / 10.2).
- All backend calls are wrapped so an unexpected error shows a safe message and
  is logged server-side, never dumped to the user (edge 8.6).

Run with:  streamlit run ui/app.py
"""

from __future__ import annotations

import logging

import streamlit as st

import config
from config import DISCLAIMER

logger = logging.getLogger(__name__)

EXAMPLE_QUESTIONS = [
    "What is the expense ratio of HDFC Large Cap Fund?",
    "What is the exit load for HDFC Small Cap Fund?",
    "What is the minimum SIP amount for HDFC Mid Cap Fund?",
]

WELCOME = (
    "Hi! I answer **factual** questions about five HDFC Mutual Fund schemes — "
    "**Large Cap, Mid Cap, Small Cap, Gold ETF FoF, and Silver ETF FoF** — using "
    "official sources (HDFC AMC, AMFI, SEBI, CAMS). Ask me about expense ratios, "
    "exit loads, minimum SIP amounts, lock-in, and similar facts.\n\n"
    "I can't give investment advice, comparisons, or return predictions."
)

_SAFE_ERROR = (
    "Sorry — something went wrong while answering that. Please try again in a "
    f"moment.\n\n{DISCLAIMER}"
)


def _startup_warnings() -> list[str]:
    """Non-fatal setup problems to surface to the user (edge 10.1 / 10.2)."""
    warnings: list[str] = []
    if not config.GROQ_API_KEY:
        warnings.append(
            "`GROQ_API_KEY` is not set, so factual answers can't be generated. "
            "Add it to your `.env` (see `.env.example`). Advisory queries are "
            "still refused correctly."
        )
    if not config.INDEX_DIR.exists():
        warnings.append(
            "The vector index was not found. Run ingestion and indexing first "
            "(`python -m vectorstore.indexer`) before asking factual questions."
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
        st.header("About")
        st.write(
            "A Retrieval-Augmented FAQ assistant for HDFC Mutual Fund schemes. "
            "Every answer is grounded in official documents and cites its source."
        )
        st.subheader("In scope")
        st.markdown(
            "- HDFC Large Cap Fund\n- HDFC Mid Cap Fund\n- HDFC Small Cap Fund\n"
            "- HDFC Gold ETF FoF\n- HDFC Silver ETF FoF"
        )
        st.info(DISCLAIMER)
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
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Looking it up…"):
            response = _safe_answer(query)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})


def main() -> None:
    st.set_page_config(page_title="Mutual Fund FAQ Assistant", page_icon="💬")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    _render_sidebar()

    st.title("Mutual Fund FAQ Assistant")
    st.caption("Facts-only answers about HDFC Mutual Fund schemes, from official sources.")

    # Persistent disclaimer banner — rendered on every run so it's always visible.
    st.info(DISCLAIMER)

    for warning in _startup_warnings():
        st.warning(warning, icon="⚠️")

    # Welcome + example questions only before the first message, to keep the
    # chat clean afterwards.
    example_clicked: str | None = None
    if not st.session_state.messages:
        st.markdown(WELCOME)
        st.markdown("**Try an example:**")
        cols = st.columns(len(EXAMPLE_QUESTIONS))
        for col, question in zip(cols, EXAMPLE_QUESTIONS):
            if col.button(question, use_container_width=True):
                example_clicked = question

    # Replay the transcript.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed = st.chat_input("Ask a factual question about an HDFC mutual fund scheme…")
    query = example_clicked or typed
    if query and query.strip():
        _handle_query(query.strip())
        # Rerun so the welcome/examples block disappears after the first turn.
        if example_clicked:
            st.rerun()


if __name__ == "__main__":
    main()
