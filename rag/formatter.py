"""Response formatter: enforces the compliant output structure (Phase 6).

Every factual answer is normalised to the same shape (edge 6.5 / 6.6 / 7.1 / 7.4):

    <answer — at most MAX_SENTENCES sentences>

    Source: <one allowlisted URL from retrieved metadata>
    Last updated from sources: <date>

This is the last of the three compliance layers (classifier → system prompt →
formatter guard). It hard-enforces what the prompt merely *requests*: it trims
to ≤3 sentences, injects exactly one citation from retrieved metadata (never a
URL the model invented), guarantees a non-empty date footer, and flags advisory
phrasing so the pipeline can refuse instead of leaking advice (edge 6.7 / 9.1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

import config
from rag.prompts import NOT_IN_CORPUS_TEMPLATE

# Advisory phrasing that must never reach the user even if the model emits it
# despite the system prompt (edge 6.7). Kept specific to avoid mangling facts.
_ADVISORY_GUARD = re.compile(
    r"\b(you should|i (would )?recommend|we recommend|i suggest|i'd suggest|"
    r"my advice|i advise|is a good (buy|investment|option|choice)|"
    r"better than|best (choice|option|fund) (for|to)|"
    r"worth (buying|investing)|must (buy|invest)|"
    r"i think you|advisable to (buy|invest))\b",
    re.IGNORECASE,
)

# Phrases the model uses when the context lacks the answer (edge 6.11 / 5.1).
_NOT_IN_CORPUS_HINT = re.compile(
    r"\b(not (available|present|found|mentioned|specified) in the "
    r"(current )?(corpus|context|provided|documents?)|"
    r"context (does not|doesn't) (contain|include|mention)|"
    r"i (don't|do not) have (enough )?(information|context|details))\b",
    re.IGNORECASE,
)

# Sentence boundary: end punctuation followed by whitespace and a capital/quote.
# The uppercase look-ahead avoids splitting decimals ("1.25%") and "Rs. 500".
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'A-Z(])")


@dataclass(frozen=True)
class FormattedResponse:
    """Result of formatting an LLM answer."""

    text: str
    ok: bool = True
    advisory_flagged: bool = False


def is_allowed_url(url: str | None) -> bool:
    """True if ``url``'s host is on the corpus domain allowlist (edge 9.4)."""
    if not url:
        return False
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return any(host == d or host.endswith("." + d) for d in config.ALLOWED_DOMAINS)


def pick_source(sources: list[dict]) -> dict | None:
    """Choose the single most relevant allowlisted source (edge 7.1)."""
    for s in sources or []:
        if is_allowed_url(s.get("source_url")):
            return s
    return None


def contains_advisory(text: str) -> bool:
    """True if the text carries advisory phrasing that must be blocked."""
    return bool(_ADVISORY_GUARD.search(text or ""))


def signals_not_in_corpus(text: str) -> bool:
    """True if the model indicated the answer isn't grounded in the context."""
    return bool(_NOT_IN_CORPUS_HINT.search(text or ""))


def enforce_sentences(text: str, max_sentences: int = config.MAX_SENTENCES) -> str:
    """Trim to at most ``max_sentences`` sentences (edge 6.5)."""
    clean = " ".join((text or "").split())
    if not clean:
        return ""
    parts = _SENTENCE_SPLIT.split(clean)
    return " ".join(parts[:max_sentences]).strip()


def clean_answer(text: str) -> str:
    """Strip any Source:/footer/quotes the model appended; keep the answer body.

    The formatter re-adds a single canonical citation + footer, so we drop the
    model's own attempt (which may invent or duplicate URLs, edge 6.6 / 7.1).
    """
    body = text or ""
    # Cut everything from the first "Source:" / footer line the model emitted.
    # The "Last updated" branch tolerates the model echoing our own footer
    # phrasing ("Last updated from sources:") so no dangling fragment survives.
    body = re.split(
        r"\n?\s*(?:Source|Sources|Reference|"
        r"Last\s+updated(?:\s+from\s+sources?)?)\s*:",
        body, maxsplit=1, flags=re.IGNORECASE)[0]
    # Strip any bare URL the model left inline in the prose: the formatter injects
    # exactly one canonical, allowlisted citation, so a model-emitted link (which
    # may be non-allowlisted or invented) must never survive in the body (edge 6.6 / 9.4).
    body = re.sub(r"https?://\S+", "", body)
    return body.strip().strip('"').strip()


def _footer_date(last_updated: str | None) -> str:
    """Never render an empty date footer (edge 7.2)."""
    date = (last_updated or "").strip()
    return date if date else "date not specified"


def format_response(answer: str, source_url: str, last_updated: str) -> str:
    """Assemble the final compliant response string.

    Enforces ≤3 sentences, injects the given (already-validated) source URL, and
    appends the freshness footer. Advisory detection is handled by the pipeline
    via :func:`contains_advisory` before this is called.
    """
    body = enforce_sentences(clean_answer(answer))
    return (
        f"{body}\n\n"
        f"Source: {source_url}\n"
        f"Last updated from sources: {_footer_date(last_updated)}"
    )


def not_in_corpus(source_url: str | None = None) -> str:
    """The grounded "no answer in corpus" fallback (edge 5.1 / 6.11)."""
    link = source_url if is_allowed_url(source_url) else config.EDUCATIONAL_LINK
    return NOT_IN_CORPUS_TEMPLATE.format(source_link=link)
