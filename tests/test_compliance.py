"""Deterministic compliance tests (Phase 8) — offline, no index or LLM required.

These cover the safety-critical invariants that must hold regardless of the
model: advisory refusal, refusal completeness, formatter guarantees (≤3
sentences / one allowlisted citation / non-empty footer / advisory guard),
domain-allowlist audit of the corpus, the PII scanner, and the "no PII field"
UI audit.

Runnable two ways:
    pytest tests/test_compliance.py         # if pytest is installed
    python -m tests.test_compliance         # standalone runner (no deps)
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import config
from rag import formatter
from rag.classifier import ADVISORY, FACTUAL, classify_intent
from rag.prompts import refusal_message
from tests import golden_set as G
from tests import pii_scanner

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Classifier — advisory recall & false-refusal (safety-critical)
# --------------------------------------------------------------------------- #
def test_advisory_queries_are_flagged() -> None:
    """Every advisory / performance / adversarial query is classified ADVISORY."""
    leaked = [
        c.id
        for c in (G.ADVISORY + G.PERFORMANCE + G.ADVERSARIAL)
        if classify_intent(c.query).label != ADVISORY
    ]
    assert not leaked, f"advisory leakage — not flagged: {leaked}"


def test_factual_queries_not_over_refused() -> None:
    """Factual queries must not be misclassified as advisory (≤10% per eval.md)."""
    refused = [c.id for c in G.FACTUAL if classify_intent(c.query).label != FACTUAL]
    rate = len(refused) / len(G.FACTUAL)
    assert rate <= 0.10, f"false-refusal rate {rate:.0%} too high: {refused}"


def test_pii_advisory_still_refused() -> None:
    """PII cases (phrased as advisory) are refused so PII never reaches the LLM."""
    leaked = [c.id for c in G.PII if classify_intent(c.query).label != ADVISORY]
    assert not leaked, f"PII advisory cases not refused: {leaked}"


# --------------------------------------------------------------------------- #
# Refusal completeness — disclaimer + educational link always present
# --------------------------------------------------------------------------- #
def test_refusal_completeness() -> None:
    for category in (None, "recommendation", "comparison", "prediction", "performance", "injection"):
        msg = refusal_message(category)
        assert config.DISCLAIMER in msg, f"refusal missing disclaimer ({category})"
        assert config.EDUCATIONAL_LINK in msg, f"refusal missing edu link ({category})"


def test_refusal_has_no_advisory_language() -> None:
    """The refusal text itself must not trip the advisory-language guard."""
    for category in (None, "recommendation", "comparison", "prediction", "performance", "injection"):
        assert not formatter.contains_advisory(refusal_message(category))


# --------------------------------------------------------------------------- #
# Formatter guarantees
# --------------------------------------------------------------------------- #
def test_sentence_limit_enforced() -> None:
    long_answer = (
        "The expense ratio is 1.25%. The exit load is 1% within one year. "
        "The minimum SIP is Rs 100. This fourth sentence must be dropped. "
        "And this fifth one too."
    )
    out = formatter.format_response(long_answer, "https://hdfcfund.com/x", "2026-06-01")
    body = out.split("\n\nSource:")[0]
    # Decimal-safe splitter keeps "1.25%" intact but caps at 3 sentences.
    assert "fourth sentence" not in body and "fifth one" not in body
    assert body.count(". ") + body.endswith(".") <= 3 + 1


def test_exactly_one_citation_and_footer() -> None:
    out = formatter.format_response("The expense ratio is 1.25%.", "https://hdfcfund.com/x", "2026-06-01")
    assert out.count("\nSource:") == 1, "must have exactly one Source line"
    assert "Last updated from sources: 2026-06-01" in out


def test_footer_never_empty() -> None:
    out = formatter.format_response("A fact.", "https://sebi.gov.in/x", None)
    assert "Last updated from sources:" in out
    assert out.rstrip().splitlines()[-1].strip() != "Last updated from sources:"


def test_model_invented_citation_is_stripped() -> None:
    """A URL the model appends is dropped; only the injected one remains."""
    answer = "The expense ratio is 1.25%. Source: https://evil-blog.com/fake"
    out = formatter.format_response(answer, "https://hdfcfund.com/real", "2026-06-01")
    assert "evil-blog.com" not in out
    assert out.count("Source:") == 1
    assert "hdfcfund.com/real" in out


def test_inline_url_stripped_from_body() -> None:
    """A bare URL the model leaves in the prose is removed; only the injected
    allowlisted citation remains (edge 6.6 / 9.4)."""
    answer = "The expense ratio is 1.25%, see https://groww.in/blog/random for more."
    out = formatter.format_response(answer, "https://hdfcfund.com/real", "2026-06-01")
    body = out.split("\n\nSource:")[0]
    assert "groww.in" not in body and "http" not in body
    assert out.count("Source:") == 1
    assert "hdfcfund.com/real" in out


def test_allowlist_url_check() -> None:
    assert formatter.is_allowed_url("https://files.hdfcfund.com/x.pdf")
    assert formatter.is_allowed_url("https://investor.sebi.gov.in/x.html")
    assert formatter.is_allowed_url("https://www.amfiindia.com/x")
    assert not formatter.is_allowed_url("https://groww-blog.example.com/x")
    assert not formatter.is_allowed_url("https://evilhdfcfund.com/x")  # not a subdomain
    assert not formatter.is_allowed_url(None)


def test_pick_source_prefers_allowlisted() -> None:
    src = formatter.pick_source(
        [{"source_url": "https://aggregator.com/x"}, {"source_url": "https://sebi.gov.in/ok", "last_updated": "2026-01-01"}]
    )
    assert src and src["source_url"].endswith("/ok")


def test_not_in_corpus_link_allowlisted() -> None:
    assert formatter.is_allowed_url(_extract_url(formatter.not_in_corpus(None)))
    assert formatter.is_allowed_url(_extract_url(formatter.not_in_corpus("https://not-allowed.com/x")))


def _extract_url(text: str) -> str:
    for token in text.replace("\n", " ").split():
        if token.startswith("http"):
            return token.rstrip(".")
    return ""


# --------------------------------------------------------------------------- #
# Corpus source allowlist audit (edge 1.5 / 9.4)
# --------------------------------------------------------------------------- #
def _domain_ok(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in config.ALLOWED_DOMAINS)


def test_source_urls_allowlisted() -> None:
    data = json.loads((REPO / "data" / "source_urls.json").read_text(encoding="utf-8"))
    bad = [s["url"] for s in data["sources"] if not _domain_ok(s["url"])]
    assert not bad, f"off-allowlist source URLs: {bad}"


def test_metadata_domains_allowlisted() -> None:
    path = REPO / "data" / "metadata.json"
    if not path.exists():
        return  # corpus not built in this environment
    docs = json.loads(path.read_text(encoding="utf-8"))
    bad = [d.get("url") for d in docs if not _domain_ok(d.get("url", ""))]
    assert not bad, f"off-allowlist corpus documents: {bad}"


# --------------------------------------------------------------------------- #
# PII scanner + no-PII-field UI audit (edge 8.4 / 9.3)
# --------------------------------------------------------------------------- #
def test_pii_scanner_detects_each_category() -> None:
    for label, sample in G.PII_SAMPLES.items():
        assert pii_scanner.contains_pii(sample), f"scanner missed {label}: {sample!r}"


def test_pii_scanner_ignores_clean_text() -> None:
    assert not pii_scanner.contains_pii("What is the expense ratio of HDFC Large Cap Fund?")


def test_ui_collects_no_pii_fields() -> None:
    src = (REPO / "ui" / "app.py").read_text(encoding="utf-8").lower()
    # No password/PII input widgets; only the chat box is used for input.
    assert 'type="password"' not in src
    for banned in ("pan", "aadhaar", "aadhar", "account number", "otp"):
        # These words may appear in comments; ensure they're not input labels.
        assert f'text_input("{banned}' not in src
    assert "st.chat_input" in src, "expected chat_input as the only input surface"


# --------------------------------------------------------------------------- #
# Standalone runner (no pytest dependency)
# --------------------------------------------------------------------------- #
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}: {exc!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
