"""End-to-end RAG orchestration: classify -> retrieve -> assemble -> LLM -> format.

This is the online-phase entry point (`answer`) that the UI (Phase 7) calls. It
wires together the compliance and cost controls built across Phases 4–6:

    query
      │
      ├─ empty?                → prompt to enter a question        (no LLM)
      ├─ advisory? (Phase 5)   → tailored refusal                 (no LLM)
      ├─ out-of-scope AMC?      → scope-limitation message         (no LLM)
      ├─ response cache hit?    → cached formatted answer          (no LLM)
      ├─ retrieve (Phase 4)     → nothing / no allowlisted source? → "not in corpus"
      └─ Groq LLM (Phase 6)     → advisory-guard / not-in-corpus checks → formatted answer

Only the final branch spends Groq quota, so refusals, scope messages, empty
inputs, out-of-corpus replies, and cache hits all cost **zero** requests/tokens
— which is what keeps the free-tier RPM/RPD/TPM/TPD budget available for genuine
factual questions (see ImplementationPlan.md §Phase 6).
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import config
from rag import formatter, prompts
from rag.classifier import classify_intent, normalize
from rag.llm import CapacityError, LLMError, estimate_tokens, get_llm
from vectorstore.retriever import (
    assemble_context,
    detect_out_of_scope,
    retrieve,
)

# --------------------------------------------------------------------------- #
# Response cache — normalized query -> formatted answer (edge: protects quota).
# --------------------------------------------------------------------------- #
_CACHE: "OrderedDict[str, str]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


def _cache_get(key: str) -> str | None:
    if not config.RESPONSE_CACHE_ENABLED:
        return None
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    return None


def _cache_put(key: str, value: str) -> None:
    if not config.RESPONSE_CACHE_ENABLED:
        return
    with _CACHE_LOCK:
        _CACHE[key] = value
        _CACHE.move_to_end(key)
        while len(_CACHE) > config.RESPONSE_CACHE_SIZE:
            _CACHE.popitem(last=False)


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def _budget_chunks(chunks: list[dict]) -> list[dict]:
    """Trim retrieved chunks to the per-call context budget (Phase 6).

    Caps the material sent to the LLM at ``config.MAX_CONTEXT_CHUNKS`` and
    ``config.MAX_CONTEXT_TOKENS`` so ``prompt + max_tokens`` stays well under the
    free-tier per-minute (TPM) budget. At least one chunk is always kept so a
    single over-long chunk still yields an answer.
    """
    kept: list[dict] = []
    used = 0
    for c in chunks[: config.MAX_CONTEXT_CHUNKS]:
        cost = estimate_tokens(c.get("text", ""))
        if kept and used + cost > config.MAX_CONTEXT_TOKENS:
            break
        kept.append(c)
        used += cost
    return kept


def answer(query: str) -> str:
    """Run the full RAG pipeline and return a compliant answer or refusal."""
    q = (query or "").strip()
    if not q:  # edge 5.6 / 8.1
        return prompts.EMPTY_QUERY_TEMPLATE

    # 1) Intent classification — refuse advisory queries before any cost (Phase 5).
    intent = classify_intent(q)
    if intent.is_advisory:
        return prompts.refusal_message(intent.category)

    # 2) Out-of-scope AMC/scheme — scope message, no retrieval/LLM (edge 5.5).
    if detect_out_of_scope(q):
        return prompts.SCOPE_TEMPLATE

    # 3) Response cache — serve repeats without spending quota.
    cache_key = normalize(q)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # 4) Retrieval (Phase 4). Empty result => not in corpus (no LLM, edge 5.1).
    chunks = retrieve(q)
    if not chunks:
        result = formatter.not_in_corpus()
        _cache_put(cache_key, result)
        return result

    # Bound the context to the per-call token budget before assembling it so a
    # single call can't blow the free-tier TPM (see ImplementationPlan §Phase 6).
    chunks = _budget_chunks(chunks)
    ctx = assemble_context(chunks)
    source = formatter.pick_source(ctx["sources"])
    if source is None:  # nothing citable from an allowlisted domain (edge 9.4)
        result = formatter.not_in_corpus()
        _cache_put(cache_key, result)
        return result

    # 5) Grounded generation via Groq (rate-limited + backoff + degrade).
    user_prompt = prompts.build_user_prompt(ctx["context"], q)
    try:
        raw = get_llm().generate(prompts.SYSTEM_PROMPT, user_prompt)
    except CapacityError:  # budget exhausted — degrade gracefully (edge 6.2)
        return prompts.BUSY_TEMPLATE
    except LLMError:  # missing key / dead backend — never fabricate (edge 6.1/6.3)
        return prompts.SERVICE_ERROR_TEMPLATE

    # 6) Post-generation compliance guards.
    if formatter.signals_not_in_corpus(raw):  # low-relevance / ungrounded (edge 6.11)
        result = formatter.not_in_corpus(source.get("source_url"))
        _cache_put(cache_key, result)
        return result
    if formatter.contains_advisory(raw):  # advice leaked past the prompt (edge 6.7)
        return prompts.refusal_message()

    last_updated = source.get("last_updated") or ctx.get("last_updated")
    result = formatter.format_response(raw, source["source_url"], last_updated)
    _cache_put(cache_key, result)
    return result
