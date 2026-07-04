"""System prompts and refusal templates.

The strict facts-only system prompt and refusal/fallback templates are defined
here so prompt wording lives in one auditable place. Wired into the pipeline in
Phases 5-6.
"""

from __future__ import annotations

from config import DISCLAIMER, EDUCATIONAL_LINK, MAX_SENTENCES

SYSTEM_PROMPT = f"""You are a facts-only Mutual Fund FAQ assistant for HDFC Mutual Fund schemes.

Rules:
1. Answer ONLY using the provided context. Do not use prior knowledge.
2. Limit your response to a maximum of {MAX_SENTENCES} sentences.
3. Do not provide investment advice, recommendations, or performance predictions.
4. End every response with exactly one source citation link from the metadata.
5. Append a footer: "Last updated from sources: <date from metadata>"
6. If the context does not contain the answer, say:
   "This information is not available in the current corpus. Please refer to <source link>."
"""

REFUSAL_TEMPLATE = (
    "I'm only able to provide factual information about mutual fund schemes — "
    "such as expense ratios, exit loads, or SIP minimums.\n\n"
    f"For guidance on investment decisions, please refer to a SEBI-registered "
    f"financial advisor or visit: {EDUCATIONAL_LINK}\n\n"
    f"{DISCLAIMER}"
)

NOT_IN_CORPUS_TEMPLATE = (
    "This information is not available in the current corpus. "
    "Please refer to {source_link}"
)

# Shown when the Groq free-tier budget is exhausted (per-minute or per-day) so
# the UI degrades gracefully instead of erroring (edge 6.2 / 10.5).
BUSY_TEMPLATE = (
    "I'm at capacity right now and can't process this question this moment. "
    "Please try again shortly.\n\n"
    f"{DISCLAIMER}"
)

# Shown when the LLM backend is unavailable/misconfigured. We never fabricate an
# ungrounded answer in this case (edge 6.1 / 6.3).
SERVICE_ERROR_TEMPLATE = (
    "The answering service is temporarily unavailable, so I can't generate a "
    "grounded answer right now. Please try again later.\n\n"
    f"{DISCLAIMER}"
)

# Shown for queries about AMCs/schemes outside the 5 in-scope HDFC funds
# (edge 5.5). Not advisory — a scope limitation.
SCOPE_TEMPLATE = (
    "I can only answer questions about these HDFC schemes: Large Cap, Mid Cap, "
    "Small Cap, Gold ETF FoF, and Silver ETF FoF. I don't have information about "
    "other fund houses or schemes.\n\n"
    f"{DISCLAIMER}"
)

# Shown for empty/whitespace input (edge 5.6 / 8.1).
EMPTY_QUERY_TEMPLATE = (
    "Please enter a question about an HDFC mutual fund scheme — for example, "
    '"What is the expense ratio of HDFC Large Cap Fund?"'
)


def build_user_prompt(context: str, query: str) -> str:
    """Assemble the user turn: retrieved context block + the question.

    The context carries per-chunk source tags (added by ``assemble_context``) so
    the model can ground its single citation in the provided material only.
    """
    return (
        "Answer the question using ONLY the context below.\n\n"
        f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
        f"Question: {query.strip()}"
    )

# One-line, category-specific lead-in prepended to the refusal. The body of the
# refusal (educational link + disclaimer) is always the same so every refusal is
# guaranteed to carry both (edge 7.5).
_REFUSAL_PREFACE: dict[str, str] = {
    "recommendation": "I can't advise whether to buy, sell, hold, or invest in a scheme.",
    "comparison": "I can't compare schemes or say which one is better.",
    "prediction": "I can't predict or estimate future returns or performance.",
    "performance": "I don't provide return figures or performance analysis — please see the official factsheet.",
    "injection": "I can only answer factual questions about HDFC mutual fund schemes.",
}


def refusal_message(category: str | None = None) -> str:
    """Build a refusal response.

    Always includes the educational link + disclaimer (via ``REFUSAL_TEMPLATE``);
    an optional ``category`` adds a tailored one-line lead-in (edge 4.1–4.4, 7.5).
    """
    preface = _REFUSAL_PREFACE.get(category or "")
    return f"{preface}\n\n{REFUSAL_TEMPLATE}" if preface else REFUSAL_TEMPLATE


# LLM fallback classifier prompt (edge 4.5 / 4.6) — only consulted for genuinely
# ambiguous queries. Kept tiny to stay well inside the Groq token budget.
CLASSIFIER_SYSTEM_PROMPT = (
    "You are an intent classifier for a facts-only HDFC mutual fund FAQ assistant.\n"
    "Classify the user's query into exactly one label:\n"
    "- ADVISORY: seeks investment advice, recommendations, suitability, fund "
    "comparisons, whether to buy/sell/hold, or predictions/opinions about future "
    "returns or performance.\n"
    "- FACTUAL: seeks an objective, documented fact (expense ratio, exit load, "
    "minimum SIP/investment, NAV, lock-in, fund manager, benchmark, category, etc.).\n"
    "Reply with ONLY one word: ADVISORY or FACTUAL."
)
