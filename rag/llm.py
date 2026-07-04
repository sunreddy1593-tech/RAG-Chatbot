"""Groq LLM client wrapper with free-tier guards (Phase 6).

Wraps the ``groq`` Python SDK with everything needed to stay inside the free-tier
limits and to degrade gracefully (see ImplementationPlan.md §Phase 6):

- **Pre-flight token gate** — estimate ``prompt + max_tokens`` and consult the
  shared :class:`~rag.ratelimit.RateLimiter` before calling. Per-minute
  exhaustion waits briefly (buckets refill); daily exhaustion fails fast.
- **429 backoff** — exponential backoff + jitter, honouring Groq's
  ``Retry-After`` header, capped at ``config.GROQ_MAX_RETRIES``.
- **Model degrade** — on sustained 429 / 5xx the primary model
  (``llama-3.3-70b-versatile``) falls back to ``llama3-8b-8192``.
- **Fail fast, never fabricate** — a missing key or a dead backend raises rather
  than returning an ungrounded answer (edge 6.1 / 6.3).
"""

from __future__ import annotations

import math
import random
import time

import config
from rag.ratelimit import RateDecision, RateLimiter, get_limiter


class LLMError(Exception):
    """Base class for LLM failures."""


class LLMUnavailable(LLMError):
    """Backend missing/misconfigured or exhausted its retries (edge 6.1/6.3)."""


class CapacityError(LLMError):
    """Rate-limit budget exhausted and not worth waiting on (edge 6.2)."""

    def __init__(self, decision: RateDecision) -> None:
        super().__init__(decision.reason or "rate limit reached")
        self.decision = decision


def estimate_tokens(text: str) -> int:
    """Cheap upper-ish token estimate (~4 chars/token) — no tokenizer needed."""
    return max(1, math.ceil(len(text or "") / 4))


class GroqLLM:
    """Thin, rate-limited Groq chat client."""

    def __init__(
        self,
        model: str = config.GROQ_MODEL,
        fallback_model: str = config.GROQ_MODEL_FALLBACK,
        limiter: RateLimiter | None = None,
        *,
        max_wait_seconds: float = 5.0,
    ) -> None:
        self.model = model
        self.fallback_model = fallback_model
        self.limiter = limiter or get_limiter()
        # Cap how long we'll block on a per-minute bucket before giving up so the
        # UI never hangs; longer waits fail fast with a "busy" message.
        self.max_wait_seconds = max_wait_seconds
        self._client = None  # lazy

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not config.GROQ_API_KEY:
            raise LLMUnavailable("GROQ_API_KEY is not set (see .env.example)")
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - import guard
            raise LLMUnavailable("groq SDK not installed") from exc
        self._client = Groq(api_key=config.GROQ_API_KEY)
        return self._client

    def _gate(self, est_tokens: int) -> None:
        """Block on a transient (per-minute) limit or fail fast on a daily one."""
        deadline = time.monotonic() + self.max_wait_seconds
        while True:
            decision = self.limiter.check(est_tokens)
            if decision.allowed:
                self.limiter.reserve(est_tokens)
                return
            if decision.is_daily:
                raise CapacityError(decision)
            # Per-minute bucket: wait if it'll clear within our budget.
            wait = min(decision.retry_after, self.max_wait_seconds)
            if time.monotonic() + wait > deadline:
                raise CapacityError(decision)
            time.sleep(max(wait, 0.05))

    @staticmethod
    def _retry_after(exc: Exception, attempt: int) -> float:
        """Backoff delay: honour Retry-After header, else exponential + jitter."""
        header = None
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                header = response.headers.get("retry-after")
            except Exception:
                header = None
        if header:
            try:
                return float(header)
            except (TypeError, ValueError):
                pass
        base = config.GROQ_BACKOFF_BASE_SECONDS * (2 ** attempt)
        return base + random.uniform(0, base * 0.25)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = config.LLM_MAX_TOKENS,
    ) -> str:
        """Return the model's completion text, or raise ``LLMError``.

        Reserves quota once (one logical call per turn), then tries the primary
        model with 429/5xx backoff before degrading to the fallback model.
        """
        client = self._get_client()  # raises LLMUnavailable if not configured
        est = estimate_tokens(system_prompt) + estimate_tokens(user_prompt) + max_tokens
        self._gate(est)  # reserves the budget on success

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Import error types lazily; fall back to broad Exception handling if the
        # SDK's surface differs across versions.
        try:
            from groq import APIConnectionError, APIStatusError, RateLimitError
            transient = (RateLimitError, APIStatusError, APIConnectionError)
        except Exception:  # pragma: no cover
            transient = (Exception,)

        models = [self.model]
        if self.fallback_model and self.fallback_model != self.model:
            models.append(self.fallback_model)

        last_exc: Exception | None = None
        for model in models:
            for attempt in range(config.GROQ_MAX_RETRIES + 1):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=config.LLM_TEMPERATURE,
                        max_tokens=max_tokens,
                    )
                    return (resp.choices[0].message.content or "").strip()
                except transient as exc:  # type: ignore[misc]
                    last_exc = exc
                    if attempt < config.GROQ_MAX_RETRIES:
                        time.sleep(self._retry_after(exc, attempt))
                        continue
                    break  # exhausted retries on this model -> try fallback
        raise LLMUnavailable(f"Groq call failed after retries: {last_exc}")


_LLM: GroqLLM | None = None


def get_llm() -> GroqLLM:
    global _LLM
    if _LLM is None:
        _LLM = GroqLLM()
    return _LLM
