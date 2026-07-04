"""Client-side rate limiting for the Groq free tier (Phase 6).

Groq meters ``llama-3.3-70b-versatile`` on four independent buckets — requests
per minute/day and tokens per minute/day (see ImplementationPlan.md §Phase 6).
The first bucket to run dry throttles the request, so we track all four with a
single sliding window of ``(timestamp, tokens)`` events and refuse locally
*before* hitting the API rather than letting Groq return 429.

Design notes:
- **Conservative reservation.** Callers reserve an *upper bound* (prompt tokens +
  ``max_tokens``) so the counters never undercount actual usage.
- **No internal sleeping.** ``check()`` is pure/side-effect-free and returns a
  ``RateDecision`` with a ``retry_after`` hint; the LLM wrapper decides whether
  to wait (per-minute buckets refill) or fail fast (daily buckets). This keeps
  all waiting/backoff logic in one place (``rag/llm.py``).
- **Thread-safe** via a lock so a multi-session Streamlit app shares one budget.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import config

_MINUTE = 60.0
_DAY = 86_400.0


@dataclass(frozen=True)
class RateDecision:
    """Outcome of a rate-limit check."""

    allowed: bool
    scope: str | None = None  # rpm | tpm | rpd | tpd
    retry_after: float = 0.0  # seconds until the offending bucket frees up
    reason: str | None = None

    @property
    def is_daily(self) -> bool:
        return self.scope in ("rpd", "tpd")


class RateLimiter:
    """Sliding-window limiter over RPM / TPM / RPD / TPD."""

    def __init__(
        self,
        rpm: int = config.GROQ_RPM,
        rpd: int = config.GROQ_RPD,
        tpm: int = config.GROQ_TPM,
        tpd: int = config.GROQ_TPD,
    ) -> None:
        self.rpm, self.rpd, self.tpm, self.tpd = rpm, rpd, tpm, tpd
        self._events: deque[tuple[float, int]] = deque()  # (ts, tokens), ascending ts
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] > _DAY:
            self._events.popleft()

    def _window(self, now: float, span: float) -> tuple[int, int]:
        """Count requests and sum tokens within the last ``span`` seconds."""
        count = tokens = 0
        for ts, tok in reversed(self._events):
            if now - ts <= span:
                count += 1
                tokens += tok
            else:
                break  # events are time-ordered; older ones are all outside
        return count, tokens

    def _oldest_within(self, now: float, span: float) -> float:
        """Timestamp of the oldest event still inside ``span`` (or ``now``)."""
        oldest = now
        for ts, _ in self._events:
            if now - ts <= span:
                oldest = ts
                break
        return oldest

    def check(self, est_tokens: int, now: float | None = None) -> RateDecision:
        """Return whether a call of ``est_tokens`` may proceed right now."""
        now = time.time() if now is None else now
        with self._lock:
            self._prune(now)
            # Daily buckets first: exhaustion here means fail-fast (won't refill
            # for a long time), so the caller shouldn't wait on them.
            rpd_c, tpd_t = self._window(now, _DAY)
            if rpd_c + 1 > self.rpd:
                return RateDecision(False, "rpd", _DAY, "daily request limit reached")
            if tpd_t + est_tokens > self.tpd:
                return RateDecision(False, "tpd", _DAY, "daily token limit reached")
            # Per-minute buckets: exhaustion is transient; hint how long to wait.
            rpm_c, tpm_t = self._window(now, _MINUTE)
            if rpm_c + 1 > self.rpm:
                wait = _MINUTE - (now - self._oldest_within(now, _MINUTE)) + 0.05
                return RateDecision(False, "rpm", max(wait, 0.0), "per-minute request limit")
            if tpm_t + est_tokens > self.tpm:
                wait = _MINUTE - (now - self._oldest_within(now, _MINUTE)) + 0.05
                return RateDecision(False, "tpm", max(wait, 0.0), "per-minute token limit")
            return RateDecision(True)

    def reserve(self, est_tokens: int, now: float | None = None) -> None:
        """Record a call's (upper-bound) token cost against the window."""
        now = time.time() if now is None else now
        with self._lock:
            self._events.append((now, max(0, est_tokens)))

    def snapshot(self, now: float | None = None) -> dict[str, int]:
        """Current usage per bucket — handy for debugging / status display."""
        now = time.time() if now is None else now
        with self._lock:
            self._prune(now)
            rpm_c, tpm_t = self._window(now, _MINUTE)
            rpd_c, tpd_t = self._window(now, _DAY)
        return {
            "rpm": rpm_c,
            "rpm_limit": self.rpm,
            "tpm": tpm_t,
            "tpm_limit": self.tpm,
            "rpd": rpd_c,
            "rpd_limit": self.rpd,
            "tpd": tpd_t,
            "tpd_limit": self.tpd,
        }


# Process-wide shared limiter (all sessions draw from the same free-tier budget).
_LIMITER: RateLimiter | None = None


def get_limiter() -> RateLimiter:
    global _LIMITER
    if _LIMITER is None:
        _LIMITER = RateLimiter()
    return _LIMITER
