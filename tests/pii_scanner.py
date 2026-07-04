"""PII scanner for Phase 8 privacy checks (eval.md "PII scanner" method).

Detects the exact categories the problem statement forbids collecting/storing:
PAN, Aadhaar, account number, OTP, email, and phone number. Used to (a) verify
detection works and (b) assert no PII is ever echoed back in a response or
persisted to disk.
"""

from __future__ import annotations

import re

# Order matters: more specific patterns (PAN, Aadhaar) are checked before the
# generic long-digit "account number" so a 12-digit Aadhaar isn't mislabelled.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Indian PAN: 5 letters, 4 digits, 1 letter.
    ("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),
    # Aadhaar: 12 digits, optionally split into 3 groups of 4.
    ("Aadhaar", re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")),
    # Email.
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    # Indian mobile: optional +91/0 prefix, then a 6-9 leading 10-digit number.
    ("phone", re.compile(r"(?:\+91[-\s]?|\b0)?[6-9]\d{9}\b")),
    # OTP: the keyword near a 4-8 digit code.
    ("OTP", re.compile(r"\b(?:otp|one[-\s]?time\s?password)\b[^\d]{0,12}\d{4,8}\b", re.IGNORECASE)),
    # Account number: a bare 9-18 digit run (checked last).
    ("account", re.compile(r"\b\d{9,18}\b")),
]


def scan(text: str) -> list[str]:
    """Return the sorted set of PII category names found in ``text``."""
    if not text:
        return []
    found: set[str] = set()
    working = text
    for label, pat in _PATTERNS:
        if pat.search(working):
            found.add(label)
            # Blank out matches so a 12-digit Aadhaar isn't also counted as an
            # account number, etc.
            working = pat.sub(" ", working)
    return sorted(found)


def contains_pii(text: str) -> bool:
    return bool(scan(text))
