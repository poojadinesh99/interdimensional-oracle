"""Input validation for questions sent to the oracle."""
from __future__ import annotations

import re
from dataclasses import dataclass

MIN_LENGTH = 3
MAX_LENGTH = 500

# Patterns aimed at the user trying to override the oracle's persona/system
# prompt rather than actually asking it a question.
INJECTION_PATTERNS = [
    re.compile(r"ignore (all|any|previous|prior|the) instructions", re.I),
    re.compile(r"disregard (all|any|previous|prior|the) instructions", re.I),
    re.compile(r"system prompt", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"act as (if )?(a|an) (?!oracle)", re.I),
    re.compile(r"reveal your (prompt|instructions)", re.I),
]


@dataclass
class ValidationResult:
    valid: bool
    reason: str | None = None


def validate(question: str) -> ValidationResult:
    if question is None:
        return ValidationResult(False, "A question is required.")

    trimmed = question.strip()

    if len(trimmed) < MIN_LENGTH:
        return ValidationResult(False, "The question is too short for the oracle to sense.")

    if len(trimmed) > MAX_LENGTH:
        return ValidationResult(False, f"Keep questions under {MAX_LENGTH} characters.")

    for pattern in INJECTION_PATTERNS:
        if pattern.search(trimmed):
            return ValidationResult(
                False, "The oracle only answers questions, not requests to change how it answers."
            )

    return ValidationResult(True)
