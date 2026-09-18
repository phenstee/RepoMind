"""Deterministic identifier-shaped candidate detection inside plain-text queries.

This is metadata extraction, not query rewriting: the original query text is
always sent to semantic and lexical retrieval unchanged. Extraction only
decides which substrings look like code identifiers, and how much confidence
that shape alone deserves, before any repository-specific lookup happens.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

_QUALIFIED_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MIN_SIMPLE_LENGTH = 2


class IdentifierConfidence(StrEnum):
    """How much the shape of a candidate alone justifies treating it as code.

    ``STRONG`` shapes (qualified dotted names, snake_case, camelCase,
    PascalCase) rarely occur in ordinary English prose. ``WEAK`` candidates
    are plain lowercase words that are also common English words (``run``,
    ``get``, ``login``); they still get looked up, but only ever as the
    lowest-priority tier, so they cannot dominate ordinary semantic/lexical
    retrieval merely by existing.
    """

    STRONG = "strong"
    WEAK = "weak"


@dataclass(frozen=True, slots=True)
class IdentifierCandidate:
    """One detected identifier-shaped substring, exact case preserved."""

    text: str
    qualified: bool
    confidence: IdentifierConfidence


def _has_internal_case_transition(word: str) -> bool:
    """Detect a lower-to-upper transition after position zero.

    A merely capitalized English word (``Where``, ``Login``) has no such
    transition and stays weak; ``ContextAssembler`` and ``camelCaseName`` do,
    and are strong.
    """

    return any(word[index - 1].islower() and word[index].isupper() for index in range(1, len(word)))


def _classify_simple(word: str) -> IdentifierConfidence:
    if "_" in word:
        return IdentifierConfidence.STRONG
    if _has_internal_case_transition(word):
        return IdentifierConfidence.STRONG
    return IdentifierConfidence.WEAK


def extract_identifier_candidates(query: str) -> tuple[IdentifierCandidate, ...]:
    """Return deduplicated identifier-shaped candidates in first-seen order.

    Qualified dotted forms (``UserService.login``) are always strong and are
    extracted whole; a plain word already covered by a qualified match is not
    also emitted as a separate simple candidate.
    """

    if not isinstance(query, str):
        raise TypeError("query must be a string")

    consumed: list[tuple[int, int]] = []
    candidates: dict[str, IdentifierCandidate] = {}
    order: list[str] = []

    for match in _QUALIFIED_RE.finditer(query):
        text = match.group(0)
        consumed.append(match.span())
        if text not in candidates:
            candidates[text] = IdentifierCandidate(
                text=text, qualified=True, confidence=IdentifierConfidence.STRONG
            )
            order.append(text)

    def _inside_consumed(start: int, end: int) -> bool:
        return any(span_start <= start and end <= span_end for span_start, span_end in consumed)

    for match in _WORD_RE.finditer(query):
        start, end = match.span()
        if _inside_consumed(start, end):
            continue
        text = match.group(0)
        if len(text) < _MIN_SIMPLE_LENGTH or text.isdigit() or text in candidates:
            continue
        candidates[text] = IdentifierCandidate(
            text=text, qualified=False, confidence=_classify_simple(text)
        )
        order.append(text)

    return tuple(candidates[key] for key in order)
