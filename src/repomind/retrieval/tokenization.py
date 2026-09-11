"""Deterministic, lightweight tokenization for source-code retrieval."""

import re
from itertools import pairwise

_COMPOUND_RE = re.compile(r"[^\W_]+(?:[_.\-/][^\W_]+)*", re.UNICODE)
_SEPARATOR_RE = re.compile(r"[_.\-/]+")

# Adjacent initialisms have no case transition (``AILLM``), so a small set of
# common software terms makes examples such as ``OpenAILLMClient`` useful while
# leaving all unknown uppercase runs intact.
_KNOWN_INITIALISMS = tuple(
    sorted(
        {
            "AI",
            "API",
            "AST",
            "BM",
            "CLI",
            "CPU",
            "CSS",
            "DB",
            "GPU",
            "HTML",
            "HTTP",
            "ID",
            "JSON",
            "JWT",
            "LLM",
            "MCP",
            "ORM",
            "RAG",
            "RRF",
            "SDK",
            "SQL",
            "UI",
            "URI",
            "URL",
            "UUID",
            "XML",
            "YAML",
        },
        key=lambda item: (-len(item), item),
    )
)


def _split_known_initialisms(value: str) -> tuple[str, ...]:
    if not value.isupper():
        return (value,)

    parts: list[str] = []
    position = 0
    while position < len(value):
        match = next(
            (
                initialism
                for initialism in _KNOWN_INITIALISMS
                if value.startswith(initialism, position)
            ),
            None,
        )
        if match is None:
            return (value,)
        parts.append(value[position : position + len(match)])
        position += len(match)
    return tuple(parts)


def _split_case_transitions(value: str) -> tuple[str, ...]:
    if not value:
        return ()

    boundaries = [0]
    for index in range(1, len(value)):
        previous = value[index - 1]
        current = value[index]
        following = value[index + 1] if index + 1 < len(value) else ""
        if (
            (previous.islower() and current.isupper())
            or (previous.isalpha() and current.isdigit())
            or (previous.isdigit() and current.isalpha())
            or (previous.isupper() and current.isupper() and following.islower())
        ):
            boundaries.append(index)
    boundaries.append(len(value))

    parts: list[str] = []
    for start, end in pairwise(boundaries):
        part = value[start:end]
        parts.extend(_split_known_initialisms(part))
    return tuple(parts)


def tokenize_code(text: str) -> tuple[str, ...]:
    """Return normalized lexical tokens while retaining compound identifiers.

    Each path-, dotted-, kebab-, or snake-like compound is retained in
    case-folded form. Its separator-delimited and case-transition components are
    also emitted once for that occurrence. Repeated occurrences remain repeated
    so BM25 can observe term frequency.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    tokens: list[str] = []
    for match in _COMPOUND_RE.finditer(text):
        compound = match.group(0)
        variants = [compound]
        atoms = tuple(part for part in _SEPARATOR_RE.split(compound) if part)
        variants.extend(atoms)
        for atom in atoms:
            variants.extend(_split_case_transitions(atom))

        seen: set[str] = set()
        for variant in variants:
            normalized = variant.casefold()
            if normalized and normalized not in seen:
                seen.add(normalized)
                tokens.append(normalized)
    return tuple(tokens)
