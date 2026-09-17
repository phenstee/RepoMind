"""Deterministic, approximate token counting for context-budget accounting.

No tokenizer dependency exists in this project. This is a documented
approximation, not an exact count for any specific model's tokenizer.
"""

from math import ceil


def estimate_tokens(text: str) -> int:
    """Approximate token count as ``ceil(len(text) / 4)``, floored at zero length."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text:
        return 0
    return ceil(len(text) / 4)
