"""Defense-in-depth host-path redaction for public, model-authored text."""

import re
from typing import Any

# This is not a general secret detector. Private payloads are excluded by response models.
_PRIVATE = re.compile(
    r"sk-[\w-]+|postgres(?:ql)?(?:\+\w+)?://[^\s]+|"
    r"(?:authorization\s*:\s*)?bearer\s+[^\s]+|"
    r"OPENAI_API_KEY(?:\s*[:=]\s*[^\s,;]+)?|"
    r"(?<![\w])[a-z]:[/\\][^\s\"'`<>]*|"
    r"(?<![\w:/])/(?:[^\s/\"'`<>]+/)+[^\s\"'`<>]*|"
    r"\\\\[^\s\"'`<>]+",
    re.IGNORECASE,
)


def public_text(value: str | None) -> str | None:
    return _PRIVATE.sub("[REDACTED]", value) if value is not None else None


def public_metadata(value: Any) -> Any:
    """Applied after the existing trace allowlist, including labels in legacy stored traces."""
    if isinstance(value, str):
        return public_text(value)
    if isinstance(value, dict):
        return {key: public_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_metadata(item) for item in value]
    return value
