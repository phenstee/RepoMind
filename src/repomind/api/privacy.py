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


def public_review_metadata(value: Any, diff_texts: tuple[str, ...]) -> Any:
    """Redact secrets, host paths, and exact changed source lines from review prose."""

    fragments: set[str] = set()
    ignored_prefixes = ("+++", "---", "@@", "diff --git", "index ")
    for diff in diff_texts:
        for line in diff.splitlines():
            if line.startswith(ignored_prefixes) or not line.startswith(("+", "-", " ")):
                continue
            source = line[1:].strip()
            if len(source) < 8:
                continue
            fragments.add(source[:256])
            fragments.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]{7,}", source)[:16])
            if len(source) > 256:
                fragments.add(source[-256:])
            if len(fragments) >= 128:
                break
        if len(fragments) >= 128:
            break

    def sanitize(item: Any) -> Any:
        if isinstance(item, str):
            result = public_text(item) or ""
            for fragment in sorted(fragments, key=len, reverse=True):
                result = result.replace(fragment, "[REDACTED SOURCE]")
            return result
        if isinstance(item, dict):
            return {key: sanitize(child) for key, child in item.items()}
        if isinstance(item, list):
            return [sanitize(child) for child in item]
        return item

    return sanitize(value)
