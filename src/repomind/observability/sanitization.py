"""Metadata allowlists; omission is the primary privacy boundary.

Redaction is defense in depth, not a claim of general secret detection.
Unknown payload fields and exception messages are deliberately omitted.
"""

import math
import re
from collections.abc import Mapping
from pathlib import PurePath, PureWindowsPath
from typing import Any

_KEYS = frozenset(
    [
        "model",
        "operation",
        "schema",
        "prompt_chars",
        "output_chars",
        "attempts",
        "retries",
        "usage",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "tool",
        "output_type",
        "failure_kind",
        "error_type",
        "message",
        "path",
        "paths",
        "start_line",
        "end_line",
        "requested_start_line",
        "requested_end_line",
        "total_lines",
        "sha256",
        "before_sha256",
        "after_sha256",
        "bytes_written",
        "bytes_before",
        "bytes_after",
        "replacements",
        "old_text_chars",
        "new_text_chars",
        "content_chars",
        "query_chars",
        "symbol_chars",
        "expected_sha256_present",
        "max_failures",
        "timeout_seconds",
        "recursive",
        "max_results",
        "case_sensitive",
        "staged",
        "max_chars",
        "truncated",
        "clean",
        "stdout_chars",
        "stderr_chars",
        "passed",
        "exit_code",
        "timed_out",
        "duration_seconds",
        "entries_count",
        "matches_count",
        "changed_files_count",
        "workspace_revision",
        "iteration",
        "action",
        "answer_chars",
        "blocker_codes",
        "completion_attempt",
        "domain_status",
        "strategy",
        "candidate_count",
        "context_chunk_count",
        "context_chars",
        "citation_count",
        "insufficient_evidence",
        "reranking_enabled",
        "case_id",
        "suite_version",
        "mode",
        "recall_at_k",
        "reciprocal_rank",
        "ndcg_at_k",
        "first_relevant_rank",
        "retrieval_recall",
        "context_recall",
        "citation_recall",
        "answer_passed",
        "task_success",
        "workflow_status",
        "oracle_passed",
        "false_positive_completion",
        "final_verification_passed",
        "recovery_observed",
        "case_count",
        "mean_recall_at_k",
        "mrr",
        "mean_ndcg_at_k",
        "mean_retrieval_recall",
        "mean_context_recall",
        "mean_citation_recall",
        "answer_pass_rate",
        "workflow_completion_rate",
        "task_success_rate",
        "false_positive_completion_rate",
        "verification_pass_rate",
        "recovery_rate",
        "trace_run_id",
        "evaluation_summary",
        "mutation_type",
        "repository_id",
        "file_count",
        "total_size_bytes",
        "chunk_count",
        "embedding_model",
    ]
)
_SECRET = re.compile(
    r"sk-[\w-]+|(?:postgres(?:ql)?(?:\+\w+)?://)[^\s]+|"
    r"(?:authorization\s*:\s*)?bearer\s+[^\s]+|"
    r"OPENAI_API_KEY(?:\s*[:=]\s*[^\s,;]+)?",
    re.IGNORECASE,
)


def redact(value: str) -> str:
    return re.sub(r"[\x00-\x1f\x7f]", " ", _SECRET.sub("[REDACTED]", value))[:256]


def _value(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return None
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, PurePath)):
        return redact(str(value))
    if isinstance(value, Mapping):
        return {
            key: _value(item, depth=depth + 1)
            for key, item in list(value.items())[:64]
            if key in _KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_value(item, depth=depth + 1) for item in value[:64]]
    return None


def sanitize_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = _value(metadata)
    return result if isinstance(result, dict) else {}


def _path(value: Any) -> str | None:
    if not isinstance(value, (str, PurePath)):
        return None
    path = str(value).replace("\\", "/")
    if path.startswith("/") or PureWindowsPath(path).drive or ".." in path.split("/"):
        return None
    return redact(path)


_ARGUMENTS = {
    "read_file": ("path", "start_line", "end_line"),
    "create_file": ("path",),
    "replace_text": ("path",),
    "run_tests": ("paths", "max_failures", "timeout_seconds"),
    "run_ruff": ("paths", "timeout_seconds"),
    "list_directory": ("path", "recursive"),
    "search_code": ("path", "max_results", "case_sensitive"),
    "find_symbol": ("path", "max_results"),
    "git_status": (),
    "git_diff": ("path", "staged", "max_chars"),
}
_OUTPUTS = {
    "read_file": ("path", "start_line", "end_line", "total_lines", "sha256"),
    "create_file": ("path", "sha256", "bytes_written"),
    "replace_text": (
        "path",
        "before_sha256",
        "after_sha256",
        "bytes_before",
        "bytes_after",
        "replacements",
    ),
    "run_tests": ("paths", "exit_code", "passed", "timed_out", "truncated", "duration_seconds"),
    "run_ruff": ("paths", "exit_code", "passed", "timed_out", "truncated", "duration_seconds"),
    "list_directory": ("path", "recursive", "truncated"),
    "search_code": ("truncated",),
    "find_symbol": ("truncated",),
    "git_status": ("clean",),
    "git_diff": ("path", "staged", "truncated"),
}


def _select(data: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    selected = {key: data[key] for key in keys if key in data}
    # Invalid arguments have not passed their tool schema yet. Do not let text
    # smuggled into a numeric/boolean/hash field become trace payload content.
    flags = {"recursive", "case_sensitive", "staged", "truncated", "clean", "passed", "timed_out"}
    for key, value in selected.items():
        if key in {"path", "paths"} or value is None:
            continue
        if key in {"sha256", "before_sha256", "after_sha256"}:
            valid = isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value)
        elif key in flags:
            valid = isinstance(value, bool)
        elif key == "duration_seconds":
            valid = type(value) in {int, float}
        else:
            valid = type(value) is int
        if not valid:
            selected[key] = None
    if "path" in selected:
        selected["path"] = _path(selected["path"])
    if "paths" in selected:
        paths = selected["paths"]
        selected["paths"] = (
            [_path(p) for p in paths[:64]] if isinstance(paths, (list, tuple)) else None
        )
    return sanitize_metadata(selected)


def sanitize_tool_arguments(name: str, data: Mapping[str, Any]) -> dict[str, Any]:
    result = _select(data, _ARGUMENTS.get(name, ()))
    fields = {
        "replace_text": ("old_text", "new_text"),
        "create_file": ("content",),
        "search_code": ("query",),
        "find_symbol": ("symbol",),
    }
    for key in fields.get(name, ()):
        if isinstance(data.get(key), str):
            result[f"{key}_chars"] = len(data[key])
    if name == "replace_text":
        result["expected_sha256_present"] = data.get("expected_sha256") is not None
    return result


def sanitize_tool_output(name: str, data: Mapping[str, Any]) -> dict[str, Any]:
    result = _select(data, _OUTPUTS.get(name, ()))
    if name not in _OUTPUTS:
        return result
    for key in ("content", "stdout", "stderr"):
        if isinstance(data.get(key), str):
            result[f"{key}_chars"] = len(data[key])
    for key in ("entries", "matches", "changed_files"):
        if isinstance(data.get(key), (tuple, list)):
            result[f"{key}_count"] = len(data[key])
    return result


def sanitize_error(error: BaseException) -> dict[str, Any]:
    return {
        "error_type": redact(type(error).__name__),
        "message": "Operation failed; inspect the caller's exception.",
    }
