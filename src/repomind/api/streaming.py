"""Public-safe semantic progress projection and standards-compatible SSE framing."""

import json
import math
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from repomind.ingestion import validate_repository_relative_path
from repomind.observability import TraceEvent
from repomind.observability.models import EventType

_EVENT_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")
_LABEL = re.compile(r"[A-Za-z0-9_.+-]{1,128}\Z")
_BLOCKER = re.compile(r"[a-z0-9_]{1,128}\Z")
_HASH = re.compile(r"[0-9a-fA-F]{64}\Z")

_NUMBERS = frozenset(
    {
        "repository_id",
        "file_count",
        "total_size_bytes",
        "chunk_count",
        "prompt_chars",
        "output_chars",
        "attempts",
        "retries",
        "candidate_count",
        "context_chunk_count",
        "context_chars",
        "citation_count",
        "answer_chars",
        "iteration",
        "workspace_revision",
        "completion_attempt",
        "exit_code",
        "timeout_seconds",
        "max_failures",
        "changed_files_count",
        "bytes_written",
        "bytes_before",
        "bytes_after",
        "replacements",
        "start_line",
        "end_line",
        "total_lines",
    }
)
_BOOLEANS = frozenset(
    {
        "reranking_enabled",
        "insufficient_evidence",
        "passed",
        "timed_out",
        "truncated",
        "clean",
    }
)
_LABELS = frozenset(
    {
        "model",
        "embedding_model",
        "operation",
        "schema",
        "tool",
        "output_type",
        "failure_kind",
        "mutation_type",
        "strategy",
        "action",
        "domain_status",
    }
)
_HASHES = frozenset({"sha256", "before_sha256", "after_sha256"})
_PATHS = frozenset({"path", "paths"})
_COMMON = frozenset({"workspace_revision", "domain_status"})
_EVENT_FIELDS: dict[str, frozenset[str]] = {
    "run.started": frozenset(),
    "run.completed": _COMMON,
    "run.failed": _COMMON,
    "index.started": frozenset({"repository_id"}),
    "ingestion.completed": frozenset({"repository_id", "file_count", "total_size_bytes"}),
    "chunking.completed": frozenset({"repository_id", "chunk_count"}),
    "embedding.started": frozenset({"repository_id", "chunk_count"}),
    "embedding.completed": frozenset({"repository_id", "chunk_count", "embedding_model"}),
    "persistence.completed": frozenset({"repository_id", "file_count", "chunk_count"}),
    "model.started": frozenset({"model", "operation", "schema", "prompt_chars"}),
    "model.attempt": frozenset({"attempts", "retries"}),
    "model.completed": frozenset({"model", "operation", "schema", "output_chars"}),
    "model.failed": frozenset({"operation", "failure_kind"}),
    "model.usage": frozenset({"attempts", "retries"}),
    "retrieval.started": frozenset({"strategy", "reranking_enabled"}),
    "retrieval.completed": frozenset({"strategy", "reranking_enabled", "candidate_count"}),
    "retrieval.failed": frozenset({"strategy", "reranking_enabled", "failure_kind"}),
    "rag.context": frozenset({"context_chunk_count", "context_chars"}),
    "rag.answer": frozenset({"citation_count", "insufficient_evidence", "answer_chars"}),
    "agent.decision": frozenset({"iteration", "action", "tool", "answer_chars"}),
    "agent.stopped": frozenset({"iteration", "domain_status", "blocker_codes"}),
    "tool.started": frozenset({"tool", "path", "paths", "workspace_revision"}),
    "tool.completed": frozenset(
        {
            "tool",
            "path",
            "paths",
            "workspace_revision",
            "passed",
            "exit_code",
            "timed_out",
            "truncated",
            "duration_seconds",
            "output_type",
        }
    ),
    "tool.failed": frozenset({"tool", "workspace_revision", "failure_kind"}),
    "tool.blocked": frozenset({"tool", "failure_kind"}),
    "file.mutated": frozenset(
        {"path", "workspace_revision", "mutation_type", "sha256", "before_sha256", "after_sha256"}
    ),
    "preflight.started": frozenset(),
    "preflight.passed": frozenset({"clean"}),
    "preflight.failed": frozenset({"blocker_codes"}),
    "verification.started": frozenset({"tool", "paths", "workspace_revision"}),
    "verification.completed": frozenset(
        {
            "tool",
            "paths",
            "workspace_revision",
            "passed",
            "exit_code",
            "timed_out",
            "truncated",
            "duration_seconds",
        }
    ),
    "verification.failed": frozenset({"tool", "workspace_revision", "failure_kind"}),
    "completion.requested": frozenset({"completion_attempt", "workspace_revision"}),
    "completion.blocked": frozenset({"completion_attempt", "workspace_revision", "blocker_codes"}),
    "completion.completed": frozenset({"workspace_revision"}),
    "final_review.started": frozenset({"workspace_revision"}),
    "final_review.completed": frozenset(
        {"workspace_revision", "changed_files_count", "truncated", "passed", "blocker_codes"}
    ),
}


class ProgressEvent(BaseModel):
    """Stable, intentionally narrow public representation of one trace event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: UUID
    sequence: int = Field(ge=1, strict=True)
    event: EventType
    timestamp: datetime
    data: dict[str, Any] = Field(default_factory=dict)


def _relative_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return validate_repository_relative_path(Path(value)).as_posix()
    except ValueError:
        return None


def _public_value(key: str, value: Any) -> Any:
    if key == "path":
        return _relative_path(value)
    if key == "paths":
        if not isinstance(value, (list, tuple)):
            return None
        paths = [_relative_path(item) for item in value[:32]]
        return paths if all(path is not None for path in paths) else None
    if key == "blocker_codes":
        if not isinstance(value, (list, tuple)):
            return None
        codes = [item for item in value[:32] if isinstance(item, str) and _BLOCKER.fullmatch(item)]
        return codes if len(codes) == len(value[:32]) else None
    if key in _HASHES:
        return value if isinstance(value, str) and _HASH.fullmatch(value) else None
    if key in _NUMBERS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return value if isinstance(value, int) or math.isfinite(value) else None
    if key == "duration_seconds":
        return value if type(value) in {int, float} and math.isfinite(value) else None
    if key in _BOOLEANS:
        return value if isinstance(value, bool) else None
    if key in _LABELS:
        return value if isinstance(value, str) and _LABEL.fullmatch(value) else None
    return None


def safe_progress_event(run_id: UUID, event: TraceEvent) -> ProgressEvent:
    """Project allowlisted trace metadata without exposing raw trace payloads."""

    data: dict[str, Any] = {}
    for key in _EVENT_FIELDS.get(event.event_type, frozenset()):
        if key not in event.metadata:
            continue
        value = _public_value(key, event.metadata[key])
        if value is not None:
            data[key] = value
    if event.duration_ms is not None:
        data["duration_ms"] = event.duration_ms
    return ProgressEvent(
        run_id=run_id,
        sequence=event.sequence,
        event=event.event_type,
        timestamp=event.timestamp,
        data=data,
    )


def encode_sse_event(
    event: str,
    data: BaseModel | Mapping[str, Any],
    *,
    event_id: int | None = None,
) -> str:
    """Serialize one event without allowing data newlines to alter SSE framing."""

    if not _EVENT_NAME.fullmatch(event):
        raise ValueError("SSE event name contains unsupported characters")
    if event_id is not None and (isinstance(event_id, bool) or event_id < 1):
        raise ValueError("SSE event ID must be a positive integer")
    payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else dict(data)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    prefix = f"id: {event_id}\n" if event_id is not None else ""
    return f"{prefix}event: {event}\ndata: {encoded}\n\n"
