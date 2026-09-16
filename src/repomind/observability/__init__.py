"""Opt-in metadata-first observability, with a lazily imported PostgreSQL adapter."""

from typing import TYPE_CHECKING, Any

from repomind.observability.models import RunStatus, RunTrace, RunType, TraceEvent
from repomind.observability.recorder import (
    InMemoryTraceRecorder,
    NoOpTraceRecorder,
    TraceContext,
    TraceEventListener,
    TraceRecorder,
)
from repomind.observability.reporting import format_run_trace

if TYPE_CHECKING:
    from repomind.observability.persistence import PostgresTraceStore

__all__ = [
    "InMemoryTraceRecorder",
    "NoOpTraceRecorder",
    "PostgresTraceStore",
    "RunStatus",
    "RunTrace",
    "RunType",
    "TraceContext",
    "TraceEvent",
    "TraceEventListener",
    "TraceRecorder",
    "format_run_trace",
]


def __getattr__(name: str) -> Any:
    if name == "PostgresTraceStore":
        from repomind.observability.persistence import PostgresTraceStore

        return PostgresTraceStore
    raise AttributeError(name)
