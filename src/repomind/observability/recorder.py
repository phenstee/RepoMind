"""Explicit run handles with best-effort recorder and storage boundaries."""

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID, uuid4

from repomind.llm.models import TokenUsage
from repomind.observability.models import EventType, RunStatus, RunTrace, RunType, TraceEvent
from repomind.observability.sanitization import sanitize_error

logger = logging.getLogger(__name__)


class TraceRecorder(Protocol):
    def start_run(self, run_type: RunType) -> RunTrace | None: ...
    def record_event(self, run_id: UUID, event: TraceEvent) -> None: ...
    def finish_run(self, trace: RunTrace) -> None: ...


class NoOpTraceRecorder:
    def start_run(self, run_type: RunType) -> None:
        return None

    def record_event(self, run_id: UUID, event: TraceEvent) -> None:
        pass

    def finish_run(self, trace: RunTrace) -> None:
        pass


class InMemoryTraceRecorder:
    """One recorder can retain several independent runs; no global context.

    Optional sink runs only after completion. Failures retain the memory copy
    and append a safe diagnostic. Use a separate recorder per concurrent task.
    """

    def __init__(
        self,
        *,
        id_provider: Callable[[], UUID] = uuid4,
        utc_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic_clock: Callable[[], float] = time.monotonic,
        sink: Callable[[RunTrace], None] | None = None,
    ) -> None:
        self.id_provider = id_provider
        self.utc_clock = utc_clock
        self.monotonic_clock = monotonic_clock
        self.sink = sink
        self.traces: dict[UUID, RunTrace] = {}
        self.diagnostics: list[dict[str, Any]] = []

    def start_run(self, run_type: RunType) -> RunTrace:
        trace = RunTrace(run_id=self.id_provider(), run_type=run_type, started_at=self.utc_clock())
        if trace.run_id in self.traces:
            raise ValueError("duplicate trace run ID")
        self.traces[trace.run_id] = trace
        return trace

    def record_event(self, run_id: UUID, event: TraceEvent) -> None:
        trace = self.traces[run_id]
        self.traces[run_id] = trace.model_copy(update={"events": (*trace.events, event)})

    def finish_run(self, trace: RunTrace) -> None:
        self.traces[trace.run_id] = trace.model_copy(deep=True)
        if self.sink is not None:
            try:
                self.sink(trace.model_copy(deep=True))
            except Exception as exc:  # noqa: BLE001 - explicit best-effort telemetry boundary
                self.diagnostics.append(sanitize_error(exc))
                logger.warning("Trace persistence failed; in-memory trace retained")


class TraceContext:
    """An explicitly injected, sequential per-run timeline.

    LLM calls count logical requests (retries stay within a call). Tool calls
    count registry requests including validation failures, excluding guards.
    Coding preflight, automatic verification and review requests are included.
    """

    def __init__(
        self, recorder: TraceRecorder | None = None, run_type: RunType = RunType.CODING_TASK
    ) -> None:
        self.recorder = recorder if recorder is not None else NoOpTraceRecorder()
        self.diagnostics: list[dict[str, Any]] = []
        self.events: list[TraceEvent] = []
        self.workspace_revision = 0
        self._finished = False
        self._trace: RunTrace | None = None
        self._utc = getattr(self.recorder, "utc_clock", lambda: datetime.now(UTC))
        self._monotonic = getattr(self.recorder, "monotonic_clock", time.monotonic)
        self._start = self.now()
        try:
            self._trace = self.recorder.start_run(RunType(run_type))
        except Exception as exc:  # noqa: BLE001 - explicit best-effort telemetry boundary
            self._diagnostic(exc)
        self.emit("run.started")

    @property
    def run_id(self) -> UUID | None:
        return self._trace.run_id if self._trace is not None else None

    def _diagnostic(self, exc: Exception) -> None:
        self.diagnostics.append(sanitize_error(exc))
        logger.warning("Trace recording failed; application operation continues")

    def project(self, factory: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """Metadata extraction is telemetry too: never fail a business result."""
        if self._trace is None:
            return {}
        try:
            return factory()
        except Exception as exc:  # noqa: BLE001 - explicit best-effort telemetry boundary
            self._diagnostic(exc)
            return {}

    def now(self) -> float:
        try:
            return self._monotonic()
        except Exception as exc:  # noqa: BLE001 - explicit best-effort telemetry boundary
            self._diagnostic(exc)
            return 0.0

    def elapsed(self, started: float) -> float:
        return max(0.0, (self.now() - started) * 1000)

    def emit(
        self, event_type: EventType, *, duration_ms: float | None = None, **metadata: Any
    ) -> None:
        if self._trace is None or self._finished:
            return
        try:
            event = TraceEvent(
                event_type=event_type,
                sequence=len(self.events) + 1,
                timestamp=self._utc(),
                duration_ms=duration_ms,
                metadata=metadata,
            )
            self.events.append(event)
            self.recorder.record_event(self._trace.run_id, event.model_copy(deep=True))
        except Exception as exc:  # noqa: BLE001 - explicit best-effort telemetry boundary
            self._diagnostic(exc)

    @contextmanager
    def operation(self, category: str, **metadata: Any) -> Iterator[dict[str, Any]]:
        started = self.now()
        self.emit(f"{category}.started", **metadata)
        result: dict[str, Any] = {}
        try:
            yield result
        except BaseException as exc:
            self.emit(
                f"{category}.failed",
                duration_ms=self.elapsed(started),
                **metadata,
                **sanitize_error(exc),
            )
            raise
        else:
            self.emit(
                f"{category}.completed", duration_ms=self.elapsed(started), **(metadata | result)
            )

    def finish(
        self,
        domain_status: str = "completed",
        *,
        evaluation_summary: dict[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        if self._trace is None or self._finished:
            return
        status = RunStatus.COMPLETED if domain_status == "completed" else RunStatus.FAILED
        self.emit(
            "run.completed" if status == RunStatus.COMPLETED else "run.failed",
            domain_status=domain_status,
            **(sanitize_error(error) if error is not None else {}),
        )
        self._finished = True
        try:
            usages = [
                e.metadata["usage"]
                for e in self.events
                if e.event_type == "model.usage" and e.metadata.get("usage") is not None
            ]
            usage = (
                TokenUsage(
                    **{
                        key: sum(item[key] for item in usages)
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    }
                )
                if usages
                else None
            )
            models = {
                e.metadata["model"]
                for e in self.events
                if e.event_type == "model.started" and e.metadata.get("model")
            }
            trace = RunTrace(
                run_id=self._trace.run_id,
                run_type=self._trace.run_type,
                status=status,
                domain_status=domain_status,
                model=next(iter(models)) if len(models) == 1 else None,
                started_at=self._trace.started_at,
                ended_at=self._utc(),
                duration_ms=self.elapsed(self._start),
                events=tuple(self.events),
                llm_calls=sum(e.event_type == "model.started" for e in self.events),
                tool_calls=sum(e.event_type == "tool.started" for e in self.events),
                successful_mutations=sum(e.event_type == "file.mutated" for e in self.events),
                errors=sum(
                    e.event_type in {"model.failed", "tool.failed", "evaluation.case.failed"}
                    for e in self.events
                ),
                token_usage=usage,
                usage_reported_calls=len(usages),
                evaluation_summary=evaluation_summary,
            )
            self.recorder.finish_run(trace)
        except Exception as exc:  # noqa: BLE001 - explicit best-effort telemetry boundary
            self._diagnostic(exc)
