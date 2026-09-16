"""Request-bound bridge from synchronous trace-producing services to SSE."""

import logging
from collections.abc import Callable
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any
from uuid import UUID

import anyio
from fastapi import Request
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from repomind.api.models import AgentRequest, CodingRequest, RAGRequest
from repomind.api.services.execution import ExecutionService
from repomind.api.streaming import ProgressEvent, encode_sse_event, safe_progress_event
from repomind.observability import InMemoryTraceRecorder, RunType, TraceContext, TraceEvent

logger = logging.getLogger(__name__)

_CHANNEL_CAPACITY = 256


class _ProgressChannel:
    """One bounded channel per request; progress loss never blocks a domain operation."""

    def __init__(self, run_id: UUID, *, capacity: int = _CHANNEL_CAPACITY) -> None:
        self.run_id = run_id
        self._events: Queue[ProgressEvent] = Queue(maxsize=capacity)
        self._done = Event()
        self._disconnected = Event()
        self._lock = Lock()
        self._result: BaseModel | None = None
        self._failed = False
        self.dropped = 0

    def receive(self, run_id: UUID, event: TraceEvent) -> None:
        if run_id != self.run_id or self._disconnected.is_set():
            return
        try:
            progress = safe_progress_event(run_id, event)
        except Exception:  # noqa: BLE001 - projection cannot affect the business operation
            logger.warning("Progress projection failed; trace retention continues")
            return
        try:
            self._events.put_nowait(progress)
        except Full:
            # A sequence gap tells a slow client that non-terminal progress was dropped.
            self.dropped += 1

    def complete(self, result: BaseModel) -> None:
        with self._lock:
            self._result = result
            self._done.set()

    def fail(self) -> None:
        with self._lock:
            self._failed = True
            self._done.set()

    def disconnect(self) -> None:
        self._disconnected.set()

    def next_event(self) -> ProgressEvent | None:
        try:
            return self._events.get(timeout=0.1)
        except Empty:
            return None

    @property
    def done(self) -> bool:
        return self._done.is_set() and self._events.empty()

    @property
    def result(self) -> BaseModel | None:
        return self._result

    @property
    def failed(self) -> bool:
        return self._failed


class StreamingService:
    """Attach a generic trace listener, then invoke the established synchronous service."""

    def __init__(self, execution: ExecutionService) -> None:
        self.execution = execution

    def _prepare(self, repository_id: int) -> None:
        # Deterministic repository/workspace failures retain normal HTTP semantics.
        self.execution.repositories.locate(repository_id)

    def index(self, request: Request, repository_id: int) -> StreamingResponse:
        self._prepare(repository_id)
        return self._response(
            request,
            RunType.INDEX,
            lambda trace: self.execution.index(repository_id, trace=trace),
        )

    def rag(self, request: Request, repository_id: int, body: RAGRequest) -> StreamingResponse:
        self._prepare(repository_id)
        return self._response(
            request,
            RunType.RAG,
            lambda trace: self.execution.rag(repository_id, body, trace=trace),
        )

    def agent(self, request: Request, repository_id: int, body: AgentRequest) -> StreamingResponse:
        self._prepare(repository_id)
        return self._response(
            request,
            RunType.READ_ONLY_AGENT,
            lambda trace: self.execution.agent(repository_id, body, trace=trace),
        )

    def coding(
        self, request: Request, repository_id: int, body: CodingRequest
    ) -> StreamingResponse:
        self._prepare(repository_id)
        return self._response(
            request,
            RunType.CODING_TASK,
            lambda trace: self.execution.coding(repository_id, body, trace=trace),
        )

    def _response(
        self,
        request: Request,
        run_type: RunType,
        operation: Callable[[TraceContext], BaseModel],
    ) -> StreamingResponse:
        # InMemoryTraceRecorder owns trace retention and still makes persistence best effort.
        channel_holder: dict[str, _ProgressChannel] = {}

        def listener(run_id: UUID, event: TraceEvent) -> None:
            channel = channel_holder.get("channel")
            if channel is not None:
                channel.receive(run_id, event)

        recorder = InMemoryTraceRecorder(
            sink=self.execution.trace_store.persist_run_trace,
            listener=listener,
        )
        trace = TraceContext(recorder, run_type)
        assert trace.run_id is not None
        channel = _ProgressChannel(trace.run_id)
        channel_holder["channel"] = channel
        # run.started was emitted during TraceContext construction before the channel existed.
        # Forward its retained event once; all later events use the generic listener.
        channel.receive(trace.run_id, recorder.traces[trace.run_id].events[0])

        def produce() -> None:
            try:
                channel.complete(operation(trace))
            except Exception:  # noqa: BLE001 - terminal SSE errors intentionally hide domain details
                channel.fail()

        async def stream() -> Any:
            producer = Thread(target=produce, name=f"repomind-stream-{trace.run_id}", daemon=False)
            producer.start()
            try:
                while True:
                    if await request.is_disconnected():
                        channel.disconnect()
                        return
                    progress = await anyio.to_thread.run_sync(
                        channel.next_event, abandon_on_cancel=True
                    )
                    if progress is not None:
                        yield encode_sse_event(progress.event, progress, event_id=progress.sequence)
                        continue
                    if channel.done:
                        if channel.failed:
                            yield encode_sse_event(
                                "error",
                                {
                                    "run_id": str(trace.run_id),
                                    "error": {
                                        "code": "operation_failed",
                                        "message": "The operation failed.",
                                    },
                                },
                            )
                        else:
                            assert channel.result is not None
                            yield encode_sse_event(
                                "result",
                                {
                                    "run_id": str(trace.run_id),
                                    "result": channel.result.model_dump(mode="json"),
                                },
                            )
                        return
            finally:
                # The producer is intentionally not cancelled: mutation/workflow integrity wins.
                channel.disconnect()

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
