"""Small reusable lifecycle and model boundaries; no business decisions."""

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any

from repomind.observability.recorder import TraceContext
from repomind.observability.sanitization import sanitize_metadata


def traced_run(run_type: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Own a lifecycle only when the caller did not supply a parent context."""

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            parent = kwargs.get("trace")
            trace = parent if parent is not None else TraceContext(kwargs.get("recorder"), run_type)
            kwargs["trace"] = trace
            try:
                result = function(*args, **kwargs)
            except BaseException as exc:
                if parent is None:
                    trace.finish("error", error=exc)
                raise
            if parent is None:
                summary = None
                if run_type == "evaluation" and trace.run_id is not None:
                    summary = trace.project(
                        lambda: sanitize_metadata(result.model_dump(mode="json"))
                    )
                trace.finish(
                    str(getattr(result, "status", "completed")), evaluation_summary=summary
                )
            return result

        return wrapped

    return decorate


def record_model_usage(
    trace: TraceContext | None, response: Any, attempts: int, *, embedding: bool = False
) -> None:
    if trace is None or trace.run_id is None:
        return
    try:
        raw = getattr(response, "usage", None)
        usage = None
        if raw is not None:
            counts = {
                key: getattr(raw, key, 0 if embedding and key == "completion_tokens" else None)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            if all(
                isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in counts.values()
            ):
                usage = counts
        trace.emit("model.usage", usage=usage, attempts=attempts, retries=attempts - 1)
    except Exception as exc:  # noqa: BLE001 - explicit best-effort telemetry boundary
        trace._diagnostic(exc)


def _model_metadata(provider: Any, args: tuple, kwargs: dict, operation: str) -> dict:
    prompt = args[0] if args else kwargs.get("prompt", "")
    schema = args[1] if len(args) > 1 else kwargs.get("response_model")
    prompt_chars = len(prompt) if isinstance(prompt, str) else None
    if operation == "embedding" and isinstance(prompt, (tuple, list)):
        prompt_chars = sum(len(text) for text in prompt if isinstance(text, str))
    return {
        "model": getattr(provider, "model", None),
        "operation": operation,
        "prompt_chars": prompt_chars,
        "schema": getattr(schema, "__name__", None),
    }


def _output_metadata(result: Any) -> dict:
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return {"output_chars": len(content)}
    if hasattr(result, "model_dump_json"):
        return {"output_chars": len(result.model_dump_json())}
    return {}


def traced_model(operation: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap sync and async requests without changing their retry machinery."""

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def sync(self: Any, *args: Any, **kwargs: Any) -> Any:
            trace = kwargs.get("trace") or getattr(self, "trace", None)
            if trace is None:
                return function(self, *args, **kwargs)
            kwargs["trace"] = trace
            with trace.operation(
                "model", **trace.project(lambda: _model_metadata(self, args, kwargs, operation))
            ) as meta:
                result = function(self, *args, **kwargs)
                if operation != "embedding":
                    meta.update(trace.project(lambda: _output_metadata(result)))
                return result

        @wraps(function)
        async def asynchronous(self: Any, *args: Any, **kwargs: Any) -> Any:
            trace = kwargs.get("trace") or getattr(self, "trace", None)
            if trace is None:
                return await function(self, *args, **kwargs)
            kwargs["trace"] = trace
            with trace.operation(
                "model", **trace.project(lambda: _model_metadata(self, args, kwargs, operation))
            ) as meta:
                result = await function(self, *args, **kwargs)
                if operation != "embedding":
                    meta.update(trace.project(lambda: _output_metadata(result)))
                return result

        wrapper = asynchronous if inspect.iscoroutinefunction(function) else sync
        wrapper._repomind_traced = True
        return wrapper

    return decorate


def generate_structured(
    provider: Any,
    prompt: str,
    response_model: type,
    *,
    trace: TraceContext | None = None,
    **kwargs: Any,
) -> Any:
    """Observe generic fake/custom providers without requiring a new protocol."""
    method = provider.generate_structured
    if trace is None or trace.run_id is None:
        return method(prompt, response_model, **kwargs)
    if getattr(method, "_repomind_traced", False):
        return method(prompt, response_model, trace=trace, **kwargs)
    with trace.operation(
        "model",
        **trace.project(
            lambda: _model_metadata(provider, (prompt, response_model), kwargs, "structured")
        ),
    ) as metadata:
        result = method(prompt, response_model, **kwargs)
        metadata.update(trace.project(lambda: _output_metadata(result)))
        return result
