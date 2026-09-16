"""Trusted-local HTTP application boundary."""

from typing import Any

__all__ = ["ProgressEvent", "create_app", "encode_sse_event"]


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from repomind.api.app import create_app

        return create_app
    if name in {"ProgressEvent", "encode_sse_event"}:
        from repomind.api.streaming import ProgressEvent, encode_sse_event

        return {"ProgressEvent": ProgressEvent, "encode_sse_event": encode_sse_event}[name]
    raise AttributeError(name)
