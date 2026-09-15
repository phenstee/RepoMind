"""Deterministic, metadata-only trace rendering."""

import json

from repomind.observability.models import RunTrace


def format_run_trace(trace: RunTrace) -> str:
    # Revalidate at the output boundary, including caller-mutated nested metadata.
    trace = RunTrace.model_validate(trace.model_dump(mode="json"))
    tokens = str(trace.token_usage.total_tokens) if trace.token_usage is not None else "unknown"
    lines = [
        f"Run: {trace.run_id}",
        f"Type: {trace.run_type}",
        f"Status: {trace.status} ({trace.domain_status})",
        f"Duration: {trace.duration_ms} ms",
        f"LLM calls: {trace.llm_calls}",
        f"Tool calls: {trace.tool_calls}",
        f"Mutations: {trace.successful_mutations}",
        f"Tokens (reported): {tokens}",
        "",
        "Timeline:",
    ]
    for event in trace.events:
        metadata = json.dumps(event.metadata, sort_keys=True, ensure_ascii=True)
        lines.append(f"{event.sequence}  {event.event_type}  {metadata}")
    return "\n".join(lines)
