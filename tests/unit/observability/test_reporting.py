from repomind.observability import InMemoryTraceRecorder, TraceContext, format_run_trace


def test_formatter_is_deterministic_and_resanitizes_mutable_metadata() -> None:
    recorder = InMemoryTraceRecorder()
    trace = TraceContext(recorder, "rag")
    trace.emit("tool.completed", tool="read_file", path="app.py", content="PRIVATE SOURCE")
    trace.finish()
    result = recorder.traces[trace.run_id]
    result.events[1].metadata["prompt"] = "PRIVATE PROMPT"
    result.events[1].metadata["message"] = "Bearer supersecret"
    report = format_run_trace(result)
    assert report == format_run_trace(result)
    assert "1  run.started" in report
    assert "2  tool.completed" in report
    assert "Tokens (reported): unknown" in report
    for private in ("PRIVATE SOURCE", "PRIVATE PROMPT", "supersecret"):
        assert private not in report
