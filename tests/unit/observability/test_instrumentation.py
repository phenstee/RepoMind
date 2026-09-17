import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

from repomind.agent import AgentConfig, AgentDecision, run_editing_agent, run_read_only_agent
from repomind.coding import (
    CodingPlan,
    CodingReview,
    CodingTask,
    CodingWorkflowConfig,
    run_coding_task,
)
from repomind.config import Settings
from repomind.ingestion import CodeChunk
from repomind.llm import LLMError, OpenAILLMClient
from repomind.observability import (
    InMemoryTraceRecorder,
    NoOpTraceRecorder,
    TraceContext,
    format_run_trace,
)
from repomind.rag import answer_repository_question_with_retriever
from repomind.rag.models import GroundedLLMResponse
from repomind.retrieval import (
    BM25Index,
    EmbeddedChunk,
    EmbeddingVector,
    OpenAIEmbeddingClient,
    hybrid_search,
)
from repomind.tools import (
    ToolContext,
    ToolValidationError,
    create_default_tool_registry,
    create_editing_tool_registry,
)


class ScriptedLLM:
    model = "scripted"

    def __init__(self, decisions):
        self.decisions = iter(decisions)

    def generate_structured(self, prompt, response_model, **kwargs):
        if response_model is CodingPlan:
            payload = json.loads(prompt)
            criteria = payload["task"]["acceptance_criteria"]
            return CodingPlan.model_validate(
                {
                    "task_summary": "Implement and verify the requested change.",
                    "steps": [
                        {
                            "step_id": 1,
                            "action": "Inspect, implement, and verify the change.",
                            "criterion_indices": list(range(len(criteria))),
                        }
                    ],
                    "acceptance_coverage": [
                        {"criterion_index": index, "step_ids": [1]}
                        for index in range(len(criteria))
                    ],
                }
            )
        if response_model is CodingReview:
            payload = json.loads(prompt)
            criteria = payload["task"]["acceptance_criteria"]
            return CodingReview.model_validate(
                {
                    "verdict": "approve",
                    "workspace_revision": payload["workspace_revision"],
                    "acceptance_results": [
                        {
                            "criterion_index": index,
                            "status": "satisfied",
                            "evidence": "The bounded evidence supports this criterion.",
                        }
                        for index in range(len(criteria))
                    ],
                }
            )
        return next(self.decisions)


def tool(name, **arguments):
    return AgentDecision(action="tool", tool_name=name, tool_arguments=arguments)


def final():
    return AgentDecision(action="final", final_answer="private final answer")


def stored(recorder):
    return next(iter(recorder.traces.values()))


def project(root: Path):
    if shutil.which("git") is None:
        pytest.skip("git unavailable")
    (root / "app.py").write_bytes(b"def value():\n    return 0\n")
    (root / "tests").mkdir()
    (root / "tests/test_app.py").write_text(
        "from app import value\n\n\ndef test_value():\n    assert value() == 2\n", encoding="utf-8"
    )
    (root / ".gitignore").write_text(
        "__pycache__/\n.pytest_cache/\n.ruff_cache/\n", encoding="utf-8"
    )
    for args in (
        ("init", "-b", "main"),
        ("config", "user.name", "Test"),
        ("config", "user.email", "test@example.invalid"),
        ("add", "."),
        ("commit", "-m", "fixture"),
    ):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def replacement(before: int, after: int):
    content = f"def value():\n    return {before}\n".encode()
    return tool(
        "replace_text",
        path="app.py",
        old_text=f"return {before}",
        new_text=f"return {after}",
        expected_sha256=hashlib.sha256(content).hexdigest(),
    )


def test_agent_timeline_matches_untraced_result(tmp_path):
    (tmp_path / "app.py").write_text("private source", encoding="utf-8")
    registry = create_default_tool_registry(ToolContext(repository_root=tmp_path))
    decisions = [tool("search_code", query="private"), tool("read_file", path="app.py"), final()]
    baseline = run_read_only_agent(
        "inspect", ScriptedLLM(decisions), registry, recorder=NoOpTraceRecorder()
    )
    recorder = InMemoryTraceRecorder()
    actual = run_read_only_agent("inspect", ScriptedLLM(decisions), registry, recorder=recorder)
    assert actual == baseline
    trace = stored(recorder)
    assert (trace.llm_calls, trace.tool_calls, trace.successful_mutations) == (3, 2, 0)
    kinds = [event.event_type for event in trace.events]
    assert kinds[0] == "run.started" and kinds[-1] == "run.completed"
    assert kinds.count("model.completed") == 3
    assert kinds.count("agent.decision") == 3
    assert trace.token_usage is None
    for value in ("private source", "private final answer"):
        assert value not in trace.model_dump_json() + format_run_trace(trace)


def test_tool_failure_recovery_and_validation(tmp_path):
    (tmp_path / "app.py").write_text("value = 1", encoding="utf-8")
    registry = create_default_tool_registry(ToolContext(repository_root=tmp_path))
    recorder = InMemoryTraceRecorder()
    run_read_only_agent(
        "inspect",
        ScriptedLLM(
            [
                tool("read_file", path="missing.py"),
                tool("search_code", query="value"),
                final(),
            ]
        ),
        registry,
        recorder=recorder,
    )
    trace = stored(recorder)
    outcomes = [e for e in trace.events if e.event_type in {"tool.failed", "tool.completed"}]
    assert [e.event_type for e in outcomes] == ["tool.failed", "tool.completed"]
    assert outcomes[0].metadata["failure_kind"] == "execution"
    assert trace.errors == 1 and trace.status == "completed"
    context = TraceContext(InMemoryTraceRecorder(), "read_only_agent")
    with pytest.raises(ToolValidationError):
        registry.execute("read_file", {"path": "../escape"}, trace=context)
    assert context.events[-1].metadata["failure_kind"] == "validation"


def test_repeated_guard_not_counted_as_registry_call_and_limit_visible(tmp_path):
    registry = create_default_tool_registry(ToolContext(repository_root=tmp_path))
    recorder = InMemoryTraceRecorder()
    run_read_only_agent(
        "inspect",
        ScriptedLLM([tool("list_directory")] * 3),
        registry,
        config=AgentConfig(max_iterations=3),
        recorder=recorder,
    )
    trace = stored(recorder)
    assert trace.tool_calls == 2
    assert trace.domain_status == "max_iterations_reached"
    assert trace.status == "failed"
    assert any(
        e.event_type == "tool.blocked" and e.metadata["failure_kind"] == "repeated_call"
        for e in trace.events
    )
    assert any(e.event_type == "agent.stopped" for e in trace.events)


def test_editing_trace_sanitizes_real_create_replace_and_verification(tmp_path):
    project(tmp_path)
    recorder = InMemoryTraceRecorder()
    result = run_editing_agent(
        "change",
        ScriptedLLM(
            [
                tool("read_file", path="app.py"),
                replacement(0, 2),
                tool("create_file", path="note.txt", content="SECRET SOURCE CODE"),
                tool("run_tests", paths=["tests"]),
                final(),
            ]
        ),
        create_editing_tool_registry(ToolContext(repository_root=tmp_path)),
        recorder=recorder,
    )
    assert result.status == "completed"
    trace = stored(recorder)
    assert trace.successful_mutations == 2
    mutations = [e for e in trace.events if e.event_type == "file.mutated"]
    assert [e.metadata["workspace_revision"] for e in mutations] == [1, 2]
    assert mutations[0].metadata["before_sha256"] != mutations[0].metadata["after_sha256"]
    assert any(
        e.event_type == "verification.completed" and e.metadata["passed"] for e in trace.events
    )
    for value in ("return 0", "return 2", "SECRET SOURCE CODE", "private final answer"):
        assert value not in trace.model_dump_json() + format_run_trace(trace)


@pytest.mark.parametrize("tracing", [False, True])
def test_coding_recovery_flagship(tmp_path, tracing):
    project(tmp_path)
    recorder = InMemoryTraceRecorder() if tracing else NoOpTraceRecorder()
    result = run_coding_task(
        CodingTask(objective="Return two"),
        ScriptedLLM(
            [
                replacement(0, 1),
                final(),
                replacement(1, 2),
                final(),
            ]
        ),
        create_editing_tool_registry(ToolContext(repository_root=tmp_path)),
        recorder=recorder,
    )
    assert result.status == "completed"
    assert result.workspace_revision == 2
    assert result.completion_attempts == 2
    assert result.verification.tests_passed and result.verification.ruff_passed
    if not tracing:
        return
    trace = stored(recorder)
    assert len(recorder.traces) == 1
    assert trace.llm_calls == 6
    assert trace.successful_mutations == 2
    event_types = [event.event_type for event in trace.events]
    assert event_types.index("planning.started") < event_types.index("planning.completed")
    assert event_types.index("planning.completed") < event_types.index("file.mutated")
    assert event_types.index("review.started") < event_types.index("review.completed")
    assert event_types.index("review.completed") < event_types.index("completion.completed")
    selected = [
        e
        for e in trace.events
        if e.event_type
        in {
            "file.mutated",
            "completion.requested",
            "verification.completed",
            "completion.blocked",
            "completion.completed",
        }
    ]
    assert [e.event_type for e in selected] == [
        "file.mutated",
        "completion.requested",
        "verification.completed",
        "verification.completed",
        "completion.blocked",
        "file.mutated",
        "completion.requested",
        "verification.completed",
        "verification.completed",
        "completion.completed",
    ]
    assert selected[2].metadata["passed"] is False
    assert selected[4].metadata["blocker_codes"] == ["tests_failed"]
    assert selected[7].metadata["passed"] is True
    assert [selected[i].metadata["workspace_revision"] for i in (2, 7)] == [1, 2]
    assert trace.tool_calls == len([e for e in trace.events if e.event_type == "tool.started"])
    assert "return 1" not in trace.model_dump_json()


def test_coding_completion_exhaustion_and_preflight(tmp_path):
    project(tmp_path)
    registry = create_editing_tool_registry(ToolContext(repository_root=tmp_path))
    recorder = InMemoryTraceRecorder()
    result = run_coding_task(
        CodingTask(objective="Return two"),
        ScriptedLLM([final()]),
        registry,
        workflow_config=CodingWorkflowConfig(max_completion_attempts=1),
        recorder=recorder,
    )
    assert result.status == "verification_failed"
    trace = stored(recorder)
    assert trace.domain_status == "verification_failed"
    blocked = next(e for e in trace.events if e.event_type == "completion.blocked")
    assert "completion_attempt_limit" in blocked.metadata["blocker_codes"]
    (tmp_path / "app.py").write_text("user changes", encoding="utf-8")
    recorder = InMemoryTraceRecorder()
    result = run_coding_task(
        CodingTask(objective="preserve changes"), ScriptedLLM([]), registry, recorder=recorder
    )
    assert stored(recorder).domain_status == "precondition_failed"
    assert stored(recorder).llm_calls == 0


def test_failing_sink_does_not_undo_edit(tmp_path):
    registry = create_editing_tool_registry(ToolContext(repository_root=tmp_path))

    def fail(trace):
        raise OSError("disk unavailable")

    recorder = InMemoryTraceRecorder(sink=fail)
    result = run_editing_agent(
        "create",
        ScriptedLLM(
            [
                tool("create_file", path="file.txt", content="retained"),
                final(),
            ]
        ),
        registry,
        recorder=recorder,
    )
    assert result.status == "completed"
    assert (tmp_path / "file.txt").read_text() == "retained"
    assert recorder.diagnostics


def test_model_text_structured_usage_and_no_double_count():
    recorder = InMemoryTraceRecorder()
    trace = TraceContext(recorder, "read_only_agent")
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="private response", parsed=final()),
                finish_reason="stop",
            )
        ],
        model="fake",
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120),
    )
    sdk = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=MagicMock(return_value=response))),
        beta=SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(parse=MagicMock(return_value=response))
            )
        ),
    )
    client = OpenAILLMClient(Settings(_env_file=None, openai_model="fake"), client=sdk)
    assert client.generate("private prompt", trace=trace).content == "private response"
    run_read_only_agent(
        "inspect",
        client,
        create_default_tool_registry(ToolContext(repository_root=Path.cwd())),
        trace=trace,
    )
    trace.finish()
    result = stored(recorder)
    assert result.llm_calls == 2
    assert result.token_usage.total_tokens == 240
    assert "private prompt" not in result.model_dump_json()
    assert "private response" not in result.model_dump_json()


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("exhausted", [False, True])
async def test_async_retry_never_uses_blocking_sleep(monkeypatch, structured, exhausted):
    import repomind.llm.client as module

    error = openai.APIConnectionError(request=MagicMock())
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="answer", parsed=final()), finish_reason="stop"
            )
        ],
        model="fake",
        usage=None,
    )
    method = AsyncMock(side_effect=[error, error if exhausted else response])
    sdk = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=method)),
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=method))),
    )
    sleeper = AsyncMock()
    monkeypatch.setattr(module.asyncio, "sleep", sleeper)
    monkeypatch.setattr(
        module.time, "sleep", MagicMock(side_effect=AssertionError("blocking sleep"))
    )
    recorder = InMemoryTraceRecorder()
    trace = TraceContext(recorder, "rag")
    client = OpenAILLMClient(Settings(_env_file=None, llm_max_retries=1), async_client=sdk)

    async def invoke():
        if structured:
            return await client.agenerate_structured("prompt", AgentDecision, trace=trace)
        return await client.agenerate("prompt", trace=trace)

    if exhausted:
        with pytest.raises(LLMError) as caught:
            await invoke()
        assert caught.value.__cause__ is error
    else:
        await invoke()
    trace.finish("error" if exhausted else "completed")
    result = stored(recorder)
    assert result.llm_calls == 1
    assert result.token_usage is None
    assert method.await_count == 2 and sleeper.await_count == 1
    assert len([e for e in result.events if e.event_type == "model.attempt"]) == 2
    assert any(
        e.event_type == ("model.failed" if exhausted else "model.completed") for e in result.events
    )


def test_embedding_usage_is_reported_without_vectors():
    recorder = InMemoryTraceRecorder()
    trace = TraceContext(recorder, "rag")
    sdk = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=MagicMock(
                return_value=SimpleNamespace(
                    data=[SimpleNamespace(index=0, embedding=[0.125, 0.875])],
                    model="fake",
                    usage=SimpleNamespace(prompt_tokens=8, total_tokens=8),
                )
            )
        )
    )
    client = OpenAIEmbeddingClient(Settings(_env_file=None), client=sdk, trace=trace)
    client.embed_text("private source content")
    trace.finish()
    result = stored(recorder)
    assert result.llm_calls == 1
    assert result.token_usage.prompt_tokens == 8
    assert result.token_usage.completion_tokens == 0
    assert "0.125" not in result.model_dump_json()
    assert "private source content" not in result.model_dump_json()


def test_hybrid_rag_metadata_and_noop_equivalence():
    chunk = CodeChunk(
        relative_path="config.py",
        language="python",
        start_line=1,
        end_line=1,
        chunk_index=0,
        content="PRIVATE_CONFIG = True\n",
    )
    embedded = [EmbeddedChunk(chunk=chunk, embedding=EmbeddingVector(values=(1, 0), model="fake"))]
    provider = SimpleNamespace(embed_text=lambda text: EmbeddingVector(values=(1, 0), model="fake"))
    index = BM25Index([chunk])

    def retrieve(query, *, top_k):
        return hybrid_search(query, embedded, index, provider, top_k=top_k)

    answer = GroundedLLMResponse(
        answer="Config is enabled", source_ids=["S1"], insufficient_evidence=False
    )
    baseline = answer_repository_question_with_retriever("config", retrieve, ScriptedLLM([answer]))
    recorder = InMemoryTraceRecorder()
    actual = answer_repository_question_with_retriever(
        "config", retrieve, ScriptedLLM([answer]), strategy="hybrid", recorder=recorder
    )
    assert baseline == actual
    trace = stored(recorder)
    retrieval = next(e for e in trace.events if e.event_type == "retrieval.completed")
    assert retrieval.metadata["strategy"] == "hybrid"
    assert retrieval.metadata["candidate_count"] == 1
    assert (
        next(e for e in trace.events if e.event_type == "rag.context").metadata[
            "context_chunk_count"
        ]
        == 1
    )
    assert (
        next(e for e in trace.events if e.event_type == "rag.answer").metadata["citation_count"]
        == 1
    )
    assert "PRIVATE_CONFIG" not in trace.model_dump_json()


def test_insufficient_evidence_is_completed_run():
    recorder = InMemoryTraceRecorder()
    answer = answer_repository_question_with_retriever(
        "unknown", lambda *a, **k: [], ScriptedLLM([]), recorder=recorder
    )
    assert answer.insufficient_evidence
    assert stored(recorder).status == "completed"
    assert stored(recorder).llm_calls == 0


def test_explicit_reranker_context_joins_answer_timeline():
    from repomind.retrieval import LLMReranker
    from repomind.retrieval.reranking import RerankLLMResponse

    chunk = CodeChunk(
        relative_path="app.py",
        language="python",
        start_line=1,
        end_line=1,
        chunk_index=0,
        content="private repository source",
    )
    recorder = InMemoryTraceRecorder()
    trace = TraceContext(recorder, "rag")
    reranker = LLMReranker(
        ScriptedLLM([RerankLLMResponse(ranked_candidate_ids=["C1"])]), trace=trace
    )

    def retrieve(query, *, top_k):
        return reranker.rerank(query, [SimpleNamespace(chunk=chunk, rank=1)], top_k=top_k)

    answer_repository_question_with_retriever(
        "query",
        retrieve,
        ScriptedLLM(
            [
                GroundedLLMResponse(
                    answer="answer", source_ids=["S1"], insufficient_evidence=False
                ),
            ]
        ),
        strategy="hybrid+rerank",
        trace=trace,
    )
    trace.finish()
    result = stored(recorder)
    assert len(recorder.traces) == 1
    assert result.llm_calls == 2
    assert "private repository source" not in result.model_dump_json()
