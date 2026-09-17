from pathlib import Path
from types import SimpleNamespace

import pytest

from repomind.coding import CodingTask, run_coding_task
from repomind.evaluation import (
    CodingBenchmarkCase,
    CodingBenchmarkSuite,
    FileTextExpectation,
    RAGBenchmarkCase,
    RAGBenchmarkSuite,
    RetrievalBenchmarkCase,
    RetrievalBenchmarkSuite,
    evaluate_coding_suite,
    evaluate_rag,
    evaluate_retrieval,
)
from repomind.ingestion import CodeChunk
from repomind.observability import InMemoryTraceRecorder
from repomind.rag.models import GroundedLLMResponse
from repomind.tools import ToolContext, create_editing_tool_registry

from .test_instrumentation import ScriptedLLM, final, project, replacement, stored


def test_retrieval_case_link_and_compact_metrics():
    suite = RetrievalBenchmarkSuite(
        cases=[
            RetrievalBenchmarkCase(
                id="exact", query="private query", relevant_chunks=(("app.py", 0, 1, 1),)
            )
        ]
    )
    recorder = InMemoryTraceRecorder()
    report = evaluate_retrieval(suite, "semantic", lambda *a, **k: [], k=1, recorder=recorder)
    case = report.case_results[0]
    assert case.trace_run_id == stored(recorder).run_id
    assert stored(recorder).evaluation_summary["mrr"] == 0
    event = next(e for e in stored(recorder).events if e.event_type == "evaluation.case.completed")
    assert event.metadata["case_id"] == "exact"
    assert event.metadata["recall_at_k"] == 0
    assert "private query" not in stored(recorder).model_dump_json()
    plain = evaluate_retrieval(suite, "semantic", lambda *a, **k: [], k=1)
    assert plain.case_results[0].trace_run_id is None
    assert plain.mrr == report.mrr


def test_evaluation_case_exception_preserved_and_trace_identifies_case():
    suite = RetrievalBenchmarkSuite(
        cases=[
            RetrievalBenchmarkCase(
                id="broken", query="private query", relevant_chunks=(("app.py", 0, 1, 1),)
            )
        ]
    )
    failure = RuntimeError("Bearer do-not-store")

    def fail(*a, **k):
        raise failure

    recorder = InMemoryTraceRecorder()
    with pytest.raises(RuntimeError) as caught:
        evaluate_retrieval(suite, "semantic", fail, k=1, recorder=recorder)
    assert caught.value is failure
    trace = stored(recorder)
    event = next(e for e in trace.events if e.event_type == "evaluation.case.failed")
    assert event.metadata["case_id"] == "broken"
    assert trace.status == "failed"
    assert "do-not-store" not in trace.model_dump_json()


def test_rag_evaluation_uses_same_timeline():
    suite = RAGBenchmarkSuite(
        cases=[
            RAGBenchmarkCase(
                id="config",
                question="enabled?",
                relevant_chunks=(("app.py", 0, 1, 1),),
                expected_answer_facts=("enabled",),
            )
        ]
    )
    chunk = CodeChunk(
        relative_path="app.py",
        language="python",
        start_line=1,
        end_line=1,
        chunk_index=0,
        content="PRIVATE CONFIG",
    )
    recorder = InMemoryTraceRecorder()
    report = evaluate_rag(
        suite,
        "hybrid",
        lambda *a, **k: [SimpleNamespace(chunk=chunk, rank=1)],
        ScriptedLLM(
            [GroundedLLMResponse(answer="enabled", source_ids=["S1"], insufficient_evidence=False)]
        ),
        recorder=recorder,
    )
    trace = stored(recorder)
    assert len(recorder.traces) == 1
    assert report.case_results[0].trace_run_id == trace.run_id
    assert trace.llm_calls == 1
    assert any(e.event_type == "rag.answer" for e in trace.events)
    assert trace.evaluation_summary["answer_pass_rate"] == 1


def test_coding_case_links_hidden_oracle_to_actual_workflow(tmp_path):
    project(tmp_path)
    suite = CodingBenchmarkSuite(
        cases=[
            CodingBenchmarkCase(
                id="return_two",
                fixture_repository=tmp_path,
                task=CodingTask(objective="Return two"),
                file_contains=(FileTextExpectation(path=Path("app.py"), text="return 2"),),
            )
        ]
    )

    def runner(task, workspace, verification_policy, *, trace=None):
        return run_coding_task(
            task,
            ScriptedLLM([replacement(0, 2), final()]),
            create_editing_tool_registry(ToolContext(repository_root=workspace)),
            verification_policy=verification_policy,
            trace=trace,
        )

    recorder = InMemoryTraceRecorder()
    report = evaluate_coding_suite(suite, runner, recorder=recorder)
    trace = stored(recorder)
    assert report.case_results[0].task_success
    assert report.case_results[0].trace_run_id == trace.run_id
    assert trace.run_type == "evaluation"
    assert trace.llm_calls == 4 and trace.successful_mutations == 1
    assert trace.evaluation_summary["task_success_rate"] == 1
    assert any(e.event_type == "completion.completed" for e in trace.events)
    assert any(e.event_type == "planning.completed" for e in trace.events)
    assert any(e.event_type == "review.completed" for e in trace.events)
    assert "return 2" not in trace.model_dump_json()
    assert (tmp_path / "app.py").read_bytes() == b"def value():\n    return 0\n"
