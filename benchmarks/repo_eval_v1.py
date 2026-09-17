"""Run the deterministic offline ``repo-eval-v1`` baseline benchmark."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel

from repomind.agent import AgentDecision, EditingAgentConfig
from repomind.coding import (
    CodingPlan,
    CodingReview,
    CodingTask,
    CodingTaskResult,
    CodingWorkflowConfig,
    VerificationPolicy,
    run_coding_task,
)
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
    format_coding_report,
    format_rag_comparison,
    format_retrieval_comparison,
)
from repomind.ingestion import CodeChunk
from repomind.rag import RAGConfig
from repomind.retrieval import (
    BM25Index,
    EmbeddedChunk,
    EmbeddingVector,
    RankedChunk,
    RerankedSearchResult,
    chunk_identity,
    hybrid_search,
    semantic_search,
)
from repomind.tools import ToolContext, create_editing_tool_registry

VERSION = "repo-eval-v1"
EXACT_QUERY = "OPENAI_EMBEDDING_MODEL"
RETRY_QUERY = "Where is retry behavior for failed model requests implemented?"


def _chunk(path: str, content: str, index: int) -> CodeChunk:
    return CodeChunk(
        relative_path=path,
        language="python",
        start_line=1,
        end_line=max(1, len(content.splitlines())),
        content=content,
        chunk_index=index,
    )


def _corpus() -> list[EmbeddedChunk]:
    chunks_and_vectors = [
        (
            _chunk(
                "src/llm/client.py",
                "async def retry_failed_request():\n    await asyncio.sleep(delay)\n",
                0,
            ),
            (0.0, 1.0, 0.0),
        ),
        (
            _chunk("tests/test_llm.py", "def test_failed_request_retry(): pass\n", 0),
            (0.0, 0.8, 0.6),
        ),
        (
            _chunk(
                "src/config.py",
                'OPENAI_EMBEDDING_MODEL = "fixture-model"\n',
                0,
            ),
            (0.55, 0.1, 0.83),
        ),
        (
            _chunk("src/embeddings.py", "def embed_chunks(texts): pass\n", 0),
            (0.95, 0.0, 0.31),
        ),
        (
            _chunk("src/db.py", "def persist_chunks_to_postgresql(): pass\n", 0),
            (0.0, 0.2, 0.98),
        ),
        (
            _chunk("src/hybrid.py", "def reciprocal_rank_fusion(): pass\n", 0),
            (0.3, 0.3, 0.9),
        ),
        (
            _chunk("web/styles.css", ".button { color: blue; }\n", 0),
            (0.1, 0.0, 0.99),
        ),
    ]
    return [
        EmbeddedChunk(
            chunk=chunk,
            embedding=EmbeddingVector(values=vector, model="offline-fixture-model"),
        )
        for chunk, vector in chunks_and_vectors
    ]


class _EmbeddingProvider:
    def embed_text(self, text: str) -> EmbeddingVector:
        values = (1.0, 0.0, 0.0) if text == EXACT_QUERY else (0.0, 1.0, 0.0)
        return EmbeddingVector(values=values, model="offline-fixture-model")


class _Reranker:
    def __call__(
        self,
        query: str,
        candidates: Sequence[RankedChunk],
        *,
        top_k: int,
    ) -> list[RerankedSearchResult]:
        preferred = "src/config.py" if query == EXACT_QUERY else "src/llm/client.py"
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                candidate.chunk.relative_path.as_posix() != preferred,
                candidate.rank,
            ),
        )[:top_k]
        return [
            RerankedSearchResult(
                chunk=candidate.chunk,
                rank=rank,
                original_rank=candidate.rank,
            )
            for rank, candidate in enumerate(ordered, start=1)
        ]


class _AnswerLLM:
    def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> BaseModel:
        sources = {
            path: source_id
            for source_id, path in re.findall(
                r'<source id="(S\d+)">.*?<path>(.*?)</path>',
                prompt,
                flags=re.DOTALL,
            )
        }
        if "src/config.py" in sources and EXACT_QUERY in prompt:
            return response_model(
                answer="OPENAI_EMBEDDING_MODEL selects the embedding model.",
                source_ids=[sources["src/config.py"]],
                insufficient_evidence=False,
            )
        if "src/llm/client.py" in sources and RETRY_QUERY in prompt:
            return response_model(
                answer="Async retry handling uses asyncio.sleep in llm/client.py.",
                source_ids=[sources["src/llm/client.py"]],
                insufficient_evidence=False,
            )
        return response_model(
            answer="The supplied context does not contain the required fact.",
            source_ids=[],
            insufficient_evidence=True,
        )


def _retrieval_inputs() -> tuple[
    RetrievalBenchmarkSuite,
    RAGBenchmarkSuite,
    dict[str, object],
]:
    corpus = _corpus()
    chunks = [item.chunk for item in corpus]
    by_path = {chunk.relative_path.as_posix(): chunk for chunk in chunks}
    bm25 = BM25Index.from_chunks(chunks)
    embedding_provider = _EmbeddingProvider()
    reranker = _Reranker()

    def semantic(query: str, *, top_k: int) -> Sequence[RankedChunk]:
        return semantic_search(query, corpus, embedding_provider, top_k=top_k)

    def lexical(query: str, *, top_k: int) -> Sequence[RankedChunk]:
        return bm25.search(query, top_k=top_k)

    def hybrid(query: str, *, top_k: int) -> Sequence[RankedChunk]:
        return hybrid_search(query, corpus, bm25, embedding_provider, top_k=top_k)

    def reranked(query: str, *, top_k: int) -> Sequence[RankedChunk]:
        candidates = hybrid_search(
            query,
            corpus,
            bm25,
            embedding_provider,
            top_k=len(corpus),
        )
        return reranker(query, candidates, top_k=top_k)

    retrieval_suite = RetrievalBenchmarkSuite(
        version=VERSION,
        cases=(
            RetrievalBenchmarkCase(
                id="exact-identifier",
                query=EXACT_QUERY,
                relevant_chunks=(chunk_identity(by_path["src/config.py"]),),
            ),
            RetrievalBenchmarkCase(
                id="natural-language-retry",
                query=RETRY_QUERY,
                relevant_chunks=(chunk_identity(by_path["src/llm/client.py"]),),
            ),
        ),
    )
    rag_suite = RAGBenchmarkSuite(
        version=VERSION,
        cases=(
            RAGBenchmarkCase(
                id="exact-identifier",
                question=EXACT_QUERY,
                relevant_chunks=(chunk_identity(by_path["src/config.py"]),),
                expected_answer_facts=("OPENAI_EMBEDDING_MODEL",),
            ),
            RAGBenchmarkCase(
                id="natural-language-retry",
                question=RETRY_QUERY,
                relevant_chunks=(chunk_identity(by_path["src/llm/client.py"]),),
                expected_answer_facts=("asyncio.sleep", "llm/client.py"),
            ),
        ),
    )
    return retrieval_suite, rag_suite, {
        "semantic": semantic,
        "bm25": lexical,
        "hybrid": hybrid,
        "hybrid+rerank": reranked,
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tool(name: str, **arguments: object) -> AgentDecision:
    return AgentDecision(action="tool", tool_name=name, tool_arguments=arguments)


def _final(message: str = "Ready for completion.") -> AgentDecision:
    return AgentDecision(action="final", final_answer=message)


class _ScriptedLLM:
    def __init__(self, responses: list[AgentDecision]) -> None:
        self.responses = responses

    def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> BaseModel:
        if response_model is CodingPlan:
            payload = json.loads(prompt)
            criteria = payload["task"]["acceptance_criteria"]
            return CodingPlan.model_validate(
                {
                    "task_summary": "Implement and verify the visible benchmark task.",
                    "steps": [
                        {
                            "step_id": 1,
                            "action": "Inspect, implement, and verify the requested change.",
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
                            "evidence": "Visible verification and bounded diff evidence pass.",
                        }
                        for index in range(len(criteria))
                    ],
                }
            )
        return self.responses.pop(0)


class _CodingRunner:
    def __call__(
        self,
        task: CodingTask,
        workspace: Path,
        verification_policy: VerificationPolicy,
    ) -> CodingTaskResult:
        original = (workspace / "app.py").read_bytes()
        if task.objective == "Correct add by returning the sum.":
            responses = [
                _tool(
                    "replace_text",
                    path="app.py",
                    old_text="return a - b",
                    new_text="return a + b",
                    expected_sha256=_sha256(original),
                ),
                _final(),
            ]
        elif task.objective == "Make the existing visible add test pass.":
            responses = [
                _tool(
                    "replace_text",
                    path="app.py",
                    old_text="return a - b",
                    new_text="return 5",
                    expected_sha256=_sha256(original),
                ),
                _final(),
            ]
        elif task.objective == "Correct add for arbitrary numeric inputs.":
            responses = [_final(), _final("Still requesting completion.")]
        else:
            incorrect = b"def add(a, b):\n    return 0\n"
            responses = [
                _tool(
                    "replace_text",
                    path="app.py",
                    old_text="return a - b",
                    new_text="return 0",
                    expected_sha256=_sha256(original),
                ),
                _final("First attempt ready."),
                _tool(
                    "replace_text",
                    path="app.py",
                    old_text="return 0",
                    new_text="return a + b",
                    expected_sha256=_sha256(incorrect),
                ),
                _final("Corrected after verification feedback."),
            ]
        return run_coding_task(
            task,
            _ScriptedLLM(responses),
            create_editing_tool_registry(ToolContext(repository_root=workspace)),
            verification_policy=verification_policy,
            agent_config=EditingAgentConfig(max_iterations=len(responses)),
            workflow_config=CodingWorkflowConfig(max_completion_attempts=2),
        )


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def _coding_suite(template: Path) -> CodingBenchmarkSuite:
    required = (Path("app.py"),)
    allowed = (Path("app.py"),)
    expected = (FileTextExpectation(path="app.py", text="return a + b"),)
    objectives = (
        ("coding-success", "Correct add by returning the sum.", False),
        ("false-positive", "Make the existing visible add test pass.", False),
        ("verification-failure", "Correct add for arbitrary numeric inputs.", False),
        ("self-correction", "Recover from verification feedback and correct add.", True),
    )
    return CodingBenchmarkSuite(
        version=VERSION,
        cases=tuple(
            CodingBenchmarkCase(
                id=case_id,
                fixture_repository=template,
                task=CodingTask(objective=objective),
                required_changed_paths=required,
                allowed_changed_paths=allowed,
                file_contains=expected,
                expects_recovery=expects_recovery,
            )
            for case_id, objective, expects_recovery in objectives
        ),
    )


def _create_coding_template(root: Path) -> Path:
    template = root / "coding-template"
    template.mkdir()
    (template / ".gitignore").write_text(
        "__pycache__/\n.pytest_cache/\n.ruff_cache/\n*.pyc\n",
        encoding="utf-8",
    )
    (template / "app.py").write_bytes(b"def add(a, b):\n    return a - b\n")
    (template / "tests").mkdir()
    (template / "tests" / "test_app.py").write_text(
        "from app import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    _git(template, "init", "-b", "main")
    _git(template, "config", "user.email", "benchmark@example.invalid")
    _git(template, "config", "user.name", "RepoMind Benchmark")
    _git(template, "add", ".")
    _git(template, "commit", "-m", "benchmark fixture")
    return template


def main() -> None:
    """Run and print the bounded offline fixture/scripted baseline."""

    retrieval_suite, rag_suite, strategies = _retrieval_inputs()
    retrieval_reports = [
        evaluate_retrieval(retrieval_suite, name, strategy, k=3)
        for name, strategy in strategies.items()
    ]
    rag_reports = [
        evaluate_rag(
            rag_suite,
            name,
            strategies[name],
            _AnswerLLM(),
            config=RAGConfig(top_k=1),
        )
        for name in ("semantic", "hybrid", "hybrid+rerank")
    ]
    with TemporaryDirectory(prefix="repomind-offline-benchmark-") as temporary:
        template = _create_coding_template(Path(temporary))
        coding_report = evaluate_coding_suite(_coding_suite(template), _CodingRunner())

    print("REPO-EVAL-V1 OFFLINE FIXTURE RETRIEVAL")
    print(format_retrieval_comparison(retrieval_reports))
    print("\nREPO-EVAL-V1 OFFLINE FIXTURE RAG")
    print(format_rag_comparison(rag_reports))
    print("\nREPO-EVAL-V1 OFFLINE SCRIPTED CODING")
    print(format_coding_report(coding_report))
    print(
        "\nThese deterministic fake embeddings and scripted model decisions test "
        "evaluation infrastructure, not real OpenAI model quality."
    )


if __name__ == "__main__":
    main()
