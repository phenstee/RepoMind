"""Offline end-to-end tests for retrieval, context, and RAG answer evaluation."""

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from repomind.evaluation import (
    RAGBenchmarkCase,
    RAGBenchmarkSuite,
    evaluate_rag,
)
from repomind.ingestion import CodeChunk
from repomind.rag import RAGConfig
from repomind.retrieval import (
    HybridSearchResult,
    RankedChunk,
    RerankedSearchResult,
    SemanticSearchResult,
    chunk_identity,
)

EXACT_QUESTION = "Which setting selects the embedding model?"
RETRY_QUESTION = "Where is retry behavior for failed model requests implemented?"


def _chunk(path: str, content: str, *, index: int = 0, start: int = 1) -> CodeChunk:
    return CodeChunk(
        relative_path=path,
        language="python",
        start_line=start,
        end_line=start + max(1, len(content.splitlines())) - 1,
        content=content,
        chunk_index=index,
    )


class _ContextAwareAnswerLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> BaseModel:
        self.prompts.append(prompt)
        if "<path>src/config.py</path>" in prompt:
            return response_model(
                answer="OPENAI_EMBEDDING_MODEL selects the embedding model.",
                source_ids=["S1"],
                insufficient_evidence=False,
            )
        if "<path>src/llm/client.py</path>" in prompt:
            return response_model(
                answer="Async retries use asyncio.sleep in llm/client.py.",
                source_ids=["S1"],
                insufficient_evidence=False,
            )
        return response_model(
            answer="The supplied context does not contain the required fact.",
            source_ids=[],
            insufficient_evidence=True,
        )


def _suite(config_chunk: CodeChunk, retry_chunk: CodeChunk) -> RAGBenchmarkSuite:
    return RAGBenchmarkSuite(
        cases=(
            RAGBenchmarkCase(
                id="embedding-setting",
                question=EXACT_QUESTION,
                relevant_chunks=(chunk_identity(config_chunk),),
                expected_answer_facts=("OPENAI_EMBEDDING_MODEL",),
            ),
            RAGBenchmarkCase(
                id="async-retry",
                question=RETRY_QUESTION,
                relevant_chunks=(chunk_identity(retry_chunk),),
                expected_answer_facts=("asyncio.sleep", "llm/client.py"),
            ),
        )
    )


def test_same_questions_compare_semantic_hybrid_and_reranked_public_rag() -> None:
    distractor = _chunk("src/embeddings.py", "def embed_text(): pass\n")
    config_chunk = _chunk(
        "src/config.py",
        'OPENAI_EMBEDDING_MODEL = "fixture-model"\n',
        start=12,
    )
    retry_chunk = _chunk(
        "src/llm/client.py",
        "async def retry():\n    await asyncio.sleep(delay)\n",
        start=40,
    )
    suite = _suite(config_chunk, retry_chunk)

    def semantic(question: str, *, top_k: int) -> Sequence[RankedChunk]:
        chunk = distractor if question == EXACT_QUESTION else retry_chunk
        return [SemanticSearchResult(chunk=chunk, score=1.0, rank=1)][:top_k]

    def hybrid(question: str, *, top_k: int) -> Sequence[RankedChunk]:
        chunk = config_chunk if question == EXACT_QUESTION else retry_chunk
        return [
            HybridSearchResult(
                chunk=chunk,
                rank=1,
                fusion_score=0.03,
                semantic_rank=1,
                lexical_rank=1,
            )
        ][:top_k]

    def reranked(question: str, *, top_k: int) -> Sequence[RankedChunk]:
        chunk = config_chunk if question == EXACT_QUESTION else retry_chunk
        return [RerankedSearchResult(chunk=chunk, rank=1, original_rank=3)][:top_k]

    reports = [
        evaluate_rag(
            suite,
            "semantic",
            semantic,
            _ContextAwareAnswerLLM(),
            config=RAGConfig(top_k=1),
        ),
        evaluate_rag(
            suite,
            "hybrid",
            hybrid,
            _ContextAwareAnswerLLM(),
            config=RAGConfig(top_k=1),
        ),
        evaluate_rag(
            suite,
            "hybrid+rerank",
            reranked,
            _ContextAwareAnswerLLM(),
            config=RAGConfig(top_k=1),
        ),
    ]

    assert [report.answer_pass_rate for report in reports] == [0.5, 1.0, 1.0]
    assert reports[0].mean_retrieval_recall == 0.5
    assert reports[1].mean_context_recall == 1.0
    assert reports[2].mean_citation_recall == 1.0
    citation = reports[1].case_results[0].cited_chunk_ids[0]
    assert citation == chunk_identity(config_chunk)
    assert reports[1].model_dump(mode="json")["mode"] == "offline_fixture"


def test_context_recall_identifies_budget_loss_after_successful_retrieval() -> None:
    distractor = _chunk("src/large.py", "x" * 500)
    target = _chunk(
        "src/config.py",
        'OPENAI_EMBEDDING_MODEL = "fixture-model"\n',
        start=12,
    )
    suite = RAGBenchmarkSuite(
        cases=(
            RAGBenchmarkCase(
                id="budget-loss",
                question=EXACT_QUESTION,
                relevant_chunks=(chunk_identity(target),),
                expected_answer_facts=("OPENAI_EMBEDDING_MODEL",),
            ),
        )
    )

    def retriever(question: str, *, top_k: int) -> Sequence[RankedChunk]:
        return [
            SemanticSearchResult(chunk=distractor, score=1.0, rank=1),
            SemanticSearchResult(chunk=target, score=0.9, rank=2),
        ][:top_k]

    report = evaluate_rag(
        suite,
        "semantic",
        retriever,
        _ContextAwareAnswerLLM(),
        config=RAGConfig(top_k=2, max_context_chars=100),
    )
    result = report.case_results[0]

    assert result.retrieval_recall == 1.0
    assert result.context_recall == 0.0
    assert result.citation_recall == 0.0
    assert result.answer_passed is False
    assert result.missing_required_facts == ("OPENAI_EMBEDDING_MODEL",)
    assert result.context_chunk_ids == (chunk_identity(distractor),)


def test_rag_result_retains_real_citation_path_and_lines() -> None:
    target = _chunk("src/llm/client.py", "await asyncio.sleep(delay)\n", start=73)
    suite = RAGBenchmarkSuite(
        cases=(
            RAGBenchmarkCase(
                id="citation",
                question=RETRY_QUESTION,
                relevant_chunks=(chunk_identity(target),),
                expected_answer_facts=("asyncio.sleep",),
            ),
        )
    )

    report = evaluate_rag(
        suite,
        "hybrid+rerank",
        lambda question, *, top_k: [
            RerankedSearchResult(chunk=target, rank=1, original_rank=3)
        ],
        _ContextAwareAnswerLLM(),
    )

    result = report.case_results[0]
    assert result.cited_chunk_ids == (("src/llm/client.py", 0, 73, 73),)
    assert Path(result.cited_chunk_ids[0][0]) == Path("src/llm/client.py")
