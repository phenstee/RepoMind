"""Tests for context-assembly evaluation, kept separate from retrieval ranking."""

from repomind.evaluation import (
    ContextAssemblyBenchmarkCase,
    ContextAssemblyBenchmarkSuite,
    evaluate_context_assembly,
    format_context_assembly_comparison,
)
from repomind.ingestion import ChunkingStrategy, ChunkKind, CodeChunk
from repomind.rag import ContextAssemblyConfig, ContextStrategy
from repomind.rag.assembly import InMemoryNeighborLoader
from repomind.retrieval import SemanticSearchResult, chunk_identity


def _chunk(index: int, *, start_line: int, content: str) -> CodeChunk:
    line_count = max(1, len(content.splitlines()))
    return CodeChunk(
        relative_path="src/a.py",
        language="python",
        start_line=start_line,
        end_line=start_line + line_count - 1,
        content=content,
        chunk_index=index,
        chunking_strategy=ChunkingStrategy.LINE,
        chunk_kind=ChunkKind.LINE,
    )


def _corpus() -> list[CodeChunk]:
    return [_chunk(i, start_line=i * 5 + 1, content=f"body {i}\n" * 3) for i in range(4)]


def _retriever(seed_index: int):
    corpus = _corpus()

    def retrieve(query: str, *, top_k: int):
        return [SemanticSearchResult(chunk=corpus[seed_index], score=0.9, rank=1)][:top_k]

    return retrieve, corpus


def test_expanded_strategy_improves_gold_coverage_over_seeds_only() -> None:
    retriever, corpus = _retriever(seed_index=1)
    needed_neighbor = corpus[2]
    suite = ContextAssemblyBenchmarkSuite(
        cases=(
            ContextAssemblyBenchmarkCase(
                id="needs-neighbor",
                query="anything",
                relevant_chunks=(chunk_identity(needed_neighbor),),
            ),
        )
    )
    loader = InMemoryNeighborLoader(corpus)

    seeds_only = evaluate_context_assembly(
        suite,
        retriever,
        top_k=1,
        config=ContextAssemblyConfig(strategy=ContextStrategy.SEEDS_ONLY),
        neighbor_loader=loader,
    )
    expanded = evaluate_context_assembly(
        suite,
        retriever,
        top_k=1,
        config=ContextAssemblyConfig(strategy=ContextStrategy.EXPANDED, neighbor_radius=1),
        neighbor_loader=loader,
    )

    assert seeds_only.mean_gold_evidence_coverage == 0.0
    assert expanded.mean_gold_evidence_coverage == 1.0
    assert expanded.case_results[0].deduplicated_count == 0

    report_text = format_context_assembly_comparison([seeds_only, expanded])
    assert "seeds_only" in report_text
    assert "expanded" in report_text


def test_budget_utilization_and_dropped_count_are_reported() -> None:
    retriever, corpus = _retriever(seed_index=1)
    suite = ContextAssemblyBenchmarkSuite(
        cases=(
            ContextAssemblyBenchmarkCase(
                id="tight-budget",
                query="anything",
                relevant_chunks=(chunk_identity(corpus[1]),),
            ),
        )
    )
    loader = InMemoryNeighborLoader(corpus)

    report = evaluate_context_assembly(
        suite,
        retriever,
        top_k=1,
        config=ContextAssemblyConfig(
            strategy=ContextStrategy.EXPANDED, neighbor_radius=1, budget_tokens=1
        ),
        neighbor_loader=loader,
    )

    result = report.case_results[0]
    assert result.packed_count == 1
    assert result.dropped_for_budget_count == 2
    assert result.budget_utilization > 1.0
