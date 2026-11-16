"""Run ``repo-eval-v5``: real-question retrieval over a real repository.

Earlier suites (``repo-eval-v1``-``v4``) use small synthetic fixtures with a
handful of queries. This one asks 44 developer questions (plus a 16-question
held-out set) about RepoMind's own source tree, frozen at a fixed commit, and
scores two things:

1. **Ranking** - does a chunk that actually contains the answer appear in the
   top k? (hit@k, target recall@k, MRR)
2. **Evidence in context** - after RAG context packing under the production
   character budget, does that evidence actually reach the model?

Gold labels are source spans, not chunk IDs, so line and structural chunking
are compared on equal terms (see ``repomind.evaluation.spans``).

Offline mode (default) is deterministic and makes no model request: it
measures the lexical (BM25) and symbol retrieval arms and context packing.
Semantic and fused strategies need real embeddings, so they only run with
``--live --confirm-live`` and a configured ``OPENAI_API_KEY``, and that mode
COSTS MONEY (roughly 1.2M embedding tokens per run, a few cents with
``text-embedding-3-small``).

Run from the repository root (needs ``git`` and the corpus commit in history):

    uv run python -m benchmarks.repo_eval_v5
    uv run python -m benchmarks.repo_eval_v5 --held-out
    uv run python -m benchmarks.repo_eval_v5 --live --confirm-live
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.repo_eval_v5_cases import CORPUS_COMMIT, HELD_OUT_SUITE, SUITE
from repomind.evaluation.spans import (
    GoldSpan,
    SpanBenchmarkSuite,
    SpanRetrievalReport,
    covered_spans,
    evaluate_span_retrieval,
    resolve_suite,
)
from repomind.ingestion import (
    ChunkingConfig,
    ChunkingStrategy,
    CodeChunk,
    RepositorySnapshot,
    chunk_repository,
    ingest_repository,
)
from repomind.rag import ContextAssemblyConfig, ContextStrategy, assemble_context
from repomind.rag.assembly import InMemoryNeighborLoader
from repomind.rag.context import build_repository_context
from repomind.rag.models import RAGConfig
from repomind.retrieval import (
    LEGACY_BM25_CONFIG,
    BM25Config,
    BM25Index,
    EmbeddedChunk,
    EmbeddingVector,
    RankedChunk,
    extract_identifier_candidates,
    fuse_ranked_sources,
    hybrid_search,
    hybrid_symbol_search,
    semantic_search,
    symbol_search,
)
from repomind.retrieval.symbols import (
    is_exact_identifier_query,
    require_weak_match_corroboration,
    select_evidence_backed_symbol_fragments,
)

Retriever = Callable[..., Sequence[RankedChunk]]
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_corpus(commit: str = CORPUS_COMMIT) -> RepositorySnapshot:
    """Ingest the repository exactly as it was at ``commit``."""

    archive = subprocess.run(
        ["git", "archive", "--format=tar", commit],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
        timeout=120,
    ).stdout
    with TemporaryDirectory(prefix="repomind-eval-v5-") as directory:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(directory, filter="data")
        return ingest_repository(directory)


def lexical_symbol_search(
    query: str,
    chunks: Sequence[CodeChunk],
    index: BM25Index,
    *,
    top_k: int,
    corroborate_weak: bool = True,
) -> list[RankedChunk]:
    """``hybrid_symbol_search`` without its semantic arm (BM25 + symbol RRF).

    ``corroborate_weak=False`` reproduces the symbol arm before weak
    (common-word) matches had to be corroborated by retrieval evidence.
    """

    lexical = index.search(query, top_k=top_k * 4)
    candidates = extract_identifier_candidates(query)
    symbols = symbol_search(candidates, chunks)
    if symbols and not is_exact_identifier_query(query, candidates):
        evidence = fuse_ranked_sources({"lexical": lexical}, top_k=max(1, len(lexical)))
        symbols = select_evidence_backed_symbol_fragments(symbols, evidence)
        if corroborate_weak:
            symbols = require_weak_match_corroboration(symbols, evidence)
    return list(fuse_ranked_sources({"lexical": lexical, "symbol": symbols}, top_k=top_k))


def offline_retrievers(
    chunks: Sequence[CodeChunk], strategy: ChunkingStrategy
) -> dict[str, Retriever]:
    legacy = BM25Index.from_chunks(chunks, LEGACY_BM25_CONFIG)
    current = BM25Index.from_chunks(chunks, BM25Config())
    retrievers: dict[str, Retriever] = {
        "bm25 (legacy)": lambda query, *, top_k: legacy.search(query, top_k=top_k),
        "bm25": lambda query, *, top_k: current.search(query, top_k=top_k),
    }
    if strategy is ChunkingStrategy.STRUCTURAL:
        retrievers["bm25 (legacy) + symbol"] = lambda query, *, top_k: lexical_symbol_search(
            query, chunks, legacy, top_k=top_k, corroborate_weak=False
        )
        retrievers["bm25 + symbol"] = lambda query, *, top_k: lexical_symbol_search(
            query, chunks, current, top_k=top_k
        )
    return retrievers


class _CachedQueryEmbedder:
    """Embed each distinct query once across every live strategy."""

    def __init__(self, client) -> None:
        self.client = client
        self._cache: dict[str, EmbeddingVector] = {}

    def embed_text(self, text: str) -> EmbeddingVector:
        if text not in self._cache:
            self._cache[text] = self.client.embed_text(text)
        return self._cache[text]


def live_retrievers(
    chunks: Sequence[CodeChunk],
    embedded: Sequence[EmbeddedChunk],
    embedder: _CachedQueryEmbedder,
    *,
    bm25_config: BM25Config,
) -> dict[str, Retriever]:
    index = BM25Index.from_chunks(chunks, bm25_config)
    return {
        "semantic": lambda query, *, top_k: semantic_search(query, embedded, embedder, top_k=top_k),
        "hybrid": lambda query, *, top_k: hybrid_search(
            query, embedded, index, embedder, top_k=top_k
        ),
        "hybrid_symbol": lambda query, *, top_k: hybrid_symbol_search(
            query, embedded, index, embedder, chunks, top_k=top_k
        ),
    }


@dataclass(frozen=True)
class ContextCoverage:
    retriever: str
    chunking: str
    top_k: int
    context_strategy: str
    hit_rate: float
    target_recall: float
    mean_context_chars: float
    mean_sources: float


def context_coverage(
    suite: SpanBenchmarkSuite,
    gold: Mapping[str, Sequence[GoldSpan]],
    retriever: Retriever,
    chunks: Sequence[CodeChunk],
    *,
    name: str,
    chunking: str,
    top_k: int,
    context_strategy: ContextStrategy,
) -> ContextCoverage:
    """Pack each case's results exactly as RAG does and check what survives."""

    config = RAGConfig(top_k=top_k, context_strategy=context_strategy)
    loader = InMemoryNeighborLoader(chunks)
    hits = recall = chars = sources = 0.0
    for case in suite.cases:
        results = retriever(case.query, top_k=top_k)
        if context_strategy is ContextStrategy.EXPANDED:
            assembled = assemble_context(
                results,
                loader,
                ContextAssemblyConfig(
                    strategy=context_strategy,
                    budget_tokens=config.context_budget_tokens,
                    neighbor_radius=config.neighbor_radius,
                ),
            )
            context = build_repository_context(
                assembled.chunks, max_context_chars=config.max_context_chars
            )
        else:
            context = build_repository_context(results, max_context_chars=config.max_context_chars)
        spans = gold[case.id]
        covered = covered_spans([source.chunk for source in context.sources], spans)
        hits += bool(covered)
        recall += len(covered) / len(spans)
        chars += len(context.text)
        sources += len(context.sources)
    count = len(suite.cases)
    return ContextCoverage(
        retriever=name,
        chunking=chunking,
        top_k=top_k,
        context_strategy=context_strategy.value,
        hit_rate=hits / count,
        target_recall=recall / count,
        mean_context_chars=chars / count,
        mean_sources=sources / count,
    )


def _ranking_rows(
    reports: Sequence[tuple[str, SpanRetrievalReport, SpanRetrievalReport]],
) -> list[str]:
    categories = sorted(
        {result.category for _, report, _ in reports for result in report.case_results}
    )
    header = (
        "| Chunking | Retriever | Hit@5 | Recall@5 | MRR@5 | Hit@10 | Recall@10 | "
        + " | ".join(f"{category} hit@5" for category in categories)
        + " |"
    )
    rows = [header, "|" + " --- |" * (header.count("|") - 1)]
    for chunking, at5, at10 in reports:
        by_category = at5.by_category()
        rows.append(
            f"| {chunking} | {at5.strategy} | {at5.hit_rate:.3f} | {at5.mean_target_recall:.3f} "
            f"| {at5.mrr:.3f} | {at10.hit_rate:.3f} | {at10.mean_target_recall:.3f} | "
            + " | ".join(f"{by_category[category].hit_rate:.3f}" for category in categories)
            + " |"
        )
    return rows


def _coverage_rows(coverages: Sequence[ContextCoverage]) -> list[str]:
    rows = [
        (
            "| Chunking | Retriever | top_k | Context | Evidence hit | Evidence recall "
            "| Mean context chars | Mean sources |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows.extend(
        f"| {c.chunking} | {c.retriever} | {c.top_k} | {c.context_strategy} | {c.hit_rate:.3f} "
        f"| {c.target_recall:.3f} | {c.mean_context_chars:,.0f} | {c.mean_sources:.1f} |"
        for c in coverages
    )
    return rows


COVERAGE_TOP_K = {ChunkingStrategy.LINE: (5,), ChunkingStrategy.STRUCTURAL: (5, 10, 15, 20)}


def run(
    *,
    held_out: bool,
    live: bool,
    confirm_live: bool,
    output: Path | None,
) -> str:
    suite = HELD_OUT_SUITE if held_out else SUITE
    live_client = None
    if live:
        from repomind.config import get_settings
        from repomind.evaluation import require_live_authorization

        settings = get_settings()
        require_live_authorization(settings, confirm_live=confirm_live)
        from repomind.retrieval import OpenAIEmbeddingClient

        live_client = OpenAIEmbeddingClient(settings)

    snapshot = load_corpus()
    gold = resolve_suite(suite, snapshot)
    ranking: list[tuple[str, SpanRetrievalReport, SpanRetrievalReport]] = []
    coverages: list[ContextCoverage] = []
    embedder = _CachedQueryEmbedder(live_client) if live_client is not None else None

    for chunking in ChunkingStrategy:
        chunks = chunk_repository(snapshot, ChunkingConfig(strategy=chunking))
        retrievers = offline_retrievers(chunks, chunking)
        if embedder is not None:
            embedded = live_client.embed_chunks(chunks)
            retrievers |= live_retrievers(chunks, embedded, embedder, bm25_config=BM25Config())
        for name, retriever in retrievers.items():
            ranking.append(
                (
                    chunking.value,
                    evaluate_span_retrieval(suite, gold, name, retriever, k=5),
                    evaluate_span_retrieval(suite, gold, name, retriever, k=10),
                )
            )
        coverage_names = (
            ["bm25 (legacy)", "bm25"]
            + (["bm25 + symbol"] if chunking is ChunkingStrategy.STRUCTURAL else [])
            + (["semantic", "hybrid", "hybrid_symbol"] if embedder is not None else [])
        )
        for name in coverage_names:
            for top_k in COVERAGE_TOP_K[chunking]:
                for context_strategy in ContextStrategy:
                    coverages.append(
                        context_coverage(
                            suite,
                            gold,
                            retrievers[name],
                            chunks,
                            name=name,
                            chunking=chunking.value,
                            top_k=top_k,
                            context_strategy=context_strategy,
                        )
                    )

    mode = "live_model" if live else "offline"
    lines = [
        f"# {suite.version} ({mode}, {len(suite.cases)} cases, corpus {CORPUS_COMMIT[:7]})",
        "",
        "## Ranking",
        "",
        *_ranking_rows(ranking),
        "",
        "## Evidence reaching the model (RAG packing, default 20,000-char budget)",
        "",
        *_coverage_rows(coverages),
    ]
    report = "\n".join(lines)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "benchmark_version": suite.version,
                    "mode": mode,
                    "corpus_commit": CORPUS_COMMIT,
                    "ranking": [
                        {
                            "chunking": chunking,
                            "retriever": at5.strategy,
                            "at_5": at5.model_dump(mode="json"),
                            "at_10": at10.model_dump(mode="json"),
                        }
                        for chunking, at5, at10 in ranking
                    ],
                    "context_coverage": [coverage.__dict__ for coverage in coverages],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--held-out", action="store_true", help="run the held-out question set")
    parser.add_argument(
        "--live",
        action="store_true",
        help="also run semantic and fused strategies with real OpenAI embeddings (costs money)",
    )
    parser.add_argument("--confirm-live", action="store_true", help="required with --live")
    parser.add_argument("--output", type=Path, help="also write the full per-case report as JSON")
    args = parser.parse_args(argv)
    print(
        run(
            held_out=args.held_out,
            live=args.live,
            confirm_live=args.confirm_live,
            output=args.output,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
