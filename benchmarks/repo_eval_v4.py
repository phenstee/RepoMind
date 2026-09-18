"""Run the deterministic offline ``repo-eval-v4`` symbol-fusion benchmark.

Milestone 24 adds symbol retrieval as a third, separately-measurable
candidate source alongside semantic and BM25. This benchmark uses a small
but deliberately *realistic* multi-file corpus with the kind of ambiguity a
single-file fixture cannot exercise: two distinct ``login`` methods on two
different classes, common-English-word method names (``run``/``get``/``set``/
``test``/``save``/``load``), snake_case/camelCase/PascalCase identifiers, a
nested function, an oversized method split into structural fragments, and
unrelated neighboring code.

No live OpenAI calls: embeddings are deterministic hashed vectors, exactly
like ``repo_eval_v2``.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

from repomind.evaluation import (
    RetrievalBenchmarkCase,
    RetrievalBenchmarkSuite,
    evaluate_retrieval,
    format_retrieval_comparison,
)
from repomind.ingestion import (
    ChunkingConfig,
    ChunkingStrategy,
    CodeChunk,
    SourceFile,
    chunk_source_file,
)
from repomind.rag import ContextAssemblyConfig, ContextStrategy, assemble_context
from repomind.rag.assembly import InMemoryNeighborLoader
from repomind.retrieval import (
    BM25Index,
    EmbeddedChunk,
    EmbeddingTextStrategy,
    EmbeddingVector,
    RankedChunk,
    RerankedSearchResult,
    chunk_identity,
    embedding_text_for_chunk,
    extract_identifier_candidates,
    hybrid_search,
    hybrid_symbol_search,
    semantic_search,
    symbol_search,
    tokenize_code,
)

VERSION = "repo-eval-v4"

_FILES = {
    "src/auth/service.py": '''import secrets

MAX_LOGIN_ATTEMPTS = 5


def normalize_credentials(username: str) -> str:
    """Normalize user input before authentication."""
    return username.strip().casefold()


class UserService:
    def login(self, username: str, password: str) -> str:
        normalized = normalize_credentials(username)
        if not normalized or not password:
            raise ValueError("credentials are required")
        if not self._rate_limit_ok(username):
            raise PermissionError("too many attempts")
        token = secrets.token_urlsafe(16)
        self._audit_login(username, token)
        return token

    def _rate_limit_ok(self, username: str) -> bool:
        return True

    def _audit_login(self, username: str, token: str) -> None:
        def audit_message() -> str:
            return f"login:{username}:{token}"
        print(audit_message())

    def logout(self, token: str) -> None:
        if not token:
            raise ValueError("token is required")
        self._revoke(token)

    def _revoke(self, token: str) -> None:
        print(f"revoked:{token}")
''',
    "src/admin/service.py": '''class AdminService:
    def login(self, username: str, password: str) -> str:
        if username != "admin":
            raise PermissionError("admin only")
        return "admin-token"

    def deactivate(self, username: str) -> None:
        print(f"deactivated:{username}")
''',
    "src/db/repositories.py": '''def load_neighbor_chunks(repository_id: int, keys: list) -> list:
    """Batch-load specific chunks for one repository in a single query."""
    return [key for key in keys if key[1] >= 0]


class RepositoryStore:
    def search(self, repository_id: int, query: str) -> list:
        return []
''',
    "src/rag/assembly.py": '''class ContextAssembler:
    """Bounded neighbor expansion and budget-aware packing."""

    def assemble(self, seeds: list) -> list:
        return list(seeds)


def buildRagContext(seeds: list) -> str:
    return "\\n".join(str(seed) for seed in seeds)
''',
    "src/jobs/store.py": '''class JobStore:
    def claim(self, worker_id: str) -> str | None:
        return None

    def cancel(self, job_id: str) -> None:
        print(f"cancel-requested:{job_id}")

    def renew_lease(self, job_id: str, worker_id: str) -> None:
        """Extend an expired lease so a stalled job can be reclaimed."""
        print(f"renew:{job_id}:{worker_id}")
''',
    "src/worker.py": '''class Worker:
    def run(self) -> None:
        self._loop()

    def _loop(self) -> None:
        pass


class Cache:
    def get(self, key: str):
        return None

    def set(self, key: str, value) -> None:
        pass


class Harness:
    def test(self) -> bool:
        return True


class Store:
    def save(self, item) -> None:
        pass

    def load(self, item_id: str):
        return None
''',
    "src/styles.py": '''BACKGROUND_COLOR = "blue"
FONT_SIZE = 14
''',
}

# (case_id, category, query, marker_substring_defining_relevance)
# A marker matches every chunk whose content contains it; this stays
# deterministic and inspectable rather than hand-picking chunk indexes that
# shift if chunking parameters change.
_CASES: tuple[tuple[str, str, str, str], ...] = (
    (
        "exact-qualified-user-login",
        "exact_qualified_symbol",
        "UserService.login",
        "def login",
    ),
    (
        "exact-qualified-admin-login",
        "exact_qualified_symbol",
        "AdminService.login",
        'if username != "admin"',
    ),
    (
        "nl-login-validates-credentials",
        "natural_language_symbol",
        "How does UserService.login validate credentials?",
        "credentials are required",
    ),
    (
        "ambiguous-login",
        "ambiguous_symbol",
        "login",
        "def login",
    ),
    (
        "snake-case-load-neighbor-chunks",
        "snake_case",
        "load_neighbor_chunks",
        "def load_neighbor_chunks",
    ),
    (
        "pascal-case-context-assembler",
        "pascal_case",
        "ContextAssembler",
        "class ContextAssembler",
    ),
    (
        "camel-case-build-rag-context",
        "camel_case",
        "buildRagContext",
        "def buildRagContext",
    ),
    (
        "common-word-run-worker-loop",
        "common_word",
        "run the worker processing loop",
        "def run",
    ),
    (
        "behavioral-lease-recovery",
        "behavioral",
        "How are expired worker leases recovered?",
        "expired lease",
    ),
    (
        "duplicate-symbol-admin-discriminates",
        "duplicate_symbol",
        "AdminService.login",
        'if username != "admin"',
    ),
)

_CATEGORY_ORDER = (
    "exact_qualified_symbol",
    "natural_language_symbol",
    "ambiguous_symbol",
    "snake_case",
    "pascal_case",
    "camel_case",
    "common_word",
    "behavioral",
    "duplicate_symbol",
)


def _chunks(strategy: ChunkingStrategy) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    for path, content in _FILES.items():
        source_file = SourceFile(
            relative_path=Path(path),
            language="python",
            content=content,
            size_bytes=len(content.encode()),
            line_count=len(content.splitlines()),
        )
        if strategy is ChunkingStrategy.STRUCTURAL:
            config = ChunkingConfig(
                strategy=ChunkingStrategy.STRUCTURAL,
                max_lines_per_chunk=4,
                overlap_lines=0,
                max_chars_per_chunk=120,
            )
        else:
            config = ChunkingConfig(max_lines_per_chunk=6, overlap_lines=1)
        chunks.extend(chunk_source_file(source_file, config))
    return chunks


def _hashed_embedding(text: str) -> tuple[float, ...]:
    # A tiny uniform baseline keeps whitespace-only structural filler chunks
    # (blank lines between definitions) from hashing to an exact zero vector,
    # which cosine similarity rejects as undefined; it is far too small to
    # meaningfully affect similarity between any two non-trivial chunks.
    values = [1e-6] * 64
    for token in tokenize_code(text):
        digest = hashlib.sha256(token.encode()).digest()
        slot = int.from_bytes(digest[:2], "big") % len(values)
        values[slot] += 1.0
    return tuple(values)


class _EmbeddingProvider:
    model = "repo-eval-v4-hash"

    def embed_text(self, text: str) -> EmbeddingVector:
        return EmbeddingVector(values=_hashed_embedding(text), model=self.model)


class _PassthroughFakeReranker:
    """A deterministic, order-preserving reranker: no live model, no fabricated scores."""

    def rerank(
        self, query: str, candidates: Sequence[RankedChunk], *, top_k: int
    ) -> list[RerankedSearchResult]:
        return [
            RerankedSearchResult(chunk=c.chunk, rank=rank, original_rank=c.rank)
            for rank, c in enumerate(candidates[:top_k], start=1)
        ]


def _embedded(chunks: Sequence[CodeChunk]) -> list[EmbeddedChunk]:
    provider = _EmbeddingProvider()
    return [
        EmbeddedChunk(
            chunk=chunk,
            embedding=provider.embed_text(
                embedding_text_for_chunk(chunk, EmbeddingTextStrategy.RAW_SOURCE)
            ),
        )
        for chunk in chunks
    ]


def _suite(chunks: Sequence[CodeChunk], category: str | None = None) -> RetrievalBenchmarkSuite:
    cases = []
    for case_id, case_category, query, marker in _CASES:
        if category is not None and case_category != category:
            continue
        relevant = tuple(chunk_identity(chunk) for chunk in chunks if marker in chunk.content)
        if not relevant:
            raise RuntimeError(f"fixture marker {marker!r} is not represented for {case_id!r}")
        cases.append(RetrievalBenchmarkCase(id=case_id, query=query, relevant_chunks=relevant))
    return RetrievalBenchmarkSuite(version=VERSION, cases=tuple(cases))


def _retrievers(chunks: Sequence[CodeChunk]) -> dict[str, object]:
    provider = _EmbeddingProvider()
    corpus = _embedded(chunks)
    bm25 = BM25Index.from_chunks(chunks)
    reranker = _PassthroughFakeReranker()

    def exact(query: str, *, top_k: int) -> Sequence[RankedChunk]:
        return semantic_search(query, corpus, provider, top_k=top_k)

    def bm25_only(query: str, *, top_k: int) -> Sequence[RankedChunk]:
        return bm25.search(query, top_k=top_k)

    def hybrid(query: str, *, top_k: int) -> Sequence[RankedChunk]:
        return hybrid_search(query, corpus, bm25, provider, top_k=top_k)

    def symbol_only(query: str, *, top_k: int) -> Sequence[RankedChunk]:
        candidates = extract_identifier_candidates(query)
        return symbol_search(candidates, chunks, limit=top_k)

    def hybrid_symbol(query: str, *, top_k: int) -> Sequence[RankedChunk]:
        return hybrid_symbol_search(query, corpus, bm25, provider, chunks, top_k=top_k)

    def hybrid_symbol_rerank(query: str, *, top_k: int) -> Sequence[RankedChunk]:
        candidates = hybrid_symbol_search(query, corpus, bm25, provider, chunks, top_k=max(20, top_k))
        return reranker.rerank(query, candidates, top_k=top_k)

    return {
        "semantic+exact": exact,
        "bm25": bm25_only,
        "semantic+bm25/rrf": hybrid,
        "symbol_only": symbol_only,
        "semantic+bm25+symbol/rrf": hybrid_symbol,
        "semantic+bm25+symbol/rrf+fake_rerank": hybrid_symbol_rerank,
    }


def _exact_symbol_hit_at_k(
    suite: RetrievalBenchmarkSuite,
    retriever,
    *,
    k: int,
    qualified_case_ids: frozenset[str],
) -> float | None:
    """Fraction of qualified-symbol cases whose defining chunk is in the top-k.

    Defined narrowly: only over cases that name a known qualified identifier,
    never mixed into the general Recall@k/MRR/nDCG@k metrics.
    """

    hits = 0
    total = 0
    for case in suite.cases:
        if case.id not in qualified_case_ids:
            continue
        total += 1
        retrieved = tuple(
            chunk_identity(result.chunk) for result in retriever(case.query, top_k=k)[:k]
        )
        if set(case.relevant_chunks) & set(retrieved):
            hits += 1
    return None if total == 0 else hits / total


def _print_fragment_recovery_check(structural_chunks: Sequence[CodeChunk]) -> None:
    """Verify bounded symbol retrieval composes with M23's expanded strategy.

    UserService.login is deliberately long enough to split into multiple
    structural fragments; symbol retrieval returns only the first fragment
    (bounded budget), and expanded context assembly should still be able to
    recover adjacent same-symbol fragments deterministically.
    """

    candidates = extract_identifier_candidates("UserService.login")
    symbol_results = symbol_search(candidates, structural_chunks, limit=10)
    login_fragment_count = sum(
        1
        for chunk in structural_chunks
        if chunk.qualified_symbol_name == "UserService.login"
    )
    result = assemble_context(
        symbol_results,
        InMemoryNeighborLoader(structural_chunks),
        ContextAssemblyConfig(strategy=ContextStrategy.EXPANDED, neighbor_radius=1),
    )
    recovered_fragments = sum(
        1
        for packed in result.chunks
        if packed.chunk.qualified_symbol_name == "UserService.login"
    )
    print(
        f"\nUserService.login fragments in source: {login_fragment_count}; "
        f"symbol-search candidates returned: {len(symbol_results)} (bounded); "
        f"fragments recovered by expanded context assembly: {recovered_fragments}"
    )


def main() -> None:
    """Print offline, deterministic symbol-fusion retrieval evidence. No live model calls."""

    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.repo_eval_v4",
        description=__doc__,
    )
    parser.parse_args()

    line_chunks = _chunks(ChunkingStrategy.LINE)
    structural_chunks = _chunks(ChunkingStrategy.STRUCTURAL)
    retrievers = _retrievers(structural_chunks)
    line_retrievers = _retrievers(line_chunks)

    qualified_case_ids = frozenset(
        case_id
        for case_id, category, _, _ in _CASES
        if category in {"exact_qualified_symbol", "duplicate_symbol"}
    )

    print(f"{VERSION.upper()} SYMBOL-AWARE RETRIEVAL FUSION")
    print("line_v1 baseline (preserved, structural symbol metadata unavailable):")
    print(
        format_retrieval_comparison(
            [evaluate_retrieval(_suite(line_chunks), "line_v1+semantic+bm25/rrf", line_retrievers["semantic+bm25/rrf"], k=3)]
        )
    )

    print("\npython_ast_v1 matrix, grouped by category:\n")
    for category in _CATEGORY_ORDER:
        suite = _suite(structural_chunks, category=category)
        if not suite.cases:
            continue
        reports = [
            evaluate_retrieval(suite, name, retriever, k=3)
            for name, retriever in retrievers.items()
        ]
        print(f"--- category: {category} ---")
        print(format_retrieval_comparison(reports))
        print()

    print("Exact-symbol-hit@3 (qualified-symbol cases only; separate from Recall@k):")
    for name, retriever in retrievers.items():
        full_suite = _suite(structural_chunks)
        hit_rate = _exact_symbol_hit_at_k(
            full_suite, retriever, k=3, qualified_case_ids=qualified_case_ids
        )
        print(f"  {name}: {hit_rate if hit_rate is None else f'{hit_rate:.3f}'}")

    _print_fragment_recovery_check(structural_chunks)

    print(
        "\nThese deterministic hashed embeddings isolate symbol-fusion plumbing "
        "and ambiguity handling; they are not evidence about OpenAI embedding "
        "quality or real-repository generalization."
    )


if __name__ == "__main__":
    main()
