"""Tests for deterministic, bounded persisted-symbol-metadata retrieval."""

import pytest

from repomind.ingestion import ChunkingStrategy, ChunkKind, CodeChunk
from repomind.retrieval import (
    BM25Index,
    EmbeddedChunk,
    EmbeddingVector,
    hybrid_search,
    hybrid_symbol_search,
)
from repomind.retrieval.identifiers import extract_identifier_candidates
from repomind.retrieval.models import SymbolMatchTier
from repomind.retrieval.symbols import SymbolSearchError, symbol_search


def _structural_chunk(
    path: str,
    index: int,
    *,
    symbol_name: str,
    qualified_symbol_name: str,
    chunk_kind: ChunkKind = ChunkKind.METHOD,
    fragment_index: int | None = None,
    fragment_count: int | None = None,
    start_line: int = 1,
    content: str | None = None,
) -> CodeChunk:
    return CodeChunk(
        relative_path=path,
        language="python",
        start_line=start_line,
        end_line=start_line + 1,
        content=content or f"def {symbol_name}(): ...\n",
        chunk_index=index,
        chunking_strategy=ChunkingStrategy.STRUCTURAL,
        chunk_kind=chunk_kind,
        symbol_name=symbol_name,
        qualified_symbol_name=qualified_symbol_name,
        fragment_index=fragment_index,
        fragment_count=fragment_count,
    )


class _FixedEmbeddingProvider:
    def embed_text(self, text: str) -> EmbeddingVector:
        return EmbeddingVector(values=(1.0, 0.0), model="test-model")


def _line_chunk(path: str, index: int) -> CodeChunk:
    return CodeChunk(
        relative_path=path,
        language="python",
        start_line=index * 10 + 1,
        end_line=index * 10 + 5,
        content="plain line content\n" * 5,
        chunk_index=index,
        chunking_strategy=ChunkingStrategy.LINE,
        chunk_kind=ChunkKind.LINE,
    )


def test_exact_qualified_match_outranks_simple_and_is_prioritized() -> None:
    login = _structural_chunk(
        "src/auth/service.py", 0, symbol_name="login", qualified_symbol_name="UserService.login"
    )
    admin_login = _structural_chunk(
        "src/admin/service.py", 0, symbol_name="login", qualified_symbol_name="AdminService.login"
    )
    candidates = extract_identifier_candidates("UserService.login")

    results = symbol_search(candidates, [login, admin_login], limit=10)

    assert results[0].chunk == login
    assert results[0].match_tier is SymbolMatchTier.QUALIFIED_SYMBOL
    assert all(r.chunk != admin_login for r in results)


def test_ambiguous_simple_name_returns_both_deterministically_and_bounded() -> None:
    login = _structural_chunk(
        "src/auth/service.py", 0, symbol_name="login", qualified_symbol_name="UserService.login"
    )
    admin_login = _structural_chunk(
        "src/admin/service.py", 0, symbol_name="login", qualified_symbol_name="AdminService.login"
    )
    candidates = extract_identifier_candidates("login")

    first = symbol_search(candidates, [login, admin_login], limit=10)
    second = symbol_search(candidates, [login, admin_login], limit=10)

    assert [r.chunk for r in first] == [admin_login, login]
    assert first == second


def test_snake_case_query_finds_the_function() -> None:
    target = _structural_chunk(
        "src/db/repositories.py",
        0,
        symbol_name="load_neighbor_chunks",
        qualified_symbol_name="load_neighbor_chunks",
        chunk_kind=ChunkKind.FUNCTION,
    )
    unrelated = _structural_chunk(
        "src/db/repositories.py",
        1,
        symbol_name="persist_chunks",
        qualified_symbol_name="persist_chunks",
        chunk_kind=ChunkKind.FUNCTION,
        start_line=20,
    )
    candidates = extract_identifier_candidates("How does load_neighbor_chunks avoid N+1 queries?")

    results = symbol_search(candidates, [target, unrelated], limit=10)

    assert len(results) == 1
    assert results[0].chunk == target
    assert results[0].match_tier is SymbolMatchTier.SIMPLE_SYMBOL_STRONG


def test_common_word_match_is_weak_tier_and_ranked_after_strong_matches() -> None:
    run_function = _structural_chunk(
        "src/jobs/worker.py", 0, symbol_name="run", qualified_symbol_name="Worker.run"
    )
    strong_match = _structural_chunk(
        "src/other.py", 0, symbol_name="load_neighbor_chunks", qualified_symbol_name="load_neighbor_chunks"
    )
    candidates = extract_identifier_candidates("run validation with load_neighbor_chunks")

    results = symbol_search(candidates, [run_function, strong_match], limit=10)

    tiers = [r.match_tier for r in results]
    assert tiers.index(SymbolMatchTier.SIMPLE_SYMBOL_STRONG) < tiers.index(
        SymbolMatchTier.SIMPLE_SYMBOL_WEAK
    )


def test_line_v1_chunks_return_no_symbol_candidates_without_error() -> None:
    chunks = [_line_chunk("src/a.py", index) for index in range(3)]
    candidates = extract_identifier_candidates("UserService.login")

    results = symbol_search(candidates, chunks, limit=10)

    assert results == []


def test_no_recognized_identifier_returns_empty_without_scanning_error() -> None:
    chunk = _structural_chunk(
        "src/a.py", 0, symbol_name="login", qualified_symbol_name="UserService.login"
    )

    results = symbol_search((), [chunk], limit=10)

    assert results == []


def test_structural_fragments_collapse_to_first_fragment() -> None:
    fragments = [
        _structural_chunk(
            "src/auth/service.py",
            index,
            symbol_name="login",
            qualified_symbol_name="UserService.login",
            chunk_kind=ChunkKind.STRUCTURAL_FRAGMENT,
            fragment_index=index + 1,
            fragment_count=30,
            start_line=index * 5 + 1,
        )
        for index in range(30)
    ]
    candidates = extract_identifier_candidates("UserService.login")

    results = symbol_search(candidates, fragments, limit=10)

    assert len(results) == 1
    assert results[0].chunk.fragment_index == 1


def test_hybrid_symbol_uses_behavioral_evidence_to_select_method_fragment() -> None:
    declaration = _structural_chunk(
        "src/auth/service.py",
        0,
        symbol_name="login",
        qualified_symbol_name="UserService.login",
        chunk_kind=ChunkKind.STRUCTURAL_FRAGMENT,
        fragment_index=1,
        fragment_count=3,
        start_line=10,
        content=(
            "def login(self, username, password):\n"
            "    normalized = normalize_credentials(username)\n"
        ),
    )
    validation = _structural_chunk(
        "src/auth/service.py",
        1,
        symbol_name="login",
        qualified_symbol_name="UserService.login",
        chunk_kind=ChunkKind.STRUCTURAL_FRAGMENT,
        fragment_index=2,
        fragment_count=3,
        start_line=12,
        content=(
            "    if not normalized or not password:\n"
            "        raise ValueError('credentials are required')\n"
        ),
    )
    token_and_audit = _structural_chunk(
        "src/auth/service.py",
        2,
        symbol_name="login",
        qualified_symbol_name="UserService.login",
        chunk_kind=ChunkKind.STRUCTURAL_FRAGMENT,
        fragment_index=3,
        fragment_count=3,
        start_line=14,
        content="    token = issue_token(normalized)\n    audit_login(username, token)\n",
    )
    semantic_distraction = _structural_chunk(
        "src/other.py",
        0,
        symbol_name="unrelated",
        qualified_symbol_name="unrelated",
        start_line=1,
        content="def unrelated():\n    return None\n",
    )
    admin_login = _structural_chunk(
        "src/admin/service.py",
        0,
        symbol_name="login",
        qualified_symbol_name="AdminService.login",
        content=(
            "def login(self, username, password):\n"
            "    if username != 'admin':\n"
            "        raise PermissionError('admin only')\n"
        ),
    )
    chunks = [declaration, validation, token_and_audit, semantic_distraction, admin_login]
    embedded = [
        EmbeddedChunk(
            chunk=chunk,
            embedding=EmbeddingVector(values=vector, model="test-model"),
        )
        for chunk, vector in (
            (declaration, (0.0, 1.0)),
            (validation, (0.8, 0.2)),
            (token_and_audit, (-1.0, 0.0)),
            (semantic_distraction, (1.0, 0.0)),
            (admin_login, (-1.0, 0.0)),
        )
    ]
    bm25 = BM25Index.from_chunks(chunks)
    provider = _FixedEmbeddingProvider()
    query = "How does UserService.login validate credentials?"

    baseline = hybrid_search(
        query,
        embedded,
        bm25,
        provider,
        top_k=3,
        semantic_candidates=2,
        lexical_candidates=2,
    )
    fused = hybrid_symbol_search(
        query,
        embedded,
        bm25,
        provider,
        chunks,
        top_k=3,
        semantic_candidates=2,
        lexical_candidates=2,
    )
    repeated = hybrid_symbol_search(
        query,
        embedded,
        bm25,
        provider,
        chunks,
        top_k=3,
        semantic_candidates=2,
        lexical_candidates=2,
    )
    exact = hybrid_symbol_search(
        "UserService.login",
        embedded,
        bm25,
        provider,
        chunks,
        top_k=3,
        semantic_candidates=5,
        lexical_candidates=5,
    )

    assert baseline[0].chunk == validation
    assert fused[0].chunk == validation
    assert fused == repeated
    assert exact[0].chunk == declaration
    assert exact[0].chunk.qualified_symbol_name != admin_login.qualified_symbol_name
    assert exact[0].symbol_rank == 1


def test_candidate_count_is_bounded_by_limit() -> None:
    many_run_functions = [
        _structural_chunk(
            f"src/mod_{index}.py", 0, symbol_name="run", qualified_symbol_name=f"Mod{index}.run"
        )
        for index in range(300)
    ]
    candidates = extract_identifier_candidates("run")

    results = symbol_search(candidates, many_run_functions, limit=5)

    assert len(results) == 5


@pytest.mark.parametrize("limit", [0, -1, True, 51])
def test_rejects_invalid_limit(limit: object) -> None:
    with pytest.raises(SymbolSearchError):
        symbol_search((), [], limit=limit)  # type: ignore[arg-type]
