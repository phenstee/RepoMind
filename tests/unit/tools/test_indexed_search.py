"""Tests for the bounded indexed-navigation tool and its registry composition."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from repomind.ingestion import ChunkingStrategy, ChunkKind, CodeChunk
from repomind.tools import (
    IndexedCodeLocation,
    IndexedCodeSearchInput,
    ToolContext,
    create_default_tool_registry,
    create_investigation_tool_registry,
    indexed_code_search,
)
from repomind.tools.registry import ToolExecutionError, ToolValidationError

READ_ONLY_TOOLS = {
    "find_symbol",
    "git_diff",
    "git_status",
    "list_directory",
    "read_file",
    "search_code",
}


def _chunk(
    path: str,
    *,
    start_line: int = 1,
    end_line: int = 2,
    chunk_kind: ChunkKind = ChunkKind.FUNCTION,
    qualified_symbol_name: str | None = None,
) -> CodeChunk:
    return CodeChunk(
        relative_path=path,
        language="python",
        start_line=start_line,
        end_line=end_line,
        content="def handled_elsewhere(): ...\n",
        chunk_index=0,
        chunking_strategy=ChunkingStrategy.STRUCTURAL,
        chunk_kind=chunk_kind,
        qualified_symbol_name=qualified_symbol_name,
    )


class _FakeResult:
    """A minimal RankedChunk-like fake, optionally with fusion provenance."""

    def __init__(self, chunk, rank, *, semantic_rank=None, lexical_rank=None, symbol_rank=None):
        self.chunk = chunk
        self.rank = rank
        if semantic_rank is not None:
            self.semantic_rank = semantic_rank
        if lexical_rank is not None:
            self.lexical_rank = lexical_rank
        if symbol_rank is not None:
            self.symbol_rank = symbol_rank


class _FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def __call__(self, query, *, top_k):
        self.calls.append((query, top_k))
        return self.results[:top_k]


def test_default_registry_has_no_indexed_tool_and_needs_no_retriever(tmp_path: Path) -> None:
    names = {
        tool.name
        for tool in create_default_tool_registry(ToolContext(repository_root=tmp_path)).list_tools()
    }
    assert names == READ_ONLY_TOOLS
    assert "indexed_code_search" not in names


def test_investigation_registry_adds_exactly_one_tool(tmp_path: Path) -> None:
    retriever = _FakeRetriever([])
    names = {
        tool.name
        for tool in create_investigation_tool_registry(
            ToolContext(repository_root=tmp_path), retriever
        ).list_tools()
    }
    assert names == READ_ONLY_TOOLS | {"indexed_code_search"}


def test_handler_maps_results_to_bounded_locations_without_content() -> None:
    chunk = _chunk("src/jobs/store.py", start_line=10, end_line=20, qualified_symbol_name="JobStore.claim")
    retriever = _FakeRetriever([_FakeResult(chunk, 1, semantic_rank=1, symbol_rank=2)])

    output = indexed_code_search(retriever, IndexedCodeSearchInput(query="claim a job", max_results=5))

    assert output.result_count == 1
    location = output.locations[0]
    assert location.relative_path == Path("src/jobs/store.py")
    assert (location.start_line, location.end_line) == (10, 20)
    assert location.qualified_symbol_name == "JobStore.claim"
    assert location.chunk_kind == "function"
    assert set(location.sources) == {"semantic", "symbol"}
    assert not hasattr(output, "content")
    assert "content" not in output.model_dump()


def test_handler_reports_sources_absent_without_fabricating_them() -> None:
    chunk = _chunk("src/a.py")
    retriever = _FakeRetriever([_FakeResult(chunk, 1)])

    output = indexed_code_search(retriever, IndexedCodeSearchInput(query="anything", max_results=5))

    assert output.locations[0].sources == ()


def test_handler_passes_bounded_top_k_to_retriever() -> None:
    retriever = _FakeRetriever([])

    indexed_code_search(retriever, IndexedCodeSearchInput(query="query text", max_results=3))

    assert retriever.calls == [("query text", 3)]


def test_empty_repository_or_no_match_returns_valid_empty_result() -> None:
    retriever = _FakeRetriever([])

    output = indexed_code_search(retriever, IndexedCodeSearchInput(query="nothing here", max_results=5))

    assert output.locations == []
    assert output.result_count == 0


@pytest.mark.parametrize("max_results", [0, -1, 11, True])
def test_max_results_is_strictly_bounded(max_results: object) -> None:
    with pytest.raises(ValidationError):
        IndexedCodeSearchInput(query="q", max_results=max_results)


@pytest.mark.parametrize("query", ["", "   "])
def test_blank_query_is_rejected(query: str) -> None:
    with pytest.raises(ValidationError):
        IndexedCodeSearchInput(query=query)


def test_code_chunk_itself_rejects_a_traversal_or_absolute_path() -> None:
    # The first line of defense: CodeChunk validates relative_path on
    # construction, so no retrieval implementation can even build a domain
    # chunk carrying an escaping path in the first place.
    with pytest.raises(ValidationError):
        CodeChunk(
            relative_path="../../etc/passwd",
            language=None,
            start_line=1,
            end_line=1,
            content="ignored\n",
            chunk_index=0,
        )


def test_indexed_code_location_independently_rejects_an_escaping_path() -> None:
    # A second, independent layer of defense at the tool-output boundary
    # itself, in case a future RankedChunk-like implementation ever supplies
    # a path without going through CodeChunk's own validation.
    with pytest.raises(ValidationError):
        IndexedCodeLocation(
            relative_path=Path("../../etc/passwd"),
            start_line=1,
            end_line=1,
            rank=1,
        )


def test_registry_execute_rejects_invalid_arguments_predictably(tmp_path: Path) -> None:
    retriever = _FakeRetriever([])
    registry = create_investigation_tool_registry(ToolContext(repository_root=tmp_path), retriever)

    with pytest.raises(ToolValidationError):
        registry.execute("indexed_code_search", {"query": "q", "max_results": 999})


def test_retriever_backend_error_becomes_predictable_tool_execution_error(tmp_path: Path) -> None:
    class _FailingRetriever:
        def __call__(self, query, *, top_k):
            raise RuntimeError("embedding provider unavailable")

    registry = create_investigation_tool_registry(
        ToolContext(repository_root=tmp_path), _FailingRetriever()
    )

    with pytest.raises(ToolExecutionError):
        registry.execute("indexed_code_search", {"query": "q", "max_results": 5})


def test_max_results_bounds_serialized_history_footprint() -> None:
    long_path = "src/" + "/".join(f"module_{i}" for i in range(10)) + "/service.py"
    long_symbol = "OuterClass." + ".".join(f"Nested{i}" for i in range(10)) + ".method"
    results = [
        _FakeResult(
            _chunk(long_path, qualified_symbol_name=long_symbol),
            rank,
            semantic_rank=rank,
            lexical_rank=rank,
            symbol_rank=rank,
        )
        for rank in range(1, 11)
    ]
    retriever = _FakeRetriever(results)

    output = indexed_code_search(
        retriever, IndexedCodeSearchInput(query="deeply nested query", max_results=10)
    )

    # One indexed_code_search observation must stay a small fraction of the
    # default 60,000-char agent history budget, even at the maximum allowed
    # result count and unusually long paths/symbol names.
    assert len(output.model_dump_json()) < 5_000


def test_line_v1_style_result_has_no_symbol_provenance_and_still_works() -> None:
    chunk = _chunk("src/a.py", chunk_kind=ChunkKind.LINE, qualified_symbol_name=None)
    retriever = _FakeRetriever([_FakeResult(chunk, 1, semantic_rank=1, lexical_rank=1)])

    output = indexed_code_search(retriever, IndexedCodeSearchInput(query="anything", max_results=5))

    assert output.locations[0].qualified_symbol_name is None
    assert set(output.locations[0].sources) == {"semantic", "bm25"}
