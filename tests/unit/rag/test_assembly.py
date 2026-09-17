"""Tests for bounded neighbor expansion, dedup, and budget-aware packing."""

import pytest

from repomind.ingestion import ChunkingStrategy, ChunkKind, CodeChunk
from repomind.rag import ContextOrigin, ContextStrategy
from repomind.rag.assembly import (
    ContextAssemblyError,
    InMemoryNeighborLoader,
    assemble_context,
)
from repomind.rag.models import ContextAssemblyConfig
from repomind.retrieval import SemanticSearchResult


def _line_chunk(path: str, index: int, *, start_line: int, content: str) -> CodeChunk:
    line_count = max(1, len(content.splitlines()))
    return CodeChunk(
        relative_path=path,
        language="python",
        start_line=start_line,
        end_line=start_line + line_count - 1,
        content=content,
        chunk_index=index,
        chunking_strategy=ChunkingStrategy.LINE,
        chunk_kind=ChunkKind.LINE,
    )


def _fragment_chunk(
    path: str,
    index: int,
    *,
    fragment_index: int,
    fragment_count: int,
    qualified_symbol_name: str,
    start_line: int,
    content: str,
) -> CodeChunk:
    return CodeChunk(
        relative_path=path,
        language="python",
        start_line=start_line,
        end_line=start_line + max(1, len(content.splitlines())) - 1,
        content=content,
        chunk_index=index,
        chunking_strategy=ChunkingStrategy.STRUCTURAL,
        chunk_kind=ChunkKind.STRUCTURAL_FRAGMENT,
        symbol_name=qualified_symbol_name.rsplit(".", 1)[-1],
        qualified_symbol_name=qualified_symbol_name,
        parent_symbol=None,
        fragment_index=fragment_index,
        fragment_count=fragment_count,
    )


def _seed(chunk: CodeChunk, rank: int) -> SemanticSearchResult:
    return SemanticSearchResult(chunk=chunk, score=1.0 - rank / 10, rank=rank)


def _line_file(path: str, count: int) -> list[CodeChunk]:
    return [
        _line_chunk(path, index, start_line=index * 10 + 1, content=f"line body {index}\n" * 5)
        for index in range(count)
    ]


def test_seeds_only_strategy_never_expands() -> None:
    chunks = _line_file("src/a.py", 3)
    seeds = [_seed(chunks[1], 1)]
    config = ContextAssemblyConfig(strategy=ContextStrategy.SEEDS_ONLY)

    result = assemble_context(seeds, InMemoryNeighborLoader(chunks), config)

    assert result.seed_count == 1
    assert result.expanded_candidate_count == 0
    assert [c.origin for c in result.chunks] == [ContextOrigin.SEED]


def test_line_v1_generic_neighbor_expansion_stays_in_same_file() -> None:
    chunks = _line_file("src/a.py", 5)
    other_file = _line_chunk("src/b.py", 1, start_line=1, content="unrelated\n")
    seeds = [_seed(chunks[2], 1)]
    loader = InMemoryNeighborLoader([*chunks, other_file])
    config = ContextAssemblyConfig(strategy=ContextStrategy.EXPANDED, neighbor_radius=1)

    result = assemble_context(seeds, loader, config)

    origins = {(c.chunk.chunk_index, c.origin) for c in result.chunks}
    assert (1, ContextOrigin.NEIGHBOR) in origins
    assert (3, ContextOrigin.NEIGHBOR) in origins
    assert all(c.chunk.relative_path.as_posix() == "src/a.py" for c in result.chunks)
    assert result.expanded_candidate_count == 2


def test_expansion_is_bounded_by_radius_and_never_recurses() -> None:
    chunks = _line_file("src/a.py", 7)
    seeds = [_seed(chunks[3], 1)]
    loader = InMemoryNeighborLoader(chunks)
    config = ContextAssemblyConfig(strategy=ContextStrategy.EXPANDED, neighbor_radius=1)

    result = assemble_context(seeds, loader, config)

    packed_indexes = {c.chunk.chunk_index for c in result.chunks}
    assert packed_indexes == {2, 3, 4}
    assert 1 not in packed_indexes
    assert 5 not in packed_indexes


def test_same_symbol_fragment_neighbor_is_classified_and_preferred() -> None:
    fragments = [
        _fragment_chunk(
            "src/service.py",
            index,
            fragment_index=index + 1,
            fragment_count=3,
            qualified_symbol_name="UserService.login",
            start_line=index * 20 + 1,
            content=f"fragment body {index}\n" * 5,
        )
        for index in range(3)
    ]
    generic_neighbor_file = _line_file("src/other.py", 3)
    seeds = [
        _seed(fragments[1], 1),
        _seed(generic_neighbor_file[1], 2),
    ]
    loader = InMemoryNeighborLoader([*fragments, *generic_neighbor_file])
    config = ContextAssemblyConfig(strategy=ContextStrategy.EXPANDED, neighbor_radius=1)

    result = assemble_context(seeds, loader, config)

    by_index = {(c.chunk.relative_path.as_posix(), c.chunk.chunk_index): c for c in result.chunks}
    assert by_index[("src/service.py", 0)].origin is ContextOrigin.SAME_SYMBOL_FRAGMENT
    assert by_index[("src/service.py", 2)].origin is ContextOrigin.SAME_SYMBOL_FRAGMENT
    assert by_index[("src/other.py", 0)].origin is ContextOrigin.NEIGHBOR
    assert by_index[("src/other.py", 2)].origin is ContextOrigin.NEIGHBOR

    # Same-symbol fragments outrank generic neighbors regardless of seed rank.
    origins_in_order = [c.origin for c in result.chunks]
    last_fragment_position = max(
        i for i, c in enumerate(result.chunks) if c.origin is ContextOrigin.SAME_SYMBOL_FRAGMENT
    )
    first_generic_position = min(
        i for i, c in enumerate(result.chunks) if c.origin is ContextOrigin.NEIGHBOR
    )
    assert last_fragment_position < first_generic_position
    assert origins_in_order.count(ContextOrigin.SEED) == 2


def test_two_seeds_expanding_to_the_same_neighbor_keep_it_once() -> None:
    chunks = _line_file("src/a.py", 3)
    seeds = [_seed(chunks[0], 1), _seed(chunks[2], 2)]
    loader = InMemoryNeighborLoader(chunks)
    config = ContextAssemblyConfig(strategy=ContextStrategy.EXPANDED, neighbor_radius=1)

    result = assemble_context(seeds, loader, config)

    indexes = [c.chunk.chunk_index for c in result.chunks]
    assert indexes.count(1) == 1
    assert result.deduplicated_count == 1


def test_neighbor_equal_to_another_seed_is_kept_once_as_seed() -> None:
    chunks = _line_file("src/a.py", 3)
    seeds = [_seed(chunks[0], 1), _seed(chunks[1], 2)]
    loader = InMemoryNeighborLoader(chunks)
    config = ContextAssemblyConfig(strategy=ContextStrategy.EXPANDED, neighbor_radius=1)

    result = assemble_context(seeds, loader, config)

    matches = [c for c in result.chunks if c.chunk.chunk_index == 1]
    assert len(matches) == 1
    assert matches[0].origin is ContextOrigin.SEED
    # seed0's neighbor at index 1 dupes seed1, and seed1's neighbor at index 0
    # dupes seed0 - both are caught by identity dedup.
    assert result.deduplicated_count == 2


def test_fully_contained_overlap_is_suppressed_but_distinct_ranges_survive() -> None:
    big = _line_chunk("src/a.py", 0, start_line=1, content="\n".join(f"l{i}" for i in range(20)))
    nested = CodeChunk(
        relative_path="src/a.py",
        language="python",
        start_line=5,
        end_line=8,
        content="l4\nl5\nl6\nl7",
        chunk_index=1,
        chunking_strategy=ChunkingStrategy.LINE,
        chunk_kind=ChunkKind.LINE,
    )
    distinct = _line_chunk("src/a.py", 2, start_line=21, content="distinct body\n" * 3)
    seeds = [_seed(big, 1)]
    loader = InMemoryNeighborLoader([big, nested, distinct])
    config = ContextAssemblyConfig(strategy=ContextStrategy.EXPANDED, neighbor_radius=2)

    result = assemble_context(seeds, loader, config)

    packed_indexes = {c.chunk.chunk_index for c in result.chunks}
    assert 1 not in packed_indexes  # fully contained in the seed's own range
    assert 2 in packed_indexes  # a distinct, non-overlapping range survives
    assert result.deduplicated_count == 1


def test_budget_keeps_higher_priority_evidence_first() -> None:
    chunks = _line_file("src/a.py", 3)
    seeds = [_seed(chunks[1], 1)]
    loader = InMemoryNeighborLoader(chunks)
    config = ContextAssemblyConfig(strategy=ContextStrategy.EXPANDED, neighbor_radius=1, budget_tokens=1)

    result = assemble_context(seeds, loader, config)

    # The single highest-priority chunk is always kept even though it alone
    # exceeds the budget; nothing else fits afterward.
    assert result.packed_count == 1
    assert result.chunks[0].chunk.chunk_index == 1
    assert result.chunks[0].origin is ContextOrigin.SEED
    assert result.dropped_for_budget_count == 2


def test_oversized_first_chunk_is_still_included() -> None:
    huge = _line_chunk("src/a.py", 0, start_line=1, content="x" * 10_000)
    seeds = [_seed(huge, 1)]
    config = ContextAssemblyConfig(strategy=ContextStrategy.SEEDS_ONLY, budget_tokens=1)

    result = assemble_context(seeds, None, config)

    assert result.packed_count == 1
    assert result.estimated_tokens > config.budget_tokens


def test_expanded_requires_neighbor_loader() -> None:
    chunks = _line_file("src/a.py", 2)
    seeds = [_seed(chunks[0], 1)]
    config = ContextAssemblyConfig(strategy=ContextStrategy.EXPANDED, neighbor_radius=1)

    with pytest.raises(ContextAssemblyError, match="neighbor_loader is required"):
        assemble_context(seeds, None, config)


def test_mismatched_chunking_strategy_neighbor_is_skipped_defensively() -> None:
    seed_chunk = _line_chunk("src/a.py", 0, start_line=1, content="seed body\n" * 3)
    stale_neighbor = _fragment_chunk(
        "src/a.py",
        1,
        fragment_index=1,
        fragment_count=2,
        qualified_symbol_name="Stale.fragment",
        start_line=20,
        content="stale body\n" * 3,
    )
    seeds = [_seed(seed_chunk, 1)]
    loader = InMemoryNeighborLoader([seed_chunk, stale_neighbor])
    config = ContextAssemblyConfig(strategy=ContextStrategy.EXPANDED, neighbor_radius=1)

    result = assemble_context(seeds, loader, config)

    assert result.packed_count == 1
    assert result.expanded_candidate_count == 0


def test_assembly_is_deterministic_across_runs() -> None:
    chunks = _line_file("src/a.py", 5)
    seeds = [_seed(chunks[1], 1), _seed(chunks[3], 2)]
    loader = InMemoryNeighborLoader(chunks)
    config = ContextAssemblyConfig(strategy=ContextStrategy.EXPANDED, neighbor_radius=1)

    first = assemble_context(seeds, loader, config)
    second = assemble_context(seeds, loader, config)

    assert first.model_dump() == second.model_dump()


def test_neighbor_radius_is_bounded() -> None:
    with pytest.raises(ValueError, match="neighbor_radius"):
        ContextAssemblyConfig(neighbor_radius=4)


def test_seed_priority_is_preserved_ahead_of_all_neighbors() -> None:
    chunks_a = _line_file("src/a.py", 3)
    chunks_c = _line_file("src/c.py", 3)
    seeds = [_seed(chunks_a[1], 1), _seed(chunks_c[1], 2)]
    loader = InMemoryNeighborLoader([*chunks_a, *chunks_c])
    config = ContextAssemblyConfig(strategy=ContextStrategy.EXPANDED, neighbor_radius=1)

    result = assemble_context(seeds, loader, config)

    seed_positions = [i for i, c in enumerate(result.chunks) if c.origin is ContextOrigin.SEED]
    neighbor_positions = [i for i, c in enumerate(result.chunks) if c.origin is not ContextOrigin.SEED]
    assert max(seed_positions) < min(neighbor_positions)
