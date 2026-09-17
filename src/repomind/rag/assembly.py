"""Bounded neighbor expansion, deduplication, and budget-aware context packing.

Retrieval answers "what chunks are relevant?" This module answers a separate
question: "what evidence should actually be sent to the model?" It never
reorders or rescopes seed relevance; it only adds bounded, clearly labeled
neighbor evidence around already-ranked seeds and then packs a deterministic,
budget-bounded final context.
"""

from collections.abc import Sequence
from typing import Protocol

from repomind.ingestion import ChunkKind, CodeChunk
from repomind.rag.models import (
    AssembledContextChunk,
    AssembledContextResult,
    ContextAssemblyConfig,
    ContextOrigin,
    ContextStrategy,
)
from repomind.rag.tokens import estimate_tokens
from repomind.retrieval import ChunkIdentity, RankedChunk, chunk_identity

NeighborKey = tuple[str, int]


class ContextAssemblyError(ValueError):
    """Raised when context-assembly inputs violate their contract."""


class NeighborLoader(Protocol):
    """Batched lookup of specific (relative_path, chunk_index) chunks."""

    def __call__(self, keys: Sequence[NeighborKey]) -> Sequence[CodeChunk]:
        """Return whichever requested chunks exist, in any order, without duplicates."""


class InMemoryNeighborLoader:
    """Neighbor lookup over an already-known in-memory chunk corpus."""

    def __init__(self, chunks: Sequence[CodeChunk]) -> None:
        self._by_key: dict[NeighborKey, CodeChunk] = {
            (chunk.relative_path.as_posix(), chunk.chunk_index): chunk for chunk in chunks
        }

    def __call__(self, keys: Sequence[NeighborKey]) -> Sequence[CodeChunk]:
        seen: set[NeighborKey] = set()
        found: list[CodeChunk] = []
        for key in keys:
            chunk = self._by_key.get(key)
            if chunk is not None and key not in seen:
                seen.add(key)
                found.append(chunk)
        return found


def _offsets_nearest_first(radius: int) -> tuple[int, ...]:
    offsets = [offset for offset in range(-radius, radius + 1) if offset != 0]
    return tuple(sorted(offsets, key=lambda offset: (abs(offset), offset)))


def _neighbor_keys(chunk: CodeChunk, radius: int) -> list[NeighborKey]:
    path = chunk.relative_path.as_posix()
    return [
        (path, chunk.chunk_index + offset)
        for offset in _offsets_nearest_first(radius)
        if chunk.chunk_index + offset >= 0
    ]


def _classify_origin(seed_chunk: CodeChunk, neighbor_chunk: CodeChunk) -> ContextOrigin:
    if (
        seed_chunk.chunk_kind is ChunkKind.STRUCTURAL_FRAGMENT
        and neighbor_chunk.chunk_kind is ChunkKind.STRUCTURAL_FRAGMENT
        and seed_chunk.qualified_symbol_name is not None
        and seed_chunk.qualified_symbol_name == neighbor_chunk.qualified_symbol_name
    ):
        return ContextOrigin.SAME_SYMBOL_FRAGMENT
    return ContextOrigin.NEIGHBOR


def _is_contained(inner: CodeChunk, outer: CodeChunk) -> bool:
    return (
        inner.relative_path == outer.relative_path
        and outer.start_line <= inner.start_line
        and inner.end_line <= outer.end_line
    )


_ORIGIN_PRIORITY = {ContextOrigin.SAME_SYMBOL_FRAGMENT: 0, ContextOrigin.NEIGHBOR: 1}


class _Candidate:
    __slots__ = ("chunk", "distance", "identity", "origin", "seed_rank")

    def __init__(
        self,
        chunk: CodeChunk,
        origin: ContextOrigin,
        seed_rank: int | None,
        distance: int,
    ) -> None:
        self.chunk = chunk
        self.origin = origin
        self.seed_rank = seed_rank
        self.distance = distance
        self.identity: ChunkIdentity = chunk_identity(chunk)


def _format_block(candidate_chunk: CodeChunk, source_id: str) -> str:
    # A cheap, stable proxy for the eventual prompt block so budget accounting
    # reflects wrapper overhead (path/lines/symbol), not raw source text alone.
    language = f"<language>{candidate_chunk.language}</language>\n" if candidate_chunk.language else ""
    symbol = (
        f"<symbol>{candidate_chunk.qualified_symbol_name}</symbol>\n"
        if candidate_chunk.qualified_symbol_name
        else ""
    )
    return (
        f'<source id="{source_id}">\n'
        f"<path>{candidate_chunk.relative_path.as_posix()}</path>\n"
        f"<lines>{candidate_chunk.start_line}-{candidate_chunk.end_line}</lines>\n"
        f"{language}{symbol}"
        '<content trust="untrusted-data" encoding="verbatim">\n'
        f"{candidate_chunk.content}"
        "</content>\n"
        "</source>"
    )


def assemble_context(
    seeds: Sequence[RankedChunk],
    neighbor_loader: NeighborLoader | None,
    config: ContextAssemblyConfig,
) -> AssembledContextResult:
    """Expand, deduplicate, and budget-pack seeds into a final ordered context.

    Seed relevance is always primary: every seed is considered for packing
    before any neighbor, regardless of budget pressure from lower-ranked
    seeds' neighbors. Neighbors are ordered same-symbol-fragment first, then
    by the rank of the seed that produced them, then by proximity.
    """

    if not isinstance(config, ContextAssemblyConfig):
        raise ContextAssemblyError("config must be a ContextAssemblyConfig")

    seed_list = list(seeds)
    included_identities: set[ChunkIdentity] = set()
    included_chunks: list[CodeChunk] = []
    seed_candidates: list[_Candidate] = []

    for seed in seed_list:
        identity = chunk_identity(seed.chunk)
        if identity in included_identities:
            continue
        included_identities.add(identity)
        included_chunks.append(seed.chunk)
        seed_candidates.append(_Candidate(seed.chunk, ContextOrigin.SEED, seed.rank, 0))

    seed_count = len(seed_candidates)
    expanded_candidate_count = 0
    deduplicated_count = 0
    neighbor_pool: list[_Candidate] = []

    if config.strategy is ContextStrategy.EXPANDED and config.neighbor_radius > 0 and seed_list:
        if neighbor_loader is None:
            raise ContextAssemblyError(
                "neighbor_loader is required when context strategy is 'expanded'"
            )
        all_keys: list[NeighborKey] = []
        seen_keys: set[NeighborKey] = set()
        for seed in seed_list:
            for key in _neighbor_keys(seed.chunk, config.neighbor_radius):
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_keys.append(key)
        loaded_by_key: dict[NeighborKey, CodeChunk] = {
            (chunk.relative_path.as_posix(), chunk.chunk_index): chunk
            for chunk in neighbor_loader(all_keys)
        }

        for seed in seed_list:
            seed_chunk = seed.chunk
            path = seed_chunk.relative_path.as_posix()
            for offset in _offsets_nearest_first(config.neighbor_radius):
                neighbor_chunk = loaded_by_key.get((path, seed_chunk.chunk_index + offset))
                if neighbor_chunk is None:
                    continue
                if neighbor_chunk.chunking_strategy != seed_chunk.chunking_strategy:
                    # A mismatched strategy means the loaded row is not part of
                    # this seed's current index snapshot; skip it defensively.
                    continue
                expanded_candidate_count += 1
                identity = chunk_identity(neighbor_chunk)
                if identity in included_identities:
                    deduplicated_count += 1
                    continue
                if any(_is_contained(neighbor_chunk, existing) for existing in included_chunks):
                    deduplicated_count += 1
                    continue
                included_identities.add(identity)
                included_chunks.append(neighbor_chunk)
                neighbor_pool.append(
                    _Candidate(
                        neighbor_chunk,
                        _classify_origin(seed_chunk, neighbor_chunk),
                        seed.rank,
                        abs(offset),
                    )
                )

    neighbor_pool.sort(key=lambda c: (_ORIGIN_PRIORITY[c.origin], c.seed_rank, c.distance, c.identity))
    ordered_candidates = seed_candidates + neighbor_pool

    packed: list[AssembledContextChunk] = []
    estimated_tokens = 0
    dropped_for_budget_count = 0
    for position, candidate in enumerate(ordered_candidates, start=1):
        block = _format_block(candidate.chunk, f"S{position}")
        tokens = estimate_tokens(block)
        if packed and estimated_tokens + tokens > config.budget_tokens:
            dropped_for_budget_count += 1
            continue
        packed.append(
            AssembledContextChunk(
                chunk=candidate.chunk,
                origin=candidate.origin,
                seed_rank=candidate.seed_rank if candidate.origin is ContextOrigin.SEED else None,
                estimated_tokens=tokens,
            )
        )
        estimated_tokens += tokens

    return AssembledContextResult(
        strategy=config.strategy,
        chunks=tuple(packed),
        budget_tokens=config.budget_tokens,
        estimated_tokens=estimated_tokens,
        seed_count=seed_count,
        expanded_candidate_count=expanded_candidate_count,
        deduplicated_count=deduplicated_count,
        dropped_for_budget_count=dropped_for_budget_count,
    )
