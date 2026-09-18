"""In-memory persisted-symbol-metadata retrieval, no AST/graph work at query time.

Symbol retrieval answers a narrow question: did the user name a code
identifier that a chunker already labeled? It only ever reads
``symbol_name``/``qualified_symbol_name`` already produced by Milestone 22
chunking; it never reparses source or fabricates a semantic similarity score.
"""

from collections.abc import Sequence

from repomind.ingestion import CodeChunk
from repomind.retrieval.identifiers import IdentifierCandidate, IdentifierConfidence
from repomind.retrieval.models import RankedChunk, SymbolMatchTier, SymbolSearchResult
from repomind.retrieval.similarity import RetrievalError

DEFAULT_SYMBOL_CANDIDATE_LIMIT = 10
MAX_SYMBOL_CANDIDATE_LIMIT = 50


class SymbolSearchError(RetrievalError):
    """Raised when symbol-search inputs are invalid."""


def is_exact_identifier_query(
    query: str, candidates: Sequence[IdentifierCandidate]
) -> bool:
    """Return whether the complete query exactly equals an extracted identifier."""

    return query.strip() in {candidate.text for candidate in candidates}


def _symbol_identity(chunk: CodeChunk) -> tuple[str, str]:
    return (
        chunk.relative_path.as_posix(),
        chunk.qualified_symbol_name or chunk.symbol_name or "",
    )


def select_evidence_backed_symbol_fragments(
    symbol_results: Sequence[SymbolSearchResult],
    evidence_results: Sequence[RankedChunk],
) -> list[SymbolSearchResult]:
    """Project each matched symbol onto its best already-retrieved fragment.

    Symbol lookup remains responsible for choosing the symbol and assigning
    its match rank. When semantic/BM25 retrieval has already found a different
    fragment of that same symbol, its fused evidence order chooses the chunk
    which receives the symbol vote. No additional chunks are scanned or added.
    """

    best_evidence: dict[tuple[str, str], CodeChunk] = {}
    for result in evidence_results:
        identity = _symbol_identity(result.chunk)
        if identity[1]:
            best_evidence.setdefault(identity, result.chunk)

    return [
        result.model_copy(
            update={"chunk": best_evidence.get(_symbol_identity(result.chunk), result.chunk)}
        )
        for result in symbol_results
    ]


def _validate_limit(limit: int) -> None:
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit <= 0
        or limit > MAX_SYMBOL_CANDIDATE_LIMIT
    ):
        raise SymbolSearchError(
            f"limit must be a positive integer no greater than {MAX_SYMBOL_CANDIDATE_LIMIT}"
        )


def _tier_and_identifier(
    chunk: CodeChunk,
    qualified_texts: frozenset[str],
    strong_texts: frozenset[str],
    weak_texts: frozenset[str],
) -> tuple[SymbolMatchTier, str] | None:
    if chunk.qualified_symbol_name is not None and chunk.qualified_symbol_name in qualified_texts:
        return SymbolMatchTier.QUALIFIED_SYMBOL, chunk.qualified_symbol_name
    if chunk.symbol_name is not None:
        if chunk.symbol_name in strong_texts:
            return SymbolMatchTier.SIMPLE_SYMBOL_STRONG, chunk.symbol_name
        if chunk.symbol_name in weak_texts:
            return SymbolMatchTier.SIMPLE_SYMBOL_WEAK, chunk.symbol_name
    return None


def symbol_search(
    candidates: Sequence[IdentifierCandidate],
    chunks: Sequence[CodeChunk],
    *,
    limit: int = DEFAULT_SYMBOL_CANDIDATE_LIMIT,
) -> list[SymbolSearchResult]:
    """Rank already-known chunks by exact persisted-symbol match tier.

    Deterministic and bounded: matches are ordered by tier, then by relative
    path and chunk index (which places a symbol's own first fragment ahead of
    its later fragments), then truncated to ``limit``. Multiple fragments of
    one symbol in one file collapse to their first fragment so one oversized
    function cannot consume the whole candidate budget; Milestone 23's
    ``expanded`` context strategy can still recover adjacent fragments later.
    """

    _validate_limit(limit)
    if not candidates or not chunks:
        return []

    qualified_texts = frozenset(c.text for c in candidates if c.qualified)
    strong_texts = frozenset(
        c.text for c in candidates if not c.qualified and c.confidence is IdentifierConfidence.STRONG
    )
    weak_texts = frozenset(
        c.text for c in candidates if not c.qualified and c.confidence is IdentifierConfidence.WEAK
    )
    if not qualified_texts and not strong_texts and not weak_texts:
        return []

    matches: list[tuple[SymbolMatchTier, str, CodeChunk]] = []
    for chunk in chunks:
        found = _tier_and_identifier(chunk, qualified_texts, strong_texts, weak_texts)
        if found is not None:
            tier, identifier = found
            matches.append((tier, identifier, chunk))

    matches.sort(
        key=lambda item: (
            item[0].value,
            item[2].relative_path.as_posix(),
            item[2].chunk_index,
        )
    )

    seen_symbols: set[tuple[str, str]] = set()
    deduplicated: list[tuple[SymbolMatchTier, str, CodeChunk]] = []
    for tier, identifier, chunk in matches:
        dedup_key = _symbol_identity(chunk)
        if dedup_key in seen_symbols:
            continue
        seen_symbols.add(dedup_key)
        deduplicated.append((tier, identifier, chunk))

    return [
        SymbolSearchResult(chunk=chunk, rank=rank, match_tier=tier, matched_identifier=identifier)
        for rank, (tier, identifier, chunk) in enumerate(deduplicated[:limit], start=1)
    ]
