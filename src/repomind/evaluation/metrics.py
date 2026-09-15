"""Small, dependency-free binary-relevance ranking metrics."""

from collections.abc import Collection, Hashable, Sequence
from math import log2


def _validate_inputs[IdentityT: Hashable](
    gold: Collection[IdentityT], k: int
) -> set[IdentityT]:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    relevant = set(gold)
    if not relevant:
        raise ValueError("gold relevance must not be empty")
    return relevant


def recall_at_k[IdentityT: Hashable](
    gold: Collection[IdentityT],
    retrieved: Sequence[IdentityT],
    *,
    k: int,
) -> float:
    """Return unique relevant items retrieved in the first k over all gold items."""

    relevant = _validate_inputs(gold, k)
    found = relevant.intersection(retrieved[:k])
    return len(found) / len(relevant)


def first_relevant_rank[IdentityT: Hashable](
    gold: Collection[IdentityT],
    retrieved: Sequence[IdentityT],
) -> int | None:
    """Return the 1-based position of the first relevant result, if any."""

    relevant = set(gold)
    if not relevant:
        raise ValueError("gold relevance must not be empty")
    return next(
        (rank for rank, identity in enumerate(retrieved, start=1) if identity in relevant),
        None,
    )


def reciprocal_rank[IdentityT: Hashable](
    gold: Collection[IdentityT],
    retrieved: Sequence[IdentityT],
) -> float:
    """Return one over the 1-based rank of the first relevant result, or zero."""

    rank = first_relevant_rank(gold, retrieved)
    return 0.0 if rank is None else 1.0 / rank


def mean_reciprocal_rank(values: Sequence[float]) -> float:
    """Return the arithmetic mean of per-case reciprocal ranks."""

    if not values:
        raise ValueError("reciprocal-rank values must not be empty")
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("reciprocal-rank values must be between zero and one")
    return sum(values) / len(values)


def ndcg_at_k[IdentityT: Hashable](
    gold: Collection[IdentityT],
    retrieved: Sequence[IdentityT],
    *,
    k: int,
) -> float:
    """Return binary nDCG using ``sum(rel_i / log2(i + 1))`` for 1-based i.

    A relevant identity contributes at most once even if a malformed retriever
    emits duplicates. IDCG places up to ``min(len(gold), k)`` relevant items first.
    """

    relevant = _validate_inputs(gold, k)
    seen: set[IdentityT] = set()
    dcg = 0.0
    for rank, identity in enumerate(retrieved[:k], start=1):
        if identity in relevant and identity not in seen:
            dcg += 1.0 / log2(rank + 1)
            seen.add(identity)

    ideal_relevant = min(len(relevant), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    return dcg / idcg
