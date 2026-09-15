"""Exact unit tests for binary-relevance ranking metrics."""

from math import log2

import pytest

from repomind.evaluation import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_perfect_retrieval_metrics_are_one() -> None:
    gold = {"A", "B"}
    retrieved = ["A", "B", "C"]

    assert recall_at_k(gold, retrieved, k=2) == 1.0
    assert reciprocal_rank(gold, retrieved) == 1.0
    assert ndcg_at_k(gold, retrieved, k=2) == pytest.approx(1.0)


def test_partial_recall_counts_unique_relevant_items() -> None:
    assert recall_at_k({"A", "B"}, ["A", "C", "D"], k=3) == 0.5
    assert recall_at_k({"A", "B"}, ["A", "A", "A"], k=3) == 0.5


def test_first_relevant_at_rank_two_has_half_reciprocal_rank() -> None:
    assert reciprocal_rank({"A"}, ["C", "A", "D"]) == 0.5


def test_no_relevant_retrieval_has_zero_scores() -> None:
    assert recall_at_k({"A"}, ["B", "C"], k=2) == 0.0
    assert reciprocal_rank({"A"}, ["B", "C"]) == 0.0
    assert ndcg_at_k({"A"}, ["B", "C"], k=2) == 0.0


def test_binary_ndcg_uses_documented_logarithmic_discount() -> None:
    expected = (1 / log2(3)) / (1 + 1 / log2(3))

    assert ndcg_at_k({"A", "B"}, ["C", "A"], k=2) == pytest.approx(expected)


def test_duplicate_relevant_result_does_not_inflate_ndcg() -> None:
    duplicate_score = ndcg_at_k({"A", "B"}, ["A", "A"], k=2)
    one_relevant_score = ndcg_at_k({"A", "B"}, ["A", "C"], k=2)

    assert duplicate_score == pytest.approx(one_relevant_score)


def test_mean_reciprocal_rank_averages_cases() -> None:
    assert mean_reciprocal_rank([1.0, 0.5, 0.0]) == pytest.approx(0.5)


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_metrics_reject_invalid_k(k: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        recall_at_k({"A"}, ["A"], k=k)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        ndcg_at_k({"A"}, ["A"], k=k)  # type: ignore[arg-type]


def test_metrics_reject_empty_gold() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        recall_at_k(set(), ["A"], k=1)
    with pytest.raises(ValueError, match="must not be empty"):
        reciprocal_rank(set(), ["A"])
    with pytest.raises(ValueError, match="must not be empty"):
        ndcg_at_k(set(), ["A"], k=1)
