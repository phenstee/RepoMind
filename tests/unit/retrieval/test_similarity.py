"""Tests for in-memory vector similarity helpers."""

from math import sqrt

import numpy as np
import pytest

from repomind.retrieval import SimilarityError, cosine_similarity


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 0.0], [-1.0, 0.0], -1.0),
        ([1.0, 2.0], [2.0, 4.0], 1.0),
        ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 32.0 / sqrt(14.0 * 77.0)),
    ],
)
def test_cosine_similarity_known_values(
    left: list[float], right: list[float], expected: float
) -> None:
    score = cosine_similarity(left, right)

    assert isinstance(score, float)
    assert score == pytest.approx(expected)
    assert -1.0 <= score <= 1.0


def test_cosine_similarity_accepts_numeric_numpy_vectors() -> None:
    score = cosine_similarity(
        np.array([1, 0], dtype=np.int32),
        np.array([1.0, 0.0], dtype=np.float32),
    )

    assert score == pytest.approx(1.0)


def test_cosine_similarity_does_not_modify_inputs() -> None:
    left = [1.0, 2.0]
    right = [3.0, 4.0]

    cosine_similarity(left, right)

    assert left == [1.0, 2.0]
    assert right == [3.0, 4.0]


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        ([1.0], [1.0, 2.0], "matching dimensions"),
        ([], [], "cannot be empty"),
        ([0.0, 0.0], [1.0, 0.0], "zero-magnitude"),
        ([1.0, 0.0], [0.0, 0.0], "zero-magnitude"),
        ([0.0, 0.0], [0.0, 0.0], "zero-magnitude"),
        ([float("nan")], [1.0], "finite"),
        ([float("inf")], [1.0], "finite"),
        ([[1.0, 0.0]], [[1.0, 0.0]], "one-dimensional"),
        (["not-a-number"], [1.0], "numeric"),
    ],
)
def test_cosine_similarity_rejects_invalid_vectors(
    left: list[object], right: list[object], message: str
) -> None:
    with pytest.raises(SimilarityError, match=message):
        cosine_similarity(left, right)  # type: ignore[arg-type]
