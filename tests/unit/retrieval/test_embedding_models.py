"""Tests for provider-independent embedding models."""

import math

import pytest
from pydantic import ValidationError

from repomind.retrieval import EmbeddingConfig, EmbeddingVector


def test_embedding_vector_accepts_valid_values_and_derives_dimensions() -> None:
    vector = EmbeddingVector(values=[0.25, -0.5, 1], model="embedding-test")

    assert vector.values == (0.25, -0.5, 1.0)
    assert vector.dimensions == 3


def test_embedding_vector_rejects_empty_values() -> None:
    with pytest.raises(ValidationError, match="cannot be empty"):
        EmbeddingVector(values=[], model="embedding-test")


def test_embedding_vector_rejects_non_sequence_values() -> None:
    with pytest.raises(ValidationError, match="sequence of numbers"):
        EmbeddingVector(values="not-a-vector", model="embedding-test")


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan, 10**1000])
def test_embedding_vector_rejects_non_finite_values(value: object) -> None:
    with pytest.raises(ValidationError, match="must be finite"):
        EmbeddingVector(values=[0.1, value], model="embedding-test")


@pytest.mark.parametrize("value", ["0.1", True, None])
def test_embedding_vector_rejects_non_numeric_values(value: object) -> None:
    with pytest.raises(ValidationError, match="must be numerical"):
        EmbeddingVector(values=[value], model="embedding-test")


def test_embedding_config_requires_positive_batch_size() -> None:
    with pytest.raises(ValidationError):
        EmbeddingConfig(batch_size=0)

    assert EmbeddingConfig(batch_size=1).batch_size == 1
