"""Typed, provider-independent models for embedding generation."""

from collections.abc import Sequence
from math import isfinite

from pydantic import BaseModel, Field, field_validator

from repomind.ingestion.models import CodeChunk


class EmbeddingConfig(BaseModel):
    """Configuration for deterministic item-count-based embedding batches."""

    batch_size: int = Field(default=64, gt=0)


class EmbeddingUsage(BaseModel):
    """Token counts reported for one or more embedding requests."""

    prompt_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class EmbeddingVector(BaseModel):
    """A validated embedding vector and the model that produced it."""

    values: tuple[float, ...]
    model: str = Field(min_length=1)

    @field_validator("values", mode="before")
    @classmethod
    def _validate_values(cls, value: object) -> tuple[float, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            # Pydantic turns ValueError into a model ValidationError for callers.
            raise ValueError("embedding values must be a sequence of numbers")  # noqa: TRY004

        values = tuple(value)
        if not values:
            raise ValueError("embedding vector cannot be empty")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in values):
            raise ValueError("embedding values must be numerical")

        try:
            normalized = tuple(float(item) for item in values)
        except OverflowError as exc:
            raise ValueError("embedding values must be finite") from exc
        if not all(isfinite(item) for item in normalized):
            raise ValueError("embedding values must be finite")
        return normalized

    @property
    def dimensions(self) -> int:
        """Return the vector dimensionality without storing redundant metadata."""

        return len(self.values)


class EmbeddingBatchResult(BaseModel):
    """Normalized vectors and aggregate usage for an embedding operation."""

    embeddings: tuple[EmbeddingVector, ...] = Field(default_factory=tuple)
    usage: EmbeddingUsage = Field(default_factory=EmbeddingUsage)


class EmbeddedChunk(BaseModel):
    """An embedding kept together with its original source chunk."""

    chunk: CodeChunk
    embedding: EmbeddingVector
