"""Typed, provider-independent models for repository retrieval."""

from collections.abc import Sequence
from enum import StrEnum
from math import isfinite
from typing import Protocol

from pydantic import BaseModel, Field, field_validator, model_validator

from repomind.ingestion.models import CodeChunk


class RankedChunk(Protocol):
    """Minimal source-ranking contract shared by retrieval, reranking, and RAG."""

    chunk: CodeChunk
    rank: int


class SemanticSearchMode(StrEnum):
    """Persisted vector-search behavior selected explicitly for evaluation."""

    EXACT = "exact"
    ANN = "ann"


class EmbeddingTextStrategy(StrEnum):
    """Explicit, benchmarkable representations sent to an embedding provider."""

    RAW_SOURCE = "raw_source"
    STRUCTURAL_CONTEXT = "structural_context"


class EmbeddingConfig(BaseModel):
    """Configuration for deterministic item-count-based embedding batches."""

    batch_size: int = Field(default=64, gt=0)
    text_strategy: EmbeddingTextStrategy = EmbeddingTextStrategy.RAW_SOURCE


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


class SemanticSearchResult(BaseModel):
    """One ranked source chunk returned by semantic search."""

    chunk: CodeChunk
    score: float = Field(ge=-1.0, le=1.0)
    rank: int = Field(ge=1)

    @field_validator("score")
    @classmethod
    def _validate_score_is_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("semantic search score must be finite")
        return value


class BM25Config(BaseModel):
    """Standard BM25 saturation and document-length parameters."""

    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)

    @field_validator("k1", "b", mode="before")
    @classmethod
    def _reject_boolean_parameters(cls, value: object) -> object:
        if isinstance(value, bool):
            # Pydantic turns ValueError into a model ValidationError for callers.
            raise ValueError(  # noqa: TRY004
                "BM25 parameters must be numerical, not boolean"
            )
        return value

    @field_validator("k1")
    @classmethod
    def _validate_k1_is_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("BM25 k1 must be finite")
        return value


class BM25SearchResult(BaseModel):
    """One lexical result; its BM25 score is a ranking signal, not confidence."""

    chunk: CodeChunk
    score: float = Field(ge=0)
    rank: int = Field(ge=1)

    @field_validator("score")
    @classmethod
    def _validate_score_is_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("BM25 score must be finite")
        return value


class HybridSearchResult(BaseModel):
    """One fused chunk with its contributing semantic and lexical ranks."""

    chunk: CodeChunk
    rank: int = Field(ge=1)
    fusion_score: float = Field(gt=0)
    semantic_rank: int | None = Field(default=None, ge=1)
    lexical_rank: int | None = Field(default=None, ge=1)

    @field_validator("fusion_score")
    @classmethod
    def _validate_fusion_score_is_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("hybrid fusion score must be finite")
        return value

    @model_validator(mode="after")
    def _validate_contributing_rank(self) -> "HybridSearchResult":
        if self.semantic_rank is None and self.lexical_rank is None:
            raise ValueError("hybrid result requires at least one contributing rank")
        return self


class SymbolMatchTier(StrEnum):
    """Deterministic, explainable confidence tiers for a persisted symbol match.

    Ordered strongest first. A qualified match (``UserService.login``) is
    always stronger evidence than a simple-name match; a simple-name match
    reached through a strong-shaped candidate (snake_case/camelCase/
    PascalCase) is stronger than one reached only through a weak, common-word
    candidate.
    """

    QUALIFIED_SYMBOL = "qualified_symbol"
    SIMPLE_SYMBOL_STRONG = "simple_symbol_strong"
    SIMPLE_SYMBOL_WEAK = "simple_symbol_weak"


class SymbolSearchResult(BaseModel):
    """One chunk matched by persisted structural symbol metadata, not semantics.

    There is no cosine similarity or lexical weight here: ``match_tier`` is
    the only ranking signal, and it is never rescaled into a fabricated score.
    """

    chunk: CodeChunk
    rank: int = Field(ge=1)
    match_tier: SymbolMatchTier
    matched_identifier: str = Field(min_length=1)


class FusedSearchResult(BaseModel):
    """One chunk fused from an arbitrary number of named ranked sources.

    ``source_ranks`` records only the sources that actually contributed a
    rank for this chunk; a source's absence does not imply irrelevance.
    """

    chunk: CodeChunk
    rank: int = Field(ge=1)
    fusion_score: float = Field(gt=0)
    source_ranks: dict[str, int] = Field(default_factory=dict)

    @field_validator("fusion_score")
    @classmethod
    def _validate_fusion_score_is_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("fusion score must be finite")
        return value

    @field_validator("source_ranks")
    @classmethod
    def _validate_source_ranks(cls, value: dict[str, int]) -> dict[str, int]:
        if not value:
            raise ValueError("fused result requires at least one contributing source")
        if any(isinstance(rank, bool) or rank < 1 for rank in value.values()):
            raise ValueError("source ranks must be positive integers")
        return value

    @property
    def semantic_rank(self) -> int | None:
        return self.source_ranks.get("semantic")

    @property
    def lexical_rank(self) -> int | None:
        return self.source_ranks.get("lexical")

    @property
    def symbol_rank(self) -> int | None:
        return self.source_ranks.get("symbol")


class RerankingConfig(BaseModel):
    """Hard limits for the candidate set sent to an LLM reranker."""

    max_candidates: int = Field(default=20, gt=0, strict=True)
    max_context_chars: int = Field(default=30_000, gt=0, strict=True)


class RerankedSearchResult(BaseModel):
    """One LLM-ordered chunk with its original retrieval position preserved."""

    chunk: CodeChunk
    rank: int = Field(ge=1)
    original_rank: int = Field(ge=1)
