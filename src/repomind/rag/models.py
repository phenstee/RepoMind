"""Validated models for grounded repository answers and their context."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from repomind.ingestion import CodeChunk, validate_repository_relative_path

# Conservative default leaving room for system prompt, question, and answer
# inside typical small-to-mid context windows; not a provider's maximum window.
DEFAULT_CONTEXT_BUDGET_TOKENS = 4_000
DEFAULT_NEIGHBOR_RADIUS = 1
MAX_NEIGHBOR_RADIUS = 3


class ContextStrategy(StrEnum):
    """Explicit, benchmarkable context-assembly behaviors."""

    SEEDS_ONLY = "seeds_only"
    EXPANDED = "expanded"


class ContextOrigin(StrEnum):
    """Why one chunk is present in assembled context, never a relevance claim."""

    SEED = "seed"
    NEIGHBOR = "neighbor"
    SAME_SYMBOL_FRAGMENT = "same_symbol_fragment"


class ContextAssemblyConfig(BaseModel):
    """Bounds for neighbor expansion and budget-aware packing."""

    strategy: ContextStrategy = ContextStrategy.SEEDS_ONLY
    budget_tokens: int = Field(default=DEFAULT_CONTEXT_BUDGET_TOKENS, gt=0, strict=True)
    neighbor_radius: int = Field(
        default=DEFAULT_NEIGHBOR_RADIUS, ge=0, le=MAX_NEIGHBOR_RADIUS, strict=True
    )


class AssembledContextChunk(BaseModel):
    """One packed chunk plus the provenance an evaluator or trace can inspect."""

    chunk: CodeChunk
    origin: ContextOrigin
    seed_rank: int | None = Field(default=None, ge=1)
    estimated_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_seed_rank(self) -> "AssembledContextChunk":
        if (self.origin is ContextOrigin.SEED) != (self.seed_rank is not None):
            raise ValueError("seed_rank must be set if and only if origin is 'seed'")
        return self


class AssembledContextResult(BaseModel):
    """Bounded assembly outcome; safe to summarize directly into observability."""

    strategy: ContextStrategy
    chunks: tuple[AssembledContextChunk, ...]
    budget_tokens: int = Field(gt=0)
    estimated_tokens: int = Field(ge=0)
    seed_count: int = Field(ge=0)
    expanded_candidate_count: int = Field(ge=0)
    deduplicated_count: int = Field(ge=0)
    dropped_for_budget_count: int = Field(ge=0)

    @property
    def packed_count(self) -> int:
        return len(self.chunks)

    @model_validator(mode="after")
    def _validate_totals(self) -> "AssembledContextResult":
        if sum(chunk.estimated_tokens for chunk in self.chunks) != self.estimated_tokens:
            raise ValueError("estimated_tokens must equal the sum of packed chunk estimates")
        return self


class RAGConfig(BaseModel):
    """Limits for ranked retrieval and deterministic context construction."""

    top_k: int = Field(default=5, gt=0, strict=True)
    max_context_chars: int = Field(default=20_000, gt=0, strict=True)
    context_strategy: ContextStrategy = ContextStrategy.SEEDS_ONLY
    context_budget_tokens: int = Field(default=DEFAULT_CONTEXT_BUDGET_TOKENS, gt=0, strict=True)
    neighbor_radius: int = Field(
        default=DEFAULT_NEIGHBOR_RADIUS, ge=0, le=MAX_NEIGHBOR_RADIUS, strict=True
    )


class SourceCitation(BaseModel):
    """A repository-owned source location supporting an answer."""

    relative_path: Path
    start_line: int = Field(ge=1, strict=True)
    end_line: int = Field(ge=1, strict=True)

    @field_validator("relative_path")
    @classmethod
    def _validate_relative_path(cls, value: Path) -> Path:
        return validate_repository_relative_path(value)

    @model_validator(mode="after")
    def _validate_line_range(self) -> "SourceCitation":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class RepositoryAnswer(BaseModel):
    """A grounded answer with validated repository citations."""

    answer: str = Field(min_length=1)
    citations: list[SourceCitation] = Field(default_factory=list)
    insufficient_evidence: bool = False

    @model_validator(mode="after")
    def _validate_grounding(self) -> "RepositoryAnswer":
        if not self.insufficient_evidence and not self.citations:
            raise ValueError("an answer with sufficient evidence must include a citation")
        return self


class ContextSource(BaseModel):
    """A retrieved chunk paired with its deterministic prompt identifier."""

    source_id: str = Field(pattern=r"^S[1-9][0-9]*$")
    chunk: CodeChunk


class BuiltRepositoryContext(BaseModel):
    """Formatted untrusted repository data and its source-ID mapping."""

    text: str
    sources: tuple[ContextSource, ...] = Field(default_factory=tuple)


class GroundedLLMResponse(BaseModel):
    """Internal structured response requested from the generation provider."""

    answer: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)
    insufficient_evidence: bool
