"""Validated models for grounded repository answers and their context."""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from repomind.ingestion import CodeChunk, validate_repository_relative_path


class RAGConfig(BaseModel):
    """Limits for ranked retrieval and deterministic context construction."""

    top_k: int = Field(default=5, gt=0, strict=True)
    max_context_chars: int = Field(default=20_000, gt=0, strict=True)


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
