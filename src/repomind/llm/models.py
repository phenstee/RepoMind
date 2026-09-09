"""Typed response models returned by the LLM layer."""

from typing import Literal

from pydantic import BaseModel, Field

MessageRole = Literal["system", "user", "assistant", "tool"]


class TokenUsage(BaseModel):
    """Token counts reported by the model provider."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """Normalized result returned by :class:`OpenAILLMClient`.

    Keeping this small model independent of the provider SDK makes the rest of
    the application easier to test and reason about.
    """

    content: str
    model: str = ""
    finish_reason: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
