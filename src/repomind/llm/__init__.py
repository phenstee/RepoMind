"""LLM client abstractions for RepoMind."""

from typing import TYPE_CHECKING, Any

from repomind.llm.models import LLMResponse, TokenUsage

if TYPE_CHECKING:
    from repomind.llm.client import LLMError, OpenAILLMClient

__all__ = ["LLMError", "LLMResponse", "OpenAILLMClient", "TokenUsage"]


def __getattr__(name: str) -> Any:
    # Keep the shared usage models independent of SDK/instrumentation imports.
    if name in {"LLMError", "OpenAILLMClient"}:
        from repomind.llm.client import LLMError, OpenAILLMClient

        return {"LLMError": LLMError, "OpenAILLMClient": OpenAILLMClient}[name]
    raise AttributeError(name)
