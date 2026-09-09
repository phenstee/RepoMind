"""Convenience functions for Pydantic-backed structured outputs."""

from pydantic import BaseModel

from repomind.llm.client import OpenAILLMClient


def generate_structured[StructuredModelT: BaseModel](
    client: OpenAILLMClient,
    prompt: str,
    response_model: type[StructuredModelT],
    *,
    system_prompt: str | None = None,
) -> StructuredModelT:
    """Generate a structured response using ``client``.

    This exists primarily as a small, discoverable entry point for callers who
    prefer a function over a method on the client.
    """

    return client.generate_structured(
        prompt,
        response_model,
        system_prompt=system_prompt,
    )
