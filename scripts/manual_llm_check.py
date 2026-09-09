"""Optional manual smoke test for the live OpenAI integration.

This script is intentionally separate from the unit tests. It makes real API
calls only when OPENAI_API_KEY is configured, so it should never run as part of
``pytest``.

Run it with::

    uv run python scripts/manual_llm_check.py
"""

from pydantic import BaseModel

from repomind.config import get_settings
from repomind.llm.client import LLMError, OpenAILLMClient


class BuildStep(BaseModel):
    title: str
    summary: str


def main() -> None:
    settings = get_settings()
    if not settings.openai_api_key:
        print("OPENAI_API_KEY is not set. Create a local .env file and try again.")
        return

    client = OpenAILLMClient(settings)

    try:
        text = client.generate("In one sentence, what is repository-aware RAG?")
        print("\nText response:\n", text.content)
        print("Usage:", text.usage.model_dump())

        step = client.generate_structured(
            "Propose the first implementation step for a code understanding agent.",
            BuildStep,
        )
        print("\nStructured response:\n", step.model_dump_json(indent=2))
    except LLMError as exc:
        print(f"Live LLM check failed: {exc}")


if __name__ == "__main__":
    main()
