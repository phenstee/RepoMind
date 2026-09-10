"""Optional tiny live smoke check for the OpenAI embeddings integration.

Run with::

    uv run python scripts/manual_embedding_check.py

The script makes one single-text request only when ``OPENAI_API_KEY`` is set.
"""

from repomind.config import get_settings
from repomind.retrieval import EmbeddingError, OpenAIEmbeddingClient


def main() -> None:
    settings = get_settings()
    if not settings.openai_api_key:
        print("OPENAI_API_KEY is not set. No embedding API request was made.")
        return

    client = OpenAIEmbeddingClient(settings)
    try:
        result = client.embed_texts(["RepoMind embedding smoke check."])
    except EmbeddingError as exc:
        print(f"Live embedding check failed: {exc}")
        return

    embedding = result.embeddings[0]
    preview = ", ".join(f"{value:.4f}" for value in embedding.values[:5])
    print(f"Embedding model: {embedding.model}")
    print(f"Dimensions: {embedding.dimensions}")
    print(f"Prompt tokens: {result.usage.prompt_tokens}")
    print(f"Total tokens: {result.usage.total_tokens}")
    print(f"Vector preview: [{preview}]")


if __name__ == "__main__":
    main()
