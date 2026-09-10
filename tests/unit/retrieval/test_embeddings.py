"""Offline tests for OpenAI-backed embedding generation."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

from repomind.config import Settings
from repomind.ingestion.models import CodeChunk
from repomind.retrieval import (
    EmbeddingConfig,
    EmbeddingError,
    OpenAIEmbeddingClient,
)


def _settings(**overrides) -> Settings:
    values = {
        "openai_api_key": "sk-test",
        "openai_embedding_model": "embedding-test",
        "llm_timeout_seconds": 1.0,
        "llm_max_retries": 2,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _response(
    vectors: list[list[float]],
    *,
    indices: list[int] | None = None,
    model: str = "embedding-test",
    prompt_tokens: int = 3,
    total_tokens: int = 3,
    include_usage: bool = True,
):
    response = SimpleNamespace(
        data=[
            SimpleNamespace(index=index, embedding=vector)
            for index, vector in zip(indices or list(range(len(vectors))), vectors, strict=True)
        ],
        model=model,
    )
    response.usage = (
        SimpleNamespace(prompt_tokens=prompt_tokens, total_tokens=total_tokens)
        if include_usage
        else None
    )
    return response


def _sync_client(response=None):
    create = MagicMock(return_value=response or _response([[0.1, 0.2]]))
    return SimpleNamespace(embeddings=SimpleNamespace(create=create))


def _async_client(response=None):
    create = AsyncMock(return_value=response or _response([[0.1, 0.2]]))
    return SimpleNamespace(embeddings=SimpleNamespace(create=create))


def _chunk(content: str, *, path: str, chunk_index: int) -> CodeChunk:
    return CodeChunk(
        relative_path=Path(path),
        language="python",
        start_line=chunk_index * 2 + 1,
        end_line=chunk_index * 2 + 2,
        content=content,
        chunk_index=chunk_index,
    )


def test_embed_single_text_returns_normalized_vector() -> None:
    sdk_client = _sync_client(_response([[0.25, -0.5, 0.75]]))
    client = OpenAIEmbeddingClient(_settings(), client=sdk_client)

    vector = client.embed_text("hello world")

    assert vector.values == (0.25, -0.5, 0.75)
    assert vector.model == "embedding-test"
    assert vector.dimensions == 3
    assert sdk_client.embeddings.create.call_args.kwargs["input"] == ["hello world"]


def test_injected_sync_client_does_not_require_api_key() -> None:
    client = OpenAIEmbeddingClient(
        _settings(openai_api_key=None),
        client=_sync_client(_response([[0.25, 0.75]])),
    )

    assert client.embed_text("hello").values == (0.25, 0.75)


def test_embed_texts_returns_vectors_and_usage() -> None:
    sdk_client = _sync_client(
        _response(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            prompt_tokens=7,
            total_tokens=7,
        )
    )
    client = OpenAIEmbeddingClient(_settings(), client=sdk_client)

    result = client.embed_texts(["alpha", "beta", "gamma"])

    assert [embedding.values for embedding in result.embeddings] == [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.5),
    ]
    assert result.usage.prompt_tokens == 7
    assert result.usage.total_tokens == 7


def test_response_indices_restore_original_input_order() -> None:
    response = _response(
        [[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]],
        indices=[2, 0, 1],
    )
    client = OpenAIEmbeddingClient(_settings(), client=_sync_client(response))

    result = client.embed_texts(["zero", "one", "two"])

    assert [embedding.values for embedding in result.embeddings] == [
        (0.0, 0.0),
        (1.0, 1.0),
        (2.0, 2.0),
    ]


@pytest.mark.parametrize(
    ("indices", "match"),
    [
        ([0, 0, 2], "duplicate index"),
        ([0, 2], "missing indices"),
        ([0, 1, 3], "out of range"),
    ],
)
def test_malformed_response_indices_raise_embedding_error(
    indices: list[int],
    match: str,
) -> None:
    vectors = [[float(index), 0.0] for index in indices]
    response = _response(vectors, indices=indices)
    client = OpenAIEmbeddingClient(_settings(), client=_sync_client(response))

    with pytest.raises(EmbeddingError, match=match):
        client.embed_texts(["zero", "one", "two"])


def test_empty_response_vector_raises_embedding_error() -> None:
    client = OpenAIEmbeddingClient(
        _settings(),
        client=_sync_client(_response([[]])),
    )

    with pytest.raises(EmbeddingError, match="invalid vector"):
        client.embed_text("hello")


def test_inconsistent_dimensions_raise_embedding_error() -> None:
    response = _response([[0.1, 0.2], [0.3]])
    client = OpenAIEmbeddingClient(_settings(), client=_sync_client(response))

    with pytest.raises(EmbeddingError, match="inconsistent vector dimensions"):
        client.embed_texts(["alpha", "beta"])


def test_inconsistent_dimensions_across_batches_raise_embedding_error() -> None:
    sdk_client = _sync_client()
    sdk_client.embeddings.create.side_effect = [
        _response([[0.1, 0.2]], prompt_tokens=1, total_tokens=1),
        _response([[0.3]], prompt_tokens=1, total_tokens=1),
    ]
    client = OpenAIEmbeddingClient(
        _settings(),
        config=EmbeddingConfig(batch_size=1),
        client=sdk_client,
    )

    with pytest.raises(EmbeddingError, match="inconsistent vector dimensions"):
        client.embed_texts(["alpha", "beta"])


def test_empty_batch_returns_without_api_call() -> None:
    sdk_client = _sync_client()
    client = OpenAIEmbeddingClient(_settings(openai_api_key=None), client=sdk_client)

    result = client.embed_texts([])

    assert result.embeddings == ()
    assert result.usage.prompt_tokens == 0
    sdk_client.embeddings.create.assert_not_called()


def test_batch_splitting_preserves_global_order_and_aggregates_usage() -> None:
    value_by_text = {"zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0}
    sdk_client = _sync_client()

    def create(*, input: list[str], model: str):
        del model
        vectors = [[value_by_text[text], 1.0] for text in reversed(input)]
        indices = list(reversed(range(len(input))))
        return _response(
            vectors,
            indices=indices,
            prompt_tokens=len(input),
            total_tokens=len(input),
        )

    sdk_client.embeddings.create.side_effect = create
    client = OpenAIEmbeddingClient(
        _settings(),
        config=EmbeddingConfig(batch_size=2),
        client=sdk_client,
    )

    result = client.embed_texts(["zero", "one", "two", "three", "four"])

    assert [call.kwargs["input"] for call in sdk_client.embeddings.create.call_args_list] == [
        ["zero", "one"],
        ["two", "three"],
        ["four"],
    ]
    assert [embedding.values[0] for embedding in result.embeddings] == [0, 1, 2, 3, 4]
    assert result.usage.prompt_tokens == 5
    assert result.usage.total_tokens == 5


def test_configured_embedding_model_is_sent_to_api() -> None:
    sdk_client = _sync_client()
    client = OpenAIEmbeddingClient(
        _settings(openai_embedding_model="configured-embedding-model"),
        client=sdk_client,
    )

    client.embed_text("hello")

    assert sdk_client.embeddings.create.call_args.kwargs["model"] == "configured-embedding-model"


def test_missing_usage_defaults_to_zero() -> None:
    response = _response([[0.1, 0.2]], include_usage=False)
    client = OpenAIEmbeddingClient(_settings(), client=_sync_client(response))

    result = client.embed_texts(["hello"])

    assert result.usage.prompt_tokens == 0
    assert result.usage.total_tokens == 0


def test_empty_text_is_rejected() -> None:
    client = OpenAIEmbeddingClient(_settings(), client=_sync_client())

    with pytest.raises(EmbeddingError, match="cannot be empty"):
        client.embed_text("")


def test_whitespace_only_text_is_preserved_and_allowed() -> None:
    sdk_client = _sync_client()
    client = OpenAIEmbeddingClient(_settings(), client=sdk_client)

    client.embed_text("  \n")

    assert sdk_client.embeddings.create.call_args.kwargs["input"] == ["  \n"]


def test_embed_chunks_preserves_content_order_and_metadata() -> None:
    chunks = [
        _chunk("def first():\n    pass\n", path="src/first.py", chunk_index=0),
        _chunk("def second():\r\n\treturn 2\r\n", path="src/second.py", chunk_index=1),
    ]
    response = _response([[1.0, 0.0], [0.0, 1.0]])
    sdk_client = _sync_client(response)
    client = OpenAIEmbeddingClient(_settings(), client=sdk_client)

    embedded = client.embed_chunks(chunks)

    assert sdk_client.embeddings.create.call_args.kwargs["input"] == [
        chunks[0].content,
        chunks[1].content,
    ]
    assert [item.chunk for item in embedded] == chunks
    assert embedded[0].chunk.relative_path == Path("src/first.py")
    assert embedded[0].chunk.language == "python"
    assert embedded[0].chunk.start_line == 1
    assert embedded[0].chunk.end_line == 2
    assert embedded[1].chunk.chunk_index == 1
    assert [item.embedding.values for item in embedded] == [(1.0, 0.0), (0.0, 1.0)]


def test_embed_empty_chunks_does_not_call_api() -> None:
    sdk_client = _sync_client()
    client = OpenAIEmbeddingClient(_settings(openai_api_key=None), client=sdk_client)

    assert client.embed_chunks([]) == []
    sdk_client.embeddings.create.assert_not_called()


def test_sync_retry_then_success(monkeypatch) -> None:
    connection_error = openai.APIConnectionError(request=MagicMock())
    sdk_client = _sync_client()
    sdk_client.embeddings.create.side_effect = [connection_error, _response([[0.1, 0.2]])]
    sleep = MagicMock()
    monkeypatch.setattr("repomind.retrieval.embeddings._sleep_for_retry", sleep)
    client = OpenAIEmbeddingClient(_settings(llm_max_retries=2), client=sdk_client)

    result = client.embed_text("hello")

    assert result.values == (0.1, 0.2)
    assert sdk_client.embeddings.create.call_count == 2
    sleep.assert_called_once_with(0, base_delay=1.0)


def test_sync_retry_exhaustion_raises_embedding_error(monkeypatch) -> None:
    connection_error = openai.APIConnectionError(request=MagicMock())
    sdk_client = _sync_client()
    sdk_client.embeddings.create.side_effect = connection_error
    monkeypatch.setattr("repomind.retrieval.embeddings._sleep_for_retry", MagicMock())
    client = OpenAIEmbeddingClient(_settings(llm_max_retries=1), client=sdk_client)

    with pytest.raises(EmbeddingError, match="failed after 1 retries"):
        client.embed_text("hello")

    assert sdk_client.embeddings.create.call_count == 2


def test_non_retryable_error_is_normalized_without_retry() -> None:
    sdk_client = _sync_client()
    sdk_client.embeddings.create.side_effect = openai.APIError(
        "bad request",
        request=MagicMock(),
        body={"error": "bad"},
    )
    client = OpenAIEmbeddingClient(_settings(), client=sdk_client)

    with pytest.raises(EmbeddingError, match="OpenAI embedding API error"):
        client.embed_text("hello")

    assert sdk_client.embeddings.create.call_count == 1


@pytest.mark.asyncio
async def test_async_embedding_works() -> None:
    sdk_client = _async_client(_response([[0.4, 0.6]]))
    client = OpenAIEmbeddingClient(_settings(), async_client=sdk_client)

    result = await client.aembed_text("hello")

    assert result.values == (0.4, 0.6)
    assert sdk_client.embeddings.create.await_count == 1


@pytest.mark.asyncio
async def test_injected_async_client_does_not_require_api_key() -> None:
    client = OpenAIEmbeddingClient(
        _settings(openai_api_key=None),
        async_client=_async_client(_response([[0.4, 0.6]])),
    )

    assert (await client.aembed_text("hello")).values == (0.4, 0.6)


@pytest.mark.asyncio
async def test_async_retry_uses_non_blocking_sleep(monkeypatch) -> None:
    connection_error = openai.APIConnectionError(request=MagicMock())
    sdk_client = _async_client()
    sdk_client.embeddings.create.side_effect = [
        connection_error,
        _response([[0.4, 0.6]]),
    ]
    async_sleep = AsyncMock()
    blocking_sleep = MagicMock(side_effect=AssertionError("time.sleep called from async code"))
    monkeypatch.setattr("repomind.retrieval.embeddings.asyncio.sleep", async_sleep)
    monkeypatch.setattr("repomind.retrieval.embeddings.time.sleep", blocking_sleep)
    client = OpenAIEmbeddingClient(_settings(llm_max_retries=2), async_client=sdk_client)

    result = await client.aembed_text("hello")

    assert result.values == (0.4, 0.6)
    assert sdk_client.embeddings.create.await_count == 2
    async_sleep.assert_awaited_once_with(1.0)
    blocking_sleep.assert_not_called()
