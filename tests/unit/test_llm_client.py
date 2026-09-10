"""Unit tests for the OpenAI LLM client.

All tests inject fake SDK objects so no network or paid API call is made.
"""

from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest
from pydantic import BaseModel

from repomind.config import Settings
from repomind.llm import LLMError, OpenAILLMClient


class Sentiment(BaseModel):
    label: Literal["positive", "negative"]
    confidence: float


class FakeCompletions:
    def __init__(self, response=None) -> None:
        self.response = response or self._make_response()
        self.calls: list[dict] = []

    @staticmethod
    def _make_response():
        choice = SimpleNamespace(
            message=SimpleNamespace(content="hello"),
            finish_reason="stop",
        )
        return SimpleNamespace(
            choices=[choice],
            model="fake-model",
            usage=SimpleNamespace(
                prompt_tokens=2,
                completion_tokens=3,
                total_tokens=5,
            ),
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeChat:
    def __init__(self, response=None) -> None:
        self.completions = FakeCompletions(response)


class FakeClient:
    def __init__(self, response=None) -> None:
        self.chat = FakeChat(response)


class FakeAsyncCompletions(FakeCompletions):
    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeAsyncChat:
    def __init__(self, response=None) -> None:
        self.completions = FakeAsyncCompletions(response)


class FakeAsyncClient:
    def __init__(self, response=None) -> None:
        self.chat = FakeAsyncChat(response)


class FakeParseCompletions:
    def __init__(self, parsed) -> None:
        self.parsed = parsed
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(parsed=self.parsed)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeAsyncParseCompletions(FakeParseCompletions):
    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(parsed=self.parsed)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeBetaClient:
    def __init__(self, parsed) -> None:
        sync_parse = FakeParseCompletions(parsed)
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(completions=sync_parse),
        )


class FakeAsyncBetaClient:
    def __init__(self, parsed) -> None:
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(
                completions=FakeAsyncParseCompletions(parsed),
            ),
        )


def _settings(**overrides) -> Settings:
    values = {
        "openai_api_key": "sk-test",
        "openai_model": "fake-model",
        "llm_timeout_seconds": 1.0,
        "llm_max_retries": 2,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_generate_returns_normalized_response() -> None:
    sync_client = FakeClient()
    client = OpenAILLMClient(
        _settings(),
        client=sync_client,
        async_client=FakeAsyncClient(),
    )

    result = client.generate(
        "Say hello",
        system_prompt="Be brief",
        temperature=0.2,
    )

    assert result.content == "hello"
    assert result.model == "fake-model"
    assert result.finish_reason == "stop"
    assert result.usage.prompt_tokens == 2
    assert result.usage.completion_tokens == 3
    assert result.usage.total_tokens == 5
    assert sync_client.chat.completions.calls[0]["model"] == "fake-model"
    assert sync_client.chat.completions.calls[0]["temperature"] == 0.2


def test_generate_structured_returns_pydantic_model() -> None:
    parsed = Sentiment(label="positive", confidence=0.94)
    sync_client = FakeBetaClient(parsed)
    client = OpenAILLMClient(
        _settings(),
        client=sync_client,
        async_client=FakeAsyncBetaClient(parsed),
    )

    result = client.generate_structured("Classify the text.", Sentiment)

    assert isinstance(result, Sentiment)
    assert result.label == "positive"
    assert result.confidence == pytest.approx(0.94)


def test_generate_retries_then_succeeds(monkeypatch) -> None:
    connection_error = openai.APIConnectionError(request=MagicMock())
    sync_client = FakeClient()
    sync_client.chat.completions.create = MagicMock(
        side_effect=[
            connection_error,
            sync_client.chat.completions.response,
        ]
    )
    monkeypatch.setattr(
        "repomind.llm.client._sleep_for_retry",
        lambda attempt, base_delay: None,
    )
    client = OpenAILLMClient(
        _settings(llm_max_retries=2),
        client=sync_client,
        async_client=FakeAsyncClient(),
    )

    result = client.generate("Hello")

    assert result.content == "hello"
    assert sync_client.chat.completions.create.call_count == 2


def test_generate_wraps_non_retryable_api_error() -> None:
    sync_client = FakeClient()
    sync_client.chat.completions.create = MagicMock(
        side_effect=openai.APIError(
            "bad request",
            request=MagicMock(),
            body={"error": "bad"},
        )
    )
    client = OpenAILLMClient(
        _settings(llm_max_retries=0),
        client=sync_client,
        async_client=FakeAsyncClient(),
    )

    with pytest.raises(LLMError, match="OpenAI API error"):
        client.generate("Hello")


def test_client_requires_api_key_when_sync_client_is_first_needed() -> None:
    client = OpenAILLMClient(_settings(openai_api_key=None))

    with pytest.raises(LLMError, match="API key is not configured"):
        client.generate("Hello")


def test_injected_sync_only_client_works_without_api_key() -> None:
    client = OpenAILLMClient(
        _settings(openai_api_key=None),
        client=FakeClient(),
    )

    assert client.generate("Hello").content == "hello"


def test_sdk_client_is_lazy_and_receives_unwrapped_api_key(monkeypatch) -> None:
    sdk_client = FakeClient()
    sdk_factory = MagicMock(return_value=sdk_client)
    monkeypatch.setattr("repomind.llm.client.OpenAI", sdk_factory)
    client = OpenAILLMClient(_settings(openai_api_key="sk-sdk-boundary-test"))

    sdk_factory.assert_not_called()
    assert client.generate("Hello").content == "hello"
    assert sdk_factory.call_args.kwargs["api_key"] == "sk-sdk-boundary-test"


def test_generate_rejects_response_without_choices() -> None:
    client = OpenAILLMClient(
        _settings(),
        client=FakeClient(SimpleNamespace(choices=[])),
    )

    with pytest.raises(LLMError, match="no completion choices"):
        client.generate("Hello")


def test_generate_rejects_response_without_usable_content() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None),
                finish_reason="stop",
            )
        ],
        model="fake-model",
        usage=None,
    )
    client = OpenAILLMClient(_settings(), client=FakeClient(response))

    with pytest.raises(LLMError, match="no usable message content"):
        client.generate("Hello")


def test_generate_structured_rejects_missing_parsed_result() -> None:
    client = OpenAILLMClient(
        _settings(),
        client=FakeBetaClient(None),
    )

    with pytest.raises(LLMError, match="no parsed structured response"):
        client.generate_structured("Classify the text.", Sentiment)


@pytest.mark.asyncio
async def test_agenerate_returns_normalized_response() -> None:
    async_client = FakeAsyncClient()
    client = OpenAILLMClient(
        _settings(),
        client=FakeClient(),
        async_client=async_client,
    )

    result = await client.agenerate("Say hello")

    assert result.content == "hello"
    assert result.usage.total_tokens == 5
    assert async_client.chat.completions.calls[0]["messages"] == [
        {"role": "user", "content": "Say hello"}
    ]


@pytest.mark.asyncio
async def test_injected_async_only_client_works_without_api_key() -> None:
    client = OpenAILLMClient(
        _settings(openai_api_key=None),
        async_client=FakeAsyncClient(),
    )

    assert (await client.agenerate("Hello")).content == "hello"


@pytest.mark.asyncio
async def test_agenerate_retries_without_blocking_sleep(monkeypatch) -> None:
    connection_error = openai.APIConnectionError(request=MagicMock())
    async_client = FakeAsyncClient()
    async_client.chat.completions.create = AsyncMock(
        side_effect=[
            connection_error,
            async_client.chat.completions.response,
        ]
    )
    async_sleep = AsyncMock()
    blocking_sleep = MagicMock(side_effect=AssertionError("time.sleep called from async code"))
    monkeypatch.setattr("repomind.llm.client.asyncio.sleep", async_sleep)
    monkeypatch.setattr("repomind.llm.client.time.sleep", blocking_sleep)
    client = OpenAILLMClient(
        _settings(llm_max_retries=2),
        async_client=async_client,
    )

    result = await client.agenerate("Hello")

    assert result.content == "hello"
    assert async_client.chat.completions.create.await_count == 2
    async_sleep.assert_awaited_once_with(1.0)
    blocking_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_agenerate_structured_returns_pydantic_model() -> None:
    parsed = Sentiment(label="negative", confidence=0.91)
    client = OpenAILLMClient(
        _settings(),
        client=FakeBetaClient(parsed),
        async_client=FakeAsyncBetaClient(parsed),
    )

    result = await client.agenerate_structured("Classify the text.", Sentiment)

    assert isinstance(result, Sentiment)
    assert result.label == "negative"


@pytest.mark.asyncio
async def test_agenerate_structured_retries_without_blocking_sleep(monkeypatch) -> None:
    connection_error = openai.APIConnectionError(request=MagicMock())
    parsed = Sentiment(label="positive", confidence=0.88)
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))]
    )
    async_client = FakeAsyncBetaClient(parsed)
    async_client.beta.chat.completions.parse = AsyncMock(
        side_effect=[connection_error, completion]
    )
    async_sleep = AsyncMock()
    blocking_sleep = MagicMock(side_effect=AssertionError("time.sleep called from async code"))
    monkeypatch.setattr("repomind.llm.client.asyncio.sleep", async_sleep)
    monkeypatch.setattr("repomind.llm.client.time.sleep", blocking_sleep)
    client = OpenAILLMClient(
        _settings(llm_max_retries=2),
        async_client=async_client,
    )

    result = await client.agenerate_structured("Classify the text.", Sentiment)

    assert result == parsed
    assert async_client.beta.chat.completions.parse.await_count == 2
    async_sleep.assert_awaited_once_with(1.0)
    blocking_sleep.assert_not_called()
