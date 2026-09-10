"""OpenAI-backed embedding generation with deterministic batching."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any

import openai
from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError

from repomind.config import Settings, get_settings
from repomind.ingestion.models import CodeChunk
from repomind.retrieval.models import (
    EmbeddedChunk,
    EmbeddingBatchResult,
    EmbeddingConfig,
    EmbeddingUsage,
    EmbeddingVector,
)

logger = logging.getLogger(__name__)

_RETRYABLE_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails or returns invalid data."""


def _retry_delay(attempt: int, base_delay: float) -> float:
    return min(base_delay * (2**attempt), 8.0)


def _sleep_for_retry(attempt: int, base_delay: float) -> None:
    delay = _retry_delay(attempt, base_delay)
    logger.warning("Retrying embedding request in %.2f seconds", delay)
    time.sleep(delay)


async def _sleep_for_retry_async(attempt: int, base_delay: float) -> None:
    delay = _retry_delay(attempt, base_delay)
    logger.warning("Retrying embedding request in %.2f seconds", delay)
    await asyncio.sleep(delay)


def _usage_from(openai_usage: Any | None) -> EmbeddingUsage:
    if openai_usage is None:
        return EmbeddingUsage()
    return EmbeddingUsage(
        prompt_tokens=getattr(openai_usage, "prompt_tokens", 0) or 0,
        total_tokens=getattr(openai_usage, "total_tokens", 0) or 0,
    )


def _validate_texts(texts: Sequence[str]) -> tuple[str, ...]:
    if isinstance(texts, (str, bytes)):
        raise EmbeddingError("Embedding batch input must be a sequence of strings")

    validated: list[str] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise EmbeddingError(f"Embedding input at index {index} must be a string")
        if text == "":
            raise EmbeddingError(f"Embedding input at index {index} cannot be empty")
        validated.append(text)
    return tuple(validated)


def _ensure_consistent_dimensions(embeddings: Sequence[EmbeddingVector]) -> None:
    dimensions = {embedding.dimensions for embedding in embeddings}
    if len(dimensions) > 1:
        raise EmbeddingError("Embedding response contains inconsistent vector dimensions")


def _normalize_response(response: Any, expected_count: int) -> EmbeddingBatchResult:
    data = getattr(response, "data", None)
    if data is None:
        raise EmbeddingError("Embedding response contains no data")

    response_model = getattr(response, "model", None)
    if not isinstance(response_model, str) or not response_model:
        raise EmbeddingError("Embedding response contains no model name")

    embeddings_by_index: dict[int, EmbeddingVector] = {}
    for item in data:
        index = getattr(item, "index", None)
        if isinstance(index, bool) or not isinstance(index, int):
            raise EmbeddingError("Embedding response contains a non-integer index")
        if index < 0 or index >= expected_count:
            raise EmbeddingError(f"Embedding response index {index} is out of range")
        if index in embeddings_by_index:
            raise EmbeddingError(f"Embedding response contains duplicate index {index}")

        try:
            vector = EmbeddingVector(
                values=getattr(item, "embedding", None),
                model=response_model,
            )
        except (TypeError, ValidationError, ValueError) as exc:
            raise EmbeddingError(
                f"Embedding response contains an invalid vector at index {index}"
            ) from exc
        embeddings_by_index[index] = vector

    expected_indices = set(range(expected_count))
    missing_indices = sorted(expected_indices - embeddings_by_index.keys())
    if missing_indices:
        raise EmbeddingError(f"Embedding response is missing indices: {missing_indices}")

    ordered = tuple(embeddings_by_index[index] for index in range(expected_count))
    _ensure_consistent_dimensions(ordered)
    try:
        usage = _usage_from(getattr(response, "usage", None))
    except ValidationError as exc:
        raise EmbeddingError("Embedding response contains invalid usage metadata") from exc
    return EmbeddingBatchResult(embeddings=ordered, usage=usage)


class OpenAIEmbeddingClient:
    """Generate normalized embeddings using the official OpenAI SDK."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        config: EmbeddingConfig | None = None,
        client: OpenAI | None = None,
        async_client: AsyncOpenAI | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.config = config or EmbeddingConfig()
        self.model = self.settings.openai_embedding_model
        self.timeout = self.settings.llm_timeout_seconds
        self.max_retries = self.settings.llm_max_retries
        self._client = client
        self._async_client = async_client

    def _client_kwargs(self) -> dict[str, Any]:
        secret = self.settings.openai_api_key
        api_key = secret.get_secret_value() if secret is not None else ""
        if not api_key:
            raise EmbeddingError(
                "OpenAI API key is not configured. Set OPENAI_API_KEY or pass a client."
            )
        return {
            "timeout": self.timeout,
            "max_retries": 0,
            "api_key": api_key,
        }

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(**self._client_kwargs())
        return self._client

    def _get_async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(**self._client_kwargs())
        return self._async_client

    def embed_text(self, text: str) -> EmbeddingVector:
        """Embed one non-empty string without normalizing its contents."""

        return self.embed_texts([text]).embeddings[0]

    def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatchResult:
        """Embed strings in deterministic item-count batches and preserve order."""

        validated = _validate_texts(texts)
        if not validated:
            return EmbeddingBatchResult()

        embeddings: list[EmbeddingVector] = []
        prompt_tokens = 0
        total_tokens = 0
        for start in range(0, len(validated), self.config.batch_size):
            batch = validated[start : start + self.config.batch_size]
            result = self._request_batch(batch)
            embeddings.extend(result.embeddings)
            prompt_tokens += result.usage.prompt_tokens
            total_tokens += result.usage.total_tokens

        _ensure_consistent_dimensions(embeddings)
        return EmbeddingBatchResult(
            embeddings=tuple(embeddings),
            usage=EmbeddingUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
            ),
        )

    async def aembed_text(self, text: str) -> EmbeddingVector:
        """Async equivalent of :meth:`embed_text`."""

        return (await self.aembed_texts([text])).embeddings[0]

    async def aembed_texts(self, texts: Sequence[str]) -> EmbeddingBatchResult:
        """Async equivalent of :meth:`embed_texts`."""

        validated = _validate_texts(texts)
        if not validated:
            return EmbeddingBatchResult()

        embeddings: list[EmbeddingVector] = []
        prompt_tokens = 0
        total_tokens = 0
        for start in range(0, len(validated), self.config.batch_size):
            batch = validated[start : start + self.config.batch_size]
            result = await self._request_batch_async(batch)
            embeddings.extend(result.embeddings)
            prompt_tokens += result.usage.prompt_tokens
            total_tokens += result.usage.total_tokens

        _ensure_consistent_dimensions(embeddings)
        return EmbeddingBatchResult(
            embeddings=tuple(embeddings),
            usage=EmbeddingUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
            ),
        )

    def embed_chunks(self, chunks: Sequence[CodeChunk]) -> list[EmbeddedChunk]:
        """Embed exact chunk contents while preserving chunk order and metadata."""

        if not chunks:
            return []
        result = self.embed_texts([chunk.content for chunk in chunks])
        return [
            EmbeddedChunk(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, result.embeddings, strict=True)
        ]

    async def aembed_chunks(self, chunks: Sequence[CodeChunk]) -> list[EmbeddedChunk]:
        """Async equivalent of :meth:`embed_chunks`."""

        if not chunks:
            return []
        result = await self.aembed_texts([chunk.content for chunk in chunks])
        return [
            EmbeddedChunk(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, result.embeddings, strict=True)
        ]

    def _request_batch(self, texts: Sequence[str]) -> EmbeddingBatchResult:
        for attempt in range(self.max_retries + 1):
            try:
                response = self._get_client().embeddings.create(
                    input=list(texts),
                    model=self.model,
                )
                return _normalize_response(response, len(texts))
            except _RETRYABLE_ERRORS as exc:
                if attempt >= self.max_retries:
                    raise EmbeddingError(
                        f"Embedding request failed after {self.max_retries} retries"
                    ) from exc
                _sleep_for_retry(attempt, base_delay=1.0)
            except openai.APIError as exc:
                raise EmbeddingError(f"OpenAI embedding API error: {exc}") from exc

        raise EmbeddingError("Embedding request failed unexpectedly")

    async def _request_batch_async(self, texts: Sequence[str]) -> EmbeddingBatchResult:
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._get_async_client().embeddings.create(
                    input=list(texts),
                    model=self.model,
                )
                return _normalize_response(response, len(texts))
            except _RETRYABLE_ERRORS as exc:
                if attempt >= self.max_retries:
                    raise EmbeddingError(
                        f"Embedding request failed after {self.max_retries} retries"
                    ) from exc
                await _sleep_for_retry_async(attempt, base_delay=1.0)
            except openai.APIError as exc:
                raise EmbeddingError(f"OpenAI embedding API error: {exc}") from exc

        raise EmbeddingError("Embedding request failed unexpectedly")
