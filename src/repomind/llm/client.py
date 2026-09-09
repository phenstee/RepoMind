"""OpenAI-backed LLM client.

The client deliberately exposes a small, testable surface. It wraps the
official OpenAI SDK and normalizes its result into RepoMind's own types.
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypeVar

import openai
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from repomind.config import Settings, get_settings
from repomind.llm.models import LLMResponse, TokenUsage

logger = logging.getLogger(__name__)

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)

_RETRYABLE_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class LLMError(RuntimeError):
    """Raised when an LLM request fails or returns an unusable response."""


def _usage_from(openai_usage: Any | None) -> TokenUsage:
    if openai_usage is None:
        return TokenUsage()

    return TokenUsage(
        prompt_tokens=getattr(openai_usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(openai_usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(openai_usage, "total_tokens", 0) or 0,
    )


def _build_messages(
    prompt: str,
    *,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _sleep_for_retry(attempt: int, base_delay: float) -> None:
    delay = min(base_delay * (2**attempt), 8.0)
    logger.warning("Retrying LLM request in %.2f seconds", delay)
    time.sleep(delay)


class OpenAILLMClient:
    """Reusable synchronous/async client around the OpenAI Python SDK."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: OpenAI | None = None,
        async_client: AsyncOpenAI | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model = self.settings.openai_model
        self.timeout = self.settings.llm_timeout_seconds
        self.max_retries = self.settings.llm_max_retries

        self._client = client
        self._async_client = async_client
        if self._client is None:
            self._client = self._make_client(OpenAI)
        if self._async_client is None:
            self._async_client = self._make_client(AsyncOpenAI)

    def _make_client(self, client_class: type[OpenAI | AsyncOpenAI]) -> OpenAI | AsyncOpenAI:
        if not self.settings.openai_api_key:
            raise LLMError(
                "OpenAI API key is not configured. Set OPENAI_API_KEY or pass a client."
            )
        kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            "max_retries": 0,
            "api_key": self.settings.openai_api_key,
        }
        return client_class(**kwargs)

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a plain text completion for ``prompt``."""

        messages = _build_messages(prompt, system_prompt=system_prompt)
        return self._complete(self._client, messages, temperature, max_tokens)

    async def agenerate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Async equivalent of :meth:`generate`."""

        messages = _build_messages(prompt, system_prompt=system_prompt)
        return await self._complete_async(
            self._async_client, messages, temperature, max_tokens
        )

    def generate_structured(
        self,
        prompt: str,
        response_model: type[StructuredModelT],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> StructuredModelT:
        """Generate a response and validate it against ``response_model``."""

        messages = _build_messages(prompt, system_prompt=system_prompt)
        kwargs: dict[str, Any] = {"messages": messages, "model": self.model}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        for attempt in range(self.max_retries + 1):
            try:
                completion = self._client.beta.chat.completions.parse(
                    response_format=response_model,
                    **kwargs,
                )
                parsed = completion.choices[0].message.parsed
                if parsed is None:
                    raise LLMError("Model returned no parsed structured response")
                return parsed
            except _RETRYABLE_ERRORS as exc:
                if attempt >= self.max_retries:
                    raise LLMError(
                        f"Structured LLM request failed after {self.max_retries} retries"
                    ) from exc
                _sleep_for_retry(attempt, base_delay=1.0)
            except openai.APIError as exc:
                raise LLMError(f"OpenAI API error: {exc}") from exc

        raise LLMError("Structured LLM request failed unexpectedly")

    async def agenerate_structured(
        self,
        prompt: str,
        response_model: type[StructuredModelT],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> StructuredModelT:
        """Async equivalent of :meth:`generate_structured`."""

        messages = _build_messages(prompt, system_prompt=system_prompt)
        kwargs: dict[str, Any] = {"messages": messages, "model": self.model}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        for attempt in range(self.max_retries + 1):
            try:
                completion = await self._async_client.beta.chat.completions.parse(
                    response_format=response_model,
                    **kwargs,
                )
                parsed = completion.choices[0].message.parsed
                if parsed is None:
                    raise LLMError("Model returned no parsed structured response")
                return parsed
            except _RETRYABLE_ERRORS as exc:
                if attempt >= self.max_retries:
                    raise LLMError(
                        f"Structured LLM request failed after {self.max_retries} retries"
                    ) from exc
                _sleep_for_retry(attempt, base_delay=1.0)
            except openai.APIError as exc:
                raise LLMError(f"OpenAI API error: {exc}") from exc

        raise LLMError("Structured LLM request failed unexpectedly")

    def _complete(
        self,
        client: OpenAI,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {"messages": messages, "model": self.model}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        for attempt in range(self.max_retries + 1):
            try:
                response = client.chat.completions.create(**kwargs)
                return self._response_to_llm_response(response)
            except _RETRYABLE_ERRORS as exc:
                if attempt >= self.max_retries:
                    raise LLMError(
                        f"LLM request failed after {self.max_retries} retries"
                    ) from exc
                _sleep_for_retry(attempt, base_delay=1.0)
            except openai.APIError as exc:
                raise LLMError(f"OpenAI API error: {exc}") from exc

        raise LLMError("LLM request failed unexpectedly")

    async def _complete_async(
        self,
        client: AsyncOpenAI,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {"messages": messages, "model": self.model}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        for attempt in range(self.max_retries + 1):
            try:
                response = await client.chat.completions.create(**kwargs)
                return self._response_to_llm_response(response)
            except _RETRYABLE_ERRORS as exc:
                if attempt >= self.max_retries:
                    raise LLMError(
                        f"LLM request failed after {self.max_retries} retries"
                    ) from exc
                _sleep_for_retry(attempt, base_delay=1.0)
            except openai.APIError as exc:
                raise LLMError(f"OpenAI API error: {exc}") from exc

        raise LLMError("LLM request failed unexpectedly")

    @staticmethod
    def _response_to_llm_response(response: Any) -> LLMResponse:
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        return LLMResponse(
            content=content,
            model=getattr(response, "model", "") or "",
            finish_reason=choice.finish_reason,
            usage=_usage_from(getattr(response, "usage", None)),
        )
