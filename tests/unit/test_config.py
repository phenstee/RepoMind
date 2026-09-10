"""Tests for centralized application configuration."""

import pytest
from pydantic import ValidationError

from repomind.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _isolated_settings_environment(monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_EMBEDDING_MODEL",
        "DATABASE_URL",
        "REDIS_URL",
        "LLM_TIMEOUT_SECONDS",
        "LLM_MAX_RETRIES",
    ):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_default_settings_are_safe() -> None:
    settings = Settings(_env_file=None)

    assert settings.openai_api_key is None
    assert settings.openai_model
    assert settings.openai_embedding_model
    assert settings.database_url.startswith("postgresql")
    assert settings.redis_url.startswith("redis")


def test_settings_read_environment_variables(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-test")

    settings = Settings(_env_file=None)

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-test"
    assert settings.openai_model == "gpt-test"
    assert settings.openai_embedding_model == "text-embedding-test"


def test_get_settings_is_cached(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_MODEL", "first-model")

    first = get_settings()
    monkeypatch.setenv("OPENAI_MODEL", "second-model")
    second = get_settings()

    assert first is second
    assert first.openai_model == "first-model"
    assert second.openai_model == "first-model"


def test_api_key_is_masked_in_repr_and_serialization() -> None:
    api_key = "sk-sensitive-test-value"
    settings = Settings(openai_api_key=api_key, _env_file=None)

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == api_key
    assert api_key not in repr(settings)
    assert api_key not in repr(settings.model_dump())
    assert api_key not in settings.model_dump_json()


@pytest.mark.parametrize("timeout", [0, -0.1])
def test_timeout_must_be_positive(timeout: float) -> None:
    with pytest.raises(ValidationError):
        Settings(llm_timeout_seconds=timeout, _env_file=None)


def test_retry_count_must_be_nonnegative() -> None:
    with pytest.raises(ValidationError):
        Settings(llm_max_retries=-1, _env_file=None)


def test_valid_timeout_and_zero_retries_are_accepted() -> None:
    settings = Settings(
        llm_timeout_seconds=0.1,
        llm_max_retries=0,
        _env_file=None,
    )

    assert settings.llm_timeout_seconds == 0.1
    assert settings.llm_max_retries == 0
