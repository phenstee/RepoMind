"""Tests for centralized application configuration."""

from repomind.config import Settings, get_settings


def test_default_settings_are_safe() -> None:
    settings = Settings()

    assert settings.openai_api_key is None
    assert settings.openai_model
    assert settings.openai_embedding_model
    assert settings.database_url.startswith("postgresql")
    assert settings.redis_url.startswith("redis")


def test_settings_read_environment_variables(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-test")

    settings = Settings()

    assert settings.openai_api_key == "sk-test"
    assert settings.openai_model == "gpt-test"
    assert settings.openai_embedding_model == "text-embedding-test"


def test_get_settings_is_cached(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("OPENAI_MODEL", "first-model")

    first = get_settings()
    monkeypatch.setenv("OPENAI_MODEL", "second-model")
    second = get_settings()

    assert first is second
    assert first.openai_model == "first-model"
    assert second.openai_model == "first-model"
