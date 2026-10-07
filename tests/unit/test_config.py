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
        "JOB_MAX_ATTEMPTS",
        "REPOMIND_ALLOWED_HOSTS",
        "REPOMIND_TRUSTED_FRONTEND_ORIGINS",
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


@pytest.mark.parametrize(
    "origin",
    [
        "*",
        "null",
        "http://*.example.com",
        "file:///tmp/index.html",
        "chrome-extension://abcdef",
        "ftp://localhost:3000",
        "localhost:3000",
        "http://localhost:3000/",
        "http://localhost:3000/app",
        "http://user@localhost:3000",
        "http://localhost:notaport",
    ],
)
def test_trusted_frontend_origins_reject_wildcards_and_non_http_origins(origin: str) -> None:
    with pytest.raises(ValidationError):
        Settings(repomind_trusted_frontend_origins=(origin,), _env_file=None)


def test_trusted_frontend_origins_accept_explicit_http_origins(monkeypatch) -> None:
    monkeypatch.setenv(
        "REPOMIND_TRUSTED_FRONTEND_ORIGINS",
        '["http://localhost:3000","https://repomind.internal:8443"]',
    )

    settings = Settings(_env_file=None)

    assert settings.repomind_trusted_frontend_origins == (
        "http://localhost:3000",
        "https://repomind.internal:8443",
    )


def test_allowed_hosts_default_to_loopback_and_compose_names() -> None:
    assert Settings(_env_file=None).repomind_allowed_hosts == (
        "localhost",
        "127.0.0.1",
        "api",
        "[::1]",
    )


@pytest.mark.parametrize("hosts", [(), ("*",), ("*.example.com",), ("localhost:8000",), ("",)])
def test_allowed_hosts_reject_wildcards_ports_and_empty_values(hosts) -> None:
    with pytest.raises(ValidationError):
        Settings(repomind_allowed_hosts=hosts, _env_file=None)


def test_job_max_attempts_defaults_reads_environment_and_is_bounded(monkeypatch) -> None:
    assert Settings(_env_file=None).job_max_attempts == 3
    monkeypatch.setenv("JOB_MAX_ATTEMPTS", "5")
    assert Settings(_env_file=None).job_max_attempts == 5
    with pytest.raises(ValidationError):
        Settings(job_max_attempts=0, _env_file=None)
