"""Centralized application configuration.

All environment-dependent values are defined here so the rest of the
application does not need to read process environment variables directly.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded from environment variables and an optional ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    database_url: str = "postgresql+psycopg://repomind:repomind@localhost:5432/repomind"
    redis_url: str = "redis://localhost:6379/0"
    repomind_workspace_root: Path | None = None
    repomind_trusted_frontend_origins: tuple[str, ...] = (
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    )

    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)
    job_lease_seconds: float = Field(default=300.0, ge=30.0, le=3600.0)
    job_poll_seconds: float = Field(default=1.0, ge=0.1, le=60.0)


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""

    return Settings()
