"""Centralized application configuration.

All environment-dependent values are defined here so the rest of the
application does not need to read process environment variables directly.
"""

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
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
    # Host header names the API answers to; anything else (DNS rebinding) is rejected.
    repomind_allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "api", "[::1]")

    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)
    job_lease_seconds: float = Field(default=300.0, ge=30.0, le=3600.0)
    job_poll_seconds: float = Field(default=1.0, ge=0.1, le=60.0)
    job_max_attempts: int = Field(default=3, ge=1, le=100)

    @field_validator("repomind_trusted_frontend_origins")
    @classmethod
    def _explicit_http_origins(cls, origins: tuple[str, ...]) -> tuple[str, ...]:
        """CORS compares exact origins, so only literal ``http(s)://host[:port]`` is useful."""

        for origin in origins:
            parts = urlsplit(origin)
            if (
                "*" in origin
                or parts.scheme not in {"http", "https"}
                or not parts.hostname
                or parts.username is not None
                or parts.path
                or parts.query
                or parts.fragment
            ):
                raise ValueError(
                    "Trusted frontend origins must be explicit http(s) origins, "
                    "for example http://localhost:3000"
                )
            parts.port  # noqa: B018 - raises ValueError for an invalid port
        return origins

    @field_validator("repomind_allowed_hosts")
    @classmethod
    def _explicit_hosts(cls, hosts: tuple[str, ...]) -> tuple[str, ...]:
        if not hosts:
            raise ValueError("At least one allowed host is required")
        for host in hosts:
            bracketed_ipv6 = host.startswith("[") and host.endswith("]")
            if (
                not host
                or "*" in host
                or "/" in host
                or (":" in host and not bracketed_ipv6)
                or any(character.isspace() for character in host)
            ):
                raise ValueError(
                    "Allowed hosts must be explicit host names without wildcards or ports"
                )
        return hosts


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""

    return Settings()
