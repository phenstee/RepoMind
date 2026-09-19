# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev --no-editable

COPY README.md ./
COPY src ./src
COPY alembic ./alembic
COPY alembic.ini ./

RUN uv sync --locked --no-dev --no-editable

FROM python:3.13-slim-bookworm AS runtime

# Deterministic, overridable non-root identity so the bind-mounted ./workspace
# directory stays writable regardless of the host user's UID/GID (Compose
# passes REPOMIND_UID/REPOMIND_GID as build args; both default to 1000).
ARG REPOMIND_UID=1000
ARG REPOMIND_GID=1000

RUN groupadd --system --gid "${REPOMIND_GID}" repomind \
    && useradd --system --uid "${REPOMIND_UID}" --gid repomind --create-home repomind

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/alembic ./alembic
COPY --from=builder /app/alembic.ini ./
COPY --from=builder /app/pyproject.toml ./
COPY --from=builder /app/README.md ./

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

RUN mkdir -p /workspace && chown repomind:repomind /workspace
USER repomind

EXPOSE 8000

CMD ["uvicorn", "repomind.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
