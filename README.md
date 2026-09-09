# RepoMind

RepoMind is an autonomous AI software engineer for understanding, modifying,
and debugging unfamiliar codebases. The project is deliberately built in small,
verifiable milestones rather than as one monolithic prototype.

## Current status

**Milestone 1: LLM foundation** is complete. It includes:

- a Python 3.13 project managed with `uv` and a `src/repomind` package layout
- centralized configuration via `pydantic-settings`
- a reusable OpenAI client with text generation and structured output
- explicit retry and error handling
- normalized, typed response models
- mocked unit tests that do not require an API key or network access
- an optional manual live-API check script

**Milestone 2: repository ingestion** is complete. RepoMind can now safely
discover, filter, decode, and represent supported source files without calling
the LLM layer.

Later milestones will add code chunking, embeddings, semantic/hybrid retrieval,
RAG, tools, the handwritten agent loop, code editing, tests/self-correction,
permissions, memory, multi-agent orchestration, observability, evaluations,
FastAPI, a Next.js frontend, Redis, Docker, and CI.

## Repository ingestion

`repomind.ingestion` traverses a local repository, skips generated or ignored
directories, filters supported source files, and returns a
`RepositorySnapshot`. `SourceFile` objects expose only a repository-relative
path, preserving privacy around local absolute paths.

Supported extensions include:

```text
.py .ts .tsx .js .jsx .go .java .rs .cpp .cc .c .h .hpp .cs
.md .json .yaml .yml .toml .sql .sh .ps1
```

Default ignored directories include `.git`, virtual environments, caches,
`node_modules`, build output, and IDE directories. `.github` is deliberately
ingested when it contains supported text files because CI/workflow
configuration is meaningful project context.

Safety behaviors:

- symlinked directories and files are skipped
- files over 1 MiB are skipped and reported
- files with null bytes in an initial sample are treated as binary and skipped
- UTF-8 and UTF-8 BOM are supported; legacy `cp1252` text is used as a fallback
- undecodable files are skipped and reported
- line counts use Python `splitlines()` semantics

## Repository layout

```text
RepoMind/
├── src/
│   └── repomind/
│       ├── config.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── language.py
│       │   ├── models.py
│       │   └── repository.py
│       └── llm/
│           ├── client.py
│           ├── models.py
│           └── structured.py
├── scripts/
│   ├── inspect_repository.py
│   └── manual_llm_check.py
├── tests/
│   └── unit/
│       ├── ingestion/
│       │   ├── test_language.py
│       │   └── test_repository.py
│       ├── test_config.py
│       └── test_llm_client.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## Architecture

The application is split into small, independently testable layers:

```mermaid
flowchart LR
    Settings[Config Settings] --> Client[OpenAI LLM Client]
    Client --> Text[Typed Text Response]
    Client --> Structured[Pydantic Structured Response]

    Root[Repository root] --> Discovery[File discovery]
    Discovery --> Filter[Validation and filtering]
    Filter --> Load[Safe text loading]
    Load --> Source[SourceFile objects]
    Source --> Snapshot[RepositorySnapshot]
```

`OpenAILLMClient` wraps the official OpenAI SDK. The rest of the codebase
depends on RepoMind's own `LLMResponse`, `TokenUsage`, and Pydantic response
models, not on raw SDK objects.

## Setup

Prerequisites:

- Python 3.13
- `uv`

Install dependencies:

```powershell
uv sync
```

Create a local environment file if you want to run live checks:

```powershell
Copy-Item .env.example .env
```

Then set `OPENAI_API_KEY` in `.env`. Never commit `.env`.

## Tests

Run the full unit test suite:

```powershell
uv run pytest
```

Run linting:

```powershell
uv run ruff check .
```

## Optional live check

With a valid `OPENAI_API_KEY` configured:

```powershell
uv run python scripts/manual_llm_check.py
```

This script makes real API calls and is not part of the automated test suite.

## Optional ingestion check

Inspect a repository without using the LLM:

```powershell
uv run python scripts/inspect_repository.py .
```

## Design decisions

- **No agent framework yet.** Milestone 1 uses only the official OpenAI SDK. The
  first agent loop will be handwritten so its mechanics remain visible.
- **Provider abstraction stays small.** Normalized response models keep the
  SDK boundary contained and testable.
- **Secrets are externalized.** All environment configuration flows through
  `repomind.config.Settings`, and `.env` is gitignored.
- **Tests are offline by default.** SDK objects are injected in tests, so CI and
  local development do not require paid API calls.

## Roadmap

The full project roadmap is described in the RepoMind engineering brief:

1. LLM foundation (complete)
2. Repository ingestion (current)
3. Code chunking
4. Embeddings
5. Semantic search
6. Basic RAG
7. PostgreSQL + pgvector persistence
8. Hybrid retrieval
9. Reranking
10. Tool system
11. Handwritten agent loop
12. Read-only software engineering agent
13. Planning
14. Controlled code editing
15. Test execution and self-correction
16. Human approval and security
17. Memory
18. Multi-agent architecture
19. MCP
20. Evaluations
21. Observability
22. FastAPI backend
23. Streaming
24. Next.js frontend
25. Redis + background workers
26. Docker
27. CI
28. Optional agent framework comparison
