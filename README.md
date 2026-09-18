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

**Milestone 3: deterministic code chunking** is complete. RepoMind can now
transform `SourceFile` objects into ordered, citation-ready `CodeChunk` objects
using a deterministic line-based baseline.

**Milestone 3.5: hardening** is complete. Async retries are event-loop safe,
configuration secrets and bounds are validated, and repository reads and path
metadata have additional safety checks.

**Milestone 4: embeddings** is complete. RepoMind can transform arbitrary text,
text batches, and `CodeChunk` objects into validated numerical vectors using a
separate synchronous/asynchronous OpenAI embedding client.

**Milestone 5: in-memory semantic search** is complete. RepoMind can embed a
natural-language query once, compare it with existing `EmbeddedChunk` vectors,
and return deterministic top-k source chunks ranked by cosine similarity.

**Milestone 6: basic repository RAG** is complete. RepoMind can turn ranked
chunks into bounded context, request a structured grounded answer, validate its
source IDs, and map them to repository-owned file and line citations.

**Milestone 7: PostgreSQL and pgvector persistence** is complete. RepoMind can
persist repository snapshots, source files, chunks, and embeddings, reconstruct
the existing domain models, and perform exact database-backed cosine retrieval.

**Milestone 8: BM25 and hybrid retrieval** is complete. RepoMind can tokenize
code-aware identifiers, rank chunks lexically with an explicit BM25
implementation, and fuse semantic and lexical rankings with Reciprocal Rank
Fusion (RRF).

**Milestone 9: LLM-based reranking** is complete. RepoMind can send a bounded
retrieval candidate set to the existing structured-output LLM interface, strictly
validate a candidate-ID ordering, and preserve original retrieval ranks.

**Milestone 10: read-only tool system** is complete. RepoMind now exposes six
bounded inspection operations through strict Pydantic inputs, structured outputs,
and a deterministic registry confined to one repository workspace.

**Milestone 11: handwritten read-only agent loop** is complete. RepoMind now asks
the existing structured-output LLM for one validated tool-or-final decision,
executes read-only tools sequentially, records observations, and repeats within
hard iteration and history limits.

**Milestone 12: controlled editing and safe verification** is complete. An
application can now explicitly opt into precise file creation/replacement and
fixed pytest/Ruff checks while the default registry and read-only agent remain
strictly read-only.

**Milestone 13: coding-task workflow and completion gates** is complete. RepoMind
now performs clean-worktree preflight, revision-aware required verification, a
final Git review, and deterministic completion evaluation around the existing
editing agent. An LLM final response is only a request to complete.

**Milestone 13.5: RAG retrieval integration hardening** is complete. The public
repository-question pipeline can now consume an injected semantic, hybrid, or
hybrid-plus-reranking path through the existing ranked-chunk boundary while the
original semantic-only API remains the default baseline.

**Milestone 14: evaluation harness and benchmarks** is complete. RepoMind now
measures retrieval ranking, RAG context/answer behavior, and isolated scripted
coding workflows with versioned cases, deterministic metrics, hidden coding
oracles, and inspectable per-case reports.

**Milestone 15: structured observability and persistent run tracing** is complete.
Optional injected recorders capture metadata-only timelines across RAG, agents,
coding workflows, and evaluation. PostgreSQL storage uses independent transactions.

**Milestone 16: FastAPI backend** is complete. Versioned, typed HTTP endpoints
compose the existing domain capabilities through application services. No retrieval,
agent-loop, editing, or completion-gate algorithms were replaced.

**Milestone 17: semantic SSE progress streaming** is complete. It projects the
existing structured trace timeline into a safe, request-bound Server-Sent Events
transport without changing the RAG, agent, indexing, or coding workflows.

**Milestone 18: Next.js + TypeScript frontend** is complete. A local developer
workspace consumes the typed HTTP and POST-SSE API for repository registration,
indexing, Ask, read-only Investigate, guarded Code, live progress, and run history.

**Milestone 19: durable jobs and worker execution** is complete. PostgreSQL owns
durable job state while Redis provides best-effort worker wakeups and live progress.

**Milestone 20: safe cooperative job cancellation** is complete. Cancellation
intent is durable in PostgreSQL and workers honor it only at explicit safe workflow
boundaries; no worker thread or in-flight blocking call is forcefully terminated.

**Milestone 21: structured planning and independent review** is complete. Each
coding task gets one bounded advisory `CodingPlan` before the existing executor;
fresh deterministic verification and Git evidence then feed a revision-bound
`CodingReview`. Reviewer corrections return to the same executor loop, while
Python-owned verification and safety gates retain final authority.

**Milestone 22: Retrieval V2** is complete. The original deterministic line
chunker and exact vector search remain explicit baselines; Python repositories
can opt into AST-anchored structural chunks, and PostgreSQL can opt into a
filtered pgvector HNSW candidate path without changing BM25, RRF, reranking, or
RAG result contracts.

**Milestone 23: Retrieval quality + context assembly V2** is complete.
Retrieval still answers "what chunks are relevant?"; a new, explicitly separate
`ContextAssembler` answers "what evidence should actually reach the model?" It
performs bounded (`radius=1` by default) same-file neighbor expansion, prefers
same-symbol structural fragments over generic neighbors, deduplicates by stable
chunk identity, conservatively suppresses fully-contained overlapping ranges,
and greedily packs a deterministic token budget. The preserved baseline
(`seeds_only`) is untouched production behavior; the new `expanded` strategy is
available per-request but is not the default. See
[Context assembly](#context-assembly) below.

**Milestone 24: symbol-aware retrieval fusion + realistic benchmark corpus**
is complete. Persisted `symbol_name`/`qualified_symbol_name` metadata from
Milestone 22 now feeds a third, bounded, separately-measurable RRF candidate
source alongside semantic and BM25, reached through the new non-default
`hybrid_symbol` strategy. It is deliberately not GraphRAG, a call graph, or a
language server: it only performs bounded, tier-ordered exact-match lookup
(one repository-scoped SQL query, two new B-tree indexes) over metadata that
already existed. RRF itself was generalized to N named sources behind an
unchanged two-source `reciprocal_rank_fusion` wrapper, so historical hybrid
retrieval is byte-for-byte unaffected. See
[Symbol-aware retrieval fusion](#symbol-aware-retrieval-fusion) below.

## Local HTTP API

The API is for **trusted local development only**. It has **no authentication or
authorization**. **Do not expose it directly to the public Internet or untrusted
networks.** It can inspect repository source and, through the separate coding
endpoint, mutate files and execute repository tests. Fixed pytest/Ruff commands
are not an OS sandbox: repository tests, plugins, and configuration must be trusted.
The API permits only configured trusted local frontend origins; it never uses a
wildcard CORS policy or credentialed cross-origin requests.

```text
Client
  |-- normal HTTP request ------------------------------------------|
  `-- POST SSE stream -> FastAPI transport -> application services -|
                                                   |
                                    Index / RAG / Agent / Coding operation
                                                   |
                                              TraceContext
                                             /            \
                           retained/persisted trace   sanitized progress -> SSE
```

Install dependencies with `uv sync`. Set `REPOMIND_WORKSPACE_ROOT` in `.env` to
an existing **absolute directory containing only repositories you trust**. Use a
dedicated workspace, not a home directory, drive root, or unrelated data directory.
Set `DATABASE_URL` for PostgreSQL; embedding/model operations also require
`OPENAI_API_KEY`. Never put secrets in requests or URLs. Then run:

```bash
uv run alembic upgrade head
uv run alembic current
uv run uvicorn repomind.api.app:app --host 127.0.0.1 --port 8000
```

The current migration head is `20260917_01`. It adds backward-compatible chunk
provenance/symbol metadata and a cosine HNSW expression index for the default
1,536-dimensional embedding shape while preserving existing repository, job,
index, and trace data.
Pre-API repositories remain unbound and return `409` for workspace
operations. They are not silently mapped to local files. Register a new unique
repository name to index a workspace through HTTP. Reusing an API-registered name
and the same canonical relative location is idempotent; changing its binding is a
conflict. Bindings are interpreted under the operator's configured workspace root.

Open [Swagger UI](http://127.0.0.1:8000/docs) or
[the OpenAPI schema](http://127.0.0.1:8000/openapi.json).
All capability endpoints use `/api/v1`:

| Method | Path after `/api/v1` | Purpose |
| --- | --- | --- |
| GET | `/health` | Process liveness; no workspace, database, or model dependency |
| POST | `/repositories` | Register `{ "name": "sample", "path": "sample" }`; returns stable ID, name, creation time |
| GET | `/repositories` | Compact registered repository metadata and safe workspace-relative bindings |
| GET | `/repositories/{id}` | Registered repository metadata |
| POST | `/repositories/{id}/index` | Synchronous ingestion, chunking, embedding, atomic persistence |
| POST | `/repositories/{id}/index/stream` | Indexing progress as SSE, then terminal index result |
| GET | `/repositories/{id}/files` | Indexed file metadata; `limit` (1–100), `offset` (0–1,000,000) |
| POST | `/repositories/{id}/rag` | Grounded repository question answering |
| POST | `/repositories/{id}/rag/stream` | RAG semantic progress as SSE, then terminal RAG result |
| POST | `/repositories/{id}/agent/runs` | Read-only agent; no editing or verification tools |
| POST | `/repositories/{id}/agent/runs/stream` | Read-only agent progress as SSE, then terminal agent result |
| POST | `/repositories/{id}/coding/runs` | Explicit opt-in controlled editing and required verification |
| POST | `/repositories/{id}/coding/runs/stream` | Controlled coding progress as SSE, then terminal coding result |
| POST | `/repositories/{id}/jobs/{index,rag,agent,coding}` | Persist a durable job and return `202` quickly |
| GET | `/jobs` | Bounded durable job summaries; optional status/type/repository filters |
| GET | `/jobs/{uuid}` | Authoritative durable status, safe terminal result/error, and trace link |
| POST | `/jobs/{uuid}/cancel` | Durably request safe cooperative cancellation |
| GET | `/jobs/{uuid}/events` | Ephemeral live semantic job progress as SSE |
| GET | `/runs` | Compact stored trace summaries; optional `run_type`, `status`, `limit` (1–100) |
| GET | `/runs/{uuid}` | Stored summary and ordered, sanitized trace events |

Paths in registration must be **workspace-relative subdirectories**. The existing
domain resolver enforces canonical confinement and rejects absolute paths,
traversal, symlinks, and junction components, even links pointing inside the root.
The API also rejects Windows stream/alias spellings. Bindings are rechecked before
local operations. Responses omit absolute workspace roots and validate relative
file/citation paths. Unexpected errors and invalid request bodies are never echoed.
Recognizable secrets and host paths in model-authored answer text are redacted as
defense in depth, not as a general secret-detection guarantee.

For a directory named `sample` under that root (examples use POSIX-shell quoting;
in PowerShell use `curl.exe` with appropriate JSON quoting or the Swagger UI):

```bash
curl http://127.0.0.1:8000/api/v1/health
curl -X POST http://127.0.0.1:8000/api/v1/repositories \
  -H 'Content-Type: application/json' -d '{"name":"sample","path":"sample"}'
# Substitute the returned repository ID; indexing below may incur embedding charges.
curl -X POST http://127.0.0.1:8000/api/v1/repositories/1/index
curl -X POST http://127.0.0.1:8000/api/v1/repositories/1/rag \
  -H 'Content-Type: application/json' \
  -d '{"question":"Where is authentication handled?","strategy":"hybrid","top_k":5,"trace":true}'
curl -X POST http://127.0.0.1:8000/api/v1/repositories/1/agent/runs \
  -H 'Content-Type: application/json' \
  -d '{"query":"Explain the entry point without changing files.","max_iterations":8,"trace":true}'
curl 'http://127.0.0.1:8000/api/v1/runs?run_type=rag&limit=10'
```

## Semantic progress streams

SSE fits RepoMind's current one-way progress need: the server reports semantic
stages while a request runs. WebSockets are intentionally not used because there
is no current requirement for persistent bidirectional interaction. Existing
non-streaming endpoints remain unchanged for scripts and simple clients.

Streaming variants are `POST` endpoints because they need structured JSON bodies.
Browser `EventSource` only supports `GET`, so use `fetch` streaming, another HTTP
streaming client, or `curl -N` instead. For example:

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/repositories/1/rag/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"Where is authentication handled?","strategy":"hybrid","top_k":5}'
```

In PowerShell, use `curl.exe -N` (rather than the `curl` alias) or Swagger UI with
a streaming-capable client. The response has `Content-Type: text/event-stream`,
`Cache-Control: no-cache`, and standard frames. Progress frames retain the trace
sequence as the SSE `id`; transport-only terminal frames intentionally have no
separate trace sequence:

```text
id: 4
event: tool.completed
data: {"run_id":"...","sequence":4,"event":"tool.completed","timestamp":"...","data":{"tool":"read_file","path":"src/app.py"}}

event: result
data: {"run_id":"...","result":{"status":"completed", "...":"same normal response contract"}}
```

The public progress model is a deliberately narrow projection of retained
`TraceEvent` objects, not a trace dump. Representative events include
`index.started`, `ingestion.completed`, `chunking.completed`,
`embedding.completed`, `retrieval.started`, `retrieval.completed`,
`context.assembled`, `agent.decision`, `tool.started`, `tool.completed`, `file.mutated`,
`verification.completed`, `completion.blocked`, `final_review.completed`,
`planning.started`, `planning.completed`, `review.started`, `review.completed`,
`review.blocked`, and `run.completed`. Event names follow observability semantics; the UI
render their structured fields rather than parse prose. Internal event families
are not automatically public API promises.

Only an explicit per-event allowlist reaches an SSE `data` object. It can include
safe labels, bounded counts, relative paths, hashes, verifier pass/fail/exit/timeout
metadata, workspace revision, and deterministic blocker codes. Streams never
include prompts, model output payloads, source contents, replacements, diffs,
test stdout/stderr, credentials, environment values, embedding vectors, or absolute
paths. Streaming does not weaken workspace confinement, the read-only-versus-coding
capability boundary, the trusted-local-only warning, or the absence of auth.

Before a stream starts, known validation/repository/workspace errors remain normal
JSON HTTP errors such as `400`, `404`, `409`, `422`, or `503`. Once SSE headers have
been sent, an unexpected runtime failure produces exactly one terminal `error`
event with `{ "code": "operation_failed", "message": "The operation failed." }`
and no `result` event. A successful stream always emits exactly one `result` event
containing the same safe business response shape as its normal endpoint.

The domain workflows remain synchronous. A per-request bridge runs the established
operation in one local thread and uses a bounded, isolated channel to forward the
existing trace listener's public projection to the async SSE response. It does not
block the event loop, does not use a global queue, and does not participate in
database/file transaction success. The 256-event buffer is well above normal
bounded RepoMind timelines; if a client is too slow, non-terminal progress may be
dropped (visible as a sequence gap), but a terminal result/error is retained.

Request-bound streams have no durable job, replay, resume token, cooperative cancel
endpoint, or `Last-Event-ID` support. If such a client disconnects, the server stops
forwarding to that connection but allows the already-running synchronous operation
to finish. Durable worker jobs instead support the explicit job cancellation API
described below. Completed traces can still be queried through
`/api/v1/runs/{uuid}` when trace persistence succeeds.

Indexing delegates to existing deterministic ingestion/chunking and embedding
providers, preserving source newlines. It rejects more than 10,000 discovered
source files, 50 MiB total discovered source bytes, or 10,000 chunks before model
calls. Existing ingestion's per-file limits still apply. Embeddings are generated
before opening the replacement transaction; a failure leaves the previous index
intact. Discovery and execution remain synchronous, not background jobs. Empty
repositories can be indexed without a model call. Index responses contain counts
and the embedding model, never source contents or vectors.

RAG accepts `semantic` (default), `hybrid`, `hybrid_rerank`, or `hybrid_symbol`
(Milestone 24 bounded symbol-metadata fusion, non-default; see
[Symbol-aware retrieval fusion](#symbol-aware-retrieval-fusion)), with `top_k`
1–20, and an independent `context_strategy` of `seeds_only` (default,
unchanged Milestone 1–22 behavior) or `expanded` (Milestone 23 bounded
neighbor expansion; see [Context assembly](#context-assembly)).
The reranking option feeds up to 20 retrieved candidates into the existing bounded
LLM reranker before the existing citation-validating RAG pipeline. Responses contain
`answer`, `insufficient_evidence`, relative path/line citations, and optional
`trace_run_id`. Empty evidence skips answer generation. The read-only endpoint
returns status, final answer, iteration/model/tool-attempt counts, and optional trace
ID; raw history and tool observations are omitted.

Coding uses a separate request contract:

```json
{
  "objective": "Fix the failing parser case with a focused change.",
  "acceptance_criteria": ["Existing tests still pass."],
  "verification": {"test_paths": ["tests"], "ruff_paths": ["."]},
  "max_iterations": 8,
  "trace": true
}
```

Posting this to `/repositories/{id}/coding/runs` explicitly permits the existing
controlled mutation workflow. Clean-worktree preflight, optimistic-concurrency
edits, fresh required pytest/Ruff evidence, final Git review, and deterministic
completion gates remain mandatory. Callers may choose bounded verification paths,
but cannot disable gates or supply executables, shell commands, or environment
overrides. Objective/query/question strings are nonblank and at most 10,000
characters; coding allows up to 50 criteria of at most 2,000 characters each.
Iteration limits are 1–20. Verification scopes allow 1–32 paths each.
Responses include workflow status, summary, compact verification evidence,
changed relative paths, a safe compact plan/review, completion and review attempts,
workspace revision, and optional trace ID, not prompts, diffs, replacements, or test
output. Review evidence is bounded and redacted for recognizable secrets, host paths,
and exact changed source fragments. A `200` can describe an incomplete or
verification-failed workflow; inspect `status`. Applied edits are not automatically
rolled back. Re-index explicitly after editing when fresh retrieval data is needed.
There is no approve/reject/resume API or Git publication capability.

Tracing is opt-in per RAG/agent/coding request. Existing Milestone 15 recorders
persist completed/failed runs independently and best-effort. A failed trace sink
does not fail the operation; its returned ID may then be unavailable in history.
History revalidates stored metadata through the existing allowlist and additional
host-path redaction. Prompts, source payloads, replacements, diffs, verifier output,
credentials, and vectors are excluded. Unknown usage remains unknown and dollar
cost is not invented. Trace records are history, not resumable job state.

Errors have the form `{"error":{"code":"...","message":"..."}}`, with optional
`trace_run_id`. Invalid paths map to `400`, missing repositories/runs to `404`,
binding/busy/preflight conflicts to `409`, index bounds to `413`, request validation
to `422`, unexpected failures to generic `500`, and unavailable storage/workspace
configuration to `503`. No raw exception messages or validation input payloads are
returned.

`from repomind.api import create_app` creates an isolated, lazy app. Tests may
inject `services`, `repository_store` (including its retrieval boundary),
`trace_store`, `llm_factory`, and `embedding_factory`, or override `get_services`
using FastAPI dependency overrides. Database sessions are short-lived and closed;
the app disposes its owned lazy engine during shutdown. Health and OpenAPI remain
available without constructing services. API tests use fake model providers and
real domain composition; PostgreSQL HTTP tests are opt-in alongside the existing
integration suite.

**Milestone 16 executes long-running operations synchronously.** Sync routes run
in the framework's thread pool; they do not call blocking workflows on the async
event loop. Run one server process: overlapping index/agent/coding/RAG operations
on the same or nested workspace return `409`. This guard does not coordinate
external editors, other processes, or direct Python callers, and path checks are
not an OS-level race-proof sandbox. A disconnected HTTP request is not guaranteed
to cancel a running workflow. No automatic retry of mutation requests is safe.

Readiness probes, evaluation execution APIs, approval/resume, auth, MCP, native tool calling,
autonomous planner/reviewer agents, multi-agent execution, arbitrary shell, Git mutation,
deployment, and retrieval experiments remain deferred. No `BackgroundTasks` are
implemented here.

## Durable jobs and worker

```text
HTTP enqueue -> PostgreSQL job -> Redis wakeup -> worker -> existing RepoMind service
                     ^                                      |
                     `----------- durable status/result -----'
                                    |
                         Redis Pub/Sub -> GET SSE -> browser

POST cancel -> PostgreSQL cancel_requested_at -> worker safe checkpoint
                                                    |
                                      cancelled/deferred state -> SSE/browser
```

PostgreSQL is the authoritative durable store for job identity, validated request
payload, status, cancellation timestamps, coding mutation phase, attempt count,
lease, safe result/error, and trace linkage. Redis
is only coordination: a Redis outage never deletes queued job state because workers
also poll PostgreSQL. Start local infrastructure with `docker compose up -d postgres redis`,
apply `uv run alembic upgrade head`, run `uv run python -m repomind.worker`, then
start the API and frontend normally.

Workers atomically claim queued rows with PostgreSQL `FOR UPDATE SKIP LOCKED` and
renew a bounded lease while executing the established index/RAG/agent/coding services.
This is at-most-one active lease, **not** exactly-once side-effect execution. Expired
RAG/agent/index jobs are requeued; an expired coding job is marked `job_interrupted`
and is never replayed automatically because it may already have changed files.

Index and coding use a PostgreSQL advisory lock per repository across API and worker
processes, so a coding job cannot overlap an index or another coding job for that
repository. Read-only work has no distributed mutation lock. Job payloads are bounded
validated application input, not trace metadata; they never contain credentials, shell
commands, environment dumps, source payloads, test output, or embedding vectors.

`GET /jobs/{id}` is authoritative after a browser refresh. `GET /jobs/{id}/events`
uses Redis Pub/Sub and has no replay guarantee: missed live progress can be recovered
only as durable status/result and the existing persisted trace timeline. “Stop viewing
progress” still does not cancel a worker operation.

Queued cancellation is immediate and prevents a later claim. For a running index,
RAG, or read-only agent job, the worker continues renewing its lease until it reaches
the next coarse safe checkpoint, then records terminal `cancelled` and emits a
terminal `cancelled` SSE frame. A blocking model, embedding, tool, or transaction
already in progress is allowed to finish first, so cancellation latency includes that
operation. Redis is deliberately not used as cancellation truth (or currently as a
cancellation wakeup); durable checkpoint reads from PostgreSQL remain correct across
Redis loss or restart.

Coding cancellation is intentionally conservative. Before the first successful file
mutation, it can stop at the next checkpoint, including immediately before or after
the bounded planner call and before review. At the first successful mutation the
worker durably records `side_effect_started_at`. Later cancel requests remain visible
as `deferred`, but do not interrupt the edit/verification/final-review workflow; the
job reaches its normal terminal result. RepoMind does not yet implement file rollback,
and will not forcefully terminate a coding operation merely to make cancellation look
immediate.

## Frontend workspace

`frontend/` is a strict TypeScript Next.js App Router application managed with
**npm**. It is a local developer console, not a browser-side AI implementation:

```text
Next.js + TypeScript UI
          |
   typed HTTP / POST-SSE client
          |
        FastAPI
          |
 RepoMind application services and domain
```

Install Node.js 20.9+ (the current local setup uses Node 25), then use two
terminals. Start PostgreSQL and apply migrations first when real persistence is
required:

```powershell
# Terminal 1: local PostgreSQL and Redis
docker compose up -d postgres redis

# Terminal 2: backend configuration/secrets stay in the repository root .env
uv run alembic upgrade head
uv run python -m repomind.worker

# Terminal 3: FastAPI
uv run uvicorn repomind.api.app:app --host 127.0.0.1 --port 8000

# Terminal 4: browser-visible configuration only
cd frontend
Copy-Item .env.local.example .env.local
npm ci
npm run dev
```

Open `http://localhost:3000`. `frontend/.env.local` contains only
`NEXT_PUBLIC_REPOMIND_API_URL`, normally `http://127.0.0.1:8000/api/v1`.
Anything prefixed `NEXT_PUBLIC_` is visible to the browser: **never copy
`OPENAI_API_KEY`, `DATABASE_URL`, or any backend secret into it.** Root `.env`
is exclusively for backend configuration and secrets.

The FastAPI CORS allowlist defaults to `http://localhost:3000` and
`http://127.0.0.1:3000`; set `REPOMIND_TRUSTED_FRONTEND_ORIGINS` to a JSON list
only when another trusted local origin is necessary. It is deliberately not
`"*"`, and CORS credentials are disabled.

The workspace flow is: register a safe workspace-relative repository binding,
select it, index it, then use **Ask**, **Investigate**, or **Code**. Ask keeps
Q&A history only in browser state, so it clears on refresh. Investigate is visibly
read-only. Code requires an explicit checkbox before it sends a controlled coding
request; it exposes only the existing relative pytest/Ruff path scopes, never
shell commands, executables, environment variables, Git controls, or diffs. Completed
coding results render the plan, criterion-to-step coverage, reviewer verdict, concise
findings, and review-attempt count without displaying raw model JSON or hidden reasoning.

The frontend submits long operations as durable jobs and observes them through a
buffered incremental SSE parser. It saves the active job ID in local storage and
recovers authoritative status after refresh; live Redis progress itself is not
replayed. The shared timeline renders safe semantic events for live progress and
persisted run history. "Stop viewing progress" aborts only the browser transport.
The separate "Cancel job" action persists cancellation intent and continues
observing until the backend reports a safe terminal or deferred outcome.

Frontend checks are deterministic and do not need FastAPI, PostgreSQL, or OpenAI:

```powershell
cd frontend
npm run lint
npm run typecheck
npm test
npm run build
```

## Observability and run tracing

```text
RAG / Agent / Coding / Evaluation
              ↓
       injected TraceContext
              ↓
        ordered trace events
              ↓
    InMemoryTraceRecorder → optional PostgresTraceStore
```

Observability describes execution; the existing workflows still decide which
actions are permitted and whether a task is complete. Omit `recorder` and `trace`
to use the no-op path. No database connection is needed for normal execution.

```python
from repomind.agent import run_read_only_agent
from repomind.observability import InMemoryTraceRecorder, format_run_trace

recorder = InMemoryTraceRecorder()
result = run_read_only_agent(
    "Where is configuration loaded?", llm_provider, tool_registry,
    recorder=recorder,
)
trace = next(iter(recorder.traces.values()))
print(format_run_trace(trace))
trace_json = trace.model_dump(mode="json")
```

The RAG entry points, editing agent, `run_coding_task`, and all three evaluation
entry points accept the same optional `recorder` and `trace` keywords. An explicit
`TraceContext(recorder, run_type)` can join nested operations into one timeline;
the caller that creates this handle calls `trace.finish()` when the run ends.
High-level entry points create and finish their own handle when only a recorder
is supplied. A coding workflow shares its handle with planning, agent decisions,
tool calls, automatic verification, final Git review, and independent review.
Planning and review remain stages inside the same durable coding job and trace,
not child jobs.

For configurable RAG, pass `strategy="semantic"`, `"hybrid"`, or `"hybrid+rerank"`
to `answer_repository_question_with_retriever` to label the configured retriever.
The label is caller-supplied metadata and does not select an algorithm. Candidate
count means the results returned by that retriever; context and citation counts
describe the actual downstream RAG stages. If a custom retrieval closure makes
model calls, explicitly bind the same context to its `OpenAIEmbeddingClient` or
`LLMReranker` using their `trace=` constructor argument. Opaque custom retrievers
do not automatically reveal their internal calls.

### Trace contracts and counters

Runs have UUIDs unrelated to user content, one of five run types (`rag`,
`read_only_agent`, `editing_agent`, `coding_task`, `evaluation`), and a small
`running` / `completed` / `failed` status. `domain_status` preserves outcomes such
as `verification_failed`, `precondition_failed`, and `max_iterations_reached`.
An insufficient-evidence RAG answer is a completed operation. A completed
evaluation may contain failed benchmark cases; its metrics describe those failures.

Events use contiguous per-run sequence numbers starting at **1**, UTC timestamps,
and monotonic durations. IDs and clocks are injectable for deterministic tests.
Use one context for a sequential run and separate contexts for concurrent runs;
there is no process-global current run. Event families cover:

- Run lifecycle, model requests/attempts/usage, and agent decisions.
- Tool start/completion/failure, validation vs execution failure, and blocked calls.
- Successful file mutations, planning, verification, preflight, completion gates,
  final Git review, and independent review.
- Retrieval/context/answer summaries and evaluation case start/completion/failure.

`llm_calls` counts logical provider requests, including failed requests; retries
are separate attempt metadata. Each embedding batch is one request. `tool_calls`
counts requests reaching the registry, including validation failures and coding
preflight/automatic verification/review. It can therefore exceed the agent's
own tool counter. Guard-blocked requests emit `tool.blocked` and do not increment
that counter. A verifier returning a failing test result is a completed tool
execution with `passed=false`; failure to execute emits a failure event.
Mutation counts increment only after successful edits. `errors` counts model,
tool, and evaluation-case failure events; nested failures can describe the same
underlying problem at multiple boundaries.

Provider usage is recorded before structured responses discard their envelope.
`TokenUsage` aggregates reported prompt, completion, and total tokens.
`usage_reported_calls` indicates coverage: totals are **reported subtotals** if
some calls lack usage. Missing usage remains unknown (`None` when none is reported),
including scripted structured providers. Embeddings report input usage where
available. Cost estimation is deferred; `estimated_cost_usd` is always `None`.

### Privacy and failure behavior

Traces store allowlisted metadata: model/operation/schema names, prompt/output
lengths, relative paths, hashes, revision numbers, result flags, and bounded
identifiers. They do **not** store chain-of-thought, API keys, request headers,
environment dumps, full prompts, final answer text, source contents, replacement
text, diff contents, test stdout/stderr contents, search lines, or embedding vectors.
Unknown tool payloads produce minimal metadata. Exception messages are omitted;
traces record an error type and fixed safe message, while normal exceptions and
their chaining remain available to the caller. Basic secret-pattern redaction
adds defense in depth and does not claim perfect general secret detection.

The formatter and persistence store revalidate/sanitize trace data at their output
boundaries. Metadata-first tracing helps explain what happened without creating
a second copy of repository source and prompts. Repository-relative paths and
caller-selected labels can still reveal project metadata; control access to traces.

Recorder failures are best effort: safe warnings/diagnostics surface the failure,
and the original result or exception is preserved. A failed persistence sink
retains the completed in-memory trace and appends to `recorder.diagnostics`.
No file edit or primary repository transaction is rolled back by that sink.

### Coding recovery timeline

This compact excerpt illustrates the tested recovery sequence (other model/tool
and review events appear between these semantic events):

```text
file.mutated            workspace_revision=1
completion.requested   workspace_revision=1
verification.completed tool=run_tests passed=false workspace_revision=1
completion.blocked     blocker_codes=[tests_failed]
file.mutated            workspace_revision=2
completion.requested   workspace_revision=2
verification.completed tool=run_tests passed=true workspace_revision=2
verification.completed tool=run_ruff passed=true workspace_revision=2
final_review.completed workspace_revision=2
completion.completed   workspace_revision=2
run.completed
```

### Persistent run history

Apply the existing Alembic workflow (`uv run alembic upgrade head`) to add migration
`20260915_01`, following `20260910_01`. It creates `trace_runs` and `trace_events`
without altering repository index tables or calling `create_all` at runtime.
Summaries include counters and nullable token counts; events contain JSONB metadata
and use `(run_id, sequence)` as their primary key. Run lookup indexes cover time
and type/status. Deleting a run cascades to its events.

```python
from repomind.db import create_database_engine, create_session_factory
from repomind.observability import InMemoryTraceRecorder
from repomind.observability.persistence import PostgresTraceStore

engine = create_database_engine()
store = PostgresTraceStore(create_session_factory(engine))
recorder = InMemoryTraceRecorder(sink=store.persist_run_trace)
# Pass recorder to a high-level operation, then inspect:
recent = store.list_run_traces(run_type="coding_task", status="failed", limit=20)
# store.get_run_trace(run_id) returns a reconstructed RunTrace or None.
```

Give the store a factory bound to an **Engine**, not an active business Session or
Connection. Every save opens, commits/rolls back, and closes its own transaction.
Direct store methods raise database errors; the recorder sink handles them safely.
Only terminal traces are persisted, atomically with all their events. Duplicate
run IDs are rejected instead of overwriting history. Listing returns recent
traces in deterministic time/ID order, with a maximum limit of 100.

Persistence currently happens after completion: process crashes can lose an
unfinished in-memory run. There is no automatic retention/deletion job; traces
remain until explicitly managed by the application/database operator. The
in-memory recorder also retains runs until its caller releases them.

### Evaluation connection and verification

Milestone 14 answers **what failed**; Milestone 15 helps inspect **why a particular
run failed**. Evaluation case results optionally include `trace_run_id`; case IDs,
suite version, mode, strategy and compact metrics locate the relevant timeline
segment without storing gold facts or full reports in each event. Suite cases
share a run ID and are distinguished by their case boundaries. A coding benchmark
runner may opt into child tracing by accepting a keyword `trace` and forwarding
it to `run_coding_task`. Existing three-argument runners remain supported.
Case exceptions retain their original propagation behavior and emit a failed
case event before the run terminates.

For example, a failed coding case can lead to a timeline showing an unsuccessful
file search, a stale edit, failed verification, and iteration exhaustion.
Tests use fake providers, clocks and IDs, real temporary fixture tools, and opt-in
PostgreSQL tests marked consistently with existing integration tests. No automated
test calls OpenAI. Real database tests require `REPOMIND_TEST_DATABASE_URL` and a
migrated PostgreSQL database; offline SQL generation is not proof of a database
roundtrip.

Streaming, background workers, external telemetry adapters,
automatic retention and dollar-cost estimation remain deferred. The existing
tool permissions and retrieval algorithms remain the execution contracts.

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

- symlinked files/directories and Windows junction directories are skipped
- discovered paths are resolved and containment-checked against the repository root
- reads are bounded; files over 1 MiB are skipped and reported
- files with null bytes anywhere in the bounded payload are treated as binary and skipped
- UTF-8 and UTF-8 BOM are supported; legacy `cp1252` text is used as a fallback
- undecodable files are skipped and reported
- line counts use Python `splitlines()` semantics

Ingestion assumes the repository is not maliciously mutated concurrently while
it is being read. Static traversal is protected, but fully race-free path
handling would require substantially more platform-specific file-handle logic.

## Code chunking

`chunk_source_file` splits a `SourceFile` into smaller `CodeChunk` objects.
`chunk_repository` applies that process across all files in a
`RepositorySnapshot` and returns one flat, deterministically ordered list.

`ChunkingStrategy.LINE` (`line_v1`) remains the production default and stable
comparison baseline:

- `max_lines_per_chunk` defaults to 120
- `overlap_lines` defaults to 20
- chunk line numbers are 1-based and inclusive
- chunk indexes restart at zero for each file
- empty files produce no chunks

Overlap exists because code near a chunk boundary often depends on surrounding
context. Repeating nearby lines helps prevent that context from being split
completely across two retrieval units.

`ChunkingStrategy.STRUCTURAL` (`python_ast_v1`) is an explicit opt-in. For valid
Python it uses the standard-library AST to anchor module regions, functions,
classes, methods, decorators, and qualified nested symbols such as
`UserService.login`. Stored content is always sliced from the original source;
it is never reconstructed with `ast.unparse`, so formatting, CRLF newlines,
docstrings, and comments inside a selected range remain exact. Imports,
constants, assignments, and other module-level content are retained in bounded
module chunks. A class is represented by bounded class context plus its methods
rather than one duplicated giant class chunk.

Structural chunks obey both `max_lines_per_chunk` and the deterministic
`max_chars_per_chunk` approximation (12,000 characters by default). An
oversized symbol is split at child-statement boundaries when possible, then by
exact line/character slices as needed. Each fragment retains its symbol,
qualified/parent symbol, and 1-based fragment position. Tiny module statements
are naturally coalesced into contiguous module regions; unrelated functions are
not merged just to hit a target size.

Unsupported languages, invalid Python, and parser failures fall back to the
unchanged line algorithm. Those chunks retain the requested `python_ast_v1`
provenance and are marked `line_fallback`, so fallback is not mistaken for AST
parsing. Every chunk persists its strategy/version, kind, and useful symbol
metadata. Re-index after changing strategy: repository replacement is atomic
and never intentionally mixes stale `line_v1` and `python_ast_v1` chunks.

Both strategies are deterministic and synchronous. They do not embed or
retrieve content. The same file and configuration produce the same ordering,
ranges, metadata, source slices, and source-domain identities.

## Embeddings

Embeddings convert text into high-dimensional numerical vectors whose geometry
captures aspects of semantic similarity. Milestone 4 implements vector
generation only:

```text
CodeChunk
    ↓ exact content
OpenAI embedding model
    ↓
EmbeddingVector
    ↓ paired with the original chunk
EmbeddedChunk
```

`OpenAIEmbeddingClient` supports single text, text batches, and `CodeChunk`
batches through synchronous and asynchronous methods. It uses the model selected
by `OPENAI_EMBEDDING_MODEL` and reuses the existing OpenAI timeout and retry
settings rather than duplicating them.

Requests are split deterministically into batches of 64 items by default. API
response indices are validated and used to restore input order, vectors must be
non-empty and finite, and dimensionality must remain consistent across the
operation. Usage returned by the provider is aggregated so future indexing cost
can be measured. Batching is currently based on item count; token-aware batching
is a future improvement.

Chunk content is sent exactly as stored by default (`raw_source`), including
indentation and newline characters. `structural_context` is a separate,
benchmarkable embedding-text option that prefixes path, qualified symbol, and
kind while leaving stored content and citations untouched. The offline v2
fixture did not show a consistent gain from that prefix, so `raw_source` remains
the default. Empty strings are rejected; whitespace-only strings are deliberately
allowed without stripping.

Vectors are immutable Python tuples held only in memory. Automated embedding
tests inject fake SDK clients and never make live API calls. Semantic search
uses temporary NumPy arrays for calculation without changing the stored vector
values.

## Semantic search

Milestone 5 adds retrieval without answer generation or persistence:

```text
Query text
    ↓ embed exactly once
EmbeddingVector
    ↓ compare with existing EmbeddedChunk vectors
cosine similarity
    ↓ sort descending
top-k SemanticSearchResult objects
```

For a query vector `q` and chunk vector `c`, RepoMind calculates:

```text
cosine_similarity(q, c) = dot(q, c) / (||q|| * ||c||)
```

Cosine similarity compares vector direction rather than raw magnitude. Higher
scores indicate more similar directions. Inputs must be one-dimensional,
non-empty, finite, equal in dimension, and non-zero in magnitude. Query and
chunk vectors must also name the same embedding model.

`rank_by_similarity` is a pure, offline function over pre-embedded chunks.
`semantic_search` validates the query, embeds its exact text once, and delegates
to that function; source chunks are never re-embedded per query. Results are
ordered by score descending, and Python's stable sort preserves original corpus
order for exact ties. No arbitrary score threshold is applied.

This baseline performs brute-force comparison in memory using NumPy. Its time
complexity is O(N × D), where N is the number of chunks and D is vector
dimensionality. It is intended for correctness and learning, not large-scale
indexing. The retrieval layer itself does not generate answers, apply keyword
or hybrid scoring internally, rerank results, or use a vector database.

## Basic repository RAG

Retrieval-Augmented Generation keeps retrieval strategy separate from context
construction and generation:

```text
Question
    |
Configurable Retriever
    |-- semantic
    `-- hybrid (semantic + BM25 + RRF)
              |
       optional reranker
              |
         RankedChunk[]
              |
       Context Builder
              |
   Structured Generation
              |
      Citation Validation
              |
      RepositoryAnswer
```

`answer_repository_question` preserves the original in-memory semantic path. It
reuses semantic search rather than duplicating ranking logic: the exact valid
question is embedded once, up to `top_k` ranked chunks are considered, and their
cosine scores are not shown to the LLM because similarity is not a calibrated
confidence measure.

`answer_repository_question_with_retriever` is the configurable high-level
entry point. Its injected callable `Retriever` receives the exact question and
`top_k`. It returns any sequence satisfying the existing `RankedChunk` contract.
A caller can therefore use semantic search, hybrid search, or hybrid search
followed by the bounded reranker without moving ranking logic into RAG. The
retriever also provides a clean boundary for persisted PostgreSQL/pgvector
hybrid results after they have been reconstructed as domain chunks; ORM objects
never enter context construction or generation. Reranking remains optional and
no retrieval configuration is selected as universally best before evaluation.

The context builder assigns deterministic IDs `S1`, `S2`, and so on in
retrieval order. Each block includes the existing relative path, line range,
language when available, and exact chunk content. Repository excerpts are
explicitly marked as untrusted data in both the context and system prompt;
instructions found in source files, comments, or documentation must not be
followed.

`RAGConfig` defaults to five retrieved chunks and a 20,000-character context
budget. Complete chunks are added in order without truncation. The highest-
ranked chunk is included even if it alone exceeds the budget; lower-ranked
chunks stop at the first block that would exceed it. Overlapping chunks remain
independent, so repeated source lines are possible in this baseline.

The structured LLM response contains only answer text, selected source IDs, and
an insufficient-evidence flag. RepoMind rejects unknown IDs, removes duplicate
IDs while preserving first-use order, and requires at least one citation for an
answer claiming sufficient evidence. The LLM never supplies paths or line
numbers. Empty retrieval returns a deterministic insufficient-evidence answer
without calling the LLM.

The original semantic-only RAG route remains available as a baseline. The
context builder accepts semantic, hybrid, and reranked chunks through the same
small `chunk` and `rank` contract; it does not know about cosine scores, BM25,
RRF, reranker internals, PostgreSQL IDs, or ORM models.

## PostgreSQL and pgvector persistence

Milestone 7 adds a durable alternative to the in-memory retrieval baseline:

```text
RepositorySnapshot
    ↓ replace named repository snapshot
repositories
    ↓ one-to-many
repository_files (metadata, exact decoded content, SHA-256)
    ↓ one-to-many
code_chunks (source ranges, structural metadata, exact content, model, vector)
    ↓
exact pgvector baseline or pgvector HNSW candidates
    ↓
SemanticSearchResult[]
```

The complete Retrieval V2 path is:

```text
source -> line_v1 or python_ast_v1 -> embedding
                                      |
                         PostgreSQL + pgvector HNSW
                                      + BM25
                                      |
                                     RRF
                                      |
                             optional reranker
                                      |
                                 RAG / agent
```

PostgreSQL is the general relational database that stores repository, file,
metadata, and text records. pgvector is a PostgreSQL extension that adds the
`vector` type and vector-distance operators. RepoMind uses them together: normal
relational filters isolate the repository and embedding model before pgvector
orders compatible vectors by cosine distance.

Persistence means an indexed repository can survive process restarts,
embeddings do not need to be regenerated for every process, and retrieval can
be scoped in the database. Only safe repository-relative file paths are stored;
the absolute local `RepositorySnapshot.root` is deliberately excluded. Decoded
source content is stored so the indexed snapshot can later be inspected or
re-chunked without rereading a potentially changed working tree.

`persist_repository_snapshot` uses repository name as the current canonical
key. Repeating it replaces that repository's complete file snapshot and removes
old chunks through cascades. `persist_chunks` and `persist_embedded_chunks`
replace the repository's complete chunk index. These functions flush but never
commit, so the caller controls one understandable transaction and can roll back
the whole operation after any failure. Incremental file reconciliation is not
implemented yet.

Files and chunks carry SHA-256 hashes of their exact UTF-8 text. These hashes
prepare for future change detection and invalidation; Milestone 7 does not use
them to implement incremental indexing or deduplication.

The vector column intentionally has no schema-wide fixed dimension. Each
embedding stores and validates its model and dimension, including a database
constraint using `vector_dims`. This supports small test vectors and different
configured embedding models without hardcoding 1,536 dimensions. Exact search
first checks the stored dimension for the requested model; no embeddings for
that model returns `[]`, while an incompatible query dimension fails clearly.

Database ranking uses ascending pgvector cosine distance and converts it back
to the existing score contract with `cosine_similarity = 1 - cosine_distance`.
Results are filtered by repository, model, and dimension. Exact ties use
relative path, start line, chunk index, and database ID as deterministic
secondary ordering. This differs deliberately from the in-memory baseline,
which preserves caller input order for ties.

Retrieval V2 adds `ix_code_chunks_embedding_hnsw_1536_cosine`, a partial HNSW
expression index over `embedding::vector(1536)` with `vector_cosine_ops`. This
shape matches the default `text-embedding-3-small` output while keeping the
underlying column dimension-flexible. Explicit ANN mode therefore requires
1,536-dimensional vectors; other dimensions continue to work in exact mode
instead of being silently coerced or rejected at persistence time.

Exact and ANN are separate modes. Exact uses the original uncast pgvector
distance expression, which cannot match the fixed-dimension expression index,
and remains the correctness/debugging reference. ANN uses the matching cast and
normal nearest-neighbor `ORDER BY ... <=> ... LIMIT k` shape. Both filter by
repository, embedding model, and dimension and reconstruct the same domain
objects. Filtered ANN enables pgvector 0.8's `strict_order` iterative scan with
transaction-local `SET LOCAL`; it never changes database-global settings.
Persisted search defaults to exact until a real-service scale run establishes a
safe default for the deployment's corpus; callers and benchmarks select ANN
explicitly through `SemanticSearchMode.ANN`.

PostgreSQL remains free to choose a sequential plan for tiny repositories. The
production query never disables sequential scans. A real-PostgreSQL integration
test temporarily uses `enable_seqscan = off` inside its rolled-back transaction
only to prove the named HNSW index is valid and planner-usable. HNSW trades
approximate recall plus index build/storage/memory for scalable candidate
lookup; exact mode remains available for Recall@k measurement.
The migration leaves pgvector's `m`, `ef_construction`, and `hnsw.ef_search`
defaults unchanged because the current fixture does not justify custom tuning.

## BM25 and hybrid retrieval

Semantic retrieval is strong when the query and source express the same idea
with different words. BM25 lexical retrieval is strong when exact code terms
such as `OPENAI_EMBEDDING_MODEL`, `persist_embedded_chunks`, or
`RepositoryIngestionError` matter. Hybrid retrieval keeps both strengths:

```text
                    ┌── semantic ranking
query ──────────────┤
                    └── BM25 lexical ranking
                              ↓
                 Reciprocal Rank Fusion
                              ↓
                     hybrid top-k chunks
```

### Code-aware tokenization

The deterministic tokenizer uses Unicode-aware case folding. It retains the
normalized compound identifier and adds separator and case-transition
components. For example:

```text
persist_embedded_chunks
→ persist_embedded_chunks, persist, embedded, chunks

OpenAILLMClient
→ openaillmclient, open, ai, llm, client

src/repomind/retrieval
→ src/repomind/retrieval, src, repomind, retrieval
```

Dotted and kebab forms receive the same treatment. This is a small lexical
tokenizer, not a parser, stemmer, AST index, or symbol table. Punctuation-only
text produces no lexical tokens, and a BM25 query with no matching terms returns
no results.

### BM25

`BM25Index` derives term frequencies, document frequencies, document lengths,
and average document length once from `CodeChunk` content. Structural chunks
also contribute their qualified symbol name, which lets a query such as
`UserService.login` match a method without adding a second symbol index. Line-v1
scoring remains unchanged. BM25 uses `k1 = 1.5`
for term-frequency saturation and `b = 0.75` for length normalization. For each
query term it calculates:

```text
idf(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

score(D, Q) = Σ idf(t) ×
    f(t,D) × (k1 + 1)
    ─────────────────────────────────────────
    f(t,D) + k1 × (1 - b + b × |D| / avgdl)
```

Only matching chunks are returned. Scores sort descending, with original corpus
order preserved for exact ties. BM25 scores are retrieval signals, not
probabilities or calibrated confidence.

### Reciprocal Rank Fusion

Cosine similarity and BM25 scores have unrelated scales, so RepoMind does not
add or hand-normalize their raw values. RRF combines rank positions instead:

```text
RRF_score(chunk) = Σ 1 / (rrf_k + rank_i(chunk))
```

The default `rrf_k` is 60. A larger value makes differences between the very top
ranks less sharp. A chunk can contribute from either ranking or both; an
intersection is not required. Fused ties use best individual rank and then the
stable source-domain identity `(relative_path, chunk_index, start_line,
end_line)`, never a database ID or Python object identity.

The high-level in-memory path embeds the unchanged query exactly once and asks
each retriever for `top_k × 4` candidates by default before fusion. That depth is
a simple correctness baseline, not an empirically optimal setting. Stored
chunks are never re-embedded.

Milestone 24 generalized this to `fuse_ranked_sources`, which fuses any number
of named ranked sources (`{"semantic": ..., "lexical": ..., "symbol": ...}`)
with the identical score formula and tie-breaking; `reciprocal_rank_fusion`
itself is now a behavior-preserving two-source wrapper over it, verified by
regression tests that a two-source call and an empty third source both
reproduce historical output exactly. See
[Symbol-aware retrieval fusion](#symbol-aware-retrieval-fusion).

For persisted repositories, `postgres_hybrid_search` combines the selected exact
or ANN pgvector semantic results with `load_chunks` → an in-memory `BM25Index`
→ RRF. The query
embedding remains an input to the database layer, so neither BM25 nor database
code calls OpenAI. Lexical indexing is currently rebuilt for each database
hybrid call; callers performing repeated in-memory searches can reuse a
`BM25Index`.

Approximate costs are O(N × D) for in-memory semantic comparison, O(N × |Q|)
for this straightforward BM25 scorer over N chunks and query terms Q, and
O(candidate results) for RRF. PostgreSQL handles exact scanning or HNSW candidate
lookup, while database-backed lexical work still loads chunks into Python. This
is not yet a production-scale persisted lexical index.

## LLM-based reranking

Retrieval and reranking solve different parts of source discovery:

```text
large repository
      ↓
semantic + BM25 retrieval
      ↓
RRF hybrid candidates
      ↓ bounded second stage
LLM relevance ordering
      ↓
reranked chunks
      ↓
existing RAG context and grounded generation
```

Retrieval broadly discovers plausible chunks using comparatively cheap vector
and lexical signals. Reranking spends one structured LLM request on only that
small candidate set so the model can compare the question with complete source
excerpts. Generation then answers from the final context. Reranking is optional:
semantic-only, BM25-only, hybrid, and RAG-without-reranking paths remain valid
baselines.

`LLMReranker` assigns deterministic IDs `C1`, `C2`, and so on in retrieval
order. The model returns only an ordered `ranked_candidate_ids` list and must
return every included ID exactly once. Unknown, duplicate, or missing IDs are
rejected. The LLM never creates paths, line ranges, chunk indexes, or database
IDs; RepoMind maps the validated IDs back to the original `CodeChunk` objects.
This mirrors the source-ID safety boundary used by grounded RAG citations.

Candidate context contains relative path, line range, optional language, and
exact complete content. It does not contain cosine, BM25, or RRF scores, which
prevents those incompatible signals from biasing the model toward simply
repeating retrieval order. `RerankedSearchResult` contains the new `rank` and
the candidate's `original_rank`, but deliberately has no artificial confidence
score.

`RerankingConfig` defaults to at most 20 candidates and 30,000 candidate-context
characters. Candidates are considered in retrieval order, never split, and
renumbered contiguously after bounding. As in RAG context construction, the
first candidate is included whole even when it alone exceeds the character
budget; a later candidate that would exceed the budget ends collection. Empty
candidate input returns `[]` without an LLM call. `top_k` is applied only after
the model has returned the complete included ID ordering.

Repository excerpts are explicitly marked as untrusted data in the candidate
context and system prompt. Instructions inside code, comments, or documentation
cannot override the ranking task. The prompt requests ordering only—no
chain-of-thought, explanation, relevance percentage, or generated source
metadata.

`hybrid_search_with_reranking` reuses hybrid retrieval rather than duplicating
it: by default up to 20 hybrid candidates become input to the bounded reranker,
which returns the best five. Semantic-only or PostgreSQL-backed hybrid results
can also be passed directly because the reranker depends only on the shared
`RankedChunk` contract. Database modules never call the LLM.

Reranking adds latency, tokens, and cost. Without it, the path is retrieval → RAG
generation; with it, the path is retrieval → reranking LLM → RAG generation.
The extra judgment may improve context ordering, but RepoMind does not claim it
always improves results—evaluation must establish that later.

### Future retrieval experiments

Retrieval V2 made AST chunking and optional structural embedding context
measurable strategies. Milestone 23 (see
[Context assembly](#context-assembly)) implemented bounded neighbor-chunk
expansion, overlap-aware deduplication, and token-aware budgeting as an
explicit assembly stage after retrieval. Milestone 24 (see
[Symbol-aware retrieval fusion](#symbol-aware-retrieval-fusion)) added
persisted-symbol-metadata matching as a third, separately-measurable RRF
input. The following ideas remain intentionally deferred:

- query rewriting or multi-query retrieval (an offline, deterministic
  experiment mode was considered for Milestone 23; current evaluation did not
  surface a query-language-mismatch failure class clear enough to justify it)
- GraphRAG, call-graph or import-graph traversal, and multi-agent retrieval
- a fuzzy/edit-distance symbol resolver, cross-language structural parsing
  beyond Python, and PostgreSQL FTS/trigram infrastructure (Milestone 24's
  exact-match symbol lookup did not need any of these)

These changes may improve some workloads, but they also change cost, latency,
recall, or context composition. They should be measured rather than assumed to
improve repository-answer quality. Each experiment should be introduced one at
a time and measured against an established baseline.

## Context assembly

Milestone 23 separates two questions that Milestones 1–22 answered together:

```text
Query
  |
Semantic / BM25 retrieval -> RRF -> optional reranker
  |
retrieved seed chunks                      <- "what chunks are relevant?"
  |
ContextAssembler
  |-- bounded same-file neighbor expansion (radius=1 by default)
  |-- same-symbol structural-fragment preference
  |-- identity + overlap-containment deduplication
  |-- deterministic, priority-ordered token-budget packing
  |
final assembled context                   <- "what evidence reaches the model?"
  |
RAG / Agent
```

`repomind.rag.assembly.assemble_context` is the one context-assembly
abstraction (`ContextAssembler` in the milestone's own terms). Its input is an
already-ranked, already-deduplicated seed list (`RankedChunk` — retrieval's
output); its output is an `AssembledContextResult` of ordered
`AssembledContextChunk` objects, each carrying its `CodeChunk`, an explicit
`origin` (`seed`, `neighbor`, or `same_symbol_fragment` — never a fabricated
relevance score), the originating seed's rank when applicable, and an
estimated token count. Retrieval logic and prompt formatting are intentionally
not mixed into this module.

Two strategies are supported (`ContextStrategy`):

- `seeds_only` — the preserved Milestone 1–22 baseline. Seeds are formatted
  directly by the unchanged `build_repository_context`, bounded only by
  `RAGConfig.max_context_chars`. This remains the default: it is not
  automatically replaced without stronger evidence.
- `expanded` — runs the assembler described above before formatting.

**Neighbor expansion** is same-file, same-chunking-snapshot, and bounded by
`neighbor_radius` (default `1`, hard-capped at `3`); it never recurses and
never expands to a whole file. For `line_v1`, a neighbor is simply the
previous/next chunk. For `python_ast_v1`, `CodeChunk.chunk_index` values for an
oversized symbol's fragments are already contiguous, so the same `±radius`
lookup naturally reaches sibling fragments; the assembler labels a neighbor
`same_symbol_fragment` (instead of the generic `neighbor`) when it shares the
seed's `qualified_symbol_name`, and such fragments are packed ahead of generic
neighbors. Retrieving one method fragment never pulls in an entire enclosing
class — only its own adjacent fragments and, generically, the immediately
adjacent chunks.

**Deduplication** uses the existing stable `chunk_identity` (relative path,
chunk index, and line range — never a database row ID), so a neighbor that
duplicates another seed, or two seeds that expand to the same neighbor, are
kept exactly once. **Overlap suppression** is deliberately conservative: a
candidate is dropped only when its line range is *fully contained* in an
already-included chunk's range on the *same file*; two distinct symbols with
merely intersecting ranges are never discarded, and identical text in
different files is never treated as the same evidence.

**Budgeting** uses a documented approximation
(`repomind.rag.tokens.estimate_tokens`, `ceil(len(text) / 4)`) because no
tokenizer dependency exists in this project yet; the estimate is applied to
the same wrapper-plus-content text that will actually be sent (path, line
range, optional language, and content), not to raw source alone. Packing is a
transparent greedy walk in priority order — every seed before any neighbor,
`same_symbol_fragment` before generic `neighbor`, both tie-broken by seed rank
and then proximity — never a knapsack solver. Chunks are never truncated
mid-body: a candidate that would overflow the remaining budget is skipped, not
cut, except that the single highest-priority chunk is always kept even if it
alone exceeds the budget (the same policy `build_repository_context` already
used for the `seeds_only` baseline).

Citations remain source-correct: an `AssembledContextChunk`'s `CodeChunk` is
the same domain object retrieval produced, so `SourceCitation` line ranges for
an included neighbor are exactly its original `start_line`/`end_line`, never
invented.

Database access for neighbor lookup is batched: `load_neighbor_chunks` issues
one row-value `IN` query for every requested `(relative_path, chunk_index)`
key across every seed, not one query per neighbor. `tests/integration/
test_postgres_context_assembly.py` asserts this against a real PostgreSQL
connection, alongside repository isolation, structural-fragment and `line_v1`
expansion, and citation integrity.

The assembler is retrieval-backend-agnostic: it accepts seeds from exact
pgvector search, HNSW ANN search, or ANN+BM25/RRF fusion identically, and never
recomputes or fabricates a score for an expanded neighbor.

The read-only agent and the coding planner/reviewer do not use retrieval at
all (they are tool-call-oriented, not RAG-context-oriented), so Milestone 23
intentionally leaves them unchanged.

`RAGRequest.context_strategy` (`"seeds_only"` default, or `"expanded"`) is the
only new public API surface; internal knobs (`neighbor_radius`,
`context_budget_tokens`) are not exposed over HTTP to keep the request surface
small. `context.assembled` is a new, safe observability event (counts and
token estimates only — never source text, paths beyond existing allowlists, or
embeddings) emitted only on the `expanded` path.

Query rewriting was evaluated as a design option and intentionally **not**
implemented: Milestone 23's own instructions require it to stay off by default
and be justified by a clear query-language-mismatch failure class, and no such
class was surfaced by the fixtures or benchmarks above. There is no GraphRAG,
call graph, dependency graph, or multi-agent retrieval, and HNSW/AST chunking
from Milestone 22 were not modified.

Run `uv run python -m benchmarks.repo_eval_v3` for the offline, deterministic
comparison of `seeds_only` vs `expanded` context assembly (gold-evidence
coverage, packed chunk count, estimated tokens, budget utilization, and
duplicates removed), plus one RAG-level comparison showing that
context-assembly gains are separate from retrieval-ranking gains: retrieval
recall is identical between the two rows because the fixture holds the
retrieved seed fixed, yet only the `expanded` row's answer passes.

## Symbol-aware retrieval fusion

Milestone 24 adds a third, independently measurable candidate source next to
semantic and BM25:

```text
Query
  |
  +--> semantic retrieval
  |
  +--> BM25
  |
  `--> structural symbol metadata (bounded, persisted lookup)
          |
          v
          RRF
          |
     optional reranker
          |
     ContextAssembler
          |
        RAG
```

Symbol retrieval is **persisted-metadata lookup, not query rewriting, not
GraphRAG, not a call/reference/import graph, not a language server, and not
another embedding query.** It reads `symbol_name`/`qualified_symbol_name`
already produced by Milestone 22 chunking; it never reparses source or
traverses an AST at query time.

**Identifier detection** (`repomind.retrieval.identifiers.extract_identifier_candidates`)
scans the *original, unmodified* query text for identifier-shaped substrings
and never rewrites it — semantic and BM25 still receive the exact question the
user asked. A qualified dotted form (`UserService.login`), snake_case
(`load_neighbor_chunks`), camelCase, or PascalCase (`ContextAssembler`) is
**strong** evidence; a plain lowercase word with no separator or case
transition (`run`, `get`, `login`) is **weak** evidence, because it is equally
likely to be ordinary English. Weak candidates are still looked up — a query
of just `login` must still find both `UserService.login` and
`AdminService.login` — but only ever at the lowest confidence tier, so a
common word cannot dominate ordinary retrieval merely by matching a real
symbol name somewhere in the repository.

**Matching** (`repomind.retrieval.symbols.symbol_search` in memory,
`repomind.db.repositories.find_symbol_candidates` over PostgreSQL) uses three
deterministic, explainable tiers, strongest first: exact `qualified_symbol_name`,
exact `symbol_name` from a strong candidate, exact `symbol_name` from a weak
candidate. There is no fuzzy/edit-distance resolution and no fabricated
cosine-style score — `SymbolSearchResult.match_tier` is the only ranking
signal, kept explicit rather than rescaled into something that looks like
semantic similarity.

**Bounding**: candidate count is capped by `symbol_candidate_limit` (default
10). Multiple structural fragments of one oversized method in one file
collapse to their first fragment, so one large function cannot consume the
whole candidate budget. During hybrid fusion, a natural-language query can
project that one symbol vote onto the highest-ranked fragment of the same
symbol already found by semantic+BM25 retrieval; this neither adds candidates
nor scans more chunks. A query consisting solely of the matched identifier
keeps the canonical first fragment. Milestone 23's `expanded` context strategy
can still recover adjacent fragments afterward. A repository with 300
functions literally named `run` still returns at most
`symbol_candidate_limit` rows.

**PostgreSQL**: `find_symbol_candidates` is one bounded, tier-ordered,
repository-scoped SQL query (a `CASE` expression computes the tier, the join
to `repository_files` enforces isolation, `LIMIT` bounds the row count) —
never a full-repository chunk scan. Two plain B-tree indexes
(`ix_code_chunks_symbol_name`, `ix_code_chunks_qualified_symbol_name`, added
in migration `20260918_01`) support it; no pgvector, FTS, or trigram
infrastructure was introduced for string identifiers. BM25 fusion still
rebuilds from every persisted chunk exactly as it already did before this
milestone (`postgres_hybrid_search`'s existing `load_chunks` call) — a real
scalability limitation this milestone documents but does not fix.

**Fusion**: `reciprocal_rank_fusion` (semantic + BM25 only) is now a thin,
behavior-preserving wrapper over a new `fuse_ranked_sources(sources: Mapping[str,
Sequence[RankedChunk]], ...)` that accepts any number of named ranked sources.
Historical two-source callers, tie-breaking (`(-score, min_contributing_rank,
chunk_identity)`), and error messages are unchanged and regression-tested;
symbol candidates join the same fusion as a `"symbol"` source with no special
weighting. For a natural-language query naming a multi-fragment symbol, the
existing semantic+BM25 RRF order selects which already-retrieved fragment of
that symbol receives its symbol rank; ties retain the historical RRF
tie-breakers and absence of evidence retains the first fragment. The symbol
still determines *which symbol* matched, while semantic/BM25 can determine
*which fragment* best answers the question. No semantic score is fabricated
and no hand-tuned constant was introduced. Fusing with an empty (or absent)
symbol source reproduces the semantic+BM25 result identically, so `line_v1`
repositories (which have no structural symbol metadata) and non-symbol queries
degrade to existing hybrid behavior with no error and no reordering.

**Compatibility**: symbol-fused candidates are ordinary `FusedSearchResult`
objects satisfying the same `RankedChunk` contract as everything else, so they
flow into the existing reranker and `ContextAssembler` unchanged — no
symbol-specific prompt formatting, no bypassed budgeting, no fabricated
reranker relevance score for a symbol match. They work identically with exact
or ANN semantic search.

`RAGRequest.strategy` gains one new, non-default value: `"hybrid_symbol"`.
`symbol.matched` is a new safe observability event (`symbol_candidate_count`,
`symbol_match_detected`, `fused_candidate_count` only — never source text,
paths, or raw SQL).

Run `uv run python -m benchmarks.repo_eval_v4` for the offline, deterministic,
category-grouped comparison (exact qualified symbol, natural-language +
symbol, ambiguous simple symbol, snake_case, PascalCase, camelCase, common
English words, behavioral/non-symbol, duplicate-symbol-across-files) across
semantic-only, BM25-only, hybrid, symbol-only, and symbol-fused strategies,
plus one exact-symbol-hit@3 metric kept separate from Recall@k/MRR/nDCG@k.
On the deterministic fixture, symbol fusion improves exact-qualified-symbol
and duplicate-symbol MRR from 0.333 to 1.000, and improves the
natural-language-plus-symbol case from the historical hybrid baseline of
0.500 to 1.000. Ambiguous-symbol, snake/Pascal/camel, common-word, and
behavioral/non-symbol categories remain at MRR 1.000. These fixture results do
not establish real-repository or real-embedding quality, and `hybrid_symbol`
therefore remains opt-in rather than becoming a default.

## Evaluation harness and benchmarks

Capability and measured performance are different claims. “RepoMind supports
hybrid retrieval” describes system capability. “Hybrid achieved MRR 1.000 on
`repo-eval-v1`” describes one measured result on one intentionally small,
deterministic fixture. The latter does not establish real-model performance.

The evaluation path remains explicit:

```text
System configuration
        |
Versioned benchmark case
        |
Actual ranked/answered/edited result
        |
Gold labels or hidden read-only oracle
        |
Direct deterministic metrics
        |
Per-case evidence + aggregate report
```

`repomind.evaluation` reuses the stable chunk identity `(relative_path,
chunk_index, start_line, end_line)` across semantic, BM25, hybrid, reranked,
PostgreSQL-originated, RAG, and gold-label data. It never uses Python object
identity or database record IDs. Suites require unique case IDs, at least one
case, and a version. Scores from different benchmark versions must not be
treated as directly comparable datasets.

### Retrieval metrics

- **Recall@k** measures how much of the relevant set appears in the top `k`.
  Duplicate retrieved identities never improve it.
- **Reciprocal rank** is `1 / rank` for the first relevant result, or zero when
  none is retrieved. **MRR** averages that value across cases.
- **nDCG@k** measures how early binary-relevant results occur using
  `sum(relevance / log2(rank + 1))`, normalized by the ideal ordering.

The same evaluator accepts callable semantic, BM25, hybrid, and
hybrid-plus-reranking strategies. It retains each retrieved ordering and first
relevant rank instead of exposing only averages or declaring a winner.

### End-to-end RAG metrics

RAG evaluation invokes the public configurable repository-question pipeline and
records three distinct stages:

```text
retrieval recall -> context recall -> citation/fact answer oracle
```

This distinguishes a retriever miss from a context-budget exclusion and from a
final answer or citation failure. The offline answer oracle checks required fact
substrings, repository-owned cited chunks, and the expected insufficient-
evidence flag. It is transparent and deliberately narrower than a general
answer-quality judge.

### Coding-agent metrics and hidden oracles

Each coding case copies a clean fixture Git repository into a fresh temporary
workspace, runs the existing coding workflow sequentially, evaluates the final
workspace with read-only hidden file/change checks, and then discards the copy.
The agent receives only `CodingTask` and application verification policy—not
`file_contains`, required/allowed changed paths, or other evaluator answers.

- **Workflow completion** measures how often mechanical gates reached
  `COMPLETED`.
- **True task success** requires both workflow completion and a passing hidden
  oracle.
- **False-positive completion** means the workflow returned `COMPLETED` while
  the hidden oracle found the result wrong.
- **Recovery** requires an observable failed verification event, a later
  successful mutation, and finally fresh passing verification plus completion.

Reports also preserve final verification, changed paths, oracle failures, and
the existing LLM-call, tool-call, successful-mutation, agent-iteration, and
completion-attempt counters. Oracles support fixed file existence, absence,
substring, and changed-path checks; they execute no Python, shell, or model.
Milestone 21 additionally records whether a plan was generated, review attempts
and blocks, final reviewer approval, completion after review, and fixture-observed
false positives prevented by review. These are descriptive benchmark measurements,
not a claim that a small scripted fixture establishes general quality improvement.

### `repo-eval-v1` offline baseline

Run the bounded deterministic benchmark with:

```powershell
uv run python -m benchmarks.repo_eval_v1
```

Its two-query retrieval fixture produced:

```text
Strategy       Recall@3  MRR    nDCG@3
semantic       1.000     0.750  0.815
bm25           1.000     0.667  0.750
hybrid         1.000     1.000  1.000
hybrid+rerank  1.000     1.000  1.000
```

The same two questions at RAG `top_k=1` produced:

```text
Strategy       Retrieval Recall  Context Recall  Citation Recall  Answer Passed
semantic       0.500             0.500           0.500            0.500
hybrid         1.000             1.000           1.000            1.000
hybrid+rerank  1.000             1.000           1.000            1.000
```

The four-case scripted coding fixture produced:

```text
Workflow completion             0.750
True task success               0.500
False-positive completion       0.250
Verification pass               0.750
Recovery rate                   1.000
Mean LLM calls                  2.500
Mean tool calls                 1.000
Mean successful mutations       1.000
Mean agent iterations           2.500
Mean completion attempts        1.500
Planner generation              1.000
Reviewer approval               0.750
Mean review attempts            0.750
Review block rate               0.000
False positives prevented       0
```

These are `offline_fixture` and `offline_scripted` infrastructure baselines.
Embeddings and model decisions are deterministic fakes; the scores do not
measure an OpenAI embedding, reranking, generation, or coding model. A
`live_model` report is a separate mode and must never be aggregated with fake-
provider results. No live benchmark runs automatically.

### `repo-eval-v2` structural retrieval

Run benchmark entry points as modules from the repository root. The benchmark
files import one another through the `benchmarks` package, so direct file-path
execution such as `python benchmarks/repo_eval_v2_postgres.py` is not supported.
To inspect the PostgreSQL benchmark CLIs without connecting to a database, run:

```powershell
uv run python -m benchmarks.repo_eval_v2_postgres --help
uv run python -m benchmarks.ann_pgvector --help
```

Run the deterministic Python-structure fixture with:

```powershell
uv run python -m benchmarks.repo_eval_v2
```

The four-case fixture covers a class with similar methods, an exact qualified
symbol query, a natural-language behavior query, a nested function, decorators/
module content, and whole-symbol containment. The current offline hashed-vector
result at `k=3` is:

```text
Strategy                      Recall@3  MRR    nDCG@3
line_v1+exact                 1.000     0.583  0.690
python_ast_v1+exact           1.000     0.750  0.831
python_ast_v1+exact+bm25_rrf  1.000     0.875  0.908
python_ast_v1+context+exact   1.000     0.750  0.795
Whole-symbol containment      4/4
```

This small fake-embedding fixture supports the structural plumbing and boundary
decision but does not establish real-model generalization. In particular, the
metadata-enriched representation did not consistently beat raw source, so raw
source remains the embedding default. The application also retains `line_v1`
as its chunking default until broader real-repository evidence is available.

Run the opt-in PostgreSQL scale/quality benchmark only against an isolated,
migrated test database:

```powershell
$env:REPOMIND_TEST_DATABASE_URL = "postgresql+psycopg://.../repomind_test"
uv run python -m benchmarks.ann_pgvector --sizes 100 500 2000 --k 10 --queries 5
```

It reports corpus size, exact and ANN elapsed time, ANN Recall@k against exact
neighbors, and whether the normal planner selected HNSW. It inserts a uniquely
named repository inside one transaction and always rolls that transaction back.
There are no hard latency assertions because machine load and PostgreSQL state
make timing unsuitable as a CI correctness contract.

With the same environment variable, the structural fixture can run the required
real-pgvector matrix:

```powershell
uv run python -m benchmarks.repo_eval_v2_postgres
```

It compares `line_v1 + exact`, `python_ast_v1 + exact`, `python_ast_v1 + ANN`,
and `python_ast_v1 + ANN + BM25/RRF`, then reports ANN neighbor Recall@3 against
exact. Because this fixture is tiny, it disables sequential scans only inside
its rollback-only transaction to exercise the eligible HNSW path. Production
never changes that planner setting. This command is opt-in and is not part of
normal pytest.

## Read-only tool system

A tool is an explicit validated Python operation, not an autonomous agent:

```text
tool name
    ↓
Pydantic input validation
    ↓
deterministic ToolRegistry dispatch
    ↓
safe read-only handler
    ↓
structured Pydantic observation
```

`create_default_tool_registry` builds an isolated registry for one immutable
`ToolContext`. It provides:

- `read_file`: reads a complete text file or a 1-based inclusive line range;
  requested end lines beyond EOF are clamped, while a start beyond EOF fails
  clearly rather than returning an ambiguous range.
- `list_directory`: lists files, directories, and link entries in deterministic
  order; recursive traversal does not follow links and prunes common generated
  directories such as `.git`, `.venv`, and `node_modules`.
- `search_code`: performs case-insensitive literal substring search by default
  over supported source files, with optional case-sensitive and scoped searches.
- `find_symbol`: locates likely Python and JavaScript/TypeScript declarations
  using conservative regular-expression heuristics. It is not AST resolution
  and does not fall back to references.
- `git_status`: returns the branch and structured working-tree changes.
- `git_diff`: returns a bounded staged or unstaged diff, optionally restricted
  to one validated repository-relative path.

Every path supplied to a tool is treated as untrusted. Absolute, drive-relative,
and parent-traversal paths are rejected. Resolution reuses ingestion's containment
checks, and paths containing symlink or Windows junction components cannot be
opened or traversed. Directory listings may report a link entry but never follow
it. Git inspection uses hardcoded argument arrays, an explicit working directory,
and `--` before the only user-controlled pathspec; it never uses `shell=True`.

`ToolConfig` centrally bounds file reads and writes (1 MiB), replacement text,
directory results (500), source matches (50), diff output, verification output,
test failures, and verification time. Directory, search, diff, and subprocess
limits report truncation where applicable; oversized reads and writes fail
instead of silently truncating. Tool outputs contain repository-relative paths
rather than machine-specific absolute paths and serialize through
`model_dump(mode="json")`.

Retrieval and tools intentionally answer different kinds of questions. Semantic,
BM25, and hybrid retrieval search a previously indexed repository representation
for conceptually relevant chunks. Live tools inspect the current checkout:
`search_code` finds exact occurrences such as `RepositoryIngestionError`, while
`read_file` can show a precise range from `client.py`. Consequently, tools can see
uncommitted changes that a persisted vector index has not ingested. Tool calls do
not automatically ingest, embed, update PostgreSQL, or synchronize that index.

Tools are currently invoked directly from Python:

```python
from repomind.tools import ToolContext, create_default_tool_registry

registry = create_default_tool_registry(ToolContext(repository_root="."))
status = registry.execute("git_status", {})
matches = registry.execute(
    "search_code",
    {"query": "RepositoryIngestionError", "path": "src"},
)
```

The registry itself has no OpenAI dependency. It can still be invoked directly
without an LLM; the agent loop composes it with structured generation separately.

## Handwritten read-only agent loop

Milestone 11 implements the agent mechanism directly rather than hiding it behind
an agent framework:

```text
user query
    ↓
LLM structured decision
    ↓
tool action ──→ ToolRegistry ──→ observation ──┐
    │                                          │
    └────────────────── LLM again ←────────────┘

final action ──→ user-facing answer
```

Each `AgentDecision` selects exactly one action. A tool action contains one
registered `tool_name` and a JSON argument object; a final action contains only
the final answer. Pydantic rejects missing, contradictory, blank, or additional
decision fields. The model therefore cannot express an executable action as vague
prose such as “I will inspect client.py.” It must return data like:

```json
{
  "action": "tool",
  "tool_name": "read_file",
  "tool_arguments": {
    "path": "src/repomind/llm/client.py",
    "start_line": 40,
    "end_line": 100
  }
}
```

RepoMind—not the model—then validates and executes that call. Tool names,
descriptions, and argument JSON schemas are generated deterministically from the
actual `ToolRegistry`; there is no parallel hand-maintained schema list and no
provider-native function calling.

An `AgentRun` preserves the exact query, terminal status, final answer, immutable
tuple of `AgentStep` values, iteration count, LLM calls, and registry execution
attempts. Each tool step contains its structured decision and a JSON-compatible
`ToolObservation`. No chain-of-thought, scratchpad, hidden reasoning, timestamp,
or random identifier is requested or stored.

Expected tool failures are observations rather than fatal run errors. For example:

```text
LLM:  read_file(path="wrong.py")
Tool: error: file does not exist
LLM:  search_code(query="OpenAILLMClient")
Tool: found src/repomind/llm/client.py
LLM:  read_file(path="src/repomind/llm/client.py")
```

Unknown tools and invalid arguments follow the same recovery path and never
bypass the registry. An identical normalized tool call may execute twice; its
third and later occurrences receive a failure observation instructing the model
to choose a different action. Normalization uses sorted-key JSON rather than dict
insertion order or object identity.

`AgentConfig` defaults to eight LLM decisions and 60,000 characters of history.
When history exceeds that budget, the prompt keeps the most recent complete tool
interactions that fit and omits older interactions; observations are never cut in
the middle of their JSON block. System instructions, the exact original query,
and current registry schemas remain present on every call. Reaching the decision
limit returns `max_iterations_reached`, never a false successful completion.

Repository content, diffs, and tool observations are explicitly marked as
untrusted data in the centralized system prompt and history boundaries.
Instructions found inside them cannot override the user task or agent safety
rules. This establishes the architectural prompt-injection boundary without
claiming perfect model-level resistance.

RAG and an agent use different control flows:

```text
RAG:   question → retrieve fixed context → one generation → answer

Agent: question → decision → live tool → observation → decision → ... → answer
```

RAG supplies a predetermined retrieved context. The agent can dynamically choose
what live working-tree evidence to inspect next, recover from failed inspection,
or answer immediately without tools. `run_read_only_agent` and the default
registry remain strictly read-only and do not expose retrieval, databases, file
mutation, tests, or arbitrary shell commands as tools.

## Controlled editing and safe verification

Editing is a separate, explicit application capability:

```text
                   User coding task
                          |
                     Agent loop
                          |
                 inspect repository
                          |
                precise controlled edit
                          |
                      git_diff
                          |
                 run_tests / run_ruff
                          |
                     observation
                          |
                 correct if necessary
                          |
                       final
```

`create_default_tool_registry(...)` remains the six-tool read-only default: it
can inspect files, search source, and inspect Git, but cannot modify files or run
checks. `create_editing_tool_registry(...)` must be selected deliberately. It
contains the read-only tools plus exactly:

- `create_file`, which writes bounded UTF-8 content only when the parent already
  exists and the target does not;
- `replace_text`, which replaces exactly one literal occurrence and requires the
  SHA-256 returned by `read_file`;
- `run_tests`, a fixed `python -m pytest` invocation over validated test paths;
- `run_ruff`, a fixed `python -m ruff check` invocation over validated paths.

There is still no arbitrary shell, command argument surface, auto-fix, delete,
rename, Git mutation, package installation, or network tool. Capabilities and
limits are supplied by the application, not accepted as LLM tool arguments.

Optimistic concurrency makes an observed file state an edit precondition:

```text
read file -> receive hash A -> request edit with hash A -> recheck -> mutate
                                                       |
                              different current hash --+-> reject and re-read
```

Hashes cover the current file bytes even when `read_file` returns only a line
range. `replace_text` preserves the existing UTF-8/UTF-8-BOM or CP1252 encoding
and exact bytes outside the literal replacement, including LF or CRLF newlines.
Both mutation tools prepare complete temporary data in the target directory and
publish it atomically; `create_file` also retains no-overwrite semantics.

Verification uses subprocess argument arrays with `shell=False`, an application
timeout ceiling, and bounded stdout/stderr. A check that runs and fails returns a
normal structured result with `passed=false`, so the loop can perform:

```text
edit -> pytest fails -> failure observation -> corrective edit -> pytest passes
```

Startup failures remain tool errors, while timeouts are structured failed
results. The subprocess inherits the parent environment for ordinary local test
execution, but no tool can set environment variables and environment contents
are never returned. Failed tests do not roll back a valid edit automatically;
the agent observes and corrects it explicitly.

`run_editing_agent(...)` reuses the Milestone 11 loop and still permits one tool
action or one final action per decision. Its prompt treats source, diffs, test
output, lint output, and observations as untrusted data. Successful mutations
are bounded separately from total decisions by `EditingAgentConfig`. The caller
must supply an editing-capable registry; the agent never constructs or escalates
capabilities itself.

Live inspection sees edits immediately, but existing persisted semantic/BM25
representations are not automatically re-indexed. Retrieval may therefore be
stale until the caller performs an explicit re-index.

## Coding-task workflow and completion gates

`repomind.coding` adds a lifecycle above the existing tools and agent loop:

```text
CodingTask
    -> preflight Git state
    -> bounded structured planner (advisory)
    -> editing agent
    -> completion request
    -> required verification
    -> final Git review
    -> bounded independent reviewer
    -> deterministic completion gate
    -> CodingTaskResult
```

The LLM does not have final authority. Its final action means “I believe this is
ready”; `run_coding_task(...)` independently checks the configured policy before
returning `completed`. The lower-level `run_read_only_agent(...)` and
`run_editing_agent(...)` APIs remain available and unchanged in capability.

A `CodingTask` separates the human objective and ordered acceptance criteria
from mechanical proof. For example, “retry logic remains nonblocking” is a
semantic acceptance criterion. A policy requiring
`tests/unit/test_llm_client.py` and `ruff check .` describes checks RepoMind can
actually execute. Passing checks provide evidence but do not pretend to prove
every natural-language requirement.

The planner receives the visible task, acceptance criteria, verification policy,
and compact repository metadata. Its strict schema limits steps, path hints,
criterion mapping, risks, and verification labels. It has no tools and cannot
mutate the workspace. The executor receives the plan as model-authored advice,
inspects the actual repository, and may adapt when a path or assumption is wrong.

The reviewer is a separate structured call, not another agent loop. It receives
the task, advisory plan, current revision, deterministic verification facts,
changed paths, and only a complete bounded Git diff. A truncated or oversized diff
produces an explicit blocker rather than an approval based on partial evidence.
It evaluates each acceptance criterion and returns `approve` or
`changes_required` with bounded findings and corrections.

`VerificationPolicy` is selected by application code before the run. It fixes
the exact required pytest and Ruff scopes and whether final status/diff evidence
is required. A narrower agent-initiated test such as `tests/unit/foo` cannot
prove a policy requiring `tests`; the workflow reruns the exact policy scope.
The LLM cannot alter policy paths, clean-worktree behavior, completion attempts,
agent iterations, or mutation limits through tool arguments.

Verification freshness uses a per-workflow logical revision rather than Git
commits:

```text
revision 0 -> edit -> revision 1 -> tests pass at revision 1
                                      |
                         another edit -> revision 2 -> old result is stale
```

Only successful structured `create_file` and `replace_text` observations advance
the revision. Failed mutations, reads, Git inspection, and verification do not.
Required tests and Ruff must pass at the current revision. Final Git review and
independent reviewer output are also tagged with the current revision. A mutation
invalidates every older approval. This avoids creating commits merely to track
ephemeral agent state.

By default, preflight requires a clean working tree. If tracked or untracked
work already exists, the workflow returns `precondition_failed` before any LLM
call or mutation. Applications may explicitly allow dirty operation; baseline
paths are then retained separately from workflow-mutated paths. Final changed
files always come from `git_status`, and changes belonging to neither the
baseline nor successful workflow mutations are reported as unexpected and block
completion. RepoMind never cleans or resets them.

After each completion request, the workflow reuses only exact-scope passing
evidence from the current revision. Otherwise it invokes the existing fixed
`run_tests` and `run_ruff` tools, then captures `git_status`, unstaged
`git_diff`, and a staged diff when staged changes exist. Diff truncation remains
visible. Test failures, timeouts, and execution errors cannot count as passing.
When a gate fails, typed workflow feedback is returned to the same editing loop:

```text
incorrect edit -> completion request -> pytest fails
    -> trusted completion feedback + untrusted test evidence
    -> corrective edit -> fresh tests/Ruff -> fresh Git review
    -> reviewer requests a correction -> same executor edits
    -> fresh tests/Ruff + fresh Git review + fresh reviewer approval -> completed
```

Completion attempts are finite. Exhaustion returns `verification_failed`; agent
iteration exhaustion returns `agent_limit_reached`. A no-change task can complete
without a meaningless edit when required checks pass and final Git review shows
no task changes. Successful completion still leaves reviewable working-tree
changes for a human—there is no automatic add, commit, or push.

Direct API usage is explicit:

```python
from repomind.agent import EditingAgentConfig
from repomind.coding import CodingTask, VerificationPolicy, run_coding_task
from repomind.tools import ToolContext, create_editing_tool_registry

registry = create_editing_tool_registry(ToolContext(repository_root="path/to/repo"))
result = run_coding_task(
    CodingTask(
        objective="Fix the failing add() test.",
        acceptance_criteria=("Existing callers remain compatible.",),
    ),
    llm_provider,
    registry,
    verification_policy=VerificationPolicy(),
    agent_config=EditingAgentConfig(max_iterations=8),
)
```

## Repository layout

```text
RepoMind/
├── src/
│   └── repomind/
│       ├── api/
│       │   ├── __init__.py
│       │   ├── app.py
│       │   ├── dependencies.py
│       │   ├── errors.py
│       │   ├── models.py
│       │   ├── privacy.py
│       │   ├── routes.py
│       │   ├── store.py
│       │   └── services/
│       │       ├── __init__.py
│       │       ├── execution.py
│       │       ├── repositories.py
│       │       └── runs.py
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── loop.py
│       │   ├── models.py
│       │   └── prompts.py
│       ├── coding/
│       │   ├── __init__.py
│       │   ├── completion.py
│       │   ├── models.py
│       │   └── workflow.py
│       ├── config.py
│       ├── db/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── hybrid.py
│       │   ├── models.py
│       │   ├── repositories.py
│       │   └── session.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── coding.py
│       │   ├── metrics.py
│       │   ├── models.py
│       │   ├── rag.py
│       │   ├── reporting.py
│       │   └── retrieval.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── chunker.py
│       │   ├── language.py
│       │   ├── models.py
│       │   └── repository.py
│       ├── llm/
│       │   ├── client.py
│       │   └── models.py
│       ├── observability/
│       │   ├── __init__.py
│       │   ├── instrumentation.py
│       │   ├── models.py
│       │   ├── persistence.py
│       │   ├── recorder.py
│       │   ├── reporting.py
│       │   └── sanitization.py
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── context.py
│       │   ├── models.py
│       │   └── pipeline.py
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── bm25.py
│       │   ├── embeddings.py
│       │   ├── hybrid.py
│       │   ├── models.py
│       │   ├── reranking.py
│       │   ├── semantic_search.py
│       │   ├── similarity.py
│       │   └── tokenization.py
│       └── tools/
│           ├── __init__.py
│           ├── editing.py
│           ├── filesystem.py
│           ├── git.py
│           ├── models.py
│           ├── registry.py
│           ├── search.py
│           └── verification.py
├── scripts/
│   ├── inspect_repository.py
│   ├── manual_agent_check.py
│   ├── manual_coding_task_check.py
│   ├── manual_editing_agent_check.py
│   ├── manual_embedding_check.py
│   ├── manual_rag_check.py
│   ├── manual_semantic_search.py
│   └── manual_llm_check.py
├── benchmarks/
│   └── repo_eval_v1.py
├── alembic/
│   ├── versions/
│   │   ├── 20260910_01_initial_pgvector_schema.py
│   │   ├── 20260915_01_run_traces.py
│   │   ├── 20260915_02_repository_workspaces.py
│   │   ├── 20260916_01_durable_jobs.py
│   │   └── 20260916_02_job_cancellation.py
│   ├── env.py
│   └── script.py.mako
├── tests/
│   ├── integration/
│   │   ├── conftest.py
│   │   ├── test_api_postgres.py
│   │   ├── test_postgres_observability.py
│   │   └── test_postgres_persistence.py
│   └── unit/
│       ├── api/
│       │   ├── conftest.py
│       │   ├── test_api_repositories.py
│       │   ├── test_execution.py
│       │   ├── test_http.py
│       │   └── test_runs.py
│       ├── agent/
│       │   ├── test_agent_models.py
│       │   ├── test_editing_loop.py
│       │   ├── test_loop.py
│       │   └── test_prompts.py
│       ├── coding/
│       │   ├── test_completion.py
│       │   ├── test_models.py
│       │   └── test_workflow.py
│       ├── db/
│       │   ├── test_db_models.py
│       │   ├── test_hashing.py
│       │   ├── test_repositories.py
│       │   └── test_session.py
│       ├── evaluation/
│       │   ├── test_coding_evaluation.py
│       │   ├── test_metrics.py
│       │   ├── test_rag_evaluation.py
│       │   ├── test_reporting.py
│       │   └── test_retrieval_evaluation.py
│       ├── ingestion/
│       │   ├── test_chunker.py
│       │   ├── test_language.py
│       │   ├── test_models.py
│       │   └── test_repository.py
│       ├── observability/
│       │   ├── test_evaluation.py
│       │   ├── test_instrumentation.py
│       │   ├── test_models.py
│       │   ├── test_persistence.py
│       │   ├── test_recorder.py
│       │   ├── test_reporting.py
│       │   └── test_sanitization.py
│       ├── rag/
│       │   ├── test_context.py
│       │   ├── test_rag_models.py
│       │   ├── test_pipeline.py
│       │   └── test_retrieval_integration.py
│       ├── retrieval/
│       │   ├── test_bm25.py
│       │   ├── test_embedding_models.py
│       │   ├── test_embeddings.py
│       │   ├── test_hybrid.py
│       │   ├── test_reranking.py
│       │   ├── test_semantic_search.py
│       │   ├── test_similarity.py
│       │   └── test_tokenization.py
│       ├── tools/
│       │   ├── test_editing_registry.py
│       │   ├── test_editing_tools.py
│       │   ├── test_filesystem_tools.py
│       │   ├── test_git_tools.py
│       │   ├── test_registry.py
│       │   ├── test_search_tools.py
│       │   ├── test_tool_models.py
│       │   └── test_verification_tools.py
│       ├── test_config.py
│       └── test_llm_client.py
├── .env.example
├── .gitignore
├── alembic.ini
├── docker-compose.yml
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
    Load --> Snapshot[RepositorySnapshot]
    Snapshot --> Source[SourceFile objects]
    Source --> Chunk[CodeChunk objects]
    Chunk --> EmbedClient[OpenAI Embedding Client]
    EmbedClient --> Vector[EmbeddingVector]
    Vector --> Embedded[EmbeddedChunk]
    Embedded --> Persist[PostgreSQL + pgvector]
    Question[User question] --> QueryEmbed[Query embedding]
    Question --> Tokenize[Code-aware tokenization]
    QueryEmbed --> Semantic[Semantic ranking]
    Persist --> Semantic
    Embedded -. in-memory baseline .-> Semantic
    Tokenize --> BM25[BM25 ranking]
    Chunk --> BM25
    Persist -. load persisted chunks .-> BM25
    Semantic --> RRF[Reciprocal Rank Fusion]
    Semantic -. semantic-only baseline .-> Results
    BM25 --> RRF
    RRF --> Results[Hybrid candidates]
    Results --> Reranker[Bounded LLM reranker]
    Results -. no-rerank baseline .-> Seeds
    Reranker --> Reranked[Reranked chunks]
    Reranked --> Seeds[Ranked seed chunks]
    Seeds -. seeds_only baseline .-> Context
    Seeds --> Assembler[ContextAssembler: expand/dedup/pack]
    Assembler --> Context[Deterministic context builder]
    Context --> Generation[Structured LLM generation]
    Client --> Generation
    Generation --> Validation[Source-ID validation]
    Validation --> Answer[RepositoryAnswer]

    Workspace[Live repository workspace] --> ToolRegistry[Tool Registry]
    ToolRegistry --> ReadFile[read_file]
    ToolRegistry --> ListDirectory[list_directory]
    ToolRegistry --> SearchCode[search_code]
    ToolRegistry --> FindSymbol[find_symbol]
    ToolRegistry --> GitStatus[git_status]
    ToolRegistry --> GitDiff[git_diff]
    ReadFile --> Observation[Structured observations]
    ListDirectory --> Observation
    SearchCode --> Observation
    FindSymbol --> Observation
    GitStatus --> Observation
    GitDiff --> Observation
    EditingMode[Explicit editing opt-in] --> EditingRegistry[Editing Registry]
    EditingRegistry --> ToolRegistry
    EditingRegistry --> CreateFile[create_file]
    EditingRegistry --> ReplaceText[replace_text]
    EditingRegistry --> RunTests[run_tests]
    EditingRegistry --> RunRuff[run_ruff]
    CreateFile --> Observation
    ReplaceText --> Observation
    RunTests --> Observation
    RunRuff --> Observation
    UserTask[User task] --> AgentLoop[Handwritten Agent Loop]
    Client --> AgentLoop
    AgentLoop --> Decision[Structured tool or final decision]
    Decision -->|tool| ToolRegistry
    Decision -->|final| AgentAnswer[Agent final answer]
    Observation --> AgentLoop
```

`OpenAILLMClient` wraps the official OpenAI SDK. The rest of the codebase
depends on RepoMind's own `LLMResponse`, `TokenUsage`, and Pydantic response
models, not on raw SDK objects. `OpenAIEmbeddingClient` is separate because
language generation and numerical representation are distinct responsibilities.

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

## Local database

Start only the PostgreSQL/pgvector dependency:

```powershell
docker compose up -d postgres
```

Apply migrations from an empty or existing development database:

```powershell
uv run alembic upgrade head
```

The Compose credentials are intentionally local-only defaults. Override
`DATABASE_URL` outside local development. Stop the service without deleting its
named data volume:

```powershell
docker compose down
```

Use `docker compose down -v` only when intentionally resetting all local
RepoMind database data.

## Tests

Run the full unit test suite:

```powershell
uv run pytest
```

Run linting:

```powershell
uv run ruff check .
```

Ordinary pytest runs remain database-independent; PostgreSQL integration tests
skip unless an explicit test URL is supplied. With the Compose database healthy
and migrated, run the real pgvector suite in PowerShell with:

```powershell
$env:REPOMIND_TEST_DATABASE_URL = `
    "postgresql+psycopg://repomind:repomind@localhost:5432/repomind"
uv run pytest -m postgres
```

These tests use manual vectors and never require an OpenAI API key. They test
real PostgreSQL and pgvector rather than substituting SQLite.

## Optional live checks

With a valid `OPENAI_API_KEY` configured:

```powershell
uv run python scripts/manual_llm_check.py
```

This script makes real API calls and is not part of the automated test suite.

Run the separate, single-text embedding smoke check with:

```powershell
uv run python scripts/manual_embedding_check.py
```

It prints the model, dimensions, usage, and only the first five vector values.
Without an API key it exits without making a request.

Run a deliberately bounded semantic-search check with:

```powershell
uv run python scripts/manual_semantic_search.py `
    "Where is retry logic implemented?" --max-chunks 10
```

It embeds at most 10 source chunks by default, embeds the query once, and prints
up to five ranked locations. Without an API key it exits before ingestion or
network activity.

Run the bounded end-to-end repository RAG check with:

```powershell
uv run python scripts/manual_rag_check.py `
    "Where is retry logic implemented?" --max-chunks 10
```

It embeds at most 10 source chunks, embeds the question once, makes one
structured generation request, and prints only validated citations. Without an
API key it exits before ingestion or network activity.

Run one bounded live query through the handwritten read-only agent with:

```powershell
uv run python scripts/manual_agent_check.py `
    "Where is async retry behavior implemented?"
```

The script prints compact tool outcomes and the final run status. It cannot edit
files, mutate Git, run tests, or execute arbitrary commands. Without an API key it
exits before constructing the OpenAI client or making a network request.

The editing demo requires an explicit repository root and interactive
confirmation. Use only a disposable or version-controlled test repository:

```powershell
uv run python scripts/manual_editing_agent_check.py `
    "Add a short docstring to foo()" --root path/to/test/repository
```

It prints the selected root and capability boundary before asking `Continue?
[y/N]`. Without an API key it exits before confirmation, mutation, or network
access. The demo can create/replace files and run fixed checks, but still cannot
run arbitrary shell commands or mutate Git.

The higher-level coding-task demo additionally enforces clean preflight, default
tests/Ruff, and final Git review:

```powershell
uv run python scripts/manual_coding_task_check.py `
    --root path/to/sandbox/repository "Fix the failing add() test."
```

It requires the same explicit confirmation and never commits. Its final output
includes status, changed paths, verification results, bounded diff evidence, the
agent answer, and any deterministic completion blockers.

## Optional ingestion check

Inspect a repository without using the LLM:

```powershell
uv run python scripts/inspect_repository.py .
```

Include deterministic chunking statistics:

```powershell
uv run python scripts/inspect_repository.py . --chunks
```

## Design decisions

- **The agent loop is handwritten.** One structured decision, registry execution,
  observation, and repeat are explicit rather than delegated to a framework.
- **Provider abstraction stays small.** Normalized response models keep the
  SDK boundary contained and testable.
- **Secrets are externalized.** All environment configuration flows through
  `repomind.config.Settings`, and `.env` is gitignored.
- **Tests are offline by default.** SDK objects are injected in tests, so CI and
  local development do not require paid API calls.
- **Line chunking remains the deterministic baseline.** Python AST chunking is
  opt-in, slices original source, bounds oversized symbols, and falls back to
  line boundaries rather than making indexing fail.
- **Embedding generation stays separate.** It uses the official OpenAI SDK
  directly, normalizes provider responses, and keeps vectors associated with
  their source chunks.
- **NumPy is limited to vector mathematics.** Raw provider vectors remain
  immutable tuples; cosine similarity converts them to temporary arrays.
- **In-memory retrieval remains available.** PostgreSQL is an optional durable
  path, while the original in-memory pipeline stays useful as a simple baseline.
- **Raw semantic ranking stays visible.** The semantic-only baseline has no
  score threshold, keyword boost, or reranker; a small injected-retriever
  protocol lets evaluation swap strategies without changing its mathematics.
- **Citations use deterministic source IDs.** The LLM selects `S1`, `S2`, and
  similar identifiers; RepoMind owns and validates the corresponding paths and
  line ranges.
- **Context uses a character budget.** This simple deterministic limit is easy
  to inspect before token-aware budgeting is justified. Chunks stay complete,
  and overlapping intervals are not merged yet.
- **Basic RAG is not an agent.** It performs one retrieval, one context build,
  and one generation request without tools, planning, iteration, or editing.
- **Domain and ORM models stay separate.** Loading persistence records rebuilds
  `CodeChunk`, `EmbeddingVector`, and `EmbeddedChunk` objects, so retrieval and
  RAG do not depend on SQLAlchemy.
- **Persistence is transaction-controlled by callers.** Low-level operations
  flush for validation but do not commit repeatedly.
- **Exact pgvector search remains the reference.** HNSW is an explicit
  1,536-dimensional acceleration path; the planner is not forced, and exact
  results measure ANN Recall@k.
- **PostgreSQL plus pgvector remains the vector database.** HNSW is an index on
  the existing persistence layer, not a reason to add another database.
- **BM25 stays explicit and dependency-free.** Its positive IDF, saturation,
  and length-normalization math remain inspectable and independently testable.
- **Tokenization understands common code forms.** Exact normalized identifiers
  are retained while snake case, separators, and case transitions add useful
  component terms without claiming full language parsing.
- **Hybrid retrieval fuses ranks, not raw scores.** RRF avoids pretending cosine
  similarity and BM25 have a naturally shared numerical scale.
- **Lexical indexes are derived in memory.** They are reusable for in-memory
  queries but rebuilt from persisted chunks for each database hybrid call until
  corpus size justifies a durable lexical index.
- **Reranking remains an optional second stage.** It sees only bounded retriever
  output, uses one structured ordering request, and preserves original ranks so
  future evaluation can compare results with and without it.
- **Candidate IDs protect source ownership.** The LLM controls ordering only;
  RepoMind retains authority over chunks, paths, line numbers, and identities.
- **No rerank confidence is invented.** Model ordering is not calibrated, and
  raw semantic, BM25, and RRF scores are deliberately absent from its prompt.
- **Tool inputs and outputs are owned schemas.** Strict Pydantic inputs reject
  undeclared arguments before handlers run, while structured outputs are stable
  JSON-ready observations for direct use by the agent loop.
- **Live inspection is separate from indexing.** Read-only tools see the current
  working tree, including uncommitted changes, without mutating or synchronizing
  persisted retrieval indexes.
- **Exact search is not retrieval.** `search_code` provides predictable literal
  matching; BM25 and semantic search remain the indexed relevance mechanisms.
- **The registry is provider-independent.** It validates and dispatches Python
  handlers without importing OpenAI or parsing model tool calls.
- **No broad execution capability exists.** Git inspection and pytest/Ruff
  verification use fixed argument arrays; no arbitrary executable, arguments,
  environment overrides, or shell command can be supplied through a tool.
- **Structured decisions control execution.** Pydantic invariants permit one tool
  call or final answer per iteration, with no free-form action parsing.
- **Tool failures enable self-correction.** Expected registry failures become
  observations so the model can select a valid alternative on its next decision.
- **Agent history stores evidence, not reasoning.** Runs preserve decisions and
  observations without requesting or retaining chain-of-thought.
- **Provider-native tool calling remains deferred.** The first loop uses the
  existing structured-output interface so validation and dispatch stay visible.
- **Finite limits are mandatory.** Eight decisions, bounded recent history, and
  repeated-call protection prevent an accidental unbounded inspection loop.
- **Read-only remains the default.** Mutation and local verification require the
  separate editing registry and editing-agent wrapper chosen by the application.
- **Replacement is narrower than overwrite.** An exact one-occurrence literal
  replacement plus an expected SHA-256 reduces ambiguous and stale edits.
- **Writes are prepared atomically.** A completed same-directory temporary file
  is published only after all encoding and size validation succeeds.
- **Failed checks are observations.** They remain available for explicit agent
  correction and do not silently roll back a successfully applied edit.
- **Git publication stays human-controlled.** No add, commit, push, reset,
  checkout, restore, or clean capability is registered.
- **Indexes can be stale after editing.** Live filesystem tools see changes at
  once, but persisted retrieval data needs a separate explicit re-index.
- **The LLM requests completion; it does not grant it.** Deterministic gates use
  structured verification and Git evidence instead of parsing confident prose.
- **Acceptance and verification remain separate.** Human criteria describe the
  intended behavior; an application policy names only checks RepoMind can run.
- **Freshness follows logical revisions.** Successful mutations invalidate older
  verification without requiring temporary commits or Git history changes.
- **Final review is repository-owned evidence.** Status and bounded diffs expose
  actual and unexpected changes before a completed result is possible.
- **Clean preflight protects user work.** Dirty operation requires explicit
  application opt-in and preserves baseline attribution.
- **Planning is advisory.** A bounded plan improves task decomposition but cannot
  expand capabilities or require blind compliance from the executor.
- **Symbol retrieval reuses persisted metadata instead of reparsing.** Milestone
  22 already chunked and labeled every symbol; Milestone 24 only had to query
  it, never rebuild it.
- **Identifier extraction is metadata extraction, not query rewriting.** The
  original query always reaches semantic and BM25 retrieval unmodified;
  detected identifiers are a parallel, additional signal.
- **Symbol SQL lookup is bounded and repository-scoped by construction.** One
  `CASE`-ordered, `LIMIT`-bounded query joined through `repository_files`,
  never a full chunk scan and never cross-repository.
- **RRF was generalized, not replaced.** Reusing the existing fusion mechanism
  for a third source avoids a second, parallel ranking algorithm and comes with
  a regression test proving historical two-source behavior is unchanged.
- **Symbol candidate counts are bounded, not weighted.** A default limit plus
  tier-ordered rank position (not a hand-tuned symbol weight) keeps common-word
  matches from dominating fusion.
- **Common lowercase identifiers stay conservative.** `run`, `get`, `login`, and
  similar words are still searched, but only ever at the lowest confidence
  tier, because they are equally likely to be ordinary English.
- **No GraphRAG.** A call graph, reference graph, or import graph is a
  materially different (and materially larger) system than bounded exact
  metadata lookup, and nothing in this milestone's evidence justified one.
- **Historical baselines are print-verified, not just assumed.** `repo_eval_v2`
  and `repo_eval_v3` are re-run and compared against their previously recorded
  numbers every time retrieval or context-assembly code changes.
- **Review is independent and revision-sensitive.** It evaluates current evidence,
  not the executor's completion claim, and an edit invalidates its older approval.
- **Models cannot override deterministic gates.** Failed, missing, stale, or
  wrong-scope tests/Ruff and incomplete Git evidence block completion regardless
  of planner, executor, or reviewer prose.
- **This is not multi-agent architecture.** Planner and reviewer are single typed
  model calls inside one existing coding workflow, without tools or independent loops.

## Roadmap

The full project roadmap is described in the RepoMind engineering brief:

1. LLM foundation (complete)
2. Repository ingestion (complete)
3. Code chunking (complete)
4. Embeddings (complete)
5. Semantic search (complete)
6. Basic RAG (complete)
7. PostgreSQL + pgvector persistence (complete)
8. BM25 + hybrid retrieval (complete)
9. LLM-based reranking (complete)
10. Read-only tool system (complete)
11. Handwritten read-only agent loop (complete)
12. Controlled editing + safe verification (complete)
13. Coding-task workflow + completion gates (complete)
13.5. RAG retrieval integration hardening (complete)
14. Evaluation harness + coding-agent benchmarks (complete)
15. Observability and persistent run tracing (complete)
16. FastAPI backend (complete)
17. Semantic SSE progress streaming (complete)
18. Next.js + TypeScript frontend (complete)
19. Redis + durable worker/job architecture (complete)
20. Cooperative cancellation / job control (complete)
21. Structured planner + independent reviewer (complete)
22. Retrieval V2: structural chunking + pgvector HNSW ANN (complete)
23. Retrieval quality + context assembly V2 (complete)
24. Symbol-aware retrieval fusion + realistic benchmark corpus (complete/current)
25. Next capability milestone (next; not yet selected)
