# Evaluation

RepoMind separates two different claims:

- **Capability** — "RepoMind supports hybrid retrieval / an indexed
  navigation tool." This is a statement about what the system does.
- **Measured result** — "Hybrid achieved MRR 1.000 on `repo-eval-v1`." This
  is one number on one versioned, intentionally small fixture. It does not
  establish real-model performance in general, and scores from different
  benchmark versions are not directly comparable.

Most benchmarks in `benchmarks/` — the versioned-case retrieval, RAG,
coding, and agent-navigation suites — follow the same evaluation path
(`ann_pgvector.py` is a corpus-size/ANN-Recall@k timing benchmark instead;
see below):

```text
system configuration -> versioned benchmark case -> actual result
    -> gold labels or hidden oracle -> deterministic metrics
    -> per-case evidence + aggregate report
```

## Offline vs. live evaluation

| | Offline (deterministic) | Live (paid) |
| --- | --- | --- |
| Model / embeddings | Fake, injected, deterministic | Real OpenAI API |
| OpenAI / model network requests | None | Real HTTP requests, costs money |
| Covered by CI | Evaluation logic via `pytest`; benchmark CLIs are not invoked directly | Never |
| Purpose | Prove orchestration, safety plumbing, and regression-free algorithm changes | Prove a real model chooses tools/answers well under the same orchestration |
| Grading | Deterministic (substring/oracle checks) | Deterministic — **no LLM judge** |

Offline benchmarks use fully scripted fake LLMs and fake embeddings; a
`live_model` report is a structurally separate mode and is never aggregated
with fake-provider results. No live benchmark runs automatically, in CI, or
as a side effect of any other command.

"OpenAI / model network requests: None" describes offline evaluation only —
it is not a claim of zero network activity in general. Some offline variants
(the opt-in `*_postgres`/`ann_pgvector` benchmarks below) connect to a real
local/test PostgreSQL database; that is unrelated to, and does not require,
any OpenAI or model-provider call.

## Live navigation benchmark: `repo-agent-eval-v3`

`benchmarks/agent_navigation_eval.py` answers: does the real read-only
investigation agent (not a scripted stand-in) navigate an unfamiliar
repository and answer correctly, with and without the optional indexed
navigation tool? `benchmarks/agent_live_eval.py` is the real-model harness
that runs the identical fixture, tasks, and tool registries through a live
`OpenAILLMClient` instead of a scripted decision queue.

The benchmark is a **small, controlled, synthetic repository-navigation
benchmark** — it is not a claim about general coding-agent accuracy. It
covers six categories, verified against `benchmarks/agent_navigation_eval.py`
and the fixture module:

- exact symbol lookup
- semantic terminology mismatch (user wording differs from source wording)
- cross-file evidence (answer requires two files)
- literal lookup (exact string search)
- indexed miss with filesystem fallback (indexed mode only)
- source prompt injection (a fixture file contains an embedded instruction
  the agent must not obey)

### Committed result

The repository commits one real run's result at
[`benchmarks/results/repo-agent-eval-v3-live.json`](../benchmarks/results/repo-agent-eval-v3-live.json).
Every number below is read directly from that file, not recomputed.

| Field | Value |
| --- | --- |
| `benchmark_version` | `repo-agent-eval-v3` |
| `harness_schema_version` | `agent-live-eval-v1` |
| `model` | `gpt-5.6-terra` |
| Code-under-test SHA | `7ffc2afea3e0288c251d121aa0dac38c4aa0df7e` |
| `max_iterations` | 8 |

| Retrieval mode | Cases | Task success | Mean tool calls | Mean indexed searches | Verified retrieval follow-up | Total tokens |
| --- | --- | --- | --- | --- | --- | --- |
| Filesystem | 5 | 1.000 | 2.600 | 0.000 | 1.000 | 26,256 |
| Indexed | 6 | 1.000 | 2.333 | 1.000 | 1.000 | 33,754 |

RepoMind passed all applicable cases in this controlled live navigation run.

**Why 5 vs. 6 cases:** the indexed-miss-fallback case only exists in indexed
mode (it tests indexed search returning zero results and the agent falling
back to filesystem tools), so filesystem mode has five applicable cases and
indexed mode has six.

**Verified retrieval follow-up = 1.0** means every case that relied on an
indexed search result read the current file at that path before finalizing
its answer — the current-source-authority policy held for every case in this
run (see [`docs/architecture.md`](architecture.md#indexed-navigation-and-current-source-authority)).

### Important nuance — do not over-read this one run

- Indexed mode used **fewer mean tool calls** (2.333 vs. 2.600) in this run.
- Indexed mode used **more total tokens** (33,754 vs. 26,256) in this run,
  because `indexed_code_search` results themselves consume prompt tokens.
- **Do not conclude indexed navigation is cheaper, faster, or more
  token-efficient overall from this.** This is one run, six cases, one model,
  one fixture repository. It shows indexed navigation can reduce
  tool-call count while still requiring current-source verification — it
  does not establish a general cost or latency advantage.

### Running it yourself

Offline (deterministic, free, network-free):

```powershell
uv run python -m benchmarks.agent_navigation_eval
```

Live (**COSTS MONEY, makes real OpenAI requests**). Write to a fresh local
file, never to the committed artifact above:

```powershell
$liveResult = Join-Path $env:TEMP "repomind-agent-live.json"

uv run python -m benchmarks.agent_live_eval `
    --retrieval-mode both `
    --confirm-live `
    --output $liveResult
```

The committed `benchmarks/results/repo-agent-eval-v3-live.json` is a frozen
evidence artifact for one specific reviewed run, tied to commit
`7ffc2afea3e0288c251d121aa0dac38c4aa0df7e`. An ad-hoc rerun — from a
different commit, model, or day — has its own provenance and must not
silently replace it; promoting a new run to the committed evidence file is a
deliberate, separate decision, not a side effect of running the harness.

`agent_live_eval` refuses to run without both a configured `OPENAI_API_KEY`
and the explicit `--confirm-live` flag; without them (or with `--help`) it
makes no network request. It is never invoked by CI, by any other benchmark,
or by any test.

## Offline benchmarks (retrieval / RAG / coding fixtures)

None of the commands below make an OpenAI or other model-provider network
request. The versioned `repo-eval-v1`–`repo-eval-v4` fixture benchmarks are
deterministic and run fully in-process. The opt-in PostgreSQL benchmarks
further below also avoid model-provider calls and roll back their own
transaction, so they too remain safe to run repeatedly; `ann_pgvector`
additionally reports local wall-clock timings, so its timing values are
expected to vary between runs.

### `repo-eval-v1` — baseline retrieval, RAG, and coding fixture

```powershell
uv run python -m benchmarks.repo_eval_v1
```

Two-query retrieval fixture:

| Strategy | Recall@3 | MRR | nDCG@3 |
| --- | --- | --- | --- |
| semantic | 1.000 | 0.750 | 0.815 |
| bm25 | 1.000 | 0.667 | 0.750 |
| hybrid | 1.000 | 1.000 | 1.000 |
| hybrid+rerank | 1.000 | 1.000 | 1.000 |

Same two questions at RAG `top_k=1`:

| Strategy | Retrieval recall | Context recall | Citation recall | Answer passed |
| --- | --- | --- | --- | --- |
| semantic | 0.500 | 0.500 | 0.500 | 0.500 |
| hybrid | 1.000 | 1.000 | 1.000 | 1.000 |
| hybrid+rerank | 1.000 | 1.000 | 1.000 | 1.000 |

Four-case scripted coding fixture measures workflow completion (0.750), true
task success (0.500 — completion **and** a passing hidden oracle),
false-positive completion (0.250), and recovery rate (1.000), among other
counters. The coding oracle sees only `CodingTask` and the verification
policy — never `file_contains`, required changed paths, or other evaluator
answers — and itself executes no Python, shell, or model.

These are `offline_fixture`/`offline_scripted` infrastructure baselines using
deterministic fakes; they do not measure a real OpenAI embedding, reranking,
generation, or coding model.

### `repo-eval-v2` — structural (AST) chunking comparison

```powershell
uv run python -m benchmarks.repo_eval_v2
```

Four-case fixture (similar methods, exact qualified-symbol query,
natural-language behavior query, nested function/decorators):

| Strategy | Recall@3 | MRR | nDCG@3 |
| --- | --- | --- | --- |
| `line_v1+exact` | 1.000 | 0.583 | 0.690 |
| `python_ast_v1+exact` | 1.000 | 0.750 | 0.831 |
| `python_ast_v1+exact+bm25_rrf` | 1.000 | 0.875 | 0.908 |
| `python_ast_v1+context+exact` | 1.000 | 0.750 | 0.795 |

`line_v1` chunking remains the application default; structural chunking is
opt-in.

Opt-in real-pgvector variants (require an isolated, migrated test database —
never run against a shared database). `repo_eval_v2_postgres` reuses the same
versioned fixture/oracle as `repo_eval_v2` above; `ann_pgvector` does not use
a versioned case suite at all — it generates a synthetic corpus at several
sizes and reports ANN Recall@k and timing against exact search:

```powershell
$env:REPOMIND_TEST_DATABASE_URL = "postgresql+psycopg://.../repomind_test"
uv run python -m benchmarks.repo_eval_v2_postgres
uv run python -m benchmarks.ann_pgvector --sizes 100 500 2000 --k 10 --queries 5
```

Both insert data inside one transaction and always roll it back. Run entry
points as modules from the repository root — the `benchmarks` package
imports across files, so `python benchmarks/repo_eval_v2_postgres.py` is not
supported.

### `repo-eval-v3` — context assembly

```powershell
uv run python -m benchmarks.repo_eval_v3
```

Holds retrieval ranking fixed and measures only whether `ContextAssembler`
delivers more of the gold evidence into the final token/char-bounded prompt
context, isolating context-assembly quality from ranking quality.

### `repo-eval-v4` — symbol fusion on a realistic corpus

```powershell
uv run python -m benchmarks.repo_eval_v4
```

A small but deliberately ambiguous multi-file corpus (two distinct `login`
methods on different classes, common-English-word method names, mixed
naming conventions) exercising symbol-fusion tie-breaking that a
single-file fixture cannot.
