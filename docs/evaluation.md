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

Per-mode aggregate totals (not per-case means), also read directly from the committed JSON:

| Retrieval mode | LLM calls | Tool calls | Prompt tokens | Completion tokens | Duration |
| --- | --- | --- | --- | --- | --- |
| Filesystem | 18 | 13 | 25,217 | 1,039 | ~35.5s |
| Indexed | 20 | 14 | 32,590 | 1,164 | ~33.6s |

RepoMind passed all applicable cases in this controlled live navigation run.

**Why 5 vs. 6 cases:** the indexed-miss-fallback case only exists in indexed
mode (it tests indexed search returning zero results and the agent falling
back to filesystem tools), so filesystem mode has five applicable cases and
indexed mode has six.

**Verified retrieval follow-up** is a per-case boolean: at least one location
surfaced by the most recent indexed search that returned results was re-read
with `read_file` before the final answer. It does not require every surfaced
path to be re-read, nor that the answer cite the re-read file. It is
vacuously true when no indexed search returned results — which includes
every filesystem-mode case, where `indexed_code_search` does not exist — so
the filesystem row's 1.000 carries no information. In indexed mode, 1.0 means
every case that received indexed results re-read at least one of those files
from the current source before finalizing — a minimal check of the
current-source-authority policy (see [`docs/architecture.md`](architecture.md#indexed-navigation-and-current-source-authority)),
not proof that every claim was verified.

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

`--output` is checked before any live request: missing parent directories are
created and the location must be writable, so a bad path fails immediately
instead of discarding a paid run. Each result records `git_commit_sha` plus
`git_dirty` (whether the working tree had uncommitted or untracked changes;
`null` when unknown). The committed artifact predates `git_dirty`, so it has
no such field and still loads with it treated as unknown.

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
request. The versioned `repo-eval-v1`–`repo-eval-v4` fixture benchmarks and
the default (offline) mode of `repo-eval-v5` are deterministic and run fully
in-process. The opt-in PostgreSQL benchmarks
further below also avoid model-provider calls and roll back their own
transaction, so they too remain safe to run repeatedly; `ann_pgvector`
additionally reports local wall-clock timings, so its timing values are
expected to vary between runs.

### `repo-eval-v5` — real questions over a real repository

```powershell
uv run python -m benchmarks.repo_eval_v5             # 44-question dev set
uv run python -m benchmarks.repo_eval_v5 --held-out  # 16-question held-out set
```

`repo-eval-v1`–`v4` use tiny synthetic fixtures with a handful of queries
each, which is why several of their rows sit at 1.000. `repo-eval-v5` asks
**44 developer questions about RepoMind's own source tree**, frozen at commit
`efd04a5` and extracted with `git archive` so local edits cannot leak in, and
holds back **16 more questions written after the retrieval changes below were
chosen**. Categories are `behavior` (natural-language "how does X work"),
`identifier` (names a class or function), and `config` (settings, Compose,
CI).

Gold labels are **source spans, not chunk IDs**: either a Python symbol
(resolved through the AST, decorators included) or an explicit line range.
A retrieved chunk counts as evidence for a span when it overlaps at least
`min(span length, 5)` lines of it, so line and structural chunking are scored
on equal terms and a shared boundary line is not evidence
(`repomind/evaluation/spans.py`). A gold target that no longer resolves fails
the run instead of silently scoring zero.

Two things are measured:

1. **Ranking:** hit@k (any gold span covered in the top k), target recall@k,
   and MRR.
2. **Evidence reaching the model:** after the production RAG context packing
   (`assemble_context` + `build_repository_context`, 20,000-character budget),
   is the gold evidence actually in the prompt? This is the number that
   matters for answers, and it is where chunking choices show up.

**Offline mode is BM25 + symbol only.** Semantic and fused strategies need
real embeddings, so the committed numbers below do **not** include the
semantic arm. `--live --confirm-live` with a configured `OPENAI_API_KEY`
adds `semantic`, `hybrid`, and `hybrid_symbol` rows; it embeds the whole
corpus (roughly 1.2M embedding tokens, a few cents with
`text-embedding-3-small`) and is never run in CI.

Ranking at k=5 (hit / target recall / MRR). "Legacy" is the previous BM25
(`LEGACY_BM25_CONFIG`: content only, no stopword filtering, tests weighted
like source):

| Chunking | Retriever | Dev (44) | Held-out (16) |
| --- | --- | --- | --- |
| `line_v1` | BM25 legacy | 0.432 / 0.409 / 0.216 | 0.812 / 0.812 / 0.460 |
| `line_v1` | BM25 | **0.659 / 0.625 / 0.419** | **0.812 / 0.812 / 0.708** |
| `python_ast_v1` | BM25 legacy | 0.295 / 0.250 / 0.224 | 0.688 / 0.656 / 0.583 |
| `python_ast_v1` | BM25 | **0.523 / 0.443 / 0.295** | **0.750 / 0.719 / 0.672** |
| `python_ast_v1` | BM25 legacy + symbol (weak matches uncorroborated) | 0.273 / 0.239 / 0.218 | 0.625 / 0.594 / 0.594 |
| `python_ast_v1` | BM25 + symbol (weak matches corroborated) | 0.523 / 0.443 / 0.266 | 0.750 / 0.719 / 0.641 |

Evidence reaching the model, top_k=5:

| Configuration | Dev evidence hit | Held-out evidence hit | Mean context chars (dev) |
| --- | --- | --- | --- |
| Previous product shape: `line_v1`, legacy BM25, `seeds_only` | 0.341 | 0.750 | 17,508 |
| `line_v1`, new BM25, `seeds_only` | 0.591 | 0.812 | 17,671 |
| `line_v1`, new BM25, `expanded` | 0.591 | 0.812 | 15,255 |
| **`python_ast_v1`, new BM25, `expanded` (new default)** | **0.636** | **0.875** | **11,863** |

What the numbers drove:

- **BM25 now indexes the file path and qualified symbol name** alongside the
  content, **drops English function words from the query** (unless nothing
  would be left), and **halves the score of test files** unless the query is
  about tests (`test`, `pytest`, `fixture`, …). Removing any one of them
  lowers structural-chunking dev hit@5 (0.523 with all three; 0.386 without
  stopword filtering, 0.295 without test demotion, 0.477 without the path).
  With line chunking the path made no hit@5 difference and helped hit@10
  (0.773 without, 0.795 with). Stemming was tried and dropped: it was mixed. `k1`/`b` were
  not tuned.
- **Weak symbol matches need corroboration.** A plain English word in the
  query (`run`, `claim`, `lease`) that happens to name a function used to get
  a full rank-1 vote in fusion; it now only counts when BM25 or semantic
  retrieval independently found the same symbol. Without that, adding the
  symbol arm made BM25 *worse* (0.295 → 0.273 dev, 0.688 → 0.625 held-out).
- **Structural chunking is the indexing default** and `expanded` is the RAG
  default. Structural chunks rank a little lower than 120-line windows on
  raw hit@5 (a 120-line window covers more spans by sheer size), but once
  neighbor expansion runs, more gold evidence reaches the model in about two
  thirds of the characters.
- **`top_k` stays at 5.** Going to 10–20 helped some configurations and
  hurt others on these sets, so there was no clear case to change it.

Caveats: 60 questions about one repository is a real but small sample, and
the author of the questions also chose the changes (the held-out set exists
to check that; it agrees in direction). The semantic arm and generated
answers are unmeasured here; the `hybrid` default is a product judgment that
fusing an improved lexical arm with semantic search beats semantic alone,
consistent with `repo-eval-v1`, not a measured `repo-eval-v5` result. Run
`--live --confirm-live` to measure it.

### `repo-eval-v1` — baseline retrieval, RAG, and coding fixture

```powershell
uv run python -m benchmarks.repo_eval_v1
```

Two-query retrieval fixture:

| Strategy | Recall@3 | MRR | nDCG@3 |
| --- | --- | --- | --- |
| semantic | 1.000 | 0.750 | 0.815 |
| bm25 | 1.000 | 0.750 | 0.815 |
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
counters. The coding runner (the `CodingTaskRunner` under test) sees only
the visible `CodingTask` and the verification policy — never
`file_contains`, required/allowed changed paths, or other hidden-oracle
answers. The hidden oracle then inspects the final copied workspace: it
checks `file_exists`/`file_not_exists`, `file_contains`/`file_not_contains`,
and required/allowed changed paths, computing the changed paths itself from
`git status` of that workspace rather than trusting the workflow's
self-reported `changed_files`. The oracle executes no fixture code, shell, or
model; its only subprocess is that fixed-argument, read-only `git status`.

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
| `python_ast_v1+exact+bm25_rrf` | 1.000 | 1.000 | 1.000 |
| `python_ast_v1+context+exact` | 1.000 | 0.750 | 0.795 |

`python_ast_v1` structural chunking is the application default (see
`repo-eval-v5` above for why); `REPOMIND_CHUNKING_STRATEGY=line_v1` reverts
to line chunking.

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
