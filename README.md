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

The current baseline is simple line-based chunking:

- `max_lines_per_chunk` defaults to 120
- `overlap_lines` defaults to 20
- chunk line numbers are 1-based and inclusive
- chunk indexes restart at zero for each file
- empty files produce no chunks

Overlap exists because code near a chunk boundary often depends on surrounding
context. Repeating nearby lines helps prevent that context from being split
completely across two retrieval units.

Chunking is intentionally deterministic and synchronous. It does not perform
embeddings, retrieval, or syntax-aware parsing.

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

Chunk content is sent exactly as stored, including indentation and newline
characters. Empty strings are rejected; whitespace-only strings are deliberately
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
code_chunks (line ranges, exact content, SHA-256, model, dimensions, vector)
    ↓
exact pgvector cosine search
    ↓
SemanticSearchResult[]
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

No approximate-nearest-neighbor index is created yet. Exact search keeps the
Milestone 5 mathematics directly comparable and avoids imposing one fixed
dimension on every embedding model. ANN design and measurement can follow when
the corpus size justifies it.

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
and average document length once from `CodeChunk` content. It uses `k1 = 1.5`
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

For persisted repositories, `postgres_hybrid_search` combines exact pgvector
semantic results with `load_chunks` → an in-memory `BM25Index` → RRF. The query
embedding remains an input to the database layer, so neither BM25 nor database
code calls OpenAI. Lexical indexing is currently rebuilt for each database
hybrid call; callers performing repeated in-memory searches can reuse a
`BM25Index`.

Approximate costs are O(N × D) for in-memory semantic comparison, O(N × |Q|)
for this straightforward BM25 scorer over N chunks and query terms Q, and
O(candidate results) for RRF. PostgreSQL handles the exact database semantic
scan, while database-backed lexical work still loads chunks into Python. This
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

The following ideas are intentionally deferred until the evaluation harness can
measure them against the current baseline:

- syntax-aware / AST chunking
- richer embedding text containing path or symbol metadata
- parent or neighboring-chunk context expansion
- overlap-aware context deduplication
- token-aware context budgeting
- query rewriting or multi-query retrieval
- exact-symbol-aware retrieval fusion

These changes may improve some workloads, but they also change cost, latency,
recall, or context composition. They should be measured rather than assumed to
improve repository-answer quality.

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
    -> editing agent
    -> completion request
    -> required verification
    -> final Git review
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
Required tests and Ruff must pass at the current revision. Final review is also
tagged with the current revision. This avoids creating commits merely to track
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
    -> corrective edit -> fresh tests/Ruff -> fresh Git review -> completed
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
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── chunker.py
│       │   ├── language.py
│       │   ├── models.py
│       │   └── repository.py
│       ├── llm/
│       │   ├── client.py
│       │   └── models.py
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
├── alembic/
│   ├── versions/
│   │   └── 20260910_01_initial_pgvector_schema.py
│   ├── env.py
│   └── script.py.mako
├── tests/
│   ├── integration/
│   │   ├── conftest.py
│   │   └── test_postgres_persistence.py
│   └── unit/
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
│       ├── ingestion/
│       │   ├── test_chunker.py
│       │   ├── test_language.py
│       │   ├── test_models.py
│       │   └── test_repository.py
│       ├── rag/
│       │   ├── test_context.py
│       │   ├── test_rag_models.py
│       │   └── test_pipeline.py
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
    Results -. no-rerank baseline .-> Context
    Reranker --> Reranked[Reranked chunks]
    Reranked --> Context[Deterministic context builder]
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
- **Chunking starts with a deterministic baseline.** Line-based chunking is
  deliberately simple so later syntax-aware strategies can be compared against
  a stable reference.
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
- **Exact pgvector search comes first.** No HNSW or IVFFlat index is claimed or
  added before corpus-scale measurements justify approximate retrieval.
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
- **No planner or reviewer model is needed yet.** One editing agent receives
  deterministic workflow feedback; completion and final review remain Python
  policy rather than another model's opinion.

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
14. Evaluation harness + coding-agent benchmarks (next)
