"""Gold questions for ``repo-eval-v5``: real questions about a real repository.

The corpus is RepoMind's own tree, frozen at :data:`CORPUS_COMMIT` (the last
commit before this benchmark existed, so no question or gold label can leak
into the text being searched). Every case is phrased the way a developer new
to the codebase would ask it. Most use behavior words rather than the code's
own identifiers, because that terminology gap is the common case and the one
lexical search handles worst.

Gold targets are resolved at run time against the frozen corpus: Python
symbols through the AST, everything else by explicit line range, so the
labels stay valid for any chunking strategy.
"""

from repomind.evaluation.spans import SpanBenchmarkCase, SpanBenchmarkSuite, SpanTarget

VERSION = "repo-eval-v5"
# "docs: align safety, provenance, and job-lease claims with the code" (2026-10).
CORPUS_COMMIT = "efd04a5ca2d0f8cbaa0c5338b6807b85a6becdad"

_SRC = "src/repomind/"


def _sym(path: str, symbol: str) -> SpanTarget:
    return SpanTarget(path=_SRC + path, symbol=symbol)


def _lines(path: str, start: int, end: int) -> SpanTarget:
    return SpanTarget(path=path, lines=(start, end))


def _case(case_id: str, category: str, query: str, *targets: SpanTarget) -> SpanBenchmarkCase:
    return SpanBenchmarkCase(id=case_id, category=category, query=query, targets=targets)


BEHAVIOR = "behavior"
IDENTIFIER = "identifier"
CONFIG = "config"

CASES: tuple[SpanBenchmarkCase, ...] = (
    _case(
        "claim-job",
        BEHAVIOR,
        "How does a worker claim a queued job without two workers taking the same one?",
        _sym("jobs/store.py", "PostgresJobStore.claim_next"),
    ),
    _case(
        "expired-lease",
        BEHAVIOR,
        "What happens to a running job when its worker dies and the lease runs out?",
        _sym("jobs/store.py", "PostgresJobStore.recover_expired"),
    ),
    _case(
        "lease-heartbeat",
        BEHAVIOR,
        "How often does a worker extend the lease on the job it is running?",
        _sym("jobs/worker.py", "JobWorker._heartbeat"),
    ),
    _case(
        "cancel-in-progress",
        BEHAVIOR,
        "How does cancelling a job stop work that is already in progress?",
        _sym("jobs/control.py", "DurableCancellationToken.checkpoint"),
        _sym("jobs/store.py", "PostgresJobStore.request_cancel"),
    ),
    _case(
        "prompt-escaping",
        BEHAVIOR,
        "How is repository text prevented from breaking out of the prompt's data block?",
        _sym("rag/context.py", "escape_untrusted_text"),
        _sym("rag/context.py", "format_source_block"),
    ),
    _case(
        "oversized-first-result",
        BEHAVIOR,
        "What happens when the best search result is too large for the context budget?",
        _sym("rag/context.py", "truncate_source_to_fit"),
        _sym("rag/context.py", "build_repository_context"),
    ),
    _case(
        "reindex-reuse",
        BEHAVIOR,
        "When can re-indexing reuse existing vectors instead of embedding everything again?",
        _sym("indexing.py", "classify_files"),
        _sym("indexing.py", "index_fingerprint"),
    ),
    _case(
        "identifier-splitting",
        BEHAVIOR,
        "How are camelCase and snake_case names split into search terms?",
        _sym("retrieval/tokenization.py", "tokenize_code"),
        _sym("retrieval/tokenization.py", "_split_case_transitions"),
    ),
    _case(
        "merge-rankings",
        BEHAVIOR,
        "How are the vector search and keyword search rankings merged into one list?",
        _sym("retrieval/hybrid.py", "fuse_ranked_sources"),
    ),
    _case(
        "cross-site-requests",
        BEHAVIOR,
        "How does the API reject requests coming from other websites?",
        _sym("api/security.py", "ClientHeaderMiddleware.__call__"),
        _sym("api/security.py", "TrustedHostMiddleware.__call__"),
    ),
    _case(
        "embedding-batches",
        BEHAVIOR,
        "How are embedding requests split so a single request stays under the provider limit?",
        _sym("retrieval/embeddings.py", "_batches"),
    ),
    _case(
        "verifier-environment",
        BEHAVIOR,
        "Which environment variables are passed to the test runner subprocess?",
        _sym("tools/verification.py", "_verification_environment"),
    ),
    _case(
        "indexed-final-answer",
        BEHAVIOR,
        "Why can't the investigation agent give its final answer right after an index search?",
        _sym("agent/indexed.py", "_IndexedGroundingPolicy.handle_final"),
    ),
    _case(
        "trace-secrets",
        BEHAVIOR,
        "How are API keys and tokens scrubbed from stored traces?",
        _sym("observability/sanitization.py", "redact"),
        _sym("observability/sanitization.py", "_replace_secret"),
    ),
    _case(
        "llm-retries",
        BEHAVIOR,
        "How are transient OpenAI failures retried?",
        _sym("llm/client.py", "OpenAILLMClient._complete"),
        _sym("llm/client.py", "_retry_delay"),
    ),
    _case(
        "minified-line",
        BEHAVIOR,
        "What happens to a minified file with one enormous line during chunking?",
        _sym("ingestion/lines.py", "split_oversized_line"),
    ),
    _case(
        "skipped-directories",
        BEHAVIOR,
        "Which directories are skipped when scanning a repository for source files?",
        _lines(_SRC + "ingestion/models.py", 8, 30),
        _sym("ingestion/repository.py", "_is_ignored_directory_name"),
    ),
    _case(
        "progress-to-browser",
        BEHAVIOR,
        "How is live job progress delivered to the browser?",
        _sym("api/services/jobs.py", "JobService.stream"),
        _sym("api/streaming.py", "encode_sse_event"),
    ),
    _case(
        "rerank-validation",
        BEHAVIOR,
        "How does the reranker check that the model returned a valid ordering?",
        _sym("retrieval/reranking.py", "_validate_response_ids"),
    ),
    _case(
        "protected-files",
        BEHAVIOR,
        "Which files is the coding agent forbidden from editing?",
        _sym("tools/editing.py", "_reject_protected_path"),
    ),
    _case(
        "neighbor-expansion",
        BEHAVIOR,
        "How are neighboring chunks added around the search hits?",
        _sym("rag/assembly.py", "assemble_context"),
        _sym("rag/assembly.py", "_neighbor_keys"),
    ),
    _case(
        "coding-complete",
        BEHAVIOR,
        "What has to be true before a coding task counts as complete?",
        _sym("coding/completion.py", "_evaluate_completion"),
        _sym("coding/completion.py", "_verification_blockers"),
    ),
    _case(
        "agent-history-budget",
        BEHAVIOR,
        "How is the agent's step history kept under the prompt size limit?",
        _sym("agent/prompts.py", "_bounded_history"),
    ),
    _case(
        "path-escape",
        BEHAVIOR,
        "How do the file tools stop a path from escaping the repository?",
        _sym("tools/filesystem.py", "_resolve_workspace_path"),
        _sym("ingestion/repository.py", "resolve_repository_path"),
    ),
    _case(
        "progress-fields",
        BEHAVIOR,
        "Which trace fields are allowed to reach the public progress stream?",
        _sym("api/streaming.py", "safe_progress_event"),
    ),
    _case(
        "python-structure",
        BEHAVIOR,
        "How is a Python file split along its functions and classes?",
        _sym("ingestion/structural.py", "chunk_python_source"),
        _sym("ingestion/structural.py", "_module_units"),
    ),
    _case(
        "worker-outage",
        BEHAVIOR,
        "How does the worker loop back off when the database is down?",
        _sym("worker.py", "run"),
    ),
    _case(
        "stream-reconnect",
        BEHAVIOR,
        "How does the frontend reconnect when the job event stream drops?",
        _lines("frontend/lib/jobs.ts", 98, 128),
    ),
    _case(
        "git-hardening",
        BEHAVIOR,
        "How are git commands run so the repository's own config cannot execute code?",
        _sym("tools/git.py", "_git_environment"),
        _sym("tools/git.py", "_git_command"),
    ),
    _case(
        "citation-validation",
        BEHAVIOR,
        "How are the answer's citations checked against the sources the model was given?",
        _sym("rag/pipeline.py", "_map_answer"),
    ),
    _case(
        "symbol-fragments",
        IDENTIFIER,
        "What does select_evidence_backed_symbol_fragments do?",
        _sym("retrieval/symbols.py", "select_evidence_backed_symbol_fragments"),
    ),
    _case(
        "strong-weak-identifiers",
        IDENTIFIER,
        "How does extract_identifier_candidates decide whether a word is a strong or weak identifier?",
        _sym("retrieval/identifiers.py", "extract_identifier_candidates"),
        _sym("retrieval/identifiers.py", "_classify_simple"),
    ),
    _case(
        "worker-run-once",
        IDENTIFIER,
        "What does JobWorker.run_once do?",
        _sym("jobs/worker.py", "JobWorker.run_once"),
    ),
    _case(
        "bm25-scoring",
        IDENTIFIER,
        "How does BM25Index score a chunk against the query?",
        _sym("retrieval/bm25.py", "BM25Index.search"),
    ),
    _case(
        "embedding-response",
        IDENTIFIER,
        "What does OpenAIEmbeddingClient check in the provider response before returning vectors?",
        _sym("retrieval/embeddings.py", "_normalize_response"),
    ),
    _case(
        "rag-system-prompt",
        IDENTIFIER,
        "What rules does RAG_SYSTEM_PROMPT give the model?",
        _lines(_SRC + "rag/pipeline.py", 34, 47),
    ),
    _case(
        "execution-lock",
        IDENTIFIER,
        "How does PostgresRepositoryExecutionLock stop two jobs mutating one repository?",
        _sym("jobs/locks.py", "PostgresRepositoryExecutionLock.hold"),
    ),
    _case(
        "hybrid-symbol-sources",
        IDENTIFIER,
        "How does postgres_hybrid_symbol_search combine its candidate sources?",
        _sym("db/hybrid.py", "postgres_hybrid_symbol_search"),
    ),
    _case(
        "chunk-size",
        CONFIG,
        "What is the default chunk size in lines, and how much do chunks overlap?",
        _sym("ingestion/models.py", "ChunkingConfig"),
    ),
    _case(
        "lease-range",
        CONFIG,
        "Where is the allowed range for the job lease length enforced?",
        _lines(_SRC + "config.py", 40, 42),
    ),
    _case(
        "rag-strategies",
        CONFIG,
        "Which retrieval strategies can a question request use, and which one is the default?",
        _lines(_SRC + "api/models.py", 28, 28),
        _sym("api/models.py", "RAGRequest"),
    ),
    _case(
        "repository-limits",
        CONFIG,
        "How large can a repository be before indexing refuses it?",
        _lines(_SRC + "api/services/repositories.py", 115, 117),
    ),
    _case(
        "run-live-eval",
        CONFIG,
        "How do I run the live agent evaluation myself?",
        _lines("docs/evaluation.md", 122, 160),
    ),
    _case(
        "compose-services",
        CONFIG,
        "Which services does docker compose start?",
        _lines("docker-compose.yml", 1, 141),
    ),
)

SUITE = SpanBenchmarkSuite(version=VERSION, cases=CASES)

# Written after the retrieval changes were chosen on CASES and before any
# strategy was run on them, so they measure whether those changes generalize
# rather than whether they were tuned to the questions above.
HELD_OUT_CASES: tuple[SpanBenchmarkCase, ...] = (
    _case(
        "repeat-tool-call",
        BEHAVIOR,
        "How does the agent avoid running the same tool call twice on an unchanged workspace?",
        _sym("agent/loop.py", "_tool_call_identity"),
    ),
    _case(
        "non-utf8-source",
        BEHAVIOR,
        "What encoding is used when a source file is not valid UTF-8?",
        _sym("ingestion/repository.py", "_decode_source"),
    ),
    _case(
        "binary-files",
        BEHAVIOR,
        "How are binary files detected and rejected while loading sources?",
        _sym("ingestion/repository.py", "load_source_file"),
    ),
    _case(
        "nested-bindings",
        BEHAVIOR,
        "Why can't two registered repositories point at the same or a nested folder?",
        _sym("api/store.py", "workspace_paths_overlap"),
    ),
    _case(
        "token-estimate",
        BEHAVIOR,
        "How are token counts estimated without a tokenizer?",
        _sym("rag/tokens.py", "estimate_tokens"),
    ),
    _case(
        "review-diff-withheld",
        BEHAVIOR,
        "When is the reviewer not shown the diff of the agent's changes?",
        _sym("coding/reasoning.py", "bounded_review_diff"),
    ),
    _case(
        "hnsw-query",
        BEHAVIOR,
        "How does vector search use the HNSW index instead of sorting every row?",
        _sym("db/repositories.py", "_semantic_search_statement"),
    ),
    _case(
        "investigation-evidence",
        BEHAVIOR,
        "Which source locations count as evidence for an investigation answer?",
        _sym("agent/evidence.py", "extract_observed_source_evidence"),
    ),
    _case(
        "client-header",
        BEHAVIOR,
        "How does the web client mark its POST requests so the API accepts them?",
        _lines("frontend/lib/api.ts", 45, 60),
    ),
    _case(
        "sse-parsing",
        BEHAVIOR,
        "How are server-sent event frames parsed in the browser?",
        _lines("frontend/lib/sse.ts", 31, 55),
    ),
    _case(
        "load-stored-chunks",
        BEHAVIOR,
        "How are stored chunks loaded back from the database for keyword search?",
        _sym("db/repositories.py", "load_chunks"),
    ),
    _case(
        "trace-emit",
        IDENTIFIER,
        "What does TraceContext.emit record?",
        _sym("observability/recorder.py", "TraceContext.emit"),
    ),
    _case(
        "public-text",
        IDENTIFIER,
        "What does public_text redact from API responses?",
        _sym("api/privacy.py", "public_text"),
        _lines(_SRC + "api/privacy.py", 10, 18),
    ),
    _case(
        "rerank-candidates",
        IDENTIFIER,
        "How does _bounded_candidates limit what the reranker sees?",
        _sym("retrieval/reranking.py", "_bounded_candidates"),
    ),
    _case(
        "embedding-batch-size",
        CONFIG,
        "What is the default embedding batch size?",
        _sym("retrieval/models.py", "EmbeddingConfig"),
    ),
    _case(
        "bm25-parameters",
        CONFIG,
        "What are the default BM25 k1 and b parameters?",
        _sym("retrieval/models.py", "BM25Config"),
    ),
)

HELD_OUT_SUITE = SpanBenchmarkSuite(version=f"{VERSION}-held-out", cases=HELD_OUT_CASES)
