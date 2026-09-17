"""Run the deterministic offline ``repo-eval-v3`` context-assembly benchmark.

Milestone 22's ``repo-eval-v2`` isolates chunking and retrieval-ranking
quality. This benchmark isolates a separate question: given an *already
fixed* seed (retrieval ranking held constant so gains cannot be misread as
ranking improvements), does context assembly deliver more of the gold
evidence into the final, budget-bounded prompt context?
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from repomind.evaluation import (
    ContextAssemblyBenchmarkCase,
    ContextAssemblyBenchmarkSuite,
    RAGBenchmarkCase,
    RAGBenchmarkSuite,
    evaluate_context_assembly,
    evaluate_rag,
    format_context_assembly_comparison,
    format_rag_comparison,
)
from repomind.ingestion import (
    ChunkingConfig,
    ChunkingStrategy,
    CodeChunk,
    SourceFile,
    chunk_source_file,
)
from repomind.rag import ContextAssemblyConfig, ContextStrategy, RAGConfig
from repomind.rag.assembly import InMemoryNeighborLoader
from repomind.retrieval import RankedChunk, SemanticSearchResult, chunk_identity

VERSION = "repo-eval-v3"

_SOURCE = '''import secrets

MAX_LOGIN_ATTEMPTS = 5


def normalize_credentials(username: str) -> str:
    """Normalize user input before authentication."""
    return username.strip().casefold()


class UserService:
    def login(self, username: str, password: str) -> str:
        normalized = normalize_credentials(username)
        if not normalized or not password:
            raise ValueError("credentials are required")
        if not self._rate_limit_ok(username):
            raise PermissionError("too many attempts")
        token = secrets.token_urlsafe(16)
        self._audit_login(username, token)
        return token

    def _rate_limit_ok(self, username: str) -> bool:
        return True

    def _audit_login(self, username: str, token: str) -> None:
        print(f"login:{username}:{token}")

    def logout(self, token: str) -> None:
        if not token:
            raise ValueError("token is required")
        self._revoke(token)

    def _revoke(self, token: str) -> None:
        print(f"revoked:{token}")
'''


def _source() -> SourceFile:
    return SourceFile(
        relative_path=Path("src/auth/service.py"),
        language="python",
        content=_SOURCE,
        size_bytes=len(_SOURCE.encode()),
        line_count=len(_SOURCE.splitlines()),
    )


def _fixed_retriever(chunk: CodeChunk) -> Sequence[RankedChunk]:
    """A deterministic, rank-frozen "retriever" isolating assembly from ranking."""

    def retrieve(query: str, *, top_k: int) -> Sequence[RankedChunk]:
        return [SemanticSearchResult(chunk=chunk, score=1.0, rank=1)][:top_k]

    return retrieve


def _line_chunks() -> list[CodeChunk]:
    return chunk_source_file(
        _source(), ChunkingConfig(max_lines_per_chunk=4, overlap_lines=0)
    )


def _structural_fragments() -> list[CodeChunk]:
    chunks = chunk_source_file(
        _source(),
        ChunkingConfig(
            strategy=ChunkingStrategy.STRUCTURAL,
            max_lines_per_chunk=4,
            overlap_lines=0,
            max_chars_per_chunk=120,
        ),
    )
    return [chunk for chunk in chunks if chunk.qualified_symbol_name == "UserService.login"]


def _context_suite() -> tuple[ContextAssemblyBenchmarkSuite, dict[str, object]]:
    line_chunks = _line_chunks()
    signature_chunk = next(c for c in line_chunks if "def login" in c.content)
    validation_chunk = next(c for c in line_chunks if "credentials are required" in c.content)

    login_fragments = _structural_fragments()
    assert len(login_fragments) >= 3, "fixture must split UserService.login into fragments"
    middle_fragment = next(f for f in login_fragments if f.fragment_index == 2)

    normalize_chunk = next(c for c in line_chunks if "def normalize_credentials" in c.content)

    suite = ContextAssemblyBenchmarkSuite(
        version=VERSION,
        cases=(
            ContextAssemblyBenchmarkCase(
                id="line-v1-signature-needs-neighbor-body",
                query="What does login validate before issuing a token?",
                relevant_chunks=(
                    chunk_identity(signature_chunk),
                    chunk_identity(validation_chunk),
                ),
            ),
            ContextAssemblyBenchmarkCase(
                id="structural-oversized-method-fragment",
                query="UserService.login",
                relevant_chunks=tuple(chunk_identity(f) for f in login_fragments),
            ),
            ContextAssemblyBenchmarkCase(
                id="unrelated-neighbor-should-not-be-required",
                query="How are usernames normalized?",
                relevant_chunks=(chunk_identity(normalize_chunk),),
            ),
        ),
    )
    fixtures = {
        "line_chunks": line_chunks,
        "signature_chunk": signature_chunk,
        "login_fragments": login_fragments,
        "middle_fragment": middle_fragment,
        "normalize_chunk": normalize_chunk,
    }
    return suite, fixtures


def _run_context_comparison() -> None:
    suite, fixtures = _context_suite()
    line_chunks: list[CodeChunk] = fixtures["line_chunks"]
    login_fragments: list[CodeChunk] = fixtures["login_fragments"]
    middle_fragment: CodeChunk = fixtures["middle_fragment"]
    normalize_chunk: CodeChunk = fixtures["normalize_chunk"]
    signature_chunk: CodeChunk = fixtures["signature_chunk"]

    retrievers = {
        suite.cases[0].id: _fixed_retriever(signature_chunk),
        suite.cases[1].id: _fixed_retriever(middle_fragment),
        suite.cases[2].id: _fixed_retriever(normalize_chunk),
    }
    loaders = {
        suite.cases[0].id: InMemoryNeighborLoader(line_chunks),
        suite.cases[1].id: InMemoryNeighborLoader(login_fragments),
        suite.cases[2].id: InMemoryNeighborLoader(line_chunks),
    }

    def evaluate(strategy: ContextStrategy) -> list:
        reports = []
        for case in suite.cases:
            one_case_suite = ContextAssemblyBenchmarkSuite(version=suite.version, cases=(case,))
            reports.append(
                evaluate_context_assembly(
                    one_case_suite,
                    retrievers[case.id],
                    top_k=1,
                    config=ContextAssemblyConfig(strategy=strategy, neighbor_radius=1),
                    neighbor_loader=loaders[case.id],
                )
            )
        return reports

    seeds_only_reports = evaluate(ContextStrategy.SEEDS_ONLY)
    expanded_reports = evaluate(ContextStrategy.EXPANDED)

    print(f"{VERSION.upper()} CONTEXT ASSEMBLY COMPARISON")
    print("(retrieval ranking is held fixed across strategies; only assembly differs)\n")
    for seeds_report, expanded_report, case in zip(
        seeds_only_reports, expanded_reports, suite.cases, strict=True
    ):
        print(f"Case: {case.id}")
        print(format_context_assembly_comparison([seeds_report, expanded_report]))
        print()


class _MarkerLLM:
    """Deterministic scripted answer: cite S-ids whose block contains a marker.

    This is not a live model call. It inspects only the already-built,
    untrusted context text for a fixed marker string, exactly like a
    reproducible oracle in an offline evaluation fixture.
    """

    def __init__(self, marker: str, fact: str) -> None:
        self.marker = marker
        self.fact = fact

    def generate_structured(self, prompt, response_model, *, system_prompt=None, temperature=None):
        source_ids = []
        for index in range(1, 10):
            open_tag = f'<source id="S{index}">'
            start = prompt.find(open_tag)
            if start == -1:
                continue
            end = prompt.find("</source>", start)
            block = prompt[start:end if end != -1 else None]
            if self.marker in block:
                source_ids.append(f"S{index}")
        found = bool(source_ids)
        return response_model(
            answer=(self.fact if found else "insufficient evidence in supplied context"),
            source_ids=source_ids[:1] if found else [],
            insufficient_evidence=not found,
        )


def _run_rag_comparison() -> None:
    _, fixtures = _context_suite()
    signature_chunk: CodeChunk = fixtures["signature_chunk"]
    line_chunks: list[CodeChunk] = fixtures["line_chunks"]
    validation_chunk: CodeChunk = next(
        c for c in line_chunks if "credentials are required" in c.content
    )

    fact = "login requires both a normalized username and a password"
    suite = RAGBenchmarkSuite(
        version=VERSION,
        cases=(
            RAGBenchmarkCase(
                id="needs-validation-neighbor",
                question="What does login validate before issuing a token?",
                relevant_chunks=(chunk_identity(validation_chunk),),
                expected_answer_facts=(fact,),
            ),
        ),
    )
    retriever = _fixed_retriever(signature_chunk)
    loader = InMemoryNeighborLoader(line_chunks)
    llm = _MarkerLLM("credentials are required", fact)

    seeds_only = evaluate_rag(
        suite,
        "seeds_only",
        retriever,
        llm,
        config=RAGConfig(top_k=1, context_strategy=ContextStrategy.SEEDS_ONLY),
    )
    expanded = evaluate_rag(
        suite,
        "expanded",
        retriever,
        llm,
        config=RAGConfig(top_k=1, context_strategy=ContextStrategy.EXPANDED, neighbor_radius=1),
        neighbor_loader=loader,
    )

    print(f"{VERSION.upper()} RAG COMPARISON (seeds-only vs assembled context)")
    print(format_rag_comparison([seeds_only, expanded]))
    print()
    print(
        "Retrieval ranking is identical between rows (same fixed seed); any "
        "answer-pass-rate difference here is attributable to context assembly, "
        "not to retrieval ranking."
    )


def main() -> None:
    """Print offline, deterministic context-assembly evidence. No live model calls."""

    _run_context_comparison()
    _run_rag_comparison()


if __name__ == "__main__":
    main()
