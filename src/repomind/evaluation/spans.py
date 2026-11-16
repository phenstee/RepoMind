"""Chunking-independent retrieval evaluation against gold source spans.

The older fixture benchmarks label relevance as exact chunk identities, which
ties every gold label to one chunking configuration. Real questions are about
code, not chunks: "how is a job claimed?" is answered by a function, wherever
the chunker happened to cut. This module labels gold evidence as source spans
(a Python symbol resolved through the AST, or an explicit line range) and
credits any retrieved chunk that actually contains enough of that span. The
same suite can therefore compare line chunking against structural chunking,
and ranking changes against context-packing changes, on equal terms.
"""

import ast
from collections.abc import Mapping, Sequence
from typing import Protocol

from pydantic import BaseModel, Field, field_validator, model_validator

from repomind.evaluation.metrics import mean_reciprocal_rank
from repomind.ingestion import CodeChunk, RepositorySnapshot

# A retrieved chunk must contain at least this many lines of a gold span (or
# the whole span, when it is shorter) to count as evidence for it. One shared
# boundary line, such as a lone ``def`` header at the end of a line window,
# is not evidence.
MIN_COVERED_LINES = 5


class SpanEvaluationError(ValueError):
    """Raised when a span suite cannot be resolved against its corpus."""


class SpanTarget(BaseModel):
    """One piece of gold evidence: a Python symbol or an explicit line range."""

    path: str = Field(min_length=1)
    symbol: str | None = Field(default=None, min_length=1)
    lines: tuple[int, int] | None = None

    @model_validator(mode="after")
    def _exactly_one_locator(self) -> "SpanTarget":
        if (self.symbol is None) == (self.lines is None):
            raise ValueError("a span target needs exactly one of symbol or lines")
        if self.lines is not None and not 1 <= self.lines[0] <= self.lines[1]:
            raise ValueError("lines must be a 1-based inclusive (start, end) range")
        return self

    def label(self) -> str:
        if self.symbol is not None:
            return f"{self.path}::{self.symbol}"
        assert self.lines is not None
        return f"{self.path}#L{self.lines[0]}-L{self.lines[1]}"


class SpanBenchmarkCase(BaseModel):
    """A natural question and every span that answers it."""

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    category: str = Field(min_length=1)
    targets: tuple[SpanTarget, ...] = Field(min_length=1)


class SpanBenchmarkSuite(BaseModel):
    version: str = Field(min_length=1)
    cases: tuple[SpanBenchmarkCase, ...] = Field(min_length=1)

    @field_validator("cases")
    @classmethod
    def _unique_ids(cls, cases: tuple[SpanBenchmarkCase, ...]) -> tuple[SpanBenchmarkCase, ...]:
        ids = [case.id for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case IDs must be unique")
        return cases


class GoldSpan(BaseModel):
    relative_path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    label: str

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


def _find_definition(
    nodes: Sequence[ast.stmt], parts: Sequence[str]
) -> ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | None:
    for node in nodes:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == parts[0]
        ):
            if len(parts) == 1:
                return node
            return _find_definition(node.body, parts[1:])
    return None


def resolve_target(target: SpanTarget, sources: Mapping[str, str]) -> GoldSpan:
    """Resolve one target to concrete lines, failing loudly when it has drifted."""

    content = sources.get(target.path)
    if content is None:
        raise SpanEvaluationError(f"gold path is not in the corpus: {target.path}")
    line_total = len(content.splitlines())
    if target.lines is not None:
        start, end = target.lines
        if end > line_total:
            raise SpanEvaluationError(f"gold lines exceed {target.path} ({line_total} lines)")
        return GoldSpan(
            relative_path=target.path, start_line=start, end_line=end, label=target.label()
        )

    assert target.symbol is not None
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        raise SpanEvaluationError(f"cannot parse gold file {target.path}") from exc
    node = _find_definition(tree.body, target.symbol.split("."))
    if node is None or node.end_lineno is None:
        raise SpanEvaluationError(f"gold symbol not found: {target.label()}")
    start = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
    return GoldSpan(
        relative_path=target.path,
        start_line=start,
        end_line=node.end_lineno,
        label=target.label(),
    )


def resolve_suite(
    suite: SpanBenchmarkSuite, snapshot: RepositorySnapshot
) -> dict[str, tuple[GoldSpan, ...]]:
    """Resolve every case's targets against one corpus snapshot."""

    sources = {source.relative_path.as_posix(): source.content for source in snapshot.files}
    return {
        case.id: tuple(resolve_target(target, sources) for target in case.targets)
        for case in suite.cases
    }


def covered_lines(chunk: CodeChunk, span: GoldSpan) -> int:
    if chunk.relative_path.as_posix() != span.relative_path:
        return 0
    return max(0, min(chunk.end_line, span.end_line) - max(chunk.start_line, span.start_line) + 1)


def chunk_covers(chunk: CodeChunk, span: GoldSpan) -> bool:
    """Whether ``chunk`` holds enough of ``span`` to count as evidence for it."""

    return covered_lines(chunk, span) >= min(span.line_count, MIN_COVERED_LINES)


def covered_spans(chunks: Sequence[CodeChunk], spans: Sequence[GoldSpan]) -> tuple[GoldSpan, ...]:
    return tuple(span for span in spans if any(chunk_covers(chunk, span) for chunk in chunks))


class RankedRetriever(Protocol):
    def __call__(self, query: str, *, top_k: int) -> Sequence[object]:
        """Return objects with a ``chunk`` attribute, best first."""


class SpanCaseResult(BaseModel):
    case_id: str
    category: str
    first_hit_rank: int | None
    targets_hit: int
    target_count: int
    retrieved: tuple[str, ...]

    @property
    def hit(self) -> bool:
        return self.first_hit_rank is not None

    @property
    def reciprocal_rank(self) -> float:
        return 0.0 if self.first_hit_rank is None else 1.0 / self.first_hit_rank

    @property
    def target_recall(self) -> float:
        return self.targets_hit / self.target_count


class SpanRetrievalReport(BaseModel):
    benchmark_version: str
    strategy: str
    k: int
    case_results: tuple[SpanCaseResult, ...]

    @property
    def hit_rate(self) -> float:
        return sum(result.hit for result in self.case_results) / len(self.case_results)

    @property
    def mean_target_recall(self) -> float:
        return sum(result.target_recall for result in self.case_results) / len(self.case_results)

    @property
    def mrr(self) -> float:
        return mean_reciprocal_rank([result.reciprocal_rank for result in self.case_results])

    def by_category(self) -> dict[str, "SpanRetrievalReport"]:
        grouped: dict[str, list[SpanCaseResult]] = {}
        for result in self.case_results:
            grouped.setdefault(result.category, []).append(result)
        return {
            category: self.model_copy(update={"case_results": tuple(results)})
            for category, results in sorted(grouped.items())
        }


def _chunk_label(chunk: CodeChunk) -> str:
    return f"{chunk.relative_path.as_posix()}:{chunk.start_line}-{chunk.end_line}"


def evaluate_span_retrieval(
    suite: SpanBenchmarkSuite,
    gold: Mapping[str, Sequence[GoldSpan]],
    strategy: str,
    retriever: RankedRetriever,
    *,
    k: int,
) -> SpanRetrievalReport:
    """Rank every case and score hit@k, target recall@k, and MRR@k."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise SpanEvaluationError("k must be a positive integer")
    results: list[SpanCaseResult] = []
    for case in suite.cases:
        spans = tuple(gold[case.id])
        chunks: list[CodeChunk] = [result.chunk for result in retriever(case.query, top_k=k)][:k]
        first_hit = next(
            (
                rank
                for rank, chunk in enumerate(chunks, start=1)
                if any(chunk_covers(chunk, span) for span in spans)
            ),
            None,
        )
        results.append(
            SpanCaseResult(
                case_id=case.id,
                category=case.category,
                first_hit_rank=first_hit,
                targets_hit=len(covered_spans(chunks, spans)),
                target_count=len(spans),
                retrieved=tuple(_chunk_label(chunk) for chunk in chunks),
            )
        )
    return SpanRetrievalReport(
        benchmark_version=suite.version,
        strategy=strategy,
        k=k,
        case_results=tuple(results),
    )
