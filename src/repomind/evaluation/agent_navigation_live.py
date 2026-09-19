"""Reproducibility metadata, safety gating, and JSON schema for live-model runs.

This module never talks to OpenAI itself. It supplies the pieces a live-model
CLI harness needs around the actual model call:

* :func:`require_live_authorization` - a hard gate that must pass, with no
  network access attempted, before an OpenAI client is even constructed.
* :func:`git_commit_sha` - best-effort provenance for reproducibility.
* :func:`build_live_navigation_harness_result` - assembles the deterministic,
  secret-free JSON result schema from the same
  :class:`~repomind.evaluation.models.AgentNavigationEvaluationReport` that
  :func:`~repomind.evaluation.agent_navigation.evaluate_agent_navigation`
  already returns for scripted runs, plus optional trace-derived call/token/
  duration metrics.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from repomind.config import Settings
from repomind.evaluation.agent_navigation import DEFAULT_LIVE_MAX_ITERATIONS
from repomind.evaluation.models import AgentNavigationEvaluationReport, EvaluationMode
from repomind.observability import RunTrace

__all__ = [
    "DEFAULT_LIVE_MAX_ITERATIONS",
    "HARNESS_SCHEMA_VERSION",
    "LiveAgentNavigationHarnessResult",
    "LiveAgentNavigationRun",
    "LiveAgentNavigationRunMetrics",
    "LiveEvaluationAuthorizationError",
    "build_live_navigation_harness_result",
    "git_commit_sha",
    "require_live_authorization",
]

HARNESS_SCHEMA_VERSION = "agent-live-eval-v1"


class LiveEvaluationAuthorizationError(RuntimeError):
    """Raised when a live model run is requested without explicit authorization."""


def require_live_authorization(settings: Settings, *, confirm_live: bool) -> None:
    """Refuse to proceed unless both an API key and explicit confirmation exist.

    Callers must call this - and must not construct an OpenAI client or
    import ``repomind.llm.client`` - before this returns without raising.
    Checking both conditions together (rather than short-circuiting) keeps
    the error message complete on the first failure. A whitespace-only key
    counts as missing: it can never authenticate, so treating it as
    configured would only defer the failure past the safety gate.
    """

    secret = settings.openai_api_key
    has_key = secret is not None and bool(secret.get_secret_value().strip())

    missing: list[str] = []
    if not has_key:
        missing.append("OPENAI_API_KEY is not configured")
    if not confirm_live:
        missing.append("the --confirm-live flag was not passed")
    if missing:
        raise LiveEvaluationAuthorizationError(
            "Refusing to make a live OpenAI request: " + "; ".join(missing) + "."
        )


def git_commit_sha(repository_root: Path) -> str | None:
    """Best-effort current commit SHA; ``None`` when git is unavailable."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    sha = completed.stdout.strip()
    return sha or None


class LiveAgentNavigationRunMetrics(BaseModel):
    """Best-effort call/token/duration counters for one retrieval-mode run.

    Sourced from the shared :class:`~repomind.observability.recorder.TraceContext`
    that :func:`~repomind.evaluation.agent_navigation.evaluate_agent_navigation`
    already maintains. A retrieval-mode run may be backed by more than one
    underlying evaluation call/trace - indexed mode evaluates each case
    separately so it can receive its own case-scoped retriever - so
    ``llm_calls``, ``tool_calls`` and ``duration_ms`` are summed across
    every contributing trace.

    Token fields are deliberately all-or-nothing: they are populated only
    when every LLM call in every contributing trace reported usage
    (``usage_reported_calls == llm_calls`` for each trace). If any call is
    unaccounted for - or no LLM call is represented at all - all three stay
    ``None`` rather than publishing a partial sum that would read as a
    complete cost figure in the committed artifact.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    llm_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    duration_ms: float | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class LiveAgentNavigationRun(BaseModel):
    """One retrieval mode's live evaluation report plus its run metrics.

    ``indexed_search_queries_by_case`` records, per case id, the exact
    ``indexed_code_search`` query strings the model chose to send (in the
    order it sent them) - not the benchmark's canonical scripted query, and
    never any evaluator-only oracle data. A case that never called
    ``indexed_code_search`` (including filesystem-mode cases, which never
    have that tool available) maps to an empty tuple.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    report: AgentNavigationEvaluationReport
    metrics: LiveAgentNavigationRunMetrics
    indexed_search_queries_by_case: dict[str, tuple[str, ...]] = Field(default_factory=dict)


class LiveAgentNavigationHarnessResult(BaseModel):
    """Deterministic, secret-free JSON result schema for one harness invocation.

    Contains no API keys, settings, prompts, or evaluator-only oracle text -
    only benchmark identity, run provenance, and the same graded case
    evidence the offline benchmark already produces.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    harness_schema_version: str = HARNESS_SCHEMA_VERSION
    benchmark_version: str
    model: str
    git_commit_sha: str | None = None
    generated_at: datetime
    max_iterations: int = Field(gt=0)
    case_ids: tuple[str, ...]
    runs: tuple[LiveAgentNavigationRun, ...] = Field(min_length=1)


def _sum_run_metrics(traces: Sequence[RunTrace]) -> LiveAgentNavigationRunMetrics:
    """Sum llm/tool calls, duration, and token usage across ALL of a run's traces.

    A retrieval-mode run can be backed by several underlying
    ``evaluate_agent_navigation`` calls (one per case, in live indexed mode),
    each with its own trace - never just the first or the last one. Callers
    must pass every contributing trace; missing ones are rejected upstream
    rather than dropped here, because a dropped trace silently under-reports
    calls, duration and tokens.
    """

    llm_calls = sum(trace.llm_calls for trace in traces)
    durations = [trace.duration_ms for trace in traces if trace.duration_ms is not None]

    # All-or-nothing token coverage: every LLM call in every trace must have
    # reported usage, otherwise a partial sum would masquerade as the run's
    # complete token cost. Zero LLM calls means nothing was measured at all,
    # which is reported as None rather than a misleading zero.
    usage_complete = llm_calls > 0 and all(
        trace.usage_reported_calls == trace.llm_calls for trace in traces
    )
    usages = [trace.token_usage for trace in traces if trace.token_usage is not None]
    return LiveAgentNavigationRunMetrics(
        llm_calls=llm_calls,
        tool_calls=sum(trace.tool_calls for trace in traces),
        duration_ms=sum(durations) if durations else None,
        prompt_tokens=sum(u.prompt_tokens for u in usages) if usage_complete else None,
        completion_tokens=sum(u.completion_tokens for u in usages) if usage_complete else None,
        total_tokens=sum(u.total_tokens for u in usages) if usage_complete else None,
    )


def build_live_navigation_harness_result(
    *,
    benchmark_version: str,
    model: str,
    git_commit_sha: str | None,
    generated_at: datetime,
    max_iterations: int,
    runs: Sequence[
        tuple[
            AgentNavigationEvaluationReport,
            Sequence[RunTrace | None],
            Mapping[str, Sequence[str]],
        ]
    ],
) -> LiveAgentNavigationHarnessResult:
    """Assemble the JSON-serializable harness result from graded live reports.

    Each entry in ``runs`` is one retrieval mode's
    (:class:`~repomind.evaluation.models.AgentNavigationEvaluationReport`,
    traces, indexed-search queries) triple:

    * ``report`` (``mode=EvaluationMode.LIVE_MODEL``, matching
      ``benchmark_version``) - a single report covering every case in that
      retrieval mode. When cases were evaluated through separate calls
      (indexed mode), callers merge them first with
      :func:`~repomind.evaluation.agent_navigation.merge_agent_navigation_reports`.
      At most one run per retrieval mode is accepted.
    * ``traces`` - every ``RunTrace`` an ``InMemoryTraceRecorder`` captured
      for a call contributing to ``report`` (one per case for a merged
      indexed-mode report; one total for a single-call filesystem-mode
      report). All are summed - never just the first or last - and the set
      of supplied ``run_id`` values must match the distinct
      ``trace_run_id`` values on the report's case results exactly, so an
      omitted, unrelated or duplicated trace is rejected rather than
      silently skewing the metrics.
    * a mapping of case id to the ``indexed_code_search`` query strings the
      model actually issued for that case, in order. Cases that never
      called it (including all filesystem-mode cases) may be omitted and
      normalize to an empty tuple, but keys naming cases outside the report
      are rejected.
    """

    if not runs:
        raise ValueError("at least one live navigation run is required")

    run_models: list[LiveAgentNavigationRun] = []
    case_ids: set[str] = set()
    seen_retrieval_modes: set[str] = set()
    for report, traces, queries_by_case in runs:
        if report.mode is not EvaluationMode.LIVE_MODEL:
            raise ValueError("live navigation harness results require mode='live_model' reports")
        if report.benchmark_version != benchmark_version:
            raise ValueError(
                f"run report benchmark_version {report.benchmark_version!r} does not match the "
                f"harness benchmark_version {benchmark_version!r}"
            )
        if report.retrieval_mode in seen_retrieval_modes:
            raise ValueError(
                f"duplicate run for retrieval_mode={report.retrieval_mode!r}; merge a mode's "
                "per-case reports into a single aggregate run instead"
            )
        seen_retrieval_modes.add(report.retrieval_mode)

        supplied_traces = list(traces)
        present_traces = [trace for trace in supplied_traces if trace is not None]
        if not supplied_traces:
            raise ValueError(
                f"live navigation run for retrieval_mode={report.retrieval_mode!r} supplied no "
                "traces; every contributing evaluate_agent_navigation call must provide one"
            )
        if len(present_traces) != len(supplied_traces):
            raise ValueError(
                f"live navigation run for retrieval_mode={report.retrieval_mode!r} is missing "
                f"{len(supplied_traces) - len(present_traces)} of {len(supplied_traces)} "
                "contributing traces; metrics would under-report calls, duration and tokens"
            )

        # Provenance: the supplied traces must be exactly the traces the report
        # says produced it. Counting alone is not enough - an omitted trace, an
        # unrelated one, or the same trace passed twice would each silently
        # skew the published metrics.
        unattributed = [
            result.case_id for result in report.case_results if result.trace_run_id is None
        ]
        if unattributed:
            raise ValueError(
                f"live navigation run for retrieval_mode={report.retrieval_mode!r} has case "
                f"results with no trace_run_id: {', '.join(sorted(unattributed))}"
            )
        supplied_ids = [trace.run_id for trace in present_traces]
        if len(set(supplied_ids)) != len(supplied_ids):
            raise ValueError(
                f"live navigation run for retrieval_mode={report.retrieval_mode!r} supplied the "
                "same trace more than once; metrics would double-count it"
            )
        expected_ids = {result.trace_run_id for result in report.case_results}
        missing_ids = sorted(str(run_id) for run_id in expected_ids - set(supplied_ids))
        unexpected_ids = sorted(str(run_id) for run_id in set(supplied_ids) - expected_ids)
        if missing_ids or unexpected_ids:
            raise ValueError(
                f"live navigation run for retrieval_mode={report.retrieval_mode!r} trace set does "
                f"not match its report: missing {missing_ids or 'none'}, "
                f"unexpected {unexpected_ids or 'none'}"
            )

        report_case_ids = [result.case_id for result in report.case_results]
        unknown_cases = sorted(set(queries_by_case) - set(report_case_ids))
        if unknown_cases:
            raise ValueError(
                f"indexed query map for retrieval_mode={report.retrieval_mode!r} names cases that "
                f"are not in its report: {', '.join(unknown_cases)}"
            )
        case_ids.update(report_case_ids)
        metrics = _sum_run_metrics(present_traces)
        normalized_queries = {
            case_id: tuple(queries_by_case.get(case_id, ())) for case_id in report_case_ids
        }
        run_models.append(
            LiveAgentNavigationRun(
                report=report,
                metrics=metrics,
                indexed_search_queries_by_case=normalized_queries,
            )
        )

    return LiveAgentNavigationHarnessResult(
        benchmark_version=benchmark_version,
        model=model,
        git_commit_sha=git_commit_sha,
        generated_at=generated_at,
        max_iterations=max_iterations,
        case_ids=tuple(sorted(case_ids)),
        runs=tuple(run_models),
    )
