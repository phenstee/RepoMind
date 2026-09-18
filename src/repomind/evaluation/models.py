"""Validated benchmark definitions and deterministic evaluation reports."""

from enum import StrEnum
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from repomind.agent import AgentDecision, AgentRunStatus
from repomind.coding import CodingTask, CodingTaskStatus, VerificationPolicy
from repomind.ingestion import validate_repository_relative_path
from repomind.rag import ContextStrategy
from repomind.retrieval import ChunkIdentity

DEFAULT_BENCHMARK_VERSION = "repo-eval-v1"


class EvaluationMode(StrEnum):
    """Provenance label that prevents fixture and live scores from mixing."""

    OFFLINE_FIXTURE = "offline_fixture"
    OFFLINE_SCRIPTED = "offline_scripted"
    LIVE_MODEL = "live_model"


class _NamedCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, max_length=200)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("benchmark case ID must not be blank")
        return value


class RetrievalBenchmarkCase(_NamedCase):
    """One query and its evaluator-only binary-relevance chunk labels."""

    query: str = Field(min_length=1, max_length=10_000)
    relevant_chunks: tuple[ChunkIdentity, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_case(self) -> "RetrievalBenchmarkCase":
        if not self.query.strip():
            raise ValueError("retrieval benchmark query must not be blank")
        if len(self.relevant_chunks) != len(set(self.relevant_chunks)):
            raise ValueError("relevant chunk identities must be unique")
        return self


class RetrievalBenchmarkSuite(BaseModel):
    """A small versioned retrieval dataset with stable case IDs."""

    model_config = ConfigDict(frozen=True)

    version: str = Field(default=DEFAULT_BENCHMARK_VERSION, min_length=1, max_length=200)
    cases: tuple[RetrievalBenchmarkCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_suite(self) -> "RetrievalBenchmarkSuite":
        if not self.version.strip():
            raise ValueError("benchmark version must not be blank")
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("retrieval benchmark case IDs must be unique")
        return self


class RetrievalCaseResult(BaseModel):
    """Per-query ranking evidence and binary-relevance metrics."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    trace_run_id: UUID | None = None
    strategy: str
    retrieved_chunk_ids: tuple[ChunkIdentity, ...]
    relevant_chunk_ids: tuple[ChunkIdentity, ...]
    recall_at_k: float = Field(ge=0, le=1)
    reciprocal_rank: float = Field(ge=0, le=1)
    ndcg_at_k: float = Field(ge=0, le=1)
    first_relevant_rank: int | None = Field(default=None, ge=1)


class RetrievalEvaluationReport(BaseModel):
    """Aggregate retrieval metrics with all case-level evidence retained."""

    model_config = ConfigDict(frozen=True)

    benchmark_version: str
    mode: EvaluationMode
    strategy: str
    k: int = Field(gt=0, strict=True)
    case_results: tuple[RetrievalCaseResult, ...] = Field(min_length=1)
    case_count: int = Field(gt=0)
    mean_recall_at_k: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    mean_ndcg_at_k: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _validate_report(self) -> "RetrievalEvaluationReport":
        if self.case_count != len(self.case_results):
            raise ValueError("retrieval case count must match case results")
        if any(result.strategy != self.strategy for result in self.case_results):
            raise ValueError("retrieval case strategies must match the report strategy")
        return self


class RAGBenchmarkCase(_NamedCase):
    """One grounded-answer question with transparent deterministic gold data."""

    question: str = Field(min_length=1, max_length=10_000)
    relevant_chunks: tuple[ChunkIdentity, ...] = Field(min_length=1)
    expected_answer_facts: tuple[str, ...] = Field(min_length=1, max_length=50)
    expected_insufficient_evidence: bool = False

    @model_validator(mode="after")
    def _validate_case(self) -> "RAGBenchmarkCase":
        if not self.question.strip():
            raise ValueError("RAG benchmark question must not be blank")
        if len(self.relevant_chunks) != len(set(self.relevant_chunks)):
            raise ValueError("relevant chunk identities must be unique")
        if any(not fact.strip() for fact in self.expected_answer_facts):
            raise ValueError("expected answer facts must not be blank")
        return self


class RAGBenchmarkSuite(BaseModel):
    """A versioned set of repository questions evaluated across configurations."""

    model_config = ConfigDict(frozen=True)

    version: str = Field(default=DEFAULT_BENCHMARK_VERSION, min_length=1, max_length=200)
    cases: tuple[RAGBenchmarkCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_suite(self) -> "RAGBenchmarkSuite":
        if not self.version.strip():
            raise ValueError("benchmark version must not be blank")
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("RAG benchmark case IDs must be unique")
        return self


class RAGCaseResult(BaseModel):
    """Separate retrieval, context, citation, and answer-oracle evidence."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    trace_run_id: UUID | None = None
    strategy: str
    answer: str
    insufficient_evidence: bool
    retrieved_chunk_ids: tuple[ChunkIdentity, ...]
    context_chunk_ids: tuple[ChunkIdentity, ...]
    cited_chunk_ids: tuple[ChunkIdentity, ...]
    relevant_chunk_ids: tuple[ChunkIdentity, ...]
    retrieval_recall: float = Field(ge=0, le=1)
    context_recall: float = Field(ge=0, le=1)
    citation_recall: float = Field(ge=0, le=1)
    required_facts_passed: bool
    missing_required_facts: tuple[str, ...]
    answer_passed: bool


class RAGEvaluationReport(BaseModel):
    """Aggregate RAG results without collapsing pipeline stages into one score."""

    model_config = ConfigDict(frozen=True)

    benchmark_version: str
    mode: EvaluationMode
    strategy: str
    case_results: tuple[RAGCaseResult, ...] = Field(min_length=1)
    case_count: int = Field(gt=0)
    mean_retrieval_recall: float = Field(ge=0, le=1)
    mean_context_recall: float = Field(ge=0, le=1)
    mean_citation_recall: float = Field(ge=0, le=1)
    answer_pass_rate: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _validate_report(self) -> "RAGEvaluationReport":
        if self.case_count != len(self.case_results):
            raise ValueError("RAG case count must match case results")
        if any(result.strategy != self.strategy for result in self.case_results):
            raise ValueError("RAG case strategies must match the report strategy")
        return self


class ContextAssemblyBenchmarkCase(_NamedCase):
    """One query and the gold chunk identities expected in packed context."""

    query: str = Field(min_length=1, max_length=10_000)
    relevant_chunks: tuple[ChunkIdentity, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_case(self) -> "ContextAssemblyBenchmarkCase":
        if not self.query.strip():
            raise ValueError("context assembly benchmark query must not be blank")
        if len(self.relevant_chunks) != len(set(self.relevant_chunks)):
            raise ValueError("relevant chunk identities must be unique")
        return self


class ContextAssemblyBenchmarkSuite(BaseModel):
    """A small versioned dataset for comparing context-assembly strategies."""

    model_config = ConfigDict(frozen=True)

    version: str = Field(default=DEFAULT_BENCHMARK_VERSION, min_length=1, max_length=200)
    cases: tuple[ContextAssemblyBenchmarkCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_suite(self) -> "ContextAssemblyBenchmarkSuite":
        if not self.version.strip():
            raise ValueError("benchmark version must not be blank")
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("context assembly benchmark case IDs must be unique")
        return self


class ContextAssemblyCaseResult(BaseModel):
    """Per-query context-assembly evidence, kept separate from retrieval ranking."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    mode: EvaluationMode
    strategy: ContextStrategy
    packed_chunk_ids: tuple[ChunkIdentity, ...]
    relevant_chunk_ids: tuple[ChunkIdentity, ...]
    seed_count: int = Field(ge=0)
    expanded_candidate_count: int = Field(ge=0)
    deduplicated_count: int = Field(ge=0)
    dropped_for_budget_count: int = Field(ge=0)
    packed_count: int = Field(ge=0)
    estimated_tokens: int = Field(ge=0)
    budget_tokens: int = Field(gt=0)
    gold_evidence_coverage: float = Field(ge=0, le=1)
    context_precision: float = Field(ge=0, le=1)
    budget_utilization: float = Field(ge=0)


class ContextAssemblyEvaluationReport(BaseModel):
    """Aggregate context-assembly metrics, separate from retrieval-ranking metrics."""

    model_config = ConfigDict(frozen=True)

    benchmark_version: str
    mode: EvaluationMode
    strategy: ContextStrategy
    case_results: tuple[ContextAssemblyCaseResult, ...] = Field(min_length=1)
    case_count: int = Field(gt=0)
    mean_gold_evidence_coverage: float = Field(ge=0, le=1)
    mean_context_precision: float = Field(ge=0, le=1)
    mean_budget_utilization: float = Field(ge=0)
    mean_packed_count: float = Field(ge=0)
    mean_deduplicated_count: float = Field(ge=0)

    @model_validator(mode="after")
    def _validate_report(self) -> "ContextAssemblyEvaluationReport":
        if self.case_count != len(self.case_results):
            raise ValueError("context assembly case count must match case results")
        if any(result.strategy != self.strategy for result in self.case_results):
            raise ValueError("context assembly case strategies must match the report strategy")
        return self


def _oracle_path(value: Path) -> Path:
    return validate_repository_relative_path(value)


class FileTextExpectation(BaseModel):
    """A fixed substring expectation for one repository-relative file."""

    model_config = ConfigDict(frozen=True)

    path: Path
    text: str = Field(min_length=1, max_length=100_000)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: Path) -> Path:
        return _oracle_path(value)


class CodingBenchmarkCase(_NamedCase):
    """One isolated coding task plus evaluator-only hidden file-state oracles."""

    fixture_repository: Path
    task: CodingTask
    verification_policy: VerificationPolicy = Field(default_factory=VerificationPolicy)
    required_changed_paths: tuple[Path, ...] = ()
    allowed_changed_paths: tuple[Path, ...] | None = None
    file_exists: tuple[Path, ...] = ()
    file_not_exists: tuple[Path, ...] = ()
    file_contains: tuple[FileTextExpectation, ...] = ()
    file_not_contains: tuple[FileTextExpectation, ...] = ()
    expects_recovery: bool = False

    @field_validator(
        "required_changed_paths",
        "allowed_changed_paths",
        "file_exists",
        "file_not_exists",
    )
    @classmethod
    def _validate_paths(cls, values: tuple[Path, ...] | None) -> tuple[Path, ...] | None:
        if values is None:
            return None
        normalized = tuple(_oracle_path(value) for value in values)
        if len(normalized) != len(set(normalized)):
            raise ValueError("coding oracle path lists must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def _validate_oracle(self) -> "CodingBenchmarkCase":
        required = set(self.required_changed_paths)
        if self.allowed_changed_paths is not None and not required.issubset(
            self.allowed_changed_paths
        ):
            raise ValueError("required changed paths must also be allowed")
        if set(self.file_exists) & set(self.file_not_exists):
            raise ValueError("a path cannot be required to both exist and not exist")
        return self


class CodingBenchmarkSuite(BaseModel):
    """A versioned sequential suite of isolated coding tasks."""

    model_config = ConfigDict(frozen=True)

    version: str = Field(default=DEFAULT_BENCHMARK_VERSION, min_length=1, max_length=200)
    cases: tuple[CodingBenchmarkCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_suite(self) -> "CodingBenchmarkSuite":
        if not self.version.strip():
            raise ValueError("benchmark version must not be blank")
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("coding benchmark case IDs must be unique")
        return self


class OracleCheckResult(BaseModel):
    """One read-only hidden-oracle assertion with debuggable evidence."""

    model_config = ConfigDict(frozen=True)

    check: str
    path: Path | None = None
    passed: bool
    message: str


class CodingOracleResult(BaseModel):
    """Combined hidden-oracle outcome for a final isolated workspace."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    checks: tuple[OracleCheckResult, ...]

    @model_validator(mode="after")
    def _validate_passed(self) -> "CodingOracleResult":
        if self.passed is not all(check.passed for check in self.checks):
            raise ValueError("coding oracle summary must match its checks")
        return self


class CodingBenchmarkCaseResult(BaseModel):
    """Workflow outcome, hidden correctness, recovery, and resource counters."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    trace_run_id: UUID | None = None
    workflow_status: CodingTaskStatus
    oracle: CodingOracleResult
    oracle_passed: bool
    task_success: bool
    false_positive_completion: bool
    final_verification_passed: bool
    recovery_observed: bool
    llm_calls: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    successful_mutations: int = Field(ge=0)
    agent_iterations: int = Field(ge=0)
    completion_attempts: int = Field(ge=0)
    planner_generated: bool = False
    review_attempts: int = Field(default=0, ge=0)
    review_blocks: int = Field(default=0, ge=0)
    review_approved: bool = False
    completion_after_review: bool = False
    false_positive_prevented_by_review: bool = False
    changed_files: tuple[Path, ...]
    oracle_failures: tuple[str, ...]
    workflow_blockers: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_outcomes(self) -> "CodingBenchmarkCaseResult":
        completed = self.workflow_status is CodingTaskStatus.COMPLETED
        if self.oracle_passed is not self.oracle.passed:
            raise ValueError("oracle summary must match the structured oracle")
        if self.task_success is not (completed and self.oracle_passed):
            raise ValueError("task success requires completion and a passing oracle")
        if self.false_positive_completion is not (completed and not self.oracle_passed):
            raise ValueError("false-positive completion summary is inconsistent")
        return self


class CodingEvaluationReport(BaseModel):
    """Aggregate coding reliability metrics retaining every hidden-oracle result."""

    model_config = ConfigDict(frozen=True)

    benchmark_version: str
    mode: EvaluationMode
    case_results: tuple[CodingBenchmarkCaseResult, ...] = Field(min_length=1)
    case_count: int = Field(gt=0)
    workflow_completion_rate: float = Field(ge=0, le=1)
    task_success_rate: float = Field(ge=0, le=1)
    false_positive_completion_rate: float = Field(ge=0, le=1)
    verification_pass_rate: float = Field(ge=0, le=1)
    recovery_rate: float | None = Field(default=None, ge=0, le=1)
    mean_llm_calls: float = Field(ge=0)
    mean_tool_calls: float = Field(ge=0)
    mean_successful_mutations: float = Field(ge=0)
    mean_agent_iterations: float = Field(ge=0)
    mean_completion_attempts: float = Field(ge=0)
    planner_generation_rate: float = Field(default=0, ge=0, le=1)
    review_approval_rate: float = Field(default=0, ge=0, le=1)
    mean_review_attempts: float = Field(default=0, ge=0)
    review_block_rate: float = Field(default=0, ge=0, le=1)
    false_positive_prevented_by_review_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_report(self) -> "CodingEvaluationReport":
        if self.case_count != len(self.case_results):
            raise ValueError("coding case count must match case results")
        return self


class AgentNavigationBenchmarkCase(_NamedCase):
    """One scripted read-only investigation task with fully deterministic decisions.

    ``decisions`` fully controls the fake LLM's behavior; the harness never
    feeds ``expected_facts``/``forbidden_facts`` back into the model or tools.
    """

    task: str = Field(min_length=1, max_length=10_000)
    category: str = Field(min_length=1, max_length=64)
    decisions: tuple[AgentDecision, ...] = Field(min_length=1)
    expected_facts: tuple[str, ...] = Field(min_length=1, max_length=50)
    forbidden_facts: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_case(self) -> "AgentNavigationBenchmarkCase":
        if not self.task.strip():
            raise ValueError("agent navigation task must not be blank")
        if self.decisions[-1].action != "final":
            raise ValueError("the last scripted decision must be a final answer")
        if any(not fact.strip() for fact in (*self.expected_facts, *self.forbidden_facts)):
            raise ValueError("expected/forbidden facts must not be blank")
        return self


class AgentNavigationBenchmarkSuite(BaseModel):
    """A versioned set of scripted read-only investigation tasks."""

    model_config = ConfigDict(frozen=True)

    version: str = Field(default=DEFAULT_BENCHMARK_VERSION, min_length=1, max_length=200)
    cases: tuple[AgentNavigationBenchmarkCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_suite(self) -> "AgentNavigationBenchmarkSuite":
        if not self.version.strip():
            raise ValueError("benchmark version must not be blank")
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("agent navigation benchmark case IDs must be unique")
        return self


class AgentNavigationCaseResult(BaseModel):
    """One case's orchestration evidence: tool usage, verification, grounding."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    category: str
    retrieval_mode: str
    trace_run_id: UUID | None = None
    status: AgentRunStatus
    task_success: bool
    final_answer_grounded: bool
    tool_calls: int = Field(ge=0)
    indexed_search_calls: int = Field(ge=0)
    read_file_calls: int = Field(ge=0)
    filesystem_search_calls: int = Field(ge=0)
    verified_retrieval_followup: bool
    final_answer: str


class AgentNavigationEvaluationReport(BaseModel):
    """Aggregate navigation-orchestration metrics for one retrieval mode."""

    model_config = ConfigDict(frozen=True)

    benchmark_version: str
    mode: EvaluationMode
    retrieval_mode: str
    case_results: tuple[AgentNavigationCaseResult, ...] = Field(min_length=1)
    case_count: int = Field(gt=0)
    task_success_rate: float = Field(ge=0, le=1)
    mean_tool_calls: float = Field(ge=0)
    mean_indexed_search_calls: float = Field(ge=0)
    mean_read_file_calls: float = Field(ge=0)
    mean_filesystem_search_calls: float = Field(ge=0)
    verified_retrieval_followup_rate: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _validate_report(self) -> "AgentNavigationEvaluationReport":
        if self.case_count != len(self.case_results):
            raise ValueError("agent navigation case count must match case results")
        if any(result.retrieval_mode != self.retrieval_mode for result in self.case_results):
            raise ValueError("case retrieval modes must match the report retrieval mode")
        return self
