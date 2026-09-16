export type JsonValue = boolean | number | string | null | JsonValue[] | { [key: string]: JsonValue };

export interface ApiErrorPayload {
  error: {
    code: string;
    message: string;
    trace_run_id?: string;
  };
}

export interface HealthResponse {
  status: "ok";
  service: "repomind";
}

export interface Repository {
  id: number;
  name: string;
  created_at: string;
  workspace_relative_path?: string | null;
}

export interface RepositoryListResponse {
  repositories: Repository[];
}

export interface RepositoryFile {
  relative_path: string;
  language: string | null;
  size_bytes: number;
  line_count: number;
}

export interface RepositoryFilesResponse {
  repository_id: number;
  files: RepositoryFile[];
  limit: number;
  offset: number;
}

export interface IndexResponse {
  repository_id: number;
  files_indexed: number;
  chunks_indexed: number;
  embedding_model: string | null;
}

export type RetrievalStrategy = "semantic" | "hybrid" | "hybrid_rerank";

export interface Citation {
  relative_path: string;
  start_line: number;
  end_line: number;
}

export interface RAGResponse {
  answer: string;
  insufficient_evidence: boolean;
  citations: Citation[];
  trace_run_id: string | null;
}

export interface AgentResponse {
  status: string;
  final_answer: string | null;
  iterations: number;
  llm_calls: number;
  tool_execution_attempts: number;
  trace_run_id: string | null;
}

export interface VerificationSummary {
  passed: boolean | null;
  workspace_revision: number | null;
  exit_code: number | null;
  timed_out: boolean | null;
  execution_failed: boolean;
}

export interface CodingResponse {
  status: string;
  final_answer: string | null;
  tests: VerificationSummary;
  ruff: VerificationSummary;
  changed_files: string[];
  completion_attempts: number;
  workspace_revision: number;
  trace_run_id: string | null;
}

export interface ProgressEvent {
  run_id: string;
  sequence: number;
  event: string;
  timestamp: string;
  data: Record<string, JsonValue>;
}

export interface TraceEvent {
  sequence: number;
  event_type: string;
  timestamp: string;
  metadata: Record<string, JsonValue>;
  duration_ms: number | null;
}

export interface RunSummary {
  run_id: string;
  run_type: string;
  status: string;
  domain_status: string | null;
  model: string | null;
  started_at: string;
  ended_at: string | null;
  duration_ms: number | null;
  llm_calls: number;
  tool_calls: number;
  successful_mutations: number;
  errors: number;
  usage_reported_calls: number;
}

export interface RunDetail extends RunSummary {
  events: TraceEvent[];
}

export interface RunListResponse {
  runs: RunSummary[];
}

export type JobType = "index" | "rag" | "agent" | "coding";
export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";
export type CancellationState =
  | "available"
  | "requested"
  | "deferred"
  | "unavailable"
  | "cancelled";

export interface JobQueuedResponse {
  job_id: string;
  job_type: JobType;
  status: JobStatus;
}

export interface JobDetail extends JobQueuedResponse {
  repository_id: number;
  attempt_count: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  trace_run_id: string | null;
  cancel_requested: boolean;
  cancel_requested_at: string | null;
  cancelled_at: string | null;
  cancellation_control: CancellationState;
  result: IndexResponse | RAGResponse | AgentResponse | CodingResponse | null;
  error: { code: string; message: string } | null;
}

export interface JobCancelResponse {
  job_id: string;
  status: JobStatus;
  cancel_requested: boolean;
  cancel_requested_at: string | null;
  cancelled_at: string | null;
  cancellation_control: CancellationState;
}
