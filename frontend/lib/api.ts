import { SSEParser } from "./sse";
import type {
  HealthResponse,
  JobCancelResponse,
  JobDetail,
  JobQueuedResponse,
  ProgressEvent,
  Repository,
  RepositoryFilesResponse,
  RepositoryListResponse,
  RunDetail,
  RunListResponse,
} from "./types";

const baseUrl = (process.env.NEXT_PUBLIC_REPOMIND_API_URL ?? "http://127.0.0.1:8000/api/v1").replace(
  /\/$/,
  "",
);

export class ApiError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly status: number | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function errorFromPayload(payload: unknown, status: number | null = null): ApiError {
  if (isRecord(payload) && isRecord(payload.error)) {
    const { code, message } = payload.error;
    if (typeof code === "string" && typeof message === "string") {
      return new ApiError(code, message, status);
    }
  }
  return new ApiError("request_failed", "The request could not be completed.", status);
}

/**
 * State-changing requests carry a custom header so browsers must preflight them; the API
 * rejects POSTs without it, which blocks cross-site "simple request" CSRF against localhost.
 */
const CLIENT_HEADER = { "X-RepoMind-Client": "web" };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const method = (init?.method ?? "GET").toUpperCase();
  const response = await fetch(`${baseUrl}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(method === "GET" || method === "HEAD" ? {} : CLIENT_HEADER),
      ...init?.headers,
    },
  });
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    throw errorFromPayload(payload, response.status);
  }
  return payload as T;
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  repositories: () => request<RepositoryListResponse>("/repositories"),
  registerRepository: (name: string, path: string) =>
    request<Repository>("/repositories", { method: "POST", body: JSON.stringify({ name, path }) }),
  files: (repositoryId: number) =>
    request<RepositoryFilesResponse>(`/repositories/${repositoryId}/files?limit=100`),
  runs: () => request<RunListResponse>("/runs?limit=30"),
  run: (runId: string) => request<RunDetail>(`/runs/${runId}`),
  job: (jobId: string) => request<JobDetail>(`/jobs/${jobId}`),
  cancelJob: (jobId: string) =>
    request<JobCancelResponse>(`/jobs/${jobId}/cancel`, { method: "POST" }),
  createJob: (repositoryId: number, type: "index" | "rag" | "agent" | "coding", body?: unknown) =>
    request<JobQueuedResponse>(`/repositories/${repositoryId}/jobs/${type}`, {
      method: "POST",
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
};

export interface StreamHandlers<T> {
  signal: AbortSignal;
  onProgress: (event: ProgressEvent) => void;
  onResult: (result: T) => void;
  onCancelled?: () => void;
}

/**
 * Follow one job's SSE stream until it emits a terminal frame.
 *
 * Throws `stream_disconnected` when the connection closes before a `result`, `error`, or
 * `cancelled` frame, so callers never mistake a dropped connection for a finished job.
 */
export async function getJobEvents<T>(
  jobId: string,
  handlers: StreamHandlers<T>,
): Promise<void> {
  const response = await fetch(`${baseUrl}/jobs/${jobId}/events`, {
    signal: handlers.signal,
    headers: { Accept: "text/event-stream" },
  });
  if (!response.ok) throw errorFromPayload(await response.json().catch(() => null), response.status);
  if (response.body === null) throw new ApiError("stream_unavailable", "The job stream is unavailable.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SSEParser();
  try {
    while (true) {
      const { done, value } = await reader.read();
      for (const frame of parser.feed(decoder.decode(value, { stream: !done }))) {
        if (frame.event === "result" && isRecord(frame.data) && "result" in frame.data) {
          handlers.onResult(frame.data.result as T);
          return;
        }
        if (frame.event === "cancelled") {
          handlers.onCancelled?.();
          return;
        }
        if (frame.event === "error") {
          throw errorFromPayload(isRecord(frame.data) && "error" in frame.data ? { error: frame.data.error } : null);
        }
        if (isProgressEvent(frame.data)) handlers.onProgress(frame.data);
      }
      if (done) {
        throw new ApiError("stream_disconnected", "The progress stream ended before the job finished.");
      }
    }
  } finally {
    // Cancelling (not just unlocking) closes the HTTP connection on every exit path, so a
    // handler error or malformed frame cannot leave a long-lived job stream open.
    await reader.cancel().catch(() => undefined);
  }
}

function isProgressEvent(value: unknown): value is ProgressEvent {
  return (
    isRecord(value) &&
    typeof value.run_id === "string" &&
    typeof value.sequence === "number" &&
    typeof value.event === "string" &&
    typeof value.timestamp === "string" &&
    isRecord(value.data)
  );
}

