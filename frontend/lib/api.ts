import { SSEParser } from "./sse";
import type {
  AgentResponse,
  ApiErrorPayload,
  HealthResponse,
  IndexResponse,
  JobCancelResponse,
  JobDetail,
  JobQueuedResponse,
  ProgressEvent,
  Repository,
  RepositoryFilesResponse,
  RepositoryListResponse,
  RAGResponse,
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
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function errorFromPayload(payload: unknown): ApiError {
  if (isRecord(payload) && isRecord(payload.error)) {
    const { code, message } = payload.error;
    if (typeof code === "string" && typeof message === "string") {
      return new ApiError(code, message);
    }
  }
  return new ApiError("request_failed", "The request could not be completed.");
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    throw errorFromPayload(payload);
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

export async function postSSE<T>(
  path: string,
  body: unknown,
  handlers: StreamHandlers<T>,
): Promise<void> {
  const response = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    signal: handlers.signal,
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw errorFromPayload(await response.json().catch(() => null));
  }
  if (response.body === null) {
    throw new ApiError("stream_unavailable", "The server did not provide a progress stream.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SSEParser();
  let terminal = false;
  try {
    while (true) {
      const { done, value } = await reader.read();
      for (const frame of parser.feed(decoder.decode(value, { stream: !done }))) {
        if (frame.event === "result" && isRecord(frame.data) && "result" in frame.data) {
          handlers.onResult(frame.data.result as T);
          terminal = true;
        } else if (frame.event === "error") {
          throw errorFromPayload(isRecord(frame.data) && "error" in frame.data ? { error: frame.data.error } : null);
        } else if (isProgressEvent(frame.data)) {
          handlers.onProgress(frame.data);
        }
      }
      if (done) {
        break;
      }
    }
  } finally {
    reader.releaseLock();
  }
  if (!terminal) {
    throw new ApiError("stream_disconnected", "The progress stream ended before a final result.");
  }
}

export async function getJobEvents<T>(
  jobId: string,
  handlers: StreamHandlers<T>,
): Promise<void> {
  const response = await fetch(`${baseUrl}/jobs/${jobId}/events`, {
    signal: handlers.signal,
    headers: { Accept: "text/event-stream" },
  });
  if (!response.ok) throw errorFromPayload(await response.json().catch(() => null));
  if (response.body === null) throw new ApiError("stream_unavailable", "The job stream is unavailable.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SSEParser();
  try {
    while (true) {
      const { done, value } = await reader.read();
      for (const frame of parser.feed(decoder.decode(value, { stream: !done }))) {
        if (frame.event === "result" && isRecord(frame.data) && "result" in frame.data) handlers.onResult(frame.data.result as T);
        else if (frame.event === "cancelled") {
          handlers.onCancelled?.();
          return;
        }
        else if (frame.event === "error") throw errorFromPayload(isRecord(frame.data) && "error" in frame.data ? { error: frame.data.error } : null);
        else if (isProgressEvent(frame.data)) handlers.onProgress(frame.data);
      }
      if (done) return;
    }
  } finally { reader.releaseLock(); }
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

export type StreamResult = IndexResponse | RAGResponse | AgentResponse;
