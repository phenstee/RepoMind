import { ApiError } from "./api";
import type { JobCancelResponse, JobDetail, JobStatus, ProgressEvent } from "./types";

interface CancellationClient {
  cancelJob(jobId: string): Promise<JobCancelResponse>;
  job(jobId: string): Promise<JobDetail>;
}

export function isTerminalJob(status: JobStatus): boolean {
  return status === "succeeded" || status === "failed" || status === "cancelled";
}

export function canCancelJob(job: JobDetail | null): boolean {
  return job !== null && !isTerminalJob(job.status) && job.cancellation_control === "available";
}

export function cancellationMessage(job: JobDetail | null): string | null {
  if (job === null) return null;
  if (job.status === "cancelled") return "Job cancelled safely.";
  if (job.cancellation_control === "deferred") {
    if (isTerminalJob(job.status)) {
      return `Cancellation was deferred after a code change began. The job finished ${job.status}.`;
    }
    return "Cancellation requested after a code change began. RepoMind will finish verification and review safely.";
  }
  if (job.cancellation_control === "requested") {
    return "Cancellation requested. Waiting for the next safe checkpoint.";
  }
  return null;
}

export async function requestJobCancellation(
  client: CancellationClient,
  jobId: string,
): Promise<JobDetail> {
  await client.cancelJob(jobId);
  return client.job(jobId);
}

export interface JobStreamHandlers<T> {
  signal: AbortSignal;
  onProgress: (event: ProgressEvent) => void;
  onResult: (result: T) => void;
  onCancelled?: () => void;
}

export interface JobFollowClient<T> {
  job(jobId: string): Promise<JobDetail>;
  events(jobId: string, handlers: JobStreamHandlers<T>): Promise<void>;
}

export interface FollowedJob<T> {
  job: JobDetail;
  result: T | undefined;
}

export interface FollowOptions {
  maxReconnects?: number;
  delayMs?: (attempt: number) => number;
}

function abortError(): DOMException {
  return new DOMException("The operation was aborted.", "AbortError");
}

function wait(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(abortError());
      return;
    }
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    function onAbort() {
      clearTimeout(timer);
      reject(abortError());
    }
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

/** A dropped connection (clean close or network failure) is worth reconnecting to. */
function isReconnectable(error: unknown): boolean {
  if (error instanceof DOMException && error.name === "AbortError") return false;
  if (error instanceof TypeError) return true;
  return error instanceof ApiError && error.code === "stream_disconnected";
}

/**
 * Follow a durable job's progress until PostgreSQL reports it terminal.
 *
 * The SSE stream is only a best-effort view of the job, so a stream that closes early (proxy
 * idle timeout, API restart, network blip) is re-subscribed with backoff instead of being
 * reported as a job failure. The authoritative terminal state always comes from `client.job`.
 */
export async function followJob<T>(
  client: JobFollowClient<T>,
  jobId: string,
  handlers: Pick<JobStreamHandlers<T>, "signal" | "onProgress">,
  { maxReconnects = 5, delayMs = (attempt) => Math.min(1000 * 2 ** attempt, 10_000) }: FollowOptions = {},
): Promise<FollowedJob<T>> {
  let result: T | undefined;
  let reconnects = 0;
  while (true) {
    try {
      await client.events(jobId, {
        signal: handlers.signal,
        onProgress: handlers.onProgress,
        onResult: (value) => {
          result = value;
        },
      });
    } catch (caught) {
      if (!isReconnectable(caught)) throw caught;
    }
    const job = await client.job(jobId);
    if (isTerminalJob(job.status) || result !== undefined) {
      return { job, result: result ?? ((job.result as T | null) ?? undefined) };
    }
    if (reconnects >= maxReconnects) {
      throw new ApiError("stream_disconnected", "The progress stream kept disconnecting before the job finished.");
    }
    await wait(delayMs(reconnects), handlers.signal);
    reconnects += 1;
  }
}
