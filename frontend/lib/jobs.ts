import type { JobCancelResponse, JobDetail, JobStatus } from "./types";

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
