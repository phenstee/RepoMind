import { describe, expect, it } from "vitest";

import { canCancelJob, cancellationMessage, isTerminalJob, requestJobCancellation } from "./jobs";
import type { JobDetail } from "./types";

function job(overrides: Partial<JobDetail> = {}): JobDetail {
  return {
    job_id: "job-1",
    job_type: "rag",
    status: "running",
    repository_id: 1,
    attempt_count: 1,
    created_at: "2026-09-16T00:00:00Z",
    started_at: "2026-09-16T00:00:01Z",
    finished_at: null,
    trace_run_id: null,
    cancel_requested: false,
    cancel_requested_at: null,
    cancelled_at: null,
    cancellation_control: "available",
    result: null,
    error: null,
    ...overrides,
  };
}

describe("durable job cancellation presentation", () => {
  it("offers cancellation only while the backend reports it available", () => {
    expect(canCancelJob(job())).toBe(true);
    expect(canCancelJob(job({ cancellation_control: "requested" }))).toBe(false);
    expect(canCancelJob(job({ status: "cancelled", cancellation_control: "cancelled" }))).toBe(false);
  });

  it("treats cancelled jobs as terminal across refreshes", () => {
    expect(isTerminalJob("cancelled")).toBe(true);
    expect(cancellationMessage(job({ status: "cancelled", cancellation_control: "cancelled" }))).toBe(
      "Job cancelled safely.",
    );
  });

  it("distinguishes pending cancellation from coding deferral", () => {
    expect(cancellationMessage(job({ cancel_requested: true, cancellation_control: "requested" }))).toContain(
      "next safe checkpoint",
    );
    expect(cancellationMessage(job({ job_type: "coding", cancel_requested: true, cancellation_control: "deferred" }))).toContain(
      "finish verification and review",
    );
    expect(
      cancellationMessage(
        job({
          job_type: "coding",
          status: "succeeded",
          cancel_requested: true,
          cancellation_control: "deferred",
        }),
      ),
    ).toContain("finished succeeded");
  });

  it("sends the cancel request and refreshes durable state", async () => {
    const cancelled = job({ status: "cancelled", cancellation_control: "cancelled" });
    const client = {
      cancelJob: async (jobId: string) => ({
        job_id: jobId,
        status: "cancelled" as const,
        cancel_requested: true,
        cancel_requested_at: cancelled.created_at,
        cancelled_at: cancelled.created_at,
        cancellation_control: "cancelled" as const,
      }),
      job: async () => cancelled,
    };

    await expect(requestJobCancellation(client, "job-1")).resolves.toBe(cancelled);
  });
});
