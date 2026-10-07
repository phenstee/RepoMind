import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, getJobEvents } from "./api";
import { followJob, type JobFollowClient } from "./jobs";
import type { JobDetail } from "./types";

afterEach(() => vi.unstubAllGlobals());

function sseResponse(body: string, onCancel?: () => void): Response {
  const encoded = new TextEncoder().encode(body);
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoded);
    },
    cancel() {
      onCancel?.();
    },
  });
  return new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

function closedSseResponse(body: string): Response {
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

const progressFrame =
  'event: run.started\nid: 1\ndata: {"run_id":"r","sequence":1,"event":"run.started","timestamp":"2026-10-07T00:00:00Z","data":{}}\n\n';

function job(overrides: Partial<JobDetail> = {}): JobDetail {
  return {
    job_id: "job-1",
    job_type: "rag",
    status: "running",
    repository_id: 1,
    attempt_count: 1,
    created_at: "2026-10-07T00:00:00Z",
    started_at: "2026-10-07T00:00:01Z",
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

describe("getJobEvents", () => {
  it("rejects a stream that closes before a terminal frame", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(closedSseResponse(progressFrame)));
    const onProgress = vi.fn();

    await expect(
      getJobEvents("job-1", { signal: new AbortController().signal, onProgress, onResult: vi.fn() }),
    ).rejects.toMatchObject({ code: "stream_disconnected" });
    expect(onProgress).toHaveBeenCalledTimes(1);
  });

  it("delivers the result frame and stops", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(closedSseResponse('event: result\ndata: {"result":{"answer":"ok"}}\n\n')),
    );
    const onResult = vi.fn();

    await getJobEvents("job-1", { signal: new AbortController().signal, onProgress: vi.fn(), onResult });

    expect(onResult).toHaveBeenCalledWith({ answer: "ok" });
  });

  it("surfaces an error frame as an ApiError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        closedSseResponse('event: error\ndata: {"error":{"code":"job_failed","message":"boom"}}\n\n'),
      ),
    );

    await expect(
      getJobEvents("job-1", { signal: new AbortController().signal, onProgress: vi.fn(), onResult: vi.fn() }),
    ).rejects.toMatchObject({ code: "job_failed" });
  });

  it("cancels the HTTP body when a handler throws", async () => {
    const cancelled = vi.fn();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(sseResponse(progressFrame, cancelled)));

    await expect(
      getJobEvents("job-1", {
        signal: new AbortController().signal,
        onProgress: () => {
          throw new Error("handler failed");
        },
        onResult: vi.fn(),
      }),
    ).rejects.toThrow("handler failed");
    expect(cancelled).toHaveBeenCalled();
  });

  it("records the HTTP status on request errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: { code: "job_not_found", message: "Job not found." } }), { status: 404 }),
      ),
    );

    const failure = getJobEvents("missing", { signal: new AbortController().signal, onProgress: vi.fn(), onResult: vi.fn() });

    await expect(failure).rejects.toBeInstanceOf(ApiError);
    await expect(failure).rejects.toMatchObject({ status: 404 });
  });
});

describe("followJob", () => {
  it("reconnects after a dropped stream until the job is terminal", async () => {
    const events = vi
      .fn<JobFollowClient<string>["events"]>()
      .mockRejectedValueOnce(new ApiError("stream_disconnected", "dropped"))
      .mockImplementationOnce(async (_jobId, handlers) => handlers.onResult("answer"));
    const jobs = vi
      .fn<JobFollowClient<string>["job"]>()
      .mockResolvedValueOnce(job({ status: "running" }))
      .mockResolvedValueOnce(job({ status: "succeeded" }));

    const followed = await followJob({ job: jobs, events }, "job-1", {
      signal: new AbortController().signal,
      onProgress: vi.fn(),
    }, { delayMs: () => 0 });

    expect(events).toHaveBeenCalledTimes(2);
    expect(followed.result).toBe("answer");
    expect(followed.job.status).toBe("succeeded");
  });

  it("uses the persisted result when the stream closed before the result frame", async () => {
    const events = vi.fn<JobFollowClient<string>["events"]>().mockRejectedValue(new ApiError("stream_disconnected", "dropped"));
    const jobs = vi.fn<JobFollowClient<string>["job"]>().mockResolvedValue(
      job({ status: "succeeded", result: { answer: "persisted", citations: [], insufficient_evidence: false } as never }),
    );

    const followed = await followJob({ job: jobs, events }, "job-1", {
      signal: new AbortController().signal,
      onProgress: vi.fn(),
    });

    expect(events).toHaveBeenCalledTimes(1);
    expect(followed.result).toMatchObject({ answer: "persisted" });
  });

  it("gives up after the reconnect budget while the job is still running", async () => {
    const events = vi.fn<JobFollowClient<string>["events"]>().mockRejectedValue(new TypeError("network down"));
    const jobs = vi.fn<JobFollowClient<string>["job"]>().mockResolvedValue(job({ status: "running" }));

    await expect(
      followJob({ job: jobs, events }, "job-1", { signal: new AbortController().signal, onProgress: vi.fn() }, {
        maxReconnects: 2,
        delayMs: () => 0,
      }),
    ).rejects.toMatchObject({ code: "stream_disconnected" });
    expect(events).toHaveBeenCalledTimes(3);
  });

  it("does not reconnect after the viewer aborts", async () => {
    const events = vi
      .fn<JobFollowClient<string>["events"]>()
      .mockRejectedValue(new DOMException("aborted", "AbortError"));
    const jobs = vi.fn<JobFollowClient<string>["job"]>();

    await expect(
      followJob({ job: jobs, events }, "job-1", { signal: new AbortController().signal, onProgress: vi.fn() }),
    ).rejects.toMatchObject({ name: "AbortError" });
    expect(jobs).not.toHaveBeenCalled();
  });
});
