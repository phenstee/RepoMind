import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "./api";

afterEach(() => vi.unstubAllGlobals());

describe("job cancellation API", () => {
  it("posts to the durable cancellation endpoint", async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          job_id: "job-1",
          status: "cancelled",
          cancel_requested: true,
          cancel_requested_at: "2026-09-16T00:00:00Z",
          cancelled_at: "2026-09-16T00:00:00Z",
          cancellation_control: "cancelled",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetch);

    const result = await api.cancelJob("job-1");

    expect(result.status).toBe("cancelled");
    expect(fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/v1/jobs/job-1/cancel",
      expect.objectContaining({ method: "POST" }),
    );
  });
});

describe("state-changing requests", () => {
  it("send the RepoMind client header so cross-site simple requests are rejected", async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ job_id: "job-1", job_type: "index", status: "queued" }), { status: 202 }),
    );
    vi.stubGlobal("fetch", fetch);

    await api.createJob(1, "index");

    const init = fetch.mock.calls[0][1] as RequestInit;
    expect(init.headers).toMatchObject({ "X-RepoMind-Client": "web" });
  });

  it("omit the header on reads", async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ repositories: [] }), { status: 200 }));
    vi.stubGlobal("fetch", fetch);

    await api.repositories();

    const init = fetch.mock.calls[0][1] as RequestInit;
    expect(init.headers).not.toHaveProperty("X-RepoMind-Client");
  });
});
