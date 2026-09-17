import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { CodingResult } from "./code-panel";
import { ProgressTimeline } from "./progress-timeline";
import type { CodingResponse, ProgressEvent } from "../lib/types";

function result(verdict: "approve" | "changes_required"): CodingResponse {
  return {
    status: verdict === "approve" ? "completed" : "verification_failed",
    final_answer: "Implemented the bounded change.",
    tests: {
      passed: true,
      workspace_revision: 2,
      exit_code: 0,
      timed_out: false,
      execution_failed: false,
    },
    ruff: {
      passed: true,
      workspace_revision: 2,
      exit_code: 0,
      timed_out: false,
      execution_failed: false,
    },
    changed_files: ["src/app.py"],
    completion_attempts: 2,
    workspace_revision: 2,
    plan: {
      task_summary: "Update retry behavior.",
      relevant_areas: ["retry client"],
      steps: [
        {
          step_id: 1,
          action: "Inspect retry behavior.",
          likely_paths: ["src/app.py"],
          criterion_indices: [0],
          verification: ["pytest"],
        },
      ],
      acceptance_coverage: [{ criterion_index: 0, step_ids: [1], uncertainty: null }],
      verification_plan: ["pytest"],
      risks: [],
      uncertainties: [],
    },
    review: {
      verdict,
      workspace_revision: 2,
      acceptance_results: [
        {
          criterion_index: 0,
          status: verdict === "approve" ? "satisfied" : "not_satisfied",
          evidence: "The bounded change was reviewed.",
        },
      ],
      findings: verdict === "approve" ? [] : ["A regression test is missing."],
      required_corrections: verdict === "approve" ? [] : ["Add the regression test."],
    },
    review_attempts: 2,
    review_blocks: verdict === "approve" ? 1 : 2,
    trace_run_id: null,
  };
}

describe("coding result", () => {
  it("renders the structured plan, criterion coverage, and approval", () => {
    const html = renderToStaticMarkup(<CodingResult result={result("approve")} />);
    expect(html).toContain("Plan");
    expect(html).toContain("Inspect retry behavior.");
    expect(html).toContain("Criterion 1");
    expect(html).toContain("Reviewer: Approved");
    expect(html).not.toContain("{&quot;verdict&quot;");
  });

  it("renders concise changes-required findings", () => {
    const html = renderToStaticMarkup(
      <CodingResult result={result("changes_required")} />,
    );
    expect(html).toContain("Reviewer: Changes required");
    expect(html).toContain("A regression test is missing.");
    expect(html).toContain("Add the regression test.");
  });
});

describe("coding corrective timeline", () => {
  it("shows review rejection followed by fresh verification and approval", () => {
    const timestamp = "2026-09-16T00:00:00Z";
    const events: ProgressEvent[] = [
      ["verification.completed", { passed: true }],
      ["review.blocked", { verdict: "changes_required" }],
      ["file.mutated", { workspace_revision: 2 }],
      ["verification.completed", { passed: true, workspace_revision: 2 }],
      ["review.completed", { verdict: "approve", workspace_revision: 2 }],
      ["completion.completed", { workspace_revision: 2 }],
    ].map(([event, data], index) => ({
      run_id: "00000000-0000-0000-0000-000000000001",
      sequence: index + 1,
      event: event as string,
      timestamp,
      data: data as Record<string, string | number | boolean>,
    }));

    const html = renderToStaticMarkup(<ProgressTimeline events={events} />);
    expect(html.indexOf("Review: changes required")).toBeLessThan(
      html.indexOf("Independent review completed"),
    );
    expect(html).toContain("Completion accepted");
  });
});
