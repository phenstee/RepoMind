import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { InvestigatePanel, InvestigationResult } from "./investigate-panel";
import type { AgentResponse, ObservedLocation } from "../lib/types";

function agentResult(overrides: Partial<AgentResponse> = {}): AgentResponse {
  return {
    status: "completed",
    final_answer: "The worker claims jobs with SKIP LOCKED.",
    iterations: 3,
    llm_calls: 3,
    tool_execution_attempts: 2,
    evidence: [],
    evidence_truncated: false,
    trace_run_id: null,
    ...overrides,
  };
}

function location(overrides: Partial<ObservedLocation> = {}): ObservedLocation {
  return {
    relative_path: "src/repomind/agent/loop.py",
    start_line: 123,
    end_line: 145,
    observed_via: "read_file",
    ...overrides,
  };
}

describe("investigate panel retrieval mode selector", () => {
  it("renders the filesystem/indexed selector defaulting to filesystem", () => {
    const html = renderToStaticMarkup(
      <InvestigatePanel disabled={false} onSubmit={async () => undefined} />,
    );

    expect(html).toContain("Code search");
    expect(html).toContain("Filesystem");
    expect(html).toContain("Indexed (experimental)");
    // React SSR marks the initially selected <option> explicitly.
    expect(html).toMatch(/<option value="filesystem"[^>]*selected[^>]*>/);
  });

  it("does not mention retrieval internals in the visible UI", () => {
    const html = renderToStaticMarkup(
      <InvestigatePanel disabled={false} onSubmit={async () => undefined} />,
    );

    for (const internal of ["RRF", "HNSW", "hybrid_symbol", "ef_search"]) {
      expect(html).not.toContain(internal);
    }
  });
});

describe("investigation evidence", () => {
  it("renders observed path/line locations", () => {
    const html = renderToStaticMarkup(
      <InvestigationResult
        result={agentResult({
          evidence: [
            location(),
            location({
              relative_path: "src/repomind/tools/search.py",
              start_line: 42,
              end_line: 42,
              observed_via: "search_code",
            }),
          ],
        })}
      />,
    );

    expect(html).toContain("Evidence");
    expect(html).toContain("src/repomind/agent/loop.py:123-145");
    // A single-line observation renders without a redundant range.
    expect(html).toContain("src/repomind/tools/search.py:42");
    expect(html).not.toContain("search.py:42-42");
  });

  it("omits the evidence section entirely when nothing was observed", () => {
    const html = renderToStaticMarkup(<InvestigationResult result={agentResult()} />);

    expect(html).not.toContain("Evidence");
  });

  it("indicates when observed locations were omitted", () => {
    const html = renderToStaticMarkup(
      <InvestigationResult
        result={agentResult({ evidence: [location()], evidence_truncated: true })}
      />,
    );

    expect(html).toContain("Additional observed locations were omitted.");
  });

  it("does not expose retrieval internals or source content in evidence", () => {
    const html = renderToStaticMarkup(
      <InvestigationResult
        result={agentResult({
          evidence: [location(), location({ observed_via: "find_symbol", start_line: 7, end_line: 7 })],
        })}
      />,
    );

    for (const internal of [
      "RRF",
      "HNSW",
      "hybrid_symbol",
      "ef_search",
      "indexed_code_search",
      "rank",
      "score",
      "chunk",
    ]) {
      expect(html).not.toContain(internal);
    }
  });
});
