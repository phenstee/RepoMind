import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { AnswerCard, AskPanel, DEFAULT_STRATEGY, STRATEGIES } from "./ask-panel";
import { PendingRun } from "./pending-run";

const noop = async () => undefined;

describe("AskPanel", () => {
  it("defaults to hybrid retrieval, matching the API default", () => {
    const html = renderToStaticMarkup(<AskPanel disabled={false} onSubmit={noop} />);
    expect(DEFAULT_STRATEGY).toBe("hybrid");
    expect(STRATEGIES[0].value).toBe("hybrid");
    expect(html).toMatch(/<option value="hybrid" selected="">Hybrid \(default\)<\/option>/);
    expect(html).not.toMatch(/<option value="semantic" selected="">/);
  });

  it("describes the selected strategy next to the selector", () => {
    const html = renderToStaticMarkup(<AskPanel disabled={false} onSubmit={noop} />);
    expect(html).toContain('aria-describedby="strategy-hint"');
    expect(html).toContain('id="strategy-hint"');
    expect(html).toContain(STRATEGIES[0].description);
  });
});

describe("AnswerCard", () => {
  it("renders each citation as a button that copies path:start-end", () => {
    const html = renderToStaticMarkup(
      <AnswerCard
        question="How are refunds issued?"
        answer={{
          answer: "Through `RefundService.issue`.",
          insufficient_evidence: false,
          citations: [
            { relative_path: "src/payments/refunds.py", start_line: 41, end_line: 88 },
            { relative_path: "src/config.py", start_line: 7, end_line: 7 },
          ],
          trace_run_id: null,
        }}
      />,
    );
    expect(html).toContain('title="Copy src/payments/refunds.py:41-88"');
    expect(html).toContain('title="Copy src/config.py:7"');
    expect(html).toContain("L41–88");
    expect(html.match(/<button type="button" class="chip chipButton"/g)).toHaveLength(2);
  });
});

describe("PendingRun", () => {
  it("narrates the latest step as a busy status where the result will appear", () => {
    const html = renderToStaticMarkup(<PendingRun title="How are refunds issued?" status="Retrieval completed" />);
    expect(html).toContain('aria-busy="true"');
    expect(html).toContain('role="status"');
    expect(html).toContain('data-text="Retrieval completed"');
    expect(html).toContain("How are refunds issued?");
  });
});
