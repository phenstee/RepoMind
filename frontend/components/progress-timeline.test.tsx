import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { eventLabel, ProgressTimeline } from "./progress-timeline";

describe("ProgressTimeline", () => {
  it("labels known events and passes unknown names through", () => {
    expect(eventLabel("retrieval.completed")).toBe("Retrieval completed");
    expect(eventLabel("custom.event")).toBe("custom.event");
  });

  it("formats long durations in seconds", () => {
    const html = renderToStaticMarkup(
      <ProgressTimeline
        events={[
          { run_id: "r", sequence: 0, event: "model.completed", timestamp: "2026-11-16T15:04:00Z", data: { duration_ms: 5120.7 } },
          { run_id: "r", sequence: 1, event: "retrieval.completed", timestamp: "2026-11-16T15:04:01Z", data: { duration_ms: 412.4 } },
        ]}
      />,
    );
    expect(html).toContain("5.1 s");
    expect(html).toContain("412 ms");
    expect(html).not.toContain("5121 ms");
  });
});
