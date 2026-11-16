import { describe, expect, it } from "vitest";

import { formatDuration, formatRelative, humanize, lineRange, runTypeLabel } from "./format";

describe("display formatting", () => {
  it("formats durations compactly", () => {
    expect(formatDuration(null)).toBe("running");
    expect(formatDuration(820)).toBe("820 ms");
    expect(formatDuration(3840)).toBe("3.8 s");
    expect(formatDuration(19_420)).toBe("19 s");
    expect(formatDuration(98_200)).toBe("1 m 38 s");
    expect(formatDuration(120_000)).toBe("2 m");
  });

  it("formats recent timestamps relatively", () => {
    const now = Date.parse("2026-10-07T12:00:00Z");
    expect(formatRelative("2026-10-07T11:59:50Z", now)).toBe("just now");
    expect(formatRelative("2026-10-07T11:55:00Z", now)).toBe("5 min ago");
    expect(formatRelative("2026-10-07T09:00:00Z", now)).toBe("3 h ago");
    expect(formatRelative("not a date", now)).toBe("");
  });

  it("labels run types, statuses, and line ranges", () => {
    expect(runTypeLabel("rag")).toBe("Ask");
    expect(runTypeLabel("custom")).toBe("custom");
    expect(humanize("verification_failed")).toBe("Verification failed");
    expect(lineRange(22, 61)).toBe("L22–61");
    expect(lineRange(7, 7)).toBe("L7");
  });
});
