import { describe, expect, it } from "vitest";

import { SSEParser } from "./sse";

describe("SSEParser", () => {
  it("buffers an incomplete CRLF frame before parsing its JSON data", () => {
    const parser = new SSEParser();
    expect(parser.feed("id: 1\r\nevent: run.started\r\ndata: {\"sequence\":1")).toEqual([]);
    expect(parser.feed("}\r\n\r\n")).toEqual([
      { id: 1, event: "run.started", data: { sequence: 1 } },
    ]);
  });

  it("parses multiple frames delivered in a single chunk", () => {
    const parser = new SSEParser();
    expect(parser.feed("event: result\ndata: {\"ok\":true}\n\nevent: error\ndata: {\"code\":\"failed\"}\n\n")).toEqual([
      { event: "result", data: { ok: true } },
      { event: "error", data: { code: "failed" } },
    ]);
  });
});
