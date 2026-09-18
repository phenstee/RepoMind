import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { InvestigatePanel } from "./investigate-panel";

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
