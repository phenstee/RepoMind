import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { RichText } from "./rich-text";

describe("RichText", () => {
  it("renders paragraphs and inline code", () => {
    const html = renderToStaticMarkup(<RichText text={"Call `issue()` first.\n\nThen retry."} />);
    expect(html).toBe('<div class="richText"><p>Call <code>issue()</code> first.</p><p>Then retry.</p></div>');
  });

  it("never renders model text as HTML", () => {
    const html = renderToStaticMarkup(<RichText text={"<img src=x onerror=alert(1)> `<b>`"} />);
    expect(html).not.toContain("<img");
    expect(html).toContain("&lt;img");
    expect(html).toContain("<code>&lt;b&gt;</code>");
  });
});
