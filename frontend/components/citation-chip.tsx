"use client";

import { useEffect, useState } from "react";

import { Icon } from "./icons";

/**
 * A source location that copies itself (`path:start-end`) for pasting into an editor or
 * terminal. The file icon swaps to a check while the copy confirmation shows.
 */
export function CitationChip({ label, meta, reference }: { label: string; meta: string; reference: string }) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const timer = window.setTimeout(() => setCopied(false), 1600);
    return () => window.clearTimeout(timer);
  }, [copied]);

  async function copy() {
    try {
      await navigator.clipboard.writeText(reference);
      setCopied(true);
    } catch {
      // Clipboard access can be blocked; the reference stays visible to select by hand.
    }
  }

  return (
    <li>
      <button type="button" className="chip chipButton" title={`Copy ${reference}`} onClick={() => void copy()}>
        <span className="t-icon-swap" data-state={copied ? "b" : "a"} aria-hidden="true">
          <span className="t-icon" data-icon="a"><Icon name="file" /></span>
          <span className="t-icon" data-icon="b"><Icon name="check" /></span>
        </span>
        <span className="srOnly">Copy </span>
        <code>{label}</code>
        <span className="chipMeta">{meta}</span>
      </button>
      <span className="srOnly" role="status">{copied ? `Copied ${reference}` : ""}</span>
    </li>
  );
}
