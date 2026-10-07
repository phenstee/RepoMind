import { Fragment, type ReactNode } from "react";

function inline(text: string): ReactNode[] {
  // Only `inline code` is recognised. Everything is rendered as React text, never HTML.
  return text.split(/(`[^`\n]+`)/g).map((part, index) =>
    part.length > 2 && part.startsWith("`") && part.endsWith("`") ? (
      <code key={index}>{part.slice(1, -1)}</code>
    ) : (
      <Fragment key={index}>{part}</Fragment>
    ),
  );
}

/** Model prose with paragraph breaks and inline code; safe for untrusted text. */
export function RichText({ text }: { text: string }) {
  const paragraphs = text.split(/\n{2,}/).filter((paragraph) => paragraph.trim().length > 0);
  return (
    <div className="richText">
      {paragraphs.map((paragraph, index) => (
        <p key={index}>{inline(paragraph)}</p>
      ))}
    </div>
  );
}
