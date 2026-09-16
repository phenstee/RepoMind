"use client";

import { useState } from "react";

import type { AgentResponse } from "../lib/types";

export function InvestigatePanel({
  disabled,
  onSubmit,
}: {
  disabled: boolean;
  onSubmit: (query: string) => Promise<AgentResponse | undefined>;
}) {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<AgentResponse | null>(null);

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const requestedQuery = query.trim();
    if (requestedQuery.length === 0) return;
    const response = await onSubmit(requestedQuery);
    if (response !== undefined) {
      setResult(response);
    }
  }

  return (
    <section className="modePanel">
      <header>
        <p className="eyebrow readOnly">Read-only</p>
        <h2>Investigate</h2>
        <p className="muted">The agent can inspect the selected repository but has no file-edit controls in this mode.</p>
      </header>
      <form className="stack" onSubmit={(event) => void submit(event)}>
        <label>
          Investigation task
          <textarea value={query} onChange={(event) => setQuery(event.target.value)} required maxLength={10_000} />
        </label>
        <button type="submit" disabled={disabled}>Run read-only investigation</button>
      </form>
      {result ? (
        <article className="answer">
          <h3>{result.status}</h3>
          <p>{result.final_answer ?? "The agent did not return a final answer."}</p>
          <small>
            {result.iterations} iterations · {result.tool_execution_attempts} tool calls · {result.llm_calls} model calls
          </small>
          {result.trace_run_id ? <small>Trace {result.trace_run_id}</small> : null}
        </article>
      ) : null}
    </section>
  );
}
