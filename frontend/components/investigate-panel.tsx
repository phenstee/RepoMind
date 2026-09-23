"use client";

import { useState } from "react";

import type { AgentResponse, AgentRetrievalMode, ObservedLocation } from "../lib/types";

function locationLabel(location: ObservedLocation) {
  return location.start_line === location.end_line
    ? `${location.relative_path}:${location.start_line}`
    : `${location.relative_path}:${location.start_line}-${location.end_line}`;
}

export function InvestigatePanel({
  disabled,
  onSubmit,
}: {
  disabled: boolean;
  onSubmit: (query: string, retrievalMode: AgentRetrievalMode) => Promise<AgentResponse | undefined>;
}) {
  const [query, setQuery] = useState("");
  const [retrievalMode, setRetrievalMode] = useState<AgentRetrievalMode>("filesystem");
  const [result, setResult] = useState<AgentResponse | null>(null);

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const requestedQuery = query.trim();
    if (requestedQuery.length === 0) return;
    const response = await onSubmit(requestedQuery, retrievalMode);
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
        <label className="inlineControl">
          Code search
          <select
            value={retrievalMode}
            onChange={(event) => setRetrievalMode(event.target.value as AgentRetrievalMode)}
          >
            <option value="filesystem">Filesystem</option>
            <option value="indexed">Indexed (experimental)</option>
          </select>
        </label>
        <button type="submit" disabled={disabled}>Run read-only investigation</button>
      </form>
      {result ? <InvestigationResult result={result} /> : null}
    </section>
  );
}

export function InvestigationResult({ result }: { result: AgentResponse }) {
  return (
    <article className="answer">
      <h3>{result.status}</h3>
      <p>{result.final_answer ?? "The agent did not return a final answer."}</p>
      <small>
        {result.iterations} iterations · {result.tool_execution_attempts} tool calls · {result.llm_calls} model calls
      </small>
      {result.evidence.length > 0 ? (
        <section className="evidence" aria-label="Evidence">
          <h4>Evidence</h4>
          <p className="muted">Locations the agent observed in the current repository.</p>
          <ul className="citations">
            {result.evidence.map((location) => (
              <li key={`${location.relative_path}:${location.start_line}-${location.end_line}`}>
                <code>{locationLabel(location)}</code>
              </li>
            ))}
          </ul>
          {result.evidence_truncated ? (
            <small>Additional observed locations were omitted.</small>
          ) : null}
        </section>
      ) : null}
      {result.trace_run_id ? <small>Trace {result.trace_run_id}</small> : null}
    </article>
  );
}
