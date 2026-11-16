"use client";

import { useState } from "react";

import { humanize } from "../lib/format";
import type { AgentResponse, AgentRetrievalMode, ObservedLocation } from "../lib/types";
import { submitOnModEnter } from "./ask-panel";
import { CitationChip } from "./citation-chip";
import { Icon, Spinner } from "./icons";
import { PendingRun } from "./pending-run";
import { RichText } from "./rich-text";

function locationLabel(location: ObservedLocation) {
  return location.start_line === location.end_line
    ? `${location.relative_path}:${location.start_line}`
    : `${location.relative_path}:${location.start_line}-${location.end_line}`;
}

const observedViaLabels: Record<ObservedLocation["observed_via"], string> = {
  read_file: "read",
  search_code: "search",
  find_symbol: "symbol",
};

export function InvestigatePanel({
  disabled,
  running,
  status = "Waiting for a worker",
  onSubmit,
}: {
  disabled: boolean;
  running?: boolean;
  status?: string;
  onSubmit: (query: string, retrievalMode: AgentRetrievalMode) => Promise<AgentResponse | undefined>;
}) {
  const [query, setQuery] = useState("");
  const [retrievalMode, setRetrievalMode] = useState<AgentRetrievalMode>("filesystem");
  const [result, setResult] = useState<AgentResponse | null>(null);
  const [pending, setPending] = useState(false);

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const requestedQuery = query.trim();
    if (requestedQuery.length === 0 || disabled) return;
    setPending(true);
    try {
      const response = await onSubmit(requestedQuery, retrievalMode);
      if (response !== undefined) setResult(response);
    } finally {
      setPending(false);
    }
  }

  return (
    <section className="modePanel">
      <header>
        <h2 className="panelTitle">
          Investigate
          <span className="badge completed"><Icon name="shield" />Read-only</span>
        </h2>
        <p>The agent can inspect the selected repository but has no file-edit controls in this mode.</p>
      </header>
      <form className="composer" onSubmit={(event) => void submit(event)}>
        <label className="composerInput">
          <span className="srOnly">Investigation task</span>
          <textarea
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={submitOnModEnter}
            required
            maxLength={10_000}
            placeholder="Where are webhook signatures verified, and is the comparison constant-time?"
          />
        </label>
        <div className="composerBar">
          <label className="inlineField">
            <span>Code search</span>
            <select
              value={retrievalMode}
              onChange={(event) => setRetrievalMode(event.target.value as AgentRetrievalMode)}
            >
              <option value="filesystem">Filesystem</option>
              <option value="indexed">Indexed (experimental)</option>
            </select>
          </label>
          <span className="composerKeys" aria-hidden="true"><kbd>Ctrl/⌘</kbd><kbd>Enter</kbd></span>
          <button type="submit" className="btn btnPrimary" disabled={disabled}>
            {running ? <Spinner /> : <Icon name="search" />}
            {running ? "Investigating…" : "Run read-only investigation"}
          </button>
        </div>
      </form>
      {running && pending ? <PendingRun title="Investigation" status={status} /> : null}
      {result && !(running && pending) ? <InvestigationResult result={result} /> : null}
    </section>
  );
}

export function InvestigationResult({ result }: { result: AgentResponse }) {
  return (
    <article className="answer" aria-live="polite">
      <div className="answerHead">
        <h3>Investigation</h3>
        <span className={`badge ${result.status}`}>{humanize(result.status)}</span>
      </div>
      <RichText text={result.final_answer ?? "The agent did not return a final answer."} />
      <div className="chipRow">
        <span className="chip">{result.iterations} iterations</span>
        <span className="chip">{result.tool_execution_attempts} tool calls</span>
        <span className="chip">{result.llm_calls} model calls</span>
      </div>
      {result.evidence.length > 0 ? (
        <section className="answerSection evidence" aria-label="Evidence">
          <h4>Evidence</h4>
          <p className="hint">Locations the agent observed in the current repository.</p>
          <ul className="chipRow citations">
            {result.evidence.map((location) => (
              <CitationChip
                key={`${location.relative_path}:${location.start_line}-${location.end_line}`}
                label={locationLabel(location)}
                meta={observedViaLabels[location.observed_via]}
                reference={locationLabel(location)}
              />
            ))}
          </ul>
          {result.evidence_truncated ? (
            <small>Additional observed locations were omitted.</small>
          ) : null}
        </section>
      ) : null}
      {result.trace_run_id ? <small className="traceId">Trace {result.trace_run_id}</small> : null}
    </article>
  );
}
