"use client";

import { useEffect, useState } from "react";

import { api, ApiError } from "../lib/api";
import type { ProgressEvent, RunDetail, RunSummary } from "../lib/types";
import { ProgressTimeline } from "./progress-timeline";

function historyEvents(run: RunDetail): ProgressEvent[] {
  return run.events.map((event) => ({
    run_id: run.run_id,
    sequence: event.sequence,
    event: event.event_type,
    timestamp: event.timestamp,
    data: event.duration_ms === null ? event.metadata : { ...event.metadata, duration_ms: event.duration_ms },
  }));
}

function safeError(error: unknown): string {
  return error instanceof ApiError ? `${error.code}: ${error.message}` : "Run history is unavailable.";
}

export function RunHistory({ refreshKey }: { refreshKey: number }) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selected, setSelected] = useState<RunDetail | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  async function refresh() {
    try {
      const response = await api.runs();
      setRuns(response.runs);
      setMessage(null);
    } catch (error) {
      setMessage(safeError(error));
    }
  }

  async function choose(runId: string) {
    try {
      setSelected(await api.run(runId));
      setMessage(null);
    } catch (error) {
      setMessage(safeError(error));
    }
  }

  useEffect(() => {
    const timer = window.setTimeout(() => void refresh(), 0);
    return () => window.clearTimeout(timer);
  }, [refreshKey]);

  return (
    <section className="historyPanel">
      <div className="panelTitle">
        <div>
          <p className="eyebrow">Persisted traces</p>
          <h2>Run history</h2>
        </div>
        <button type="button" onClick={() => void refresh()}>Refresh</button>
      </div>
      {message ? <p className="error" role="alert">{message}</p> : null}
      <div className="historyGrid">
        <ul className="runList">
          {runs.map((run) => (
            <li key={run.run_id}>
              <button type="button" onClick={() => void choose(run.run_id)}>
                <strong>{run.run_type}</strong>
                <span>{run.status} · {run.duration_ms === null ? "running" : `${Math.round(run.duration_ms)} ms`}</span>
                <small>{run.llm_calls} model / {run.tool_calls} tool calls</small>
              </button>
            </li>
          ))}
        </ul>
        <div>
          {selected ? (
            <>
              <p className="muted">{selected.run_id}</p>
              <ProgressTimeline events={historyEvents(selected)} emptyLabel="This run has no public timeline events." />
            </>
          ) : <p className="muted">Select a completed run to inspect its sanitized timeline.</p>}
        </div>
      </div>
    </section>
  );
}
