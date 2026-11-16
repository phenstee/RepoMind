"use client";

import { useEffect, useRef, useState } from "react";

import { api, ApiError } from "../lib/api";
import { formatDuration, formatRelative, humanize, runTypeLabel } from "../lib/format";
import type { ProgressEvent, RunDetail, RunSummary } from "../lib/types";
import { Icon, type IconName } from "./icons";
import { ProgressTimeline } from "./progress-timeline";

const runIcons: Record<string, IconName> = {
  rag: "message",
  agent: "search",
  coding: "code",
  index: "database",
};

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
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const latestChoice = useRef<string | null>(null);

  async function refresh() {
    setLoading(true);
    try {
      const response = await api.runs();
      setRuns(response.runs);
      setMessage(null);
    } catch (error) {
      setMessage(safeError(error));
    } finally {
      setLoading(false);
    }
  }

  async function choose(runId: string) {
    latestChoice.current = runId;
    setSelectedId(runId);
    try {
      const run = await api.run(runId);
      // Ignore responses for runs the user has already clicked away from.
      if (latestChoice.current !== runId) return;
      setSelected(run);
      setMessage(null);
    } catch (error) {
      if (latestChoice.current === runId) setMessage(safeError(error));
    }
  }

  useEffect(() => {
    const timer = window.setTimeout(() => void refresh(), 0);
    return () => window.clearTimeout(timer);
  }, [refreshKey]);

  return (
    <section className="card historyPanel" aria-label="Run history">
      <header className="cardHeader">
        <div>
          <Icon name="clock" />
          <div>
            <p className="eyebrow">Persisted traces</p>
            <h2>Run history</h2>
          </div>
        </div>
        <button
          type="button"
          className={loading ? "btn btnGhost btnSm isSpinning" : "btn btnGhost btnSm"}
          onClick={() => void refresh()}
          disabled={loading}
        >
          <Icon name="refresh" size={14} />
          Refresh
        </button>
      </header>
      {message ? <p className="callout danger" role="alert" style={{ margin: "1rem 1.2rem 0" }}><Icon name="alert" />{message}</p> : null}
      <div className="historyGrid">
        {runs.length === 0 ? (
          <div className="emptyBlock">
            <Icon name="clock" />
            <p>No runs yet.</p>
            <small>Ask, investigate, or index to record a trace.</small>
          </div>
        ) : (
          <ul className="runList stagger">
            {runs.map((run) => {
              const active = run.run_id === selectedId;
              return (
                <li key={run.run_id}>
                  <button
                    type="button"
                    className={active ? "runItem active" : "runItem"}
                    aria-pressed={active}
                    onClick={() => void choose(run.run_id)}
                  >
                    <span className={`runIcon ${run.run_type}`} aria-hidden="true">
                      <Icon name={runIcons[run.run_type] ?? "sparkles"} />
                    </span>
                    <span className="runMain">
                      <strong>
                        {runTypeLabel(run.run_type)}
                        <span className={`statusDot ${run.ended_at === null ? "running" : run.status}`} aria-hidden="true" />
                        <span className="srOnly">{run.status}</span>
                      </strong>
                      <small>
                        {run.llm_calls} model · {run.tool_calls} tool calls
                        {run.domain_status ? ` · ${humanize(run.domain_status).toLowerCase()}` : ""}
                      </small>
                    </span>
                    <span className="runSide">
                      <span>{formatDuration(run.duration_ms)}</span>
                      <span>{formatRelative(run.started_at)}</span>
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
        <div className="runDetail">
          {selected ? (
            <>
              <div className="runDetailHead">
                <div>
                  <h3>{runTypeLabel(selected.run_type)} run</h3>
                  <small className="traceId">{selected.run_id}</small>
                </div>
                <span className={`badge ${selected.status}`}>{humanize(selected.status)}</span>
              </div>
              <div className="statGrid stagger">
                <div className="stat"><span>Duration</span><strong>{formatDuration(selected.duration_ms)}</strong></div>
                <div className="stat"><span>Model calls</span><strong>{selected.llm_calls}</strong></div>
                <div className="stat"><span>Tool calls</span><strong>{selected.tool_calls}</strong></div>
                <div className="stat"><span>Errors</span><strong>{selected.errors}</strong></div>
              </div>
              <ProgressTimeline key={selected.run_id} events={historyEvents(selected)} emptyLabel="This run has no public timeline events." />
            </>
          ) : (
            <div className="emptyBlock">
              <Icon name="eye" />
              <p>Select a run to inspect its sanitized timeline.</p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
