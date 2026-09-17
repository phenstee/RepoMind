"use client";

import { useState } from "react";

import type { CodingResponse } from "../lib/types";

interface CodingInput {
  objective: string;
  acceptance_criteria: string[];
  verification: { test_paths: string[]; ruff_paths: string[] };
}

function splitPaths(value: string): string[] {
  return value.split("\n").map((item) => item.trim()).filter(Boolean);
}

export function CodePanel({
  disabled,
  onSubmit,
}: {
  disabled: boolean;
  onSubmit: (request: CodingInput) => Promise<CodingResponse | undefined>;
}) {
  const [objective, setObjective] = useState("");
  const [criteria, setCriteria] = useState<string[]>([""]);
  const [testPaths, setTestPaths] = useState("tests");
  const [ruffPaths, setRuffPaths] = useState(".");
  const [confirmed, setConfirmed] = useState(false);
  const [result, setResult] = useState<CodingResponse | null>(null);

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!confirmed) return;
    const response = await onSubmit({
      objective: objective.trim(),
      acceptance_criteria: criteria.map((item) => item.trim()).filter(Boolean),
      verification: { test_paths: splitPaths(testPaths), ruff_paths: splitPaths(ruffPaths) },
    });
    if (response !== undefined) setResult(response);
  }

  return (
    <section className="modePanel dangerPanel">
      <header>
        <p className="eyebrow danger">Controlled mutation</p>
        <h2>Code</h2>
        <p className="warning">This operation may modify files in the selected repository and run configured verification checks.</p>
      </header>
      <form className="stack" onSubmit={(event) => void submit(event)}>
        <label>
          Objective
          <textarea value={objective} onChange={(event) => setObjective(event.target.value)} required maxLength={10_000} />
        </label>
        <fieldset className="criteria">
          <legend>Acceptance criteria</legend>
          {criteria.map((criterion, index) => (
            <div className="criterion" key={index}>
              <input
                aria-label={`Acceptance criterion ${index + 1}`}
                value={criterion}
                onChange={(event) => setCriteria((items) => items.map((item, i) => i === index ? event.target.value : item))}
                maxLength={2000}
              />
              <button
                type="button"
                disabled={criteria.length === 1}
                onClick={() => setCriteria((items) => items.filter((_, i) => i !== index))}
              >
                Remove
              </button>
            </div>
          ))}
          <button type="button" onClick={() => setCriteria((items) => [...items, ""])} disabled={criteria.length >= 50}>
            Add criterion
          </button>
        </fieldset>
        <div className="verificationGrid">
          <label>
            Pytest paths (one relative path per line)
            <textarea value={testPaths} onChange={(event) => setTestPaths(event.target.value)} required />
          </label>
          <label>
            Ruff paths (one relative path per line)
            <textarea value={ruffPaths} onChange={(event) => setRuffPaths(event.target.value)} required />
          </label>
        </div>
        <label className="confirmation">
          <input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} />
          I understand this controlled workflow may modify repository files.
        </label>
        <button type="submit" disabled={disabled || !confirmed}>Start controlled coding run</button>
      </form>
      {result ? <CodingResult result={result} /> : null}
    </section>
  );
}

export function CodingResult({ result }: { result: CodingResponse }) {
  return (
    <article className="answer">
      <h3>{result.status}</h3>
      <p>{result.final_answer ?? "No final coding answer was returned."}</p>
      <p>Revision {result.workspace_revision} · {result.completion_attempts} completion attempts</p>
      <p>Tests: {verificationLabel(result.tests)} · Ruff: {verificationLabel(result.ruff)}</p>
      {result.plan ? (
        <section aria-label="Coding plan">
          <h4>Plan</h4>
          <p>{result.plan.task_summary}</p>
          <ol>
            {result.plan.steps.map((step) => <li key={step.step_id}>{step.action}</li>)}
          </ol>
          {result.plan.acceptance_coverage.length > 0 ? (
            <ul>
              {result.plan.acceptance_coverage.map((coverage) => (
                <li key={coverage.criterion_index}>
                  Criterion {coverage.criterion_index + 1} → {coverage.step_ids.length > 0
                    ? `Step${coverage.step_ids.length === 1 ? "" : "s"} ${coverage.step_ids.join(", ")}`
                    : coverage.uncertainty}
                </li>
              ))}
            </ul>
          ) : null}
        </section>
      ) : null}
      {result.review ? (
        <section aria-label="Coding review">
          <h4>Reviewer: {result.review.verdict === "approve" ? "Approved" : "Changes required"}</h4>
          <p>{result.review_attempts} review attempt{result.review_attempts === 1 ? "" : "s"}</p>
          {result.review.findings.length > 0 ? (
            <ul>{result.review.findings.map((finding) => <li key={finding}>{finding}</li>)}</ul>
          ) : null}
          {result.review.required_corrections.length > 0 ? (
            <ul>{result.review.required_corrections.map((item) => <li key={item}>{item}</li>)}</ul>
          ) : null}
        </section>
      ) : null}
      {result.changed_files.length > 0 ? <ul className="citations">{result.changed_files.map((path) => <li key={path}><code>{path}</code></li>)}</ul> : null}
      {result.trace_run_id ? <small>Trace {result.trace_run_id}</small> : null}
    </article>
  );
}

function verificationLabel(summary: CodingResponse["tests"]): string {
  if (summary.execution_failed) return "execution failed";
  if (summary.passed === true) return "passed";
  if (summary.passed === false) return "failed";
  return "not run";
}
