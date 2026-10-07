"use client";

import { useState } from "react";

import { humanize } from "../lib/format";
import type { CodingResponse } from "../lib/types";
import { Icon, Spinner } from "./icons";
import { RichText } from "./rich-text";

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
  running,
  onSubmit,
}: {
  disabled: boolean;
  running?: boolean;
  onSubmit: (request: CodingInput) => Promise<CodingResponse | undefined>;
}) {
  const [objective, setObjective] = useState("");
  const [criteria, setCriteria] = useState<string[]>([""]);
  const [testPaths, setTestPaths] = useState("tests");
  const [ruffPaths, setRuffPaths] = useState(".");
  const [confirmed, setConfirmed] = useState(false);
  const [result, setResult] = useState<CodingResponse | null>(null);

  const pathsValid = splitPaths(testPaths).length > 0 && splitPaths(ruffPaths).length > 0;

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!confirmed || !pathsValid || disabled) return;
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
        <p>Plans, edits, verifies with pytest and Ruff, then passes an independent review before completing.</p>
      </header>
      <p className="callout warning">
        <Icon name="alert" />
        This operation may modify files in the selected repository and run configured verification checks.
      </p>
      <form className="stack" onSubmit={(event) => void submit(event)}>
        <label>
          Objective
          <textarea
            value={objective}
            onChange={(event) => setObjective(event.target.value)}
            required
            maxLength={10_000}
            placeholder="Reject partial refunds that exceed the remaining refundable balance."
          />
        </label>
        <fieldset>
          <legend>Acceptance criteria</legend>
          <div className="criteriaList">
            {criteria.map((criterion, index) => (
              <div className="criterion" key={index}>
                <span aria-hidden="true">{index + 1}</span>
                <input
                  aria-label={`Acceptance criterion ${index + 1}`}
                  value={criterion}
                  onChange={(event) => setCriteria((items) => items.map((item, i) => i === index ? event.target.value : item))}
                  maxLength={2000}
                  placeholder="Describe one observable outcome"
                />
                <button
                  type="button"
                  className="btn btnGhost btnIcon btnSm"
                  aria-label={`Remove criterion ${index + 1}`}
                  title="Remove"
                  disabled={criteria.length === 1}
                  onClick={() => setCriteria((items) => items.filter((_, i) => i !== index))}
                >
                  <Icon name="trash" size={14} />
                </button>
              </div>
            ))}
            <div>
              <button
                type="button"
                className="btn btnSecondary btnSm"
                onClick={() => setCriteria((items) => [...items, ""])}
                disabled={criteria.length >= 50}
              >
                <Icon name="plus" size={14} />
                Add criterion
              </button>
            </div>
          </div>
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
        {!pathsValid ? (
          <p className="callout danger">
            <Icon name="alert" />
            Enter at least one Pytest path and one Ruff path.
          </p>
        ) : null}
        <label className="confirmation">
          <input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} />
          I understand this controlled workflow may modify repository files.
        </label>
        <div>
          <button type="submit" className="btn btnDanger" disabled={disabled || !confirmed || !pathsValid}>
            {running ? <Spinner /> : <Icon name="play" size={14} />}
            {running ? "Coding run in progress…" : "Start controlled coding run"}
          </button>
        </div>
      </form>
      {result ? <CodingResult result={result} /> : null}
    </section>
  );
}

export function CodingResult({ result }: { result: CodingResponse }) {
  return (
    <article className="answer" aria-live="polite">
      <div className="answerHead">
        <h3>Coding run</h3>
        <span className={`badge ${result.status === "completed" ? "completed" : "failed"}`}>
          {humanize(result.status)}
        </span>
      </div>
      <RichText text={result.final_answer ?? "No final coding answer was returned."} />
      <div className="chipRow">
        <VerificationChip label="Tests" summary={result.tests} />
        <VerificationChip label="Ruff" summary={result.ruff} />
        <span className="chip">Revision {result.workspace_revision}</span>
        <span className="chip">
          {result.completion_attempts} completion attempt{result.completion_attempts === 1 ? "" : "s"}
        </span>
      </div>
      {result.plan ? (
        <section className="answerSection" aria-label="Coding plan">
          <h4>Plan</h4>
          <p className="muted">{result.plan.task_summary}</p>
          <ol className="stepList">
            {result.plan.steps.map((step) => <li key={step.step_id}><span>{step.action}</span></li>)}
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
        <section className="answerSection" aria-label="Coding review">
          <div className="answerHead">
            <h4>Reviewer: {result.review.verdict === "approve" ? "Approved" : "Changes required"}</h4>
            <span className={`badge ${result.review.verdict === "approve" ? "completed" : "blocked"}`}>
              {result.review_attempts} review attempt{result.review_attempts === 1 ? "" : "s"}
            </span>
          </div>
          {result.review.findings.length > 0 ? (
            <ul>{result.review.findings.map((finding) => <li key={finding}>{finding}</li>)}</ul>
          ) : null}
          {result.review.required_corrections.length > 0 ? (
            <ul>{result.review.required_corrections.map((item) => <li key={item}>{item}</li>)}</ul>
          ) : null}
        </section>
      ) : null}
      {result.changed_files.length > 0 ? (
        <section className="answerSection" aria-label="Changed files">
          <h4>Changed files</h4>
          <ul className="chipRow citations">
            {result.changed_files.map((path) => (
              <li key={path} className="chip">
                <Icon name="file" />
                <code>{path}</code>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
      {result.trace_run_id ? <small className="traceId">Trace {result.trace_run_id}</small> : null}
    </article>
  );
}

function VerificationChip({ label, summary }: { label: string; summary: CodingResponse["tests"] }) {
  const state = verificationLabel(summary);
  const tone = state === "passed" ? "completed" : state === "not run" ? "" : "failed";
  return (
    <span className={`badge ${tone}`}>
      <Icon name={state === "passed" ? "check" : state === "not run" ? "clock" : "x"} />
      {label}: {state}
    </span>
  );
}

function verificationLabel(summary: CodingResponse["tests"]): string {
  if (summary.execution_failed) return "execution failed";
  if (summary.passed === true) return "passed";
  if (summary.passed === false) return "failed";
  return "not run";
}
