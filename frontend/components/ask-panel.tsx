"use client";

import { useState } from "react";

import { lineRange } from "../lib/format";
import type { Citation, RAGResponse, RetrievalStrategy } from "../lib/types";
import { CitationChip } from "./citation-chip";
import { Icon, Spinner } from "./icons";
import { PendingRun } from "./pending-run";
import { RichText } from "./rich-text";

/** Matches the API default (repo-eval-v5, docs/evaluation.md); listed first. */
export const DEFAULT_STRATEGY: RetrievalStrategy = "hybrid";

export const STRATEGIES: ReadonlyArray<{ value: RetrievalStrategy; label: string; description: string }> = [
  {
    value: "hybrid",
    label: "Hybrid (default)",
    description: "Meaning-based and keyword search, fused. Finds both concepts and exact names.",
  },
  {
    value: "hybrid_rerank",
    label: "Hybrid + rerank",
    description: "Hybrid, then a model reorders the candidates. Adds one model call.",
  },
  {
    value: "hybrid_symbol",
    label: "Hybrid + symbol (experimental)",
    description: "Hybrid plus matches on class and function names you mention.",
  },
  {
    value: "semantic",
    label: "Semantic only",
    description: "Meaning-based search alone. Can miss exact identifiers and config keys.",
  },
];

function citationReference(citation: Citation): string {
  const range = citation.start_line === citation.end_line
    ? `${citation.start_line}`
    : `${citation.start_line}-${citation.end_line}`;
  return `${citation.relative_path}:${range}`;
}

/** Cmd/Ctrl+Enter submits the surrounding form from a textarea. */
export function submitOnModEnter(event: React.KeyboardEvent<HTMLTextAreaElement>) {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    event.currentTarget.form?.requestSubmit();
  }
}

export function AskPanel({
  disabled,
  running,
  status = "Waiting for a worker",
  onSubmit,
}: {
  disabled: boolean;
  running?: boolean;
  /** Latest progress step, shown on the pending answer while the job runs. */
  status?: string;
  onSubmit: (question: string, strategy: RetrievalStrategy) => Promise<RAGResponse | undefined>;
}) {
  const [question, setQuestion] = useState("");
  const [strategy, setStrategy] = useState<RetrievalStrategy>(DEFAULT_STRATEGY);
  const [pending, setPending] = useState<string | null>(null);
  const [history, setHistory] = useState<Array<{ question: string; answer: RAGResponse }>>([]);
  const strategyHint = STRATEGIES.find((item) => item.value === strategy)?.description;

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const requestedQuestion = question.trim();
    if (requestedQuestion.length === 0 || disabled) return;
    setPending(requestedQuestion);
    let answer: RAGResponse | undefined;
    try {
      answer = await onSubmit(requestedQuestion, strategy);
    } finally {
      setPending(null);
    }
    if (answer !== undefined) {
      // Newest first, so the answer appears right under the form.
      setHistory((items) => [{ question: requestedQuestion, answer }, ...items]);
      setQuestion("");
    }
  }

  return (
    <section className="modePanel">
      <header>
        <h2>Ask</h2>
        <p>Grounded answers with line-level citations from the indexed repository.</p>
      </header>
      <form className="composer" onSubmit={(event) => void submit(event)}>
        <label className="composerInput">
          <span className="srOnly">Question</span>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={submitOnModEnter}
            required
            maxLength={10_000}
            placeholder="How are refunds issued, and what stops one from exceeding the original payment?"
          />
        </label>
        <div className="composerBar">
          <label className="inlineField">
            <span>Retrieval</span>
            <select
              value={strategy}
              aria-describedby="strategy-hint"
              onChange={(event) => setStrategy(event.target.value as RetrievalStrategy)}
            >
              {STRATEGIES.map((item) => (
                <option key={item.value} value={item.value}>{item.label}</option>
              ))}
            </select>
          </label>
          <span className="composerKeys" aria-hidden="true"><kbd>Ctrl/⌘</kbd><kbd>Enter</kbd></span>
          <button type="submit" className="btn btnPrimary" disabled={disabled}>
            {running ? <Spinner /> : <Icon name="sparkles" />}
            {running ? "Answering…" : "Ask repository"}
          </button>
        </div>
      </form>
      <p id="strategy-hint" className="fieldHint">{strategyHint} <span className="srOnly">Press Ctrl or Command with Enter to ask.</span></p>
      <AskResults history={history} pending={running ? pending : null} status={status} />
    </section>
  );
}

function AskResults({
  history,
  pending,
  status,
}: {
  history: Array<{ question: string; answer: RAGResponse }>;
  pending: string | null;
  status: string;
}) {
  if (history.length === 0 && pending === null) {
    return <p className="hint">Q&amp;A history is stored only in this browser tab.</p>;
  }
  return (
    <div className="resultList" aria-live="polite">
      {pending !== null ? <PendingRun title={pending} status={status} /> : null}
      {history.map((item, index) => (
        <AnswerCard key={`${history.length - index}-${item.question}`} question={item.question} answer={item.answer} />
      ))}
    </div>
  );
}

export function AnswerCard({ question, answer }: { question: string; answer: RAGResponse }) {
  return (
    <article className="answer">
      <div className="answerHead">
        <h3>{question}</h3>
        {answer.insufficient_evidence ? (
          <span className="badge cancelled">Low evidence</span>
        ) : (
          <span className="badge completed"><Icon name="check" />Grounded</span>
        )}
      </div>
      {answer.insufficient_evidence ? (
        <p className="callout warning">
          <Icon name="alert" />
          Insufficient repository evidence.
        </p>
      ) : null}
      <RichText text={answer.answer} />
      {answer.citations.length > 0 ? (
        <div className="answerSection">
          <h4>Sources</h4>
          <ul className="chipRow citations">
            {answer.citations.map((citation) => (
              <CitationChip
                key={`${citation.relative_path}:${citation.start_line}`}
                label={citation.relative_path}
                meta={lineRange(citation.start_line, citation.end_line)}
                reference={citationReference(citation)}
              />
            ))}
          </ul>
        </div>
      ) : null}
      {answer.trace_run_id ? <small className="traceId">Trace {answer.trace_run_id}</small> : null}
    </article>
  );
}
