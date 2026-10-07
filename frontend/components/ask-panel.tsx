"use client";

import { useState } from "react";

import { lineRange } from "../lib/format";
import type { RAGResponse, RetrievalStrategy } from "../lib/types";
import { Icon, Spinner } from "./icons";
import { RichText } from "./rich-text";

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
  onSubmit,
}: {
  disabled: boolean;
  running?: boolean;
  onSubmit: (question: string, strategy: RetrievalStrategy) => Promise<RAGResponse | undefined>;
}) {
  const [question, setQuestion] = useState("");
  const [strategy, setStrategy] = useState<RetrievalStrategy>("semantic");
  const [history, setHistory] = useState<Array<{ question: string; answer: RAGResponse }>>([]);

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const requestedQuestion = question.trim();
    if (requestedQuestion.length === 0 || disabled) return;
    const answer = await onSubmit(requestedQuestion, strategy);
    if (answer !== undefined) {
      // Newest first, so the answer appears right under the form.
      setHistory((items) => [{ question: requestedQuestion, answer }, ...items]);
      setQuestion("");
    }
  }

  return (
    <section className="modePanel">
      <header>
        <p className="eyebrow">Repository Q&amp;A</p>
        <h2>Ask</h2>
        <p>Grounded answers with line-level citations from the indexed repository.</p>
      </header>
      <form className="stack" onSubmit={(event) => void submit(event)}>
        <label>
          Question
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={submitOnModEnter}
            required
            maxLength={10_000}
            placeholder="How are refunds issued, and what stops one from exceeding the original payment?"
          />
        </label>
        <div className="formFooter">
          <label>
            Retrieval strategy
            <select value={strategy} onChange={(event) => setStrategy(event.target.value as RetrievalStrategy)}>
              <option value="semantic">Semantic</option>
              <option value="hybrid">Hybrid</option>
              <option value="hybrid_rerank">Hybrid + rerank</option>
              <option value="hybrid_symbol">Hybrid + symbol (experimental)</option>
            </select>
          </label>
          <button type="submit" className="btn btnPrimary" disabled={disabled}>
            {running ? <Spinner /> : <Icon name="sparkles" />}
            {running ? "Answering…" : "Ask repository"}
          </button>
        </div>
        <p className="hint"><kbd>Ctrl</kbd> / <kbd>⌘</kbd> + <kbd>Enter</kbd> to ask</p>
      </form>
      <AskResults history={history} />
    </section>
  );
}

function AskResults({ history }: { history: Array<{ question: string; answer: RAGResponse }> }) {
  if (history.length === 0) {
    return <p className="hint">Q&amp;A history is stored only in this browser tab.</p>;
  }
  return (
    <div className="resultList" aria-live="polite">
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
              <li key={`${citation.relative_path}:${citation.start_line}`} className="chip" title={citation.relative_path}>
                <Icon name="file" />
                <code>{citation.relative_path}</code>
                <span className="chipMeta">{lineRange(citation.start_line, citation.end_line)}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {answer.trace_run_id ? <small className="traceId">Trace {answer.trace_run_id}</small> : null}
    </article>
  );
}
