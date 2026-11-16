/**
 * Placeholder that holds the result's place while a job runs, so progress shows up where the
 * answer will land instead of only in the timeline further down the page.
 */
export function PendingRun({ title, status }: { title: string; status: string }) {
  return (
    <article className="answer pending" aria-busy="true">
      <div className="answerHead">
        <h3>{title}</h3>
      </div>
      <p className="thinking" role="status">
        {/* Keyed by status: each new step remounts and plays the enter swap. */}
        <span key={status} className="t-think-text" data-text={status}>
          {status}
        </span>
      </p>
      <div className="skeletonLines" aria-hidden="true">
        <span className="skeleton" style={{ width: "94%" }} />
        <span className="skeleton" style={{ width: "86%" }} />
        <span className="skeleton" style={{ width: "58%" }} />
      </div>
    </article>
  );
}
