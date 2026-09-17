import type { JsonValue, ProgressEvent } from "../lib/types";

const labels: Record<string, string> = {
  "run.started": "Run started",
  "run.completed": "Run completed",
  "run.failed": "Run failed",
  "index.started": "Indexing started",
  "ingestion.completed": "Repository ingested",
  "chunking.completed": "Chunks created",
  "embedding.started": "Embedding started",
  "embedding.completed": "Embedding completed",
  "persistence.completed": "Index persisted",
  "retrieval.started": "Retrieval started",
  "retrieval.completed": "Retrieval completed",
  "rag.context": "Context constructed",
  "rag.answer": "Answer generated",
  "agent.decision": "Agent decision",
  "tool.started": "Tool started",
  "tool.completed": "Tool completed",
  "tool.failed": "Tool failed",
  "file.mutated": "File changed",
  "preflight.started": "Preflight started",
  "preflight.passed": "Preflight passed",
  "preflight.failed": "Preflight blocked",
  "planning.started": "Planning started",
  "planning.completed": "Planning completed",
  "planning.failed": "Planning failed",
  "verification.started": "Verification started",
  "verification.completed": "Verification completed",
  "completion.requested": "Completion requested",
  "completion.blocked": "Completion blocked",
  "completion.completed": "Completion accepted",
  "final_review.started": "Final review started",
  "final_review.completed": "Final review completed",
  "review.started": "Independent review started",
  "review.completed": "Independent review completed",
  "review.blocked": "Review: changes required",
  "review.failed": "Independent review failed",
};

const blockerLabels: Record<string, string> = {
  tests_failed: "Tests failed",
  tests_stale: "Tests need to be re-run",
  ruff_failed: "Ruff failed",
  review_stale: "Final review needs to be re-run",
  unexpected_files: "Unexpected files changed",
  review_changes_required: "Independent review requires changes",
  review_context_too_large: "Review context is too large",
  review_failed: "Independent review failed",
};

function stateFor(event: string): string {
  if (event.includes("failed")) return "failure";
  if (event.includes("blocked")) return "blocked";
  if (event.includes("completed") || event.includes("passed")) return "success";
  return "running";
}

function displayValue(value: JsonValue): string | null {
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value) && value.every((item) => typeof item === "string")) {
    return value
      .map((item) => blockerLabels[item] ?? item)
      .join(", ");
  }
  return null;
}

function eventMetadata(data: Record<string, JsonValue>): string[] {
  const visible = [
    "tool",
    "path",
    "paths",
    "strategy",
    "workspace_revision",
    "completion_attempt",
    "blocker_codes",
    "file_count",
    "chunk_count",
    "embedding_model",
    "passed",
    "verdict",
    "criteria_satisfied",
    "criteria_unsatisfied",
    "finding_count",
  ];
  return visible.flatMap((key) => {
    const value = data[key];
    const rendered = value === undefined ? null : displayValue(value);
    return rendered === null ? [] : [`${key.replaceAll("_", " ")}: ${rendered}`];
  });
}

export function ProgressTimeline({
  events,
  emptyLabel = "Progress events will appear here.",
}: {
  events: ProgressEvent[];
  emptyLabel?: string;
}) {
  if (events.length === 0) {
    return <p className="muted">{emptyLabel}</p>;
  }
  return (
    <ol className="timeline" aria-label="Progress timeline">
      {events.map((entry) => (
        <li className={`timelineItem ${stateFor(entry.event)}`} key={`${entry.run_id}-${entry.sequence}`}>
          <div className="timelineHead">
            <strong>{labels[entry.event] ?? entry.event}</strong>
            <time dateTime={entry.timestamp}>{new Date(entry.timestamp).toLocaleTimeString()}</time>
          </div>
          {eventMetadata(entry.data).map((item) => (
            <span className="metadata" key={item}>
              {item}
            </span>
          ))}
        </li>
      ))}
    </ol>
  );
}
