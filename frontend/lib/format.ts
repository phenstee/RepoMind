const RUN_TYPE_LABELS: Record<string, string> = {
  rag: "Ask",
  agent: "Investigate",
  coding: "Code",
  index: "Index",
};

/** Human label for a persisted run/job type; unknown types pass through unchanged. */
export function runTypeLabel(type: string): string {
  return RUN_TYPE_LABELS[type] ?? type;
}

/** Compact duration: "820 ms", "3.8 s", "42 s", "1 m 38 s"; null means still running. */
export function formatDuration(ms: number | null): string {
  if (ms === null) return "running";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 10_000) return `${(ms / 1000).toFixed(1)} s`;
  if (ms < 60_000) return `${Math.round(ms / 1000)} s`;
  const totalSeconds = Math.round(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return seconds === 0 ? `${minutes} m` : `${minutes} m ${seconds} s`;
}

/** Relative time for recent timestamps, falling back to a short date. */
export function formatRelative(iso: string, now: number = Date.now()): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const seconds = Math.max(0, Math.round((now - then) / 1000));
  if (seconds < 45) return "just now";
  if (seconds < 3600) return `${Math.max(1, Math.round(seconds / 60))} min ago`;
  if (seconds < 86_400) return `${Math.round(seconds / 3600)} h ago`;
  return new Date(then).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/** "verification_failed" → "Verification failed". */
export function humanize(value: string): string {
  const text = value.replaceAll("_", " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** "src/app.py" lines 22–61 → "L22–61"; a single line → "L22". */
export function lineRange(start: number, end: number): string {
  return start === end ? `L${start}` : `L${start}–${end}`;
}
