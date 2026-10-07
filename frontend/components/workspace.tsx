"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { api, ApiError, getJobEvents } from "../lib/api";
import { canCancelJob, cancellationMessage, followJob, isTerminalJob, requestJobCancellation } from "../lib/jobs";
import type {
  AgentResponse,
  CodingResponse,
  IndexResponse,
  JobDetail,
  JobType,
  ProgressEvent,
  RAGResponse,
  Repository,
  RepositoryFile,
} from "../lib/types";
import { AnswerCard, AskPanel } from "./ask-panel";
import { CodePanel, CodingResult } from "./code-panel";
import { InvestigatePanel, InvestigationResult } from "./investigate-panel";
import { ProgressTimeline } from "./progress-timeline";
import { RepositoryPanel } from "./repository-panel";
import { RunHistory } from "./run-history";

type Mode = "ask" | "investigate" | "code";
type OperationState = "running" | "completed" | "failed" | "cancelled" | "disconnected";

interface Operation {
  id: number;
  kind: JobType;
  state: OperationState;
}

type RecoveredResult =
  | { jobType: "rag"; result: RAGResponse }
  | { jobType: "agent"; result: AgentResponse }
  | { jobType: "coding"; result: CodingResponse };

const ACTIVE_JOB_KEY = "repomind.activeJob";

function rememberActiveJob(jobId: string | null) {
  try {
    if (jobId === null) localStorage.removeItem(ACTIVE_JOB_KEY);
    else localStorage.setItem(ACTIVE_JOB_KEY, jobId);
  } catch {
    // Storage can be unavailable (private mode, blocked site data); recovery is best-effort.
  }
}

function rememberedActiveJob(): string | null {
  try {
    return localStorage.getItem(ACTIVE_JOB_KEY);
  } catch {
    return null;
  }
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function safeError(error: unknown): string {
  if (error instanceof ApiError) return `${error.code}: ${error.message}`;
  return "The request could not be completed. Check the local API connection and try again.";
}

function terminalState(job: JobDetail, hasResult: boolean): OperationState {
  if (job.status === "cancelled") return "cancelled";
  if (job.status === "succeeded" || hasResult) return "completed";
  if (job.status === "failed") return "failed";
  return "disconnected";
}

function recoveredResultFor(jobType: JobType, result: unknown): RecoveredResult | null {
  if (result === null || result === undefined || jobType === "index") return null;
  return { jobType, result } as RecoveredResult;
}

/** Keep a fresher terminal snapshot instead of overwriting it with a stale in-flight one. */
function newerJob(current: JobDetail | null, next: JobDetail): JobDetail {
  if (current !== null && current.job_id === next.job_id && isTerminalJob(current.status) && !isTerminalJob(next.status)) {
    return current;
  }
  return next;
}

export function Workspace() {
  const [repositories, setRepositories] = useState<Repository[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [files, setFiles] = useState<RepositoryFile[]>([]);
  const [loadingFiles, setLoadingFiles] = useState(false);
  const [mode, setMode] = useState<Mode>("ask");
  const [health, setHealth] = useState<"checking" | "online" | "offline">("checking");
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [operation, setOperation] = useState<Operation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [indexResult, setIndexResult] = useState<IndexResponse | null>(null);
  const [historyKey, setHistoryKey] = useState(0);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [trackedJob, setTrackedJob] = useState<JobDetail | null>(null);
  const [recoveredResult, setRecoveredResult] = useState<RecoveredResult | null>(null);
  const activeOperation = useRef(0);
  const controller = useRef<AbortController | null>(null);

  const refreshRepositories = useCallback(async () => {
    try {
      const response = await api.repositories();
      setRepositories(response.repositories);
      setSelectedId((current) => current ?? response.repositories[0]?.id ?? null);
      setError(null);
    } catch (caught) {
      setError(safeError(caught));
    }
  }, []);

  /** Start a new operation generation; results from older generations are ignored. */
  const beginOperation = useCallback((kind: JobType): { id: number; signal: AbortSignal } => {
    controller.current?.abort();
    const id = activeOperation.current + 1;
    activeOperation.current = id;
    const nextController = new AbortController();
    controller.current = nextController;
    setOperation({ id, kind, state: "running" });
    return { id, signal: nextController.signal };
  }, []);

  /**
   * Follow a durable job to a terminal state. Stream drops are reconnected inside followJob;
   * the stored job pointer is cleared only once PostgreSQL reports the job terminal.
   */
  const followActiveJob = useCallback(async <T,>(
    jobId: string,
    jobType: JobType,
    id: number,
    signal: AbortSignal,
  ): Promise<T | undefined> => {
    const isCurrent = () => activeOperation.current === id;
    const refreshJob = () => {
      void api.job(jobId)
        .then((latest) => { if (isCurrent()) setTrackedJob((current) => newerJob(current, latest)); })
        .catch(() => undefined);
    };
    try {
      const { job, result } = await followJob<T>({ job: api.job, events: getJobEvents }, jobId, {
        signal,
        onProgress: (event) => {
          if (!isCurrent()) return;
          setEvents((items) => [...items, event]);
          // Cancellation availability changes as the job progresses (e.g. once files mutate).
          if (event.event.startsWith("job.") || event.event === "file.mutated") refreshJob();
        },
      });
      if (!isCurrent()) return undefined;
      setTrackedJob(job);
      setOperation({ id, kind: jobType, state: terminalState(job, result !== undefined) });
      setHistoryKey((value) => value + 1);
      if (isTerminalJob(job.status)) {
        rememberActiveJob(null);
        setActiveJobId(null);
      }
      return result;
    } catch (caught) {
      if (!isCurrent()) return undefined;
      if (isAbort(caught)) {
        setOperation({ id, kind: jobType, state: "disconnected" });
        return undefined;
      }
      const latest = await api.job(jobId).catch(() => null);
      if (!isCurrent()) return undefined;
      if (latest !== null) setTrackedJob(latest);
      if (latest !== null && isTerminalJob(latest.status)) {
        rememberActiveJob(null);
        setActiveJobId(null);
        setHistoryKey((value) => value + 1);
        setOperation({ id, kind: jobType, state: terminalState(latest, false) });
      } else {
        setOperation({ id, kind: jobType, state: "disconnected" });
      }
      setError(safeError(caught));
      return undefined;
    } finally {
      if (isCurrent()) controller.current = null;
    }
  }, []);

  const attachToJob = useCallback(async (job: JobDetail) => {
    const { id, signal } = beginOperation(job.job_type);
    const result = await followActiveJob<unknown>(job.job_id, job.job_type, id, signal);
    if (activeOperation.current !== id) return;
    if (job.job_type === "index" && result !== undefined) setIndexResult(result as IndexResponse);
    else setRecoveredResult(recoveredResultFor(job.job_type, result));
  }, [beginOperation, followActiveJob]);

  useEffect(() => {
    let current = true;
    const timer = window.setTimeout(() => {
      void refreshRepositories();
      void api.health().then(() => setHealth("online")).catch(() => setHealth("offline"));
      const remembered = rememberedActiveJob();
      if (remembered) {
        void api.job(remembered).then(async (job) => {
          if (!current) return;
          setTrackedJob(job);
          setSelectedId(job.repository_id);
          if (isTerminalJob(job.status)) {
            rememberActiveJob(null);
            setRecoveredResult(recoveredResultFor(job.job_type, job.result));
            return;
          }
          setActiveJobId(job.job_id);
          await attachToJob(job);
        }).catch((caught) => {
          // Forget the pointer only when the API says the job does not exist; keep it when the
          // API is merely unreachable so a later reload can still recover the job.
          if (caught instanceof ApiError && caught.status !== null && caught.status >= 400 && caught.status < 500) {
            rememberActiveJob(null);
          }
        });
      }
    }, 0);
    return () => {
      current = false;
      window.clearTimeout(timer);
      activeOperation.current += 1;
      controller.current?.abort();
    };
  }, [attachToJob, refreshRepositories]);

  useEffect(() => {
    if (selectedId === null) {
      return;
    }
    let current = true;
    const timer = window.setTimeout(() => {
      setLoadingFiles(true);
      void api.files(selectedId)
        .then((response) => { if (current) setFiles(response.files); })
        .catch((caught) => { if (current) setError(safeError(caught)); })
        .finally(() => { if (current) setLoadingFiles(false); });
    }, 0);
    return () => {
      current = false;
      window.clearTimeout(timer);
    };
  }, [selectedId]);

  const runJob = useCallback(async <T,>(jobType: JobType, body?: unknown): Promise<T | undefined> => {
    if (selectedId === null || operation?.state === "running") return undefined;
    const { id, signal } = beginOperation(jobType);
    setError(null);
    setEvents([]);
    setTrackedJob(null);
    setRecoveredResult(null);
    let jobId: string;
    try {
      jobId = (await api.createJob(selectedId, jobType, body)).job_id;
    } catch (caught) {
      if (activeOperation.current === id) {
        setOperation({ id, kind: jobType, state: "failed" });
        setError(safeError(caught));
        controller.current = null;
      }
      return undefined;
    }
    rememberActiveJob(jobId);
    if (activeOperation.current !== id) return undefined;
    setActiveJobId(jobId);
    // Subscribe immediately: progress is pub/sub without replay, so any extra round trip before
    // subscribing loses early events. The job snapshot (for cancel controls) loads in parallel.
    void api.job(jobId)
      .then((job) => { if (activeOperation.current === id) setTrackedJob((current) => newerJob(current, job)); })
      .catch(() => undefined);
    return followActiveJob<T>(jobId, jobType, id, signal);
  }, [beginOperation, followActiveJob, operation?.state, selectedId]);

  function selectRepository(id: number | null) {
    // Invalidate the in-flight operation first so its abort cannot write stale state back.
    activeOperation.current += 1;
    controller.current?.abort();
    controller.current = null;
    setSelectedId(id);
    setFiles([]);
    setLoadingFiles(false);
    setIndexResult(null);
    setEvents([]);
    setOperation(null);
    setActiveJobId(null);
    setTrackedJob(null);
    setRecoveredResult(null);
  }

  async function registerRepository(name: string, path: string) {
    try {
      const repository = await api.registerRepository(name, path);
      await refreshRepositories();
      selectRepository(repository.id);
    } catch (caught) {
      setError(safeError(caught));
    }
  }

  async function indexRepository() {
    if (selectedId === null) return;
    const repositoryId = selectedId;
    const result = await runJob<IndexResponse>("index");
    if (result !== undefined) {
      setIndexResult(result);
      const response = await api.files(repositoryId).catch(() => null);
      if (response !== null) setFiles(response.files);
    }
  }

  async function cancelActiveJob() {
    if (activeJobId === null) return;
    try {
      setTrackedJob(await requestJobCancellation(api, activeJobId));
    } catch (caught) {
      setError(safeError(caught));
    }
  }

  function resumeProgress() {
    if (trackedJob === null || trackedJob.job_id !== activeJobId) return;
    setError(null);
    void attachToJob(trackedJob);
  }

  const selectedRepository = repositories.find((repository) => repository.id === selectedId) ?? null;
  const running = operation?.state === "running";
  const jobPending = activeJobId !== null && trackedJob !== null && !isTerminalJob(trackedJob.status);

  return (
    <main className="appShell">
      <header className="topbar">
        <div><span className="brandMark">RM</span><strong>RepoMind</strong><small>trusted local developer workspace</small></div>
        <button className={`health ${health}`} type="button" onClick={() => void api.health().then(() => setHealth("online")).catch(() => setHealth("offline"))}>
          <span /> API {health}
        </button>
      </header>
      {error ? <p className="error globalError" role="alert">{error}</p> : null}
      <div className="workbench">
        <RepositoryPanel
          repositories={repositories}
          selectedId={selectedId}
          files={files}
          loadingFiles={loadingFiles}
          running={running}
          indexResult={indexResult}
          onSelect={selectRepository}
          onRegister={registerRepository}
          onIndex={() => void indexRepository()}
        />
        <section className="workspacePanel">
          <div className="workspaceHeader">
            <div>
              <p className="eyebrow">Workspace</p>
              <h1>{selectedRepository?.name ?? "Select a repository"}</h1>
              {selectedRepository?.workspace_relative_path ? <p className="muted">{selectedRepository.workspace_relative_path}</p> : null}
            </div>
            {running ? <button type="button" className="secondary" onClick={() => controller.current?.abort()}>Stop viewing progress</button> : null}
            {!running && jobPending && operation?.state === "disconnected" ? <button type="button" className="secondary" onClick={resumeProgress}>Resume progress</button> : null}
            {jobPending && canCancelJob(trackedJob) ? <button type="button" className="secondary" onClick={() => void cancelActiveJob()}>Cancel job</button> : null}
          </div>
          {operation?.state === "disconnected" ? <p className="warning">Progress viewing stopped. The backend operation may still be running.</p> : null}
          {activeJobId ? <p className="muted">Durable job: {activeJobId}</p> : null}
          {cancellationMessage(trackedJob) ? <p className="warning">{cancellationMessage(trackedJob)}</p> : null}
          {trackedJob ? <p className="muted">Job {trackedJob.status}{trackedJob.trace_run_id ? ` · trace ${trackedJob.trace_run_id}` : ""}</p> : null}
          <nav className="modeTabs" aria-label="Workspace mode">
            {(["ask", "investigate", "code"] as Mode[]).map((item) => (
              <button key={item} type="button" className={mode === item ? "active" : ""} aria-pressed={mode === item} onClick={() => setMode(item)} disabled={running}>
                {item === "ask" ? "Ask" : item === "investigate" ? "Investigate" : "Code"}
              </button>
            ))}
          </nav>
          {selectedId === null ? <p className="emptyState">Register or select a repository to begin.</p> : null}
          {selectedId !== null && mode === "ask" ? <AskPanel key={selectedId} disabled={running} onSubmit={(question, strategy) => runJob<RAGResponse>("rag", { question, strategy })} /> : null}
          {selectedId !== null && mode === "investigate" ? <InvestigatePanel key={selectedId} disabled={running} onSubmit={(query, retrievalMode) => runJob<AgentResponse>("agent", { query, retrieval_mode: retrievalMode })} /> : null}
          {selectedId !== null && mode === "code" ? <CodePanel key={selectedId} disabled={running} onSubmit={(request) => runJob<CodingResponse>("coding", request)} /> : null}
          {recoveredResult !== null ? (
            <section className="modePanel" aria-live="polite">
              <header><p className="eyebrow">Recovered job result</p></header>
              {recoveredResult.jobType === "rag" ? <AnswerCard question="Recovered answer" answer={recoveredResult.result} /> : null}
              {recoveredResult.jobType === "agent" ? <InvestigationResult result={recoveredResult.result} /> : null}
              {recoveredResult.jobType === "coding" ? <CodingResult result={recoveredResult.result} /> : null}
            </section>
          ) : null}
          <section className="liveProgress">
            <div className="panelTitle"><div><p className="eyebrow">Live progress</p><h2>{operation?.kind ?? "No active operation"}</h2></div><span className={`status ${operation?.state ?? "idle"}`} role="status" aria-live="polite">{operation?.state ?? "idle"}</span></div>
            <ProgressTimeline events={events} />
          </section>
        </section>
      </div>
      <RunHistory refreshKey={historyKey} />
    </main>
  );
}
