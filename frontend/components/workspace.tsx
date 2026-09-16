"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { api, ApiError, getJobEvents } from "../lib/api";
import type {
  AgentResponse,
  CodingResponse,
  IndexResponse,
  JobDetail,
  ProgressEvent,
  RAGResponse,
  Repository,
  RepositoryFile,
  RetrievalStrategy,
} from "../lib/types";
import { AskPanel } from "./ask-panel";
import { CodePanel } from "./code-panel";
import { InvestigatePanel } from "./investigate-panel";
import { ProgressTimeline } from "./progress-timeline";
import { RepositoryPanel } from "./repository-panel";
import { RunHistory } from "./run-history";

type Mode = "ask" | "investigate" | "code";
type OperationState = "running" | "completed" | "failed" | "disconnected";

interface Operation {
  id: number;
  kind: string;
  state: OperationState;
}

function safeError(error: unknown): string {
  if (error instanceof ApiError) return `${error.code}: ${error.message}`;
  return "The request could not be completed. Check the local API connection and try again.";
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
  const [recoveredJob, setRecoveredJob] = useState<JobDetail | null>(null);
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

  useEffect(() => {
    let current = true;
    const timer = window.setTimeout(() => {
      void refreshRepositories();
      void api.health().then(() => setHealth("online")).catch(() => setHealth("offline"));
      const remembered = localStorage.getItem("repomind.activeJob");
      if (remembered) {
        void api.job(remembered).then(async (job) => {
          if (!current) return;
          setActiveJobId(job.job_id);
          setRecoveredJob(job);
          if (job.status === "succeeded" || job.status === "failed") {
            localStorage.removeItem("repomind.activeJob");
            setActiveJobId(null);
            return;
          }

          const id = activeOperation.current + 1;
          activeOperation.current = id;
          const nextController = new AbortController();
          controller.current = nextController;
          setOperation({ id, kind: job.job_type, state: "running" });
          try {
            await getJobEvents(job.job_id, {
              signal: nextController.signal,
              onProgress: (event) => {
                if (current && activeOperation.current === id) setEvents((items) => [...items, event]);
              },
              onResult: () => undefined,
            });
            const completed = await api.job(job.job_id);
            if (!current || activeOperation.current !== id) return;
            setRecoveredJob(completed);
            setOperation({ id, kind: job.job_type, state: completed.status === "succeeded" ? "completed" : "failed" });
            if (completed.status === "succeeded" || completed.status === "failed") {
              localStorage.removeItem("repomind.activeJob");
              setActiveJobId(null);
              setHistoryKey((value) => value + 1);
            }
          } catch (caught) {
            if (!current || (caught instanceof DOMException && caught.name === "AbortError")) return;
            const latest = await api.job(job.job_id).catch(() => null);
            if (!current || activeOperation.current !== id) return;
            if (latest !== null) {
              setRecoveredJob(latest);
              setOperation({ id, kind: job.job_type, state: latest.status === "failed" ? "failed" : "disconnected" });
              if (latest.status === "succeeded" || latest.status === "failed") {
                localStorage.removeItem("repomind.activeJob");
                setActiveJobId(null);
              }
            } else {
              setOperation({ id, kind: job.job_type, state: "disconnected" });
            }
          }
        }).catch(() => localStorage.removeItem("repomind.activeJob"));
      }
    }, 0);
    return () => {
      current = false;
      window.clearTimeout(timer);
      controller.current?.abort();
    };
  }, [refreshRepositories]);

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

  const runStream = useCallback(async <T,>(path: string, body: unknown): Promise<T | undefined> => {
    if (selectedId === null || operation?.state === "running") return undefined;
    const id = activeOperation.current + 1;
    activeOperation.current = id;
    const nextController = new AbortController();
    controller.current = nextController;
    setError(null);
    setEvents([]);
    const jobType = path.includes("index") ? "index" : mode === "ask" ? "rag" : mode === "investigate" ? "agent" : "coding";
    setOperation({ id, kind: jobType, state: "running" });
    let result: T | undefined;
    try {
      const job = await api.createJob(selectedId, jobType, jobType === "index" ? undefined : body);
      localStorage.setItem("repomind.activeJob", job.job_id);
      setActiveJobId(job.job_id);
      await getJobEvents<T>(job.job_id, {
        signal: nextController.signal,
        onProgress: (event) => {
          if (activeOperation.current === id) setEvents((items) => [...items, event]);
        },
        onResult: (value) => { result = value; },
      });
      if (activeOperation.current === id) {
        setOperation({ id, kind: path, state: "completed" });
        setHistoryKey((value) => value + 1);
        localStorage.removeItem("repomind.activeJob");
        setActiveJobId(null);
        setRecoveredJob(await api.job(job.job_id));
      }
      return result;
    } catch (caught) {
      if (activeOperation.current === id) {
        if (caught instanceof DOMException && caught.name === "AbortError") {
          setOperation({ id, kind: path, state: "disconnected" });
        } else {
          setOperation({ id, kind: path, state: "failed" });
          setError(safeError(caught));
        }
      }
      return undefined;
    } finally {
      if (activeOperation.current === id) controller.current = null;
    }
  }, [mode, operation?.state, selectedId]);

  function selectRepository(id: number | null) {
    controller.current?.abort();
    setSelectedId(id);
    setFiles([]);
    setLoadingFiles(false);
    setIndexResult(null);
    setEvents([]);
    setOperation(null);
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
    const result = await runStream<IndexResponse>(`/repositories/${selectedId}/index/stream`, {});
    if (result !== undefined) {
      setIndexResult(result);
      const response = await api.files(selectedId).catch(() => null);
      if (response !== null) setFiles(response.files);
    }
  }

  const selectedRepository = repositories.find((repository) => repository.id === selectedId) ?? null;
  const running = operation?.state === "running";

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
          </div>
          {operation?.state === "disconnected" ? <p className="warning">Progress viewing stopped. The backend operation may still be running.</p> : null}
          {activeJobId ? <p className="muted">Durable job: {activeJobId}</p> : null}
          {recoveredJob ? <p className="muted">Recovered job {recoveredJob.status}{recoveredJob.trace_run_id ? ` · trace ${recoveredJob.trace_run_id}` : ""}</p> : null}
          <nav className="modeTabs" aria-label="Workspace mode">
            {(["ask", "investigate", "code"] as Mode[]).map((item) => (
              <button key={item} type="button" className={mode === item ? "active" : ""} onClick={() => setMode(item)} disabled={running}>
                {item === "ask" ? "Ask" : item === "investigate" ? "Investigate" : "Code"}
              </button>
            ))}
          </nav>
          {selectedId === null ? <p className="emptyState">Register or select a repository to begin.</p> : null}
          {selectedId !== null && mode === "ask" ? <AskPanel disabled={running} onSubmit={(question, strategy) => runStream<RAGResponse>(`/repositories/${selectedId}/rag/stream`, { question, strategy })} /> : null}
          {selectedId !== null && mode === "investigate" ? <InvestigatePanel disabled={running} onSubmit={(query) => runStream<AgentResponse>(`/repositories/${selectedId}/agent/runs/stream`, { query })} /> : null}
          {selectedId !== null && mode === "code" ? <CodePanel disabled={running} onSubmit={(request) => runStream<CodingResponse>(`/repositories/${selectedId}/coding/runs/stream`, request)} /> : null}
          <section className="liveProgress">
            <div className="panelTitle"><div><p className="eyebrow">Live progress</p><h2>{operation?.kind ?? "No active operation"}</h2></div><span className={`status ${operation?.state ?? "idle"}`}>{operation?.state ?? "idle"}</span></div>
            <ProgressTimeline events={events} />
          </section>
        </section>
      </div>
      <RunHistory refreshKey={historyKey} />
    </main>
  );
}
