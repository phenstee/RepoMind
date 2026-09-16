"use client";

import { useState } from "react";

import type { IndexResponse, Repository, RepositoryFile } from "../lib/types";

interface RepositoryPanelProps {
  repositories: Repository[];
  selectedId: number | null;
  files: RepositoryFile[];
  loadingFiles: boolean;
  running: boolean;
  indexResult: IndexResponse | null;
  onSelect: (id: number) => void;
  onRegister: (name: string, path: string) => Promise<void>;
  onIndex: () => void;
}

export function RepositoryPanel({
  repositories,
  selectedId,
  files,
  loadingFiles,
  running,
  indexResult,
  onSelect,
  onRegister,
  onIndex,
}: RepositoryPanelProps) {
  const [name, setName] = useState("");
  const [path, setPath] = useState("");
  const [registering, setRegistering] = useState(false);

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setRegistering(true);
    try {
      await onRegister(name, path);
      setName("");
      setPath("");
    } finally {
      setRegistering(false);
    }
  }

  return (
    <aside className="repositoryPanel" aria-label="Repository controls">
      <div className="panelTitle">
        <h2>Repositories</h2>
        <span>{repositories.length}</span>
      </div>
      <label>
        Selected repository
        <select value={selectedId ?? ""} onChange={(event) => onSelect(Number(event.target.value))}>
          <option value="">Select a registered repository</option>
          {repositories.map((repository) => (
            <option key={repository.id} value={repository.id}>
              {repository.name}
            </option>
          ))}
        </select>
      </label>

      <form className="stack" onSubmit={(event) => void submit(event)}>
        <h3>Register repository</h3>
        <label>
          Name
          <input value={name} onChange={(event) => setName(event.target.value)} required maxLength={255} />
        </label>
        <label>
          Workspace-relative path
          <input
            value={path}
            onChange={(event) => setPath(event.target.value)}
            placeholder="example-project"
            required
            maxLength={1024}
          />
        </label>
        <button type="submit" disabled={registering || running}>
          {registering ? "Registering…" : "Register"}
        </button>
      </form>

      <section className="stack">
        <div className="panelTitle">
          <h3>Indexed files</h3>
          <button type="button" onClick={onIndex} disabled={selectedId === null || running}>
            Index repository
          </button>
        </div>
        {loadingFiles ? <p className="muted">Loading file metadata…</p> : null}
        {!loadingFiles && files.length === 0 ? <p className="muted">Index a repository to see files.</p> : null}
        <ul className="fileList">
          {files.map((file) => (
            <li key={file.relative_path}>
              <code>{file.relative_path}</code>
              <small>
                {file.language ?? "text"} · {file.line_count} lines
              </small>
            </li>
          ))}
        </ul>
        {indexResult !== null ? (
          <p className="resultSummary">
            Indexed {indexResult.files_indexed} files / {indexResult.chunks_indexed} chunks
            {indexResult.embedding_model === null ? "" : ` · ${indexResult.embedding_model}`}
          </p>
        ) : null}
      </section>
    </aside>
  );
}
