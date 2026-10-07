"use client";

import { useState } from "react";

import type { IndexResponse, Repository, RepositoryFile } from "../lib/types";
import { Icon, Spinner } from "./icons";

interface RepositoryPanelProps {
  repositories: Repository[];
  selectedId: number | null;
  files: RepositoryFile[];
  loadingFiles: boolean;
  running: boolean;
  indexing: boolean;
  indexResult: IndexResponse | null;
  onSelect: (id: number | null) => void;
  onRegister: (name: string, path: string) => Promise<boolean>;
  onIndex: () => void;
}

function initials(name: string): string {
  const parts = name.split(/[\s._-]+/).filter(Boolean);
  return (parts.length > 1 ? parts[0][0] + parts[1][0] : name.slice(0, 2)) || "?";
}

function languageClass(language: string | null): string {
  return (language ?? "text").toLowerCase().replace(/[^a-z0-9]/g, "");
}

export function RepositoryPanel({
  repositories,
  selectedId,
  files,
  loadingFiles,
  running,
  indexing,
  indexResult,
  onSelect,
  onRegister,
  onIndex,
}: RepositoryPanelProps) {
  const [name, setName] = useState("");
  const [path, setPath] = useState("");
  const [registering, setRegistering] = useState(false);
  const [registerOpen, setRegisterOpen] = useState(false);

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setRegistering(true);
    try {
      if (await onRegister(name.trim(), path.trim())) {
        setName("");
        setPath("");
        setRegisterOpen(false);
      }
    } finally {
      setRegistering(false);
    }
  }

  return (
    <aside className="sidebar" aria-label="Repository controls">
      <section className="card">
        <header className="cardHeader">
          <div>
            <Icon name="folder" />
            <h2>Repositories</h2>
          </div>
          <span className="countBadge" aria-label={`${repositories.length} registered`}>{repositories.length}</span>
        </header>
        {repositories.length === 0 ? (
          <div className="emptyBlock">
            <Icon name="folder" />
            <p>No repositories yet.</p>
            <small>Register one from your workspace directory below.</small>
          </div>
        ) : (
          <ul className="repoList stagger">
            {repositories.map((repository) => {
              const active = repository.id === selectedId;
              return (
                <li key={repository.id}>
                  <button
                    type="button"
                    className={active ? "repoItem active" : "repoItem"}
                    aria-pressed={active}
                    onClick={() => onSelect(repository.id)}
                  >
                    <span className="repoAvatar" aria-hidden="true">{initials(repository.name)}</span>
                    <span className="repoMeta">
                      <strong>{repository.name}</strong>
                      <small>{repository.workspace_relative_path ?? "workspace root"}</small>
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
        <details className="register" open={registerOpen} onToggle={(event) => setRegisterOpen(event.currentTarget.open)}>
          <summary>
            <Icon name="plus" />
            Register repository
          </summary>
          <form className="stack" onSubmit={(event) => void submit(event)}>
            <label>
              Name
              <input
                value={name}
                onChange={(event) => setName(event.target.value)}
                required
                maxLength={255}
                pattern="[A-Za-z0-9][A-Za-z0-9 ._\-]*"
                title="Start with a letter or digit; then letters, digits, spaces, dots, underscores, or hyphens."
                placeholder="payments-service"
              />
            </label>
            <label>
              Workspace-relative path
              <input
                value={path}
                onChange={(event) => setPath(event.target.value)}
                placeholder="example-project"
                required
                maxLength={1024}
                className="mono"
              />
            </label>
            <button type="submit" className="btn btnPrimary" disabled={registering || running}>
              {registering ? <Spinner /> : <Icon name="plus" />}
              {registering ? "Registering…" : "Register"}
            </button>
          </form>
        </details>
      </section>

      <section className="card">
        <header className="cardHeader">
          <div>
            <Icon name="file" />
            <h2>Indexed files</h2>
            {files.length > 0 ? <span className="countBadge">{files.length}</span> : null}
          </div>
          <button
            type="button"
            className="btn btnSecondary btnSm"
            onClick={onIndex}
            disabled={selectedId === null || running}
            aria-label="Index repository"
            title="Index repository"
          >
            {indexing ? <Spinner /> : <Icon name="database" />}
            {indexing ? "Indexing…" : "Index"}
          </button>
        </header>
        {loadingFiles ? (
          <div className="skeletonList" aria-label="Loading file metadata">
            {[78, 62, 85, 54, 70].map((width) => (
              <div key={width} className="skeleton" style={{ width: `${width}%` }} />
            ))}
          </div>
        ) : null}
        {!loadingFiles && files.length === 0 ? (
          <div className="emptyBlock">
            <Icon name="database" />
            <p>Nothing indexed yet.</p>
            <small>Index the repository to enable Ask and indexed search.</small>
          </div>
        ) : null}
        {!loadingFiles && files.length > 0 ? (
          <ul className="fileList">
            {files.map((file) => (
              <li key={file.relative_path} className="fileItem" title={`${file.relative_path} · ${file.language ?? "text"}`}>
                <span className={`langDot ${languageClass(file.language)}`} aria-hidden="true" />
                <code>{file.relative_path}</code>
                <small>{file.line_count} lines</small>
              </li>
            ))}
          </ul>
        ) : null}
        {indexResult !== null ? (
          <p className="indexSummary" role="status">
            <Icon name="check" />
            Indexed {indexResult.files_indexed} files / {indexResult.chunks_indexed} chunks
            {indexResult.embedding_model === null ? "" : ` · ${indexResult.embedding_model}`}
          </p>
        ) : null}
      </section>
    </aside>
  );
}
