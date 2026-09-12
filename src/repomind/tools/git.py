"""Read-only Git inspection implemented with fixed subprocess argument arrays."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from repomind.tools.filesystem import _resolve_workspace_path
from repomind.tools.models import (
    GitChangedFile,
    GitDiffInput,
    GitDiffOutput,
    GitStatusInput,
    GitStatusOutput,
    ToolConfig,
    ToolContext,
)
from repomind.tools.registry import ToolExecutionError


def _run_git(context: ToolContext, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    command = _git_command(context, arguments)
    try:
        return subprocess.run(
            command,
            cwd=context.repository_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise ToolExecutionError("Git executable is not available") from exc
    except subprocess.CalledProcessError as exc:
        raise ToolExecutionError("Git status inspection failed") from exc
    except OSError as exc:
        raise ToolExecutionError("Could not execute Git inspection") from exc


def _git_command(context: ToolContext, arguments: list[str]) -> list[str]:
    return [
        "git",
        "-c",
        "core.quotepath=false",
        "-c",
        f"safe.directory={context.repository_root.as_posix()}",
        *arguments,
    ]


def _run_bounded_git_diff(
    context: ToolContext,
    arguments: list[str],
    max_chars: int,
) -> tuple[str, bool]:
    process: subprocess.Popen[str] | None = None
    try:
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file:
            process = subprocess.Popen(
                _git_command(context, arguments),
                cwd=context.repository_root,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if process.stdout is None:  # pragma: no cover - guaranteed by PIPE
                raise ToolExecutionError("Could not capture Git diff output")
            content = process.stdout.read(max_chars + 1)
            truncated = len(content) > max_chars
            if truncated:
                process.kill()
            return_code = process.wait()
            process.stdout.close()
            if not truncated and return_code != 0:
                raise ToolExecutionError("Git diff inspection failed")
            return content[:max_chars], truncated
    except FileNotFoundError as exc:
        raise ToolExecutionError("Git executable is not available") from exc
    except OSError as exc:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise ToolExecutionError("Could not execute Git diff inspection") from exc


def _branch_name(header: str) -> str | None:
    value = header.removeprefix("## ")
    if value.startswith("No commits yet on "):
        return value.removeprefix("No commits yet on ") or None
    if value.startswith("HEAD (no branch)"):
        return None
    return value.split("...", maxsplit=1)[0] or None


def _parse_status(output: str) -> tuple[str | None, list[GitChangedFile]]:
    records = output.split("\0")
    branch: str | None = None
    changed: list[GitChangedFile] = []
    index = 0
    if records and records[0].startswith("## "):
        branch = _branch_name(records[0])
        index = 1

    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        status = record[:2]
        path_text = record[3:]
        changed.append(GitChangedFile(path=Path(path_text), status=status))
        if "R" in status or "C" in status:
            index += 1

    changed.sort(key=lambda item: (item.path.as_posix().casefold(), item.path.as_posix()))
    return branch, changed


def git_status(
    context: ToolContext,
    arguments: GitStatusInput,
    *,
    config: ToolConfig | None = None,
) -> GitStatusOutput:
    """Inspect the current branch and working tree without modifying Git state."""

    del arguments, config
    result = _run_git(context, ["status", "--porcelain=v1", "--branch", "-z"])
    branch, changed = _parse_status(result.stdout)
    return GitStatusOutput(branch=branch, changed_files=changed, clean=not changed)


def git_diff(
    context: ToolContext,
    arguments: GitDiffInput,
    *,
    config: ToolConfig | None = None,
) -> GitDiffOutput:
    """Inspect a bounded staged or unstaged diff using only fixed Git options."""

    resolved_config = config or ToolConfig()
    command = ["diff", "--no-ext-diff", "--no-color"]
    if arguments.staged:
        command.append("--cached")
    if arguments.path is not None:
        _resolve_workspace_path(context, arguments.path, allow_root=True)
        command.extend(["--", arguments.path.as_posix()])

    max_chars = min(
        arguments.max_chars or resolved_config.max_diff_chars,
        resolved_config.max_diff_chars,
    )
    content, truncated = _run_bounded_git_diff(context, command, max_chars)
    return GitDiffOutput(
        content=content,
        truncated=truncated,
        staged=arguments.staged,
        path=arguments.path,
    )
