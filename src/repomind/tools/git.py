"""Read-only Git inspection implemented with fixed subprocess argument arrays."""

from __future__ import annotations

import subprocess
import tempfile
import threading
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
from repomind.tools.verification import _verification_environment

GIT_TIMEOUT_SECONDS = 30
# Upper bound on raw porcelain/ls-files output read into memory; entries past
# it (or past ToolConfig.max_status_entries) are reported as truncated.
_MAX_GIT_LISTING_CHARS = 1_048_576


def _git_environment() -> dict[str, str]:
    """Run Git without inherited secrets, repository overrides, or prompts."""

    environment = _verification_environment()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    return environment


def _git_command(context: ToolContext, arguments: list[str]) -> list[str]:
    # safe.directory stays per call: Docker bind-mounts the workspace with a
    # different owner. The remaining overrides keep repository configuration
    # from executing commands during read-only inspection.
    return [
        "git",
        "-c",
        "core.quotepath=false",
        "-c",
        f"safe.directory={context.repository_root.as_posix()}",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.untrackedCache=false",
        *arguments,
    ]


def _run_bounded_git(
    context: ToolContext,
    arguments: list[str],
    max_chars: int,
    *,
    operation: str,
) -> tuple[str, bool]:
    """Run one fixed Git command, keeping at most ``max_chars`` of stdout.

    The process is killed once the bound is exceeded or after
    ``GIT_TIMEOUT_SECONDS``; a timeout is a tool error, never a partial result.
    """

    process: subprocess.Popen[str] | None = None
    timed_out = threading.Event()
    timer: threading.Timer | None = None
    try:
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file:
            process = subprocess.Popen(
                _git_command(context, arguments),
                cwd=context.repository_root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=_git_environment(),
            )
            running = process

            def expire() -> None:
                timed_out.set()
                running.kill()

            timer = threading.Timer(GIT_TIMEOUT_SECONDS, expire)
            timer.daemon = True
            timer.start()
            if process.stdout is None:  # pragma: no cover - guaranteed by PIPE
                raise ToolExecutionError(f"Could not capture Git {operation} output")
            content = process.stdout.read(max_chars + 1)
            truncated = len(content) > max_chars
            if truncated:
                process.kill()
            return_code = process.wait()
            process.stdout.close()
            if timed_out.is_set():
                raise ToolExecutionError(
                    f"Git {operation} inspection timed out after {GIT_TIMEOUT_SECONDS} seconds"
                )
            if not truncated and return_code != 0:
                raise ToolExecutionError(f"Git {operation} inspection failed")
            return content[:max_chars], truncated
    except FileNotFoundError as exc:
        raise ToolExecutionError("Git executable is not available") from exc
    except OSError as exc:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise ToolExecutionError(f"Could not execute Git {operation} inspection") from exc
    finally:
        if timer is not None:
            timer.cancel()


def _branch_name(header: str) -> str | None:
    value = header.removeprefix("## ")
    if value.startswith("No commits yet on "):
        return value.removeprefix("No commits yet on ") or None
    if value.startswith("HEAD (no branch)"):
        return None
    return value.split("...", maxsplit=1)[0] or None


def _parse_status(
    output: str,
    *,
    max_entries: int,
    output_truncated: bool = False,
) -> tuple[str | None, list[GitChangedFile], bool]:
    records = output.split("\0")
    if output_truncated and records:
        # The final record may have been cut mid-path by the output bound.
        records.pop()
    truncated = output_truncated
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
        if len(changed) == max_entries:
            truncated = True
            break
        status = record[:2]
        path_text = record[3:]
        changed.append(GitChangedFile(path=Path(path_text), status=status))
        if "R" in status or "C" in status:
            index += 1

    changed.sort(key=lambda item: (item.path.as_posix().casefold(), item.path.as_posix()))
    return branch, changed, truncated


def git_status(
    context: ToolContext,
    arguments: GitStatusInput,
    *,
    config: ToolConfig | None = None,
) -> GitStatusOutput:
    """Inspect the current branch and working tree without modifying Git state."""

    del arguments
    resolved_config = config or ToolConfig()
    output, output_truncated = _run_bounded_git(
        context,
        ["status", "--porcelain=v1", "--branch", "-z"],
        _MAX_GIT_LISTING_CHARS,
        operation="status",
    )
    branch, changed, truncated = _parse_status(
        output,
        max_entries=resolved_config.max_status_entries,
        output_truncated=output_truncated,
    )
    # A truncated listing proves the tree is dirty even if no entry survived.
    return GitStatusOutput(
        branch=branch,
        changed_files=changed,
        clean=not changed and not truncated,
        truncated=truncated,
    )


def _render_untracked_file(
    context: ToolContext,
    relative: Path,
    max_bytes: int,
) -> tuple[str, bool]:
    """Render one untracked path as a Git-style new-file diff.

    Returns the rendering and whether it shows the file's complete content.
    """

    display = relative.as_posix()
    header = f"diff --git a/{display} b/{display}\n"
    try:
        target = _resolve_workspace_path(context, relative, allow_root=False)
    except ToolExecutionError:
        return f"{header}(untracked path not rendered: link or unsafe path)\n", False
    if not target.is_file():
        return f"{header}(untracked path not rendered: not a regular file)\n", False
    try:
        mode = "100755" if target.stat().st_mode & 0o111 else "100644"
        with target.open("rb") as source:
            data = source.read(max_bytes + 1)
    except OSError:
        return f"{header}(untracked file could not be read)\n", False
    header += f"new file mode {mode}\n"
    if len(data) > max_bytes:
        return (
            f"{header}(untracked file exceeds {max_bytes} bytes; content not rendered)\n",
            False,
        )
    if b"\x00" in data:
        return f"{header}Binary files /dev/null and b/{display} differ\n", True

    lines = data.decode("utf-8", errors="replace").split("\n")
    missing_final_newline = lines[-1] != ""
    if not missing_final_newline:
        lines.pop()
    rendered = f"{header}--- /dev/null\n+++ b/{display}\n"
    if lines:
        rendered += f"@@ -0,0 +1,{len(lines)} @@\n" + "".join(f"+{line}\n" for line in lines)
    if missing_final_newline:
        rendered += "\\ No newline at end of file\n"
    return rendered, True


def _untracked_diff(
    context: ToolContext,
    path: Path | None,
    max_chars: int,
    config: ToolConfig,
) -> tuple[str, bool]:
    """Render untracked, non-ignored files as new-file diffs within ``max_chars``.

    ``git diff`` never shows untracked files, so without this a newly created
    file would be invisible to diff-based review.
    """

    command = ["ls-files", "--others", "--exclude-standard", "-z"]
    if path is not None:
        command.extend(["--", path.as_posix()])
    listing, truncated = _run_bounded_git(
        context, command, _MAX_GIT_LISTING_CHARS, operation="untracked file"
    )
    names = [name for name in listing.split("\0") if name]
    if truncated and names:
        names.pop()

    parts: list[str] = []
    used = 0
    for name in names:
        rendered, complete = _render_untracked_file(
            context, Path(name), config.max_file_bytes
        )
        # Unrendered content is incomplete evidence, exactly like a cut diff.
        truncated = truncated or not complete
        if used + len(rendered) > max_chars:
            parts.append(rendered[: max_chars - used])
            return "".join(parts), True
        parts.append(rendered)
        used += len(rendered)
    return "".join(parts), truncated


def git_diff(
    context: ToolContext,
    arguments: GitDiffInput,
    *,
    config: ToolConfig | None = None,
) -> GitDiffOutput:
    """Inspect a bounded staged or unstaged diff using only fixed Git options."""

    resolved_config = config or ToolConfig()
    command = ["diff", "--no-ext-diff", "--no-textconv", "--no-color"]
    if arguments.staged:
        command.append("--cached")
    if arguments.path is not None:
        _resolve_workspace_path(context, arguments.path, allow_root=True)
        command.extend(["--", arguments.path.as_posix()])

    max_chars = min(
        arguments.max_chars or resolved_config.max_diff_chars,
        resolved_config.max_diff_chars,
    )
    content, truncated = _run_bounded_git(context, command, max_chars, operation="diff")
    if arguments.include_untracked and not arguments.staged and not truncated:
        untracked, truncated = _untracked_diff(
            context,
            arguments.path,
            max_chars - len(content),
            resolved_config,
        )
        content += untracked
    return GitDiffOutput(
        content=content,
        truncated=truncated,
        staged=arguments.staged,
        path=arguments.path,
    )
