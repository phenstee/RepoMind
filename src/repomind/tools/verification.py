"""Fixed, bounded local verification tools without a general command surface."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

from repomind.tools.filesystem import _resolve_workspace_path
from repomind.tools.models import (
    RunRuffInput,
    RunRuffOutput,
    RunTestsInput,
    RunTestsOutput,
    ToolConfig,
    ToolContext,
    VerificationOutput,
)
from repomind.tools.registry import ToolExecutionError

_VERIFICATION_ENVIRONMENT_ALLOWLIST = (
    # Executable discovery and Windows process startup.
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    # Writable temporary locations.
    "TEMP",
    "TMP",
    "TMPDIR",
    # Standard user/config locations used by Python and developer tooling.
    "HOME",
    "HOMEDRIVE",
    "HOMEPATH",
    "USERPROFILE",
    "LOCALAPPDATA",
    "APPDATA",
    "VIRTUAL_ENV",
    # Explicit Python text and locale behavior.
    "PYTHONUTF8",
    "PYTHONIOENCODING",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
)

# ``python -P`` keeps the repository root (the working directory) off
# ``sys.path`` while pytest itself is imported, so a repository-level
# pytest.py (or a module shadowing one of pytest's own dependencies) cannot
# replace the verifier. The root is then restored at the front of ``sys.path``
# exactly as ``python -m pytest`` would, so flat-layout test imports still work.
_PYTEST_BOOTSTRAP = (
    "import os, runpy, sys; import pytest; sys.path.insert(0, os.getcwd()); "
    "runpy.run_module('pytest', run_name='__main__', alter_sys=True)"
)

# Hard cap on what one verifier stream may spool to disk before the whole
# process tree is terminated; only ``max_output_chars`` is ever read back.
_MAX_SPOOLED_OUTPUT_BYTES = 64 * 1024 * 1024
_PROCESS_POLL_SECONDS = 0.1


def _verification_environment() -> dict[str, str]:
    """Build the minimal non-secret environment shared by verifier children."""

    return {
        name: value
        for name in _VERIFICATION_ENVIRONMENT_ALLOWLIST
        if (value := os.environ.get(name)) is not None
    }


def _validated_paths(
    context: ToolContext,
    paths: Sequence[Path],
    *,
    tests_only: bool,
) -> list[Path]:
    validated: list[Path] = []
    for path in paths:
        resolved = _resolve_workspace_path(context, path, allow_root=not tests_only)
        if not resolved.exists():
            raise ToolExecutionError(f"Verification path does not exist: {path.as_posix()}")
        if not resolved.is_file() and not resolved.is_dir():
            raise ToolExecutionError(
                f"Verification path is not a file or directory: {path.as_posix()}"
            )
        if tests_only:
            parts = tuple(part.casefold() for part in path.parts)
            is_test_file = resolved.is_file() and (
                (path.name.casefold().startswith("test_") or path.name.casefold().endswith("_test.py"))
                and path.suffix.casefold() == ".py"
            )
            is_test_directory = resolved.is_dir() and "tests" in parts
            if not is_test_file and not is_test_directory:
                raise ToolExecutionError(
                    f"Path is outside the conservative test scope: {path.as_posix()}"
                )
        validated.append(path)
    return validated


def _read_bounded_output(stream: object, max_chars: int) -> tuple[str, bool]:
    stream.seek(0)  # type: ignore[attr-defined]
    data = stream.read(max_chars + 1)  # type: ignore[attr-defined]
    truncated = len(data) > max_chars
    return data[:max_chars].decode("utf-8", errors="replace"), truncated


def _spooled_bytes(stream: object) -> int:
    try:
        return os.fstat(stream.fileno()).st_size  # type: ignore[attr-defined]
    except OSError:
        return 0


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Kill the verifier and, on POSIX, every process left in its session.

    The verifier is started as a session leader, so its process-group ID is
    its PID; killing the group also reaps grandchildren a test may spawn,
    which a plain ``kill`` of the direct child would orphan.
    """

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            process.kill()
    elif process.poll() is None:
        process.kill()


def _wait_bounded(
    process: subprocess.Popen[bytes],
    *,
    deadline: float,
    streams: Sequence[object],
) -> tuple[int | None, bool, bool]:
    """Wait for exit, a deadline, or a spool overflow; return exit/timeout/overflow."""

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, True, False
        try:
            return process.wait(timeout=min(remaining, _PROCESS_POLL_SECONDS)), False, False
        except subprocess.TimeoutExpired:
            pass
        if any(_spooled_bytes(stream) > _MAX_SPOOLED_OUTPUT_BYTES for stream in streams):
            return None, False, True


def _run_fixed_verifier(
    command: list[str],
    *,
    context: ToolContext,
    paths: list[Path],
    timeout_seconds: int,
    max_output_chars: int,
    environment: dict[str, str] | None = None,
) -> VerificationOutput:
    started = time.monotonic()
    with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(
        mode="w+b"
    ) as stderr_file:
        try:
            process = subprocess.Popen(
                command,
                cwd=context.repository_root,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                shell=False,
                env=environment,
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            raise ToolExecutionError("Could not start local verification process") from exc
        except subprocess.SubprocessError as exc:
            raise ToolExecutionError("Local verification process failed internally") from exc

        try:
            exit_code, timed_out, output_overflow = _wait_bounded(
                process,
                deadline=started + timeout_seconds,
                streams=(stdout_file, stderr_file),
            )
        finally:
            _terminate_process_tree(process)
            process.wait()

        stdout, stdout_truncated = _read_bounded_output(
            stdout_file, max_output_chars
        )
        stderr, stderr_truncated = _read_bounded_output(
            stderr_file, max_output_chars
        )

    return VerificationOutput(
        paths=paths,
        exit_code=exit_code,
        passed=exit_code == 0 and not timed_out and not output_overflow,
        stdout=stdout,
        stderr=stderr,
        truncated=stdout_truncated or stderr_truncated or output_overflow,
        timed_out=timed_out,
        output_limit_exceeded=output_overflow,
        duration_seconds=time.monotonic() - started,
    )


def run_tests(
    context: ToolContext,
    arguments: RunTestsInput,
    *,
    config: ToolConfig | None = None,
) -> RunTestsOutput:
    """Run pytest through the current Python interpreter with fixed arguments."""

    resolved_config = config or ToolConfig()
    paths = _validated_paths(context, arguments.paths, tests_only=True)
    timeout = min(
        arguments.timeout_seconds or resolved_config.verification_timeout_seconds,
        resolved_config.verification_timeout_seconds,
    )
    max_failures = min(arguments.max_failures, resolved_config.max_test_failures)
    command = [
        sys.executable,
        "-P",
        "-c",
        _PYTEST_BOOTSTRAP,
        *(path.as_posix() for path in paths),
        f"--maxfail={max_failures}",
        "-q",
    ]
    environment = _verification_environment()
    with tempfile.TemporaryDirectory(prefix="repomind-pycache-") as pycache_prefix:
        environment["PYTHONPYCACHEPREFIX"] = pycache_prefix
        output = _run_fixed_verifier(
            command,
            context=context,
            paths=paths,
            timeout_seconds=timeout,
            max_output_chars=resolved_config.max_verification_output_chars,
            environment=environment,
        )
    return RunTestsOutput.model_validate(output.model_dump())


def run_ruff(
    context: ToolContext,
    arguments: RunRuffInput,
    *,
    config: ToolConfig | None = None,
) -> RunRuffOutput:
    """Run Ruff check through the current Python interpreter without auto-fix."""

    resolved_config = config or ToolConfig()
    paths = _validated_paths(context, arguments.paths, tests_only=False)
    timeout = min(
        arguments.timeout_seconds or resolved_config.verification_timeout_seconds,
        resolved_config.verification_timeout_seconds,
    )
    # -P keeps the repository root off sys.path so a repository-level ruff.py
    # cannot shadow the real Ruff entry point.
    command = [
        sys.executable,
        "-P",
        "-m",
        "ruff",
        "check",
        *(path.as_posix() for path in paths),
    ]
    output = _run_fixed_verifier(
        command,
        context=context,
        paths=paths,
        timeout_seconds=timeout,
        max_output_chars=resolved_config.max_verification_output_chars,
        environment=_verification_environment(),
    )
    return RunRuffOutput.model_validate(output.model_dump())
