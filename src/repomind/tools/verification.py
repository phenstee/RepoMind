"""Fixed, bounded local verification tools without a general command surface."""

from __future__ import annotations

import os
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
    timed_out = False
    exit_code: int | None = None
    with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(
        mode="w+b"
    ) as stderr_file:
        try:
            completed = subprocess.run(
                command,
                cwd=context.repository_root,
                stdout=stdout_file,
                stderr=stderr_file,
                timeout=timeout_seconds,
                check=False,
                shell=False,
                env=environment,
            )
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
        except OSError as exc:
            raise ToolExecutionError("Could not start local verification process") from exc
        except subprocess.SubprocessError as exc:
            raise ToolExecutionError("Local verification process failed internally") from exc

        stdout, stdout_truncated = _read_bounded_output(
            stdout_file, max_output_chars
        )
        stderr, stderr_truncated = _read_bounded_output(
            stderr_file, max_output_chars
        )

    return VerificationOutput(
        paths=paths,
        exit_code=exit_code,
        passed=exit_code == 0 and not timed_out,
        stdout=stdout,
        stderr=stderr,
        truncated=stdout_truncated or stderr_truncated,
        timed_out=timed_out,
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
        "-m",
        "pytest",
        *(path.as_posix() for path in paths),
        f"--maxfail={max_failures}",
        "-q",
    ]
    environment = os.environ.copy()
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
    command = [
        sys.executable,
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
    )
    return RunRuffOutput.model_validate(output.model_dump())
