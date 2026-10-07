"""Tests for fixed-command, read-only Git inspection tools."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from repomind.tools import (
    GitDiffInput,
    GitStatusInput,
    ToolConfig,
    ToolContext,
    ToolExecutionError,
    git_diff,
    git_status,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is unavailable")


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "RepoMind Tests")
    _git(root, "config", "user.email", "tests@example.invalid")
    (root / "tracked.py").write_text("before\n", encoding="utf-8")
    _git(root, "add", "tracked.py")
    _git(root, "commit", "-m", "initial")
    return root


def test_git_status_reports_clean_branch_modified_and_untracked(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    context = ToolContext(repository_root=root)
    clean = git_status(context, GitStatusInput())
    assert clean.clean and clean.branch == "main" and clean.changed_files == []

    (root / "tracked.py").write_text("after\n", encoding="utf-8")
    (root / "untracked.py").write_text("new\n", encoding="utf-8")
    dirty = git_status(context, GitStatusInput())

    assert not dirty.clean
    assert [(item.path.as_posix(), item.status) for item in dirty.changed_files] == [
        ("tracked.py", " M"),
        ("untracked.py", "??"),
    ]
    assert dirty.model_dump(mode="json")["branch"] == "main"


def test_git_diff_supports_unstaged_staged_and_safe_path_filtering(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    context = ToolContext(repository_root=root)
    (root / "tracked.py").write_text("after\n", encoding="utf-8")
    (root / "other.py").write_text("other\n", encoding="utf-8")

    unstaged = git_diff(context, GitDiffInput(path="tracked.py"))
    assert "+after" in unstaged.content and "-before" in unstaged.content
    assert not unstaged.staged and not unstaged.truncated

    _git(root, "add", "tracked.py")
    assert git_diff(context, GitDiffInput(path="tracked.py")).content == ""
    staged = git_diff(context, GitDiffInput(staged=True, path="tracked.py"))
    assert "+after" in staged.content and staged.staged
    assert "other.py" not in staged.content


def test_git_diff_reports_deterministic_truncation(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "tracked.py").write_text("x" * 500 + "\n", encoding="utf-8")
    output = git_diff(
        ToolContext(repository_root=root),
        GitDiffInput(max_chars=40),
        config=ToolConfig(max_diff_chars=100),
    )
    assert len(output.content) == 40
    assert output.truncated


def test_git_diff_path_cannot_be_interpreted_as_an_option(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    option_named = root / "--output=stolen"
    option_named.write_text("new\n", encoding="utf-8")

    output = git_diff(
        ToolContext(repository_root=root),
        GitDiffInput(path="--output=stolen"),
    )
    assert output.content == ""
    assert not (root / "stolen").exists()


def test_repository_config_cannot_execute_commands_during_inspection(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    marker = tmp_path / "executed"
    command = f"touch {marker.as_posix()}; false"
    _git(root, "config", "core.fsmonitor", command)
    _git(root, "config", "diff.external", command)
    _git(root, "config", "diff.cat.textconv", command)
    (root / ".gitattributes").write_text("*.py diff=cat\n", encoding="utf-8")
    (root / "tracked.py").write_text("after\n", encoding="utf-8")
    context = ToolContext(repository_root=root)

    status = git_status(context, GitStatusInput())
    diff = git_diff(context, GitDiffInput(include_untracked=True))

    assert not marker.exists()
    assert ("tracked.py", " M") in [
        (item.path.as_posix(), item.status) for item in status.changed_files
    ]
    assert "+after" in diff.content


def test_git_runs_with_hardened_options_and_scrubbed_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-parent-secret")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere"))
    launches: list[tuple[list[str], dict[str, object]]] = []
    real_popen = subprocess.Popen

    def recording_popen(command: list[str], **kwargs: object) -> subprocess.Popen[str]:
        launches.append((command, kwargs))
        return real_popen(command, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr("repomind.tools.git.subprocess.Popen", recording_popen)
    context = ToolContext(repository_root=root)

    assert git_status(context, GitStatusInput()).clean
    git_diff(context, GitDiffInput())

    (status_command, status_kwargs), (diff_command, _) = launches
    for setting in (
        f"safe.directory={root.resolve().as_posix()}",
        "core.fsmonitor=false",
        "core.hooksPath=/dev/null",
        "core.untrackedCache=false",
    ):
        assert setting in status_command and setting in diff_command
    assert "--no-ext-diff" in diff_command and "--no-textconv" in diff_command
    environment = status_kwargs["env"]
    assert isinstance(environment, dict)
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert "OPENAI_API_KEY" not in environment
    assert "GIT_DIR" not in environment
    assert environment.get("PATH")


def test_git_inspection_times_out_as_a_tool_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path)
    monkeypatch.setattr("repomind.tools.git.GIT_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(
        "repomind.tools.git._git_command",
        lambda context, arguments: [sys.executable, "-c", "import time; time.sleep(30)"],
    )

    with pytest.raises(ToolExecutionError, match="timed out"):
        git_status(ToolContext(repository_root=root), GitStatusInput())
    with pytest.raises(ToolExecutionError, match="timed out"):
        git_diff(ToolContext(repository_root=root), GitDiffInput())


def test_git_status_caps_entries_and_reports_truncation(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    for index in range(7):
        (root / f"new_{index}.py").write_text("x\n", encoding="utf-8")
    context = ToolContext(repository_root=root)

    capped = git_status(context, GitStatusInput(), config=ToolConfig(max_status_entries=3))
    complete = git_status(context, GitStatusInput())

    assert capped.truncated and not capped.clean
    assert [item.path.as_posix() for item in capped.changed_files] == [
        "new_0.py",
        "new_1.py",
        "new_2.py",
    ]
    assert not complete.truncated and len(complete.changed_files) == 7


def test_git_diff_can_render_untracked_files_as_new_file_diffs(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    _git(root, "add", ".gitignore")
    _git(root, "commit", "-m", "ignore")
    (root / "tracked.py").write_text("after\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "conftest.py").write_text(
        "def pytest_sessionfinish(session):\n    session.exitstatus = 0", encoding="utf-8"
    )
    (root / "ignored.py").write_text("hidden\n", encoding="utf-8")
    context = ToolContext(repository_root=root)

    plain = git_diff(context, GitDiffInput())
    full = git_diff(context, GitDiffInput(include_untracked=True))

    assert "conftest" not in plain.content
    assert "+after" in full.content
    assert (
        "diff --git a/tests/conftest.py b/tests/conftest.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/conftest.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def pytest_sessionfinish(session):\n"
        "+    session.exitstatus = 0\n"
        "\\ No newline at end of file\n"
    ) in full.content
    assert "ignored.py" not in full.content
    assert not full.truncated


def test_untracked_diff_counts_toward_truncation(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "big_new.py").write_text("y" * 500 + "\n", encoding="utf-8")

    output = git_diff(
        ToolContext(repository_root=root),
        GitDiffInput(max_chars=120, include_untracked=True),
    )

    assert output.truncated
    assert len(output.content) == 120
    assert output.content.startswith("diff --git a/big_new.py b/big_new.py\n")


def test_unrenderable_untracked_content_is_reported_as_truncated(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "huge.py").write_text("z" * 200 + "\n", encoding="utf-8")

    output = git_diff(
        ToolContext(repository_root=root),
        GitDiffInput(include_untracked=True),
        config=ToolConfig(max_file_bytes=100),
    )

    assert output.truncated
    assert "content not rendered" in output.content


def test_git_diff_path_rejects_git_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Git metadata"):
        GitDiffInput(path=".git/config")
