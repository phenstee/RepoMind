"""Tests for fixed-command, read-only Git inspection tools."""

import shutil
import subprocess
from pathlib import Path

import pytest

from repomind.tools import (
    GitDiffInput,
    GitStatusInput,
    ToolConfig,
    ToolContext,
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
