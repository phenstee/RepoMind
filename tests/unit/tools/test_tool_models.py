"""Validation and serialization tests for tool models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from repomind.tools import (
    GitDiffInput,
    ListDirectoryInput,
    ReadFileInput,
    SearchCodeInput,
    ToolConfig,
    ToolContext,
)


def test_tool_context_resolves_and_freezes_repository_root(tmp_path: Path) -> None:
    context = ToolContext(repository_root=tmp_path)

    assert context.repository_root == tmp_path.resolve()
    with pytest.raises(ValidationError):
        context.repository_root = tmp_path / "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "values",
    [
        {"max_file_bytes": 0},
        {"max_directory_entries": -1},
        {"max_search_results": True},
        {"max_diff_chars": 1.5},
    ],
)
def test_tool_config_requires_strict_positive_limits(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ToolConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "path",
    [
        "../outside.txt",
        "../../.ssh/id_rsa",
        "C:outside.txt",
        r"C:\Windows\system.ini",
        "/root/.ssh/id_rsa",
    ],
)
def test_tool_inputs_reject_hostile_paths(path: str) -> None:
    for model, field in (
        (ReadFileInput, {"path": path}),
        (ListDirectoryInput, {"path": path}),
        (SearchCodeInput, {"query": "x", "path": path}),
        (GitDiffInput, {"path": path}),
    ):
        with pytest.raises(ValidationError):
            model(**field)


def test_root_path_is_only_allowed_for_directory_scoped_inputs() -> None:
    assert ListDirectoryInput().path == Path(".")
    assert SearchCodeInput(query="needle").path == Path(".")
    with pytest.raises(ValidationError):
        ReadFileInput(path=".")


def test_tool_models_serialize_paths_for_future_observations() -> None:
    dumped = SearchCodeInput(query="snow", path="src").model_dump(mode="json")
    assert dumped == {
        "query": "snow",
        "path": "src",
        "max_results": 50,
        "case_sensitive": False,
    }
