import json
from pathlib import Path

import pytest

from repomind.observability.sanitization import (
    sanitize_error,
    sanitize_metadata,
    sanitize_tool_arguments,
    sanitize_tool_output,
)


@pytest.mark.parametrize(
    "tool,arguments",
    [
        (
            "replace_text",
            {
                "path": "app.py",
                "old_text": "SECRET SOURCE CODE",
                "new_text": "MORE SECRET CODE",
                "expected_sha256": "a" * 64,
            },
        ),
        ("create_file", {"path": "app.py", "content": "SECRET SOURCE CODE"}),
        ("search_code", {"query": "SECRET SOURCE CODE"}),
    ],
)
def test_arguments_keep_lengths_without_content(tool, arguments) -> None:
    metadata = sanitize_tool_arguments(tool, arguments)
    assert "SECRET" not in json.dumps(metadata)
    assert any(key.endswith("_chars") for key in metadata)


@pytest.mark.parametrize(
    "tool", ["git_diff", "read_file", "run_tests", "run_ruff", "search_code", "find_symbol"]
)
def test_outputs_omit_source_diffs_stdout_and_matches(tool) -> None:
    data = {
        "content": "private source",
        "stdout": "sk-test-secret",
        "stderr": "postgresql://user:password@localhost/db",
        "matches": [{"line": "private source"}],
        "passed": False,
        "exit_code": 1,
        "timed_out": False,
        "truncated": True,
    }
    metadata = sanitize_tool_output(tool, data)
    assert "private source" not in json.dumps(metadata)
    assert "sk-test-secret" not in json.dumps(metadata)
    assert "password" not in json.dumps(metadata)
    if tool in {"run_tests", "run_ruff"}:
        assert metadata["passed"] is False
        assert metadata["exit_code"] == 1
        assert metadata["stdout_chars"] == len(data["stdout"])


@pytest.mark.parametrize(
    "secret",
    [
        "sk-test-secret",
        "OPENAI_API_KEY=secret",
        "postgresql://user:password@localhost/db",
        "Authorization: Bearer supersecret",
    ],
)
def test_defense_in_depth_redacts_known_secret_patterns(secret) -> None:
    assert secret not in json.dumps(sanitize_metadata({"message": secret, "path": secret}))
    assert secret not in json.dumps(sanitize_error(RuntimeError(secret)))


def test_unknown_tools_and_fields_fail_closed() -> None:
    assert sanitize_tool_arguments("custom", {"path": "secret", "content": "secret"}) == {}
    assert sanitize_tool_output("custom", {"stdout": "secret", "content": "secret"}) == {}
    assert sanitize_metadata({"api_key": "secret", "embedding": [1, 2], "prompt": "secret"}) == {}


def test_invalid_field_types_cannot_store_payload_text() -> None:
    metadata = sanitize_tool_arguments("read_file", {"start_line": "PRIVATE SOURCE"})
    assert metadata == {"start_line": None}
    assert sanitize_tool_output("replace_text", {"before_sha256": "PRIVATE SOURCE"}) == {
        "before_sha256": None
    }


@pytest.mark.parametrize("path", ["/private/root.py", "C:\\private\\root.py", "../root.py"])
def test_absolute_or_escaping_paths_not_recorded(path) -> None:
    assert sanitize_tool_arguments("read_file", {"path": path}) == {"path": None}


def test_relative_paths_and_bounded_nested_metadata() -> None:
    assert sanitize_tool_arguments("read_file", {"path": Path("src/app.py")}) == {
        "path": "src/app.py"
    }
    metadata = sanitize_metadata({"message": "x" * 10000, "paths": ["x"] * 1000})
    assert len(metadata["message"]) == 256
    assert len(metadata["paths"]) == 64
    assert sanitize_metadata({"message": "before\x00\x1bafter"}) == {"message": "before  after"}
