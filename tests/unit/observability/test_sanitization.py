import json
from pathlib import Path

import pytest

from repomind.observability.sanitization import (
    redact,
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


@pytest.mark.parametrize(
    "value",
    [
        "coding-task-01",
        "src/tasks/task-runner.py",
        "task-runner.py",
        "risk-model",
        "disk-usage-report",
        "sk-learn",
        "gpt-4.1-mini",
        "prompt_tokens=12 total_tokens: 30 max_tokens=5",
        "https://example.com:8443/docs/a@b",
        "basic configuration and Basic settings",
    ],
)
def test_ordinary_identifiers_and_paths_are_not_redacted(value) -> None:
    assert redact(value) == value
    assert sanitize_metadata({"message": value}) == {"message": value}


def test_relative_task_paths_survive_tool_argument_sanitization() -> None:
    for path in ("src/tasks/task-runner.py", "cases/coding-task-01.json", "risk-model/app.py"):
        assert sanitize_tool_arguments("read_file", {"path": path}) == {"path": path}


# Built at runtime so no literal token-shaped string sits in the source tree.
_JWT_HEADER, _JWT_PAYLOAD, _JWT_SIGNATURE = "eyJhbGciOiJub25lIn0", "eyJzdWIiOiJ0ZXN0In0", "c2ln"
_FAKE_JWT = f"{_JWT_HEADER}.{_JWT_PAYLOAD}.{_JWT_SIGNATURE}"


@pytest.mark.parametrize(
    "secret",
    [
        "sk-" + "proj-" + "a1B2" * 6,
        "sk-" + "ant-" + "api03-" + "x9" * 10,
        "ghp_" + "A1b2" * 9,
        "gho_" + "Z" * 24,
        "github_pat_" + "11AB" * 6,
        "AKIA" + "IOSFODNN7EXAMPLE",
        "xoxb-" + "0123456789-" + "fakeslacktoken",
        _FAKE_JWT,
        "Basic " + "YWRtaW46aHVudGVyMg==",
    ],
)
def test_provider_token_shapes_are_redacted(secret) -> None:
    redacted = redact(f"before {secret} after")
    assert secret not in redacted
    assert redacted.startswith("before ")
    assert redacted.endswith(" after")
    assert "[REDACTED]" in redacted


def test_authorization_basic_header_is_redacted() -> None:
    assert redact("Authorization: Basic dXNlcjpwYXNz") == "[REDACTED]"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://user:hunter2@example.com/repo.git", "https://[REDACTED]@example.com/repo.git"),
        ("redis://:s3cret@localhost:6379/0", "redis://[REDACTED]@localhost:6379/0"),
        ("amqp://guest:guest-pass@rabbit:5672/", "amqp://[REDACTED]@rabbit:5672/"),
    ],
)
def test_url_userinfo_credentials_are_redacted_in_any_scheme(url, expected) -> None:
    assert redact(url) == expected


@pytest.mark.parametrize(
    "assignment,expected",
    [
        ("api_key=abc123", "api_key=[REDACTED]"),
        ("X-API-KEY: abc123", "X-API-KEY: [REDACTED]"),
        ("token = xyz", "token = [REDACTED]"),
        ("GITHUB_TOKEN=ghx", "GITHUB_TOKEN=[REDACTED]"),
        ("client_secret: 'zzz'", "client_secret: [REDACTED]"),
        ('"password": "hunter2"', '"password": [REDACTED]'),
        ("passwd=foo", "passwd=[REDACTED]"),
    ],
)
def test_secret_assignments_keep_the_name_and_redact_the_value(assignment, expected) -> None:
    assert redact(assignment) == expected


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


def test_redaction_is_linear_time_on_adversarial_identifier_runs() -> None:
    import time

    started = time.perf_counter()
    for value in ("a-" * 50_000, "http://" + "a:" * 50_000, "x_" * 50_000):
        redact(value)
    assert time.perf_counter() - started < 2.0
