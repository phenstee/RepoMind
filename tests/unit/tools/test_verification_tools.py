"""Tests for fixed, bounded pytest and Ruff verification tools."""

import json
import os
import py_compile
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from repomind.tools import (
    RunRuffInput,
    RunTestsInput,
    ToolConfig,
    ToolContext,
    ToolExecutionError,
    run_ruff,
    run_tests,
)


def _write_test(root: Path, body: str, name: str = "test_sample.py") -> Path:
    tests = root / "tests"
    tests.mkdir(exist_ok=True)
    target = tests / name
    target.write_text(body, encoding="utf-8")
    return target


def test_run_tests_reports_passing_and_failing_checks_as_results(tmp_path: Path) -> None:
    target = _write_test(tmp_path, "def test_value():\n    assert 1 == 1\n")
    context = ToolContext(repository_root=tmp_path)

    passed = run_tests(context, RunTestsInput(paths=["tests/test_sample.py"]))
    target.write_text("def test_value():\n    assert 1 == 2\n", encoding="utf-8")
    failed = run_tests(context, RunTestsInput(paths=["tests"]))

    assert passed.passed and passed.exit_code == 0 and not passed.timed_out
    assert not failed.passed and failed.exit_code != 0 and not failed.timed_out
    assert "failed" in (failed.stdout + failed.stderr).casefold()
    assert json.loads(failed.model_dump_json())["paths"] == ["tests"]


def test_run_tests_does_not_reuse_stale_same_size_module_bytecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PYTHONPYCACHEPREFIX", raising=False)
    module = tmp_path / "sample.py"
    old_source = "def value():\n    return 1\n"
    new_source = "def value():\n    return 2\n"
    assert len(old_source) == len(new_source)
    module.write_text(old_source, encoding="utf-8")
    _write_test(
        tmp_path,
        "from sample import value\n\ndef test_value():\n    assert value() == 2\n",
    )
    context = ToolContext(repository_root=tmp_path)
    arguments = RunTestsInput(paths=["tests/test_sample.py"])

    initial = run_tests(context, arguments)
    assert not initial.passed

    stale_cache = Path(
        py_compile.compile(
            str(module),
            doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
        )
    )
    assert stale_cache.is_file()
    module.write_text(new_source, encoding="utf-8")

    corrected = run_tests(context, arguments)

    assert corrected.passed


def test_run_tests_child_receives_only_allowlisted_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-parent-secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://parent-secret")
    monkeypatch.setenv("REPOMIND_TEST_DATABASE_URL", "postgresql://test-secret")
    monkeypatch.setenv("REDIS_URL", "redis://parent-secret")
    monkeypatch.setenv("REDIS_TEST_URL", "redis://test-secret")
    monkeypatch.setenv("REPOMIND_UNKNOWN_PARENT_SECRET", "unknown-secret")
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    assert os.environ.get("PATH")

    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "conftest.py").write_text(
        """import os

import pytest


@pytest.fixture(autouse=True)
def isolated_verifier_environment():
    blocked = {
        \"OPENAI_API_KEY\",
        \"DATABASE_URL\",
        \"REPOMIND_TEST_DATABASE_URL\",
        \"REDIS_URL\",
        \"REDIS_TEST_URL\",
        \"REPOMIND_UNKNOWN_PARENT_SECRET\",
    }
    assert blocked.isdisjoint(os.environ)
    assert os.environ[\"PYTHONIOENCODING\"] == \"utf-8\"
    assert os.environ.get(\"PATH\")
""",
        encoding="utf-8",
    )
    _write_test(tmp_path, "def test_child_environment():\n    assert True\n")

    output = run_tests(
        ToolContext(repository_root=tmp_path),
        RunTestsInput(paths=["tests/test_sample.py"]),
    )

    assert output.passed, output.stdout + output.stderr


def test_run_tests_bounds_captured_output(tmp_path: Path) -> None:
    _write_test(
        tmp_path,
        "def test_noisy():\n    print('x' * 2000)\n    assert False\n",
    )
    output = run_tests(
        ToolContext(repository_root=tmp_path),
        RunTestsInput(),
        config=ToolConfig(max_verification_output_chars=100),
    )

    assert not output.passed
    assert output.truncated
    assert len(output.stdout) <= 100
    assert len(output.stderr) <= 100


def test_run_tests_timeout_is_a_structured_failure(tmp_path: Path) -> None:
    _write_test(
        tmp_path,
        "import time\n\ndef test_slow():\n    time.sleep(5)\n",
    )
    output = run_tests(
        ToolContext(repository_root=tmp_path),
        RunTestsInput(timeout_seconds=1),
        config=ToolConfig(verification_timeout_seconds=1),
    )

    assert output.timed_out
    assert not output.passed
    assert output.exit_code is None
    assert output.duration_seconds < 4


@pytest.mark.parametrize(
    "path",
    ["../tests", "-q", "tests/--collect-only", "C:tests"],
)
def test_verification_models_reject_traversal_and_option_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        RunTestsInput(paths=[path])
    with pytest.raises(ValidationError):
        RunRuffInput(paths=[path])


def test_run_tests_rejects_missing_and_non_test_scopes(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("value = 1\n", encoding="utf-8")
    context = ToolContext(repository_root=tmp_path)

    with pytest.raises(ToolExecutionError, match="does not exist"):
        run_tests(context, RunTestsInput(paths=["tests/test_missing.py"]))
    with pytest.raises(ToolExecutionError, match="conservative test scope"):
        run_tests(context, RunTestsInput(paths=["src/app.py"]))


def test_run_tests_uses_argument_array_and_never_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_test(tmp_path, "def test_value():\n    assert True\n")
    inherited_cache = tmp_path / "inherited-cache"
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(inherited_cache))
    monkeypatch.setenv("REPOMIND_TEST_ENVIRONMENT_SENTINEL", "must-not-pass-through")
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("repomind.tools.verification.subprocess.run", fake_run)
    output = run_tests(
        ToolContext(repository_root=tmp_path),
        RunTestsInput(paths=["tests/test_sample.py"], max_failures=2),
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[1:3] == ["-m", "pytest"]
    assert command[-2:] == ["--maxfail=2", "-q"]
    assert captured["shell"] is False
    environment = captured["env"]
    assert isinstance(environment, dict)
    isolated_cache = Path(environment["PYTHONPYCACHEPREFIX"])
    assert isolated_cache != inherited_cache
    assert "REPOMIND_TEST_ENVIRONMENT_SENTINEL" not in environment
    assert not isolated_cache.exists()
    assert output.passed


def test_pytest_and_ruff_use_the_same_allowlisted_base_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_test(tmp_path, "def test_value():\n    assert True\n")
    source = tmp_path / "clean.py"
    source.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-parent-secret")
    monkeypatch.setenv("REPOMIND_ARBITRARY_SECRET", "private")
    monkeypatch.setenv("PYTHONUTF8", "1")
    captured_environments: list[dict[str, str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        captured_environments.append(environment.copy())
        assert kwargs["shell"] is False
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("repomind.tools.verification.subprocess.run", fake_run)
    context = ToolContext(repository_root=tmp_path)

    assert run_tests(context, RunTestsInput()).passed
    assert run_ruff(context, RunRuffInput(paths=["clean.py"])).passed

    pytest_environment, ruff_environment = captured_environments
    pycache_prefix = pytest_environment.pop("PYTHONPYCACHEPREFIX")
    assert pycache_prefix
    assert pytest_environment == ruff_environment
    assert pytest_environment["PYTHONUTF8"] == "1"
    assert pytest_environment.get("PATH")
    assert "OPENAI_API_KEY" not in pytest_environment
    assert "REPOMIND_ARBITRARY_SECRET" not in pytest_environment


def test_verifier_startup_failure_is_a_tool_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_test(tmp_path, "def test_value():\n    assert True\n")

    def fail_start(command: list[str], **kwargs: object) -> None:
        raise OSError("missing executable")

    monkeypatch.setattr("repomind.tools.verification.subprocess.run", fail_start)
    with pytest.raises(ToolExecutionError, match="Could not start"):
        run_tests(ToolContext(repository_root=tmp_path), RunTestsInput())


def test_run_ruff_reports_clean_and_lint_failure_without_fixing(tmp_path: Path) -> None:
    clean = tmp_path / "clean.py"
    bad = tmp_path / "bad.py"
    clean.write_text("value = 1\n", encoding="utf-8")
    bad.write_text("import os\n", encoding="utf-8")
    context = ToolContext(repository_root=tmp_path)

    clean_output = run_ruff(context, RunRuffInput(paths=["clean.py"]))
    bad_output = run_ruff(context, RunRuffInput(paths=["bad.py"]))

    assert clean_output.passed and clean_output.exit_code == 0
    assert not bad_output.passed and bad_output.exit_code != 0
    assert "F401" in bad_output.stdout + bad_output.stderr
    assert bad.read_text(encoding="utf-8") == "import os\n"


def test_run_ruff_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(ToolExecutionError, match="does not exist"):
        run_ruff(
            ToolContext(repository_root=tmp_path),
            RunRuffInput(paths=["missing.py"]),
        )
