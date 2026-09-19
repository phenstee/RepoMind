"""Tests for the benchmarks.agent_live_eval CLI safety gating and I/O.

These tests prove the harness never contacts OpenAI when it should not: no
OPENAI_API_KEY, no --confirm-live, or plain --help must all make zero live
calls. Constructing OpenAILLMClient is stubbed to raise, so any accidental
construction fails the test loudly instead of silently attempting a real
request.
"""

import json
from pathlib import Path

import pytest

from benchmarks import agent_live_eval
from repomind.config import Settings


class _ClientConstructedError(AssertionError):
    """Raised by the stub client if it is ever constructed; must never happen."""


class _StubOpenAILLMClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise _ClientConstructedError(
            "OpenAILLMClient must not be constructed without passing authorization"
        )


@pytest.fixture(autouse=True)
def _no_openai_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub repomind.llm.OpenAILLMClient everywhere this module might import it."""

    import repomind.llm as repomind_llm

    monkeypatch.setattr(repomind_llm, "OpenAILLMClient", _StubOpenAILLMClient, raising=False)


def _patch_settings(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> Settings:
    """Isolate tests from the developer's real .env file and the get_settings()
    lru_cache singleton, which would otherwise leak a real OPENAI_API_KEY into
    these safety-gating tests.
    """

    values: dict[str, object] = {"openai_api_key": None}
    values.update(overrides)
    settings = Settings(_env_file=None, **values)
    monkeypatch.setattr(agent_live_eval, "get_settings", lambda: settings)
    return settings


def test_help_makes_no_live_calls_and_exits_zero(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as excinfo:
        agent_live_eval.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--confirm-live" in out


def test_missing_api_key_causes_zero_live_calls_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _patch_settings(monkeypatch, openai_api_key=None)
    exit_code = agent_live_eval.main(["--confirm-live"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err


def test_whitespace_only_api_key_causes_zero_live_calls_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    # The autouse stub raises if OpenAILLMClient is constructed at all, so
    # reaching exit code 2 proves the whitespace-only key was rejected by the
    # gate rather than passed through to client construction.
    _patch_settings(monkeypatch, openai_api_key="   \t ")
    exit_code = agent_live_eval.main(["--confirm-live"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err


def test_missing_confirm_flag_causes_zero_live_calls_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _patch_settings(monkeypatch, openai_api_key="sk-test-not-real")
    exit_code = agent_live_eval.main([])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "confirm-live" in err


def test_dry_invocation_with_neither_makes_zero_live_calls(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _patch_settings(monkeypatch, openai_api_key=None)
    exit_code = agent_live_eval.main([])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err
    assert "confirm-live" in err


def test_import_alone_makes_no_live_calls() -> None:
    # The fixture-time import of benchmarks.agent_live_eval, plus a fresh
    # reload here, must not construct any client or touch the network.
    import importlib

    importlib.reload(agent_live_eval)


def test_invalid_max_iterations_rejected_before_authorization_check(
    capsys: pytest.CaptureFixture,
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        agent_live_eval.main(["--max-iterations", "0", "--confirm-live"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--max-iterations" in err


def test_invalid_limit_rejected(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as excinfo:
        agent_live_eval.main(["--limit", "0", "--confirm-live"])
    assert excinfo.value.code == 2


def test_unknown_case_id_raises_before_any_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch, openai_api_key="sk-test-not-real")

    # Authorization passes, but the stub client would raise if constructed;
    # replace it with a client that raises only if *used*, so we can prove
    # the unknown-case-id failure happens before any generate_structured call.
    class _UnusedClient:
        model = "fake-model"

        def generate_structured(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("must not call the model for an unknown case id")

    import repomind.llm as repomind_llm

    monkeypatch.setattr(repomind_llm, "OpenAILLMClient", lambda *a, **k: _UnusedClient())

    with pytest.raises(SystemExit, match="Unknown case id"):
        agent_live_eval.main(["--confirm-live", "--case", "does-not-exist"])


def test_select_cases_filters_by_id_and_limit() -> None:
    from repomind.evaluation.agent_navigation_fixtures import benchmark_cases

    cases = benchmark_cases(indexed=False)
    all_ids = [case.id for case in cases]

    selected = agent_live_eval._select_cases(
        cases, retrieval_mode="filesystem", case_ids=[all_ids[0]], limit=None
    )
    assert [case.id for case in selected] == [all_ids[0]]

    limited = agent_live_eval._select_cases(
        cases, retrieval_mode="filesystem", case_ids=None, limit=2
    )
    assert len(limited) == 2


def test_build_parser_default_max_iterations_matches_shared_default() -> None:
    from repomind.evaluation import DEFAULT_LIVE_MAX_ITERATIONS

    parser = agent_live_eval.build_parser()
    args = parser.parse_args([])
    assert args.max_iterations == DEFAULT_LIVE_MAX_ITERATIONS
    assert args.retrieval_mode == "filesystem"
    assert args.confirm_live is False


def test_existing_offline_scripted_benchmark_remains_valid(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    # Prompt C's refactor renamed private benchmarks.agent_navigation_eval
    # helpers to public ones so the live harness can reuse the exact same
    # dataset; this proves the offline scripted benchmark still runs
    # end-to-end afterward, unchanged.
    from benchmarks import agent_navigation_eval

    monkeypatch.setattr("sys.argv", ["agent_navigation_eval.py"])
    agent_navigation_eval.main()

    out = capsys.readouterr().out
    assert "REPO-AGENT-EVAL-V1 AGENT NAVIGATION COMPARISON" in out
    assert "STALE-INDEX SAFETY DEMONSTRATION" in out
    assert "task_success=True" in out


def test_full_run_with_fake_client_writes_expected_json_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    _patch_settings(monkeypatch, openai_api_key="sk-test-not-real")

    class _FakeClient:
        model = "fake-model"

        def generate_structured(self, prompt, response_model, *, system_prompt=None, temperature=None):
            return response_model.model_validate(
                {"action": "final", "final_answer": "The rate-limit error code is ERR_RATE_LIMIT_EXCEEDED_42."}
            )

    import repomind.llm as repomind_llm

    monkeypatch.setattr(repomind_llm, "OpenAILLMClient", lambda *a, **k: _FakeClient())

    output_path = tmp_path / "results.json"
    exit_code = agent_live_eval.main(
        [
            "--confirm-live",
            "--case",
            "literal-error-string",
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["harness_schema_version"] == "agent-live-eval-v1"
    assert payload["benchmark_version"] == "repo-agent-eval-v1"
    assert payload["model"] == "fake-model"
    assert payload["case_ids"] == ["literal-error-string"]
    assert len(payload["runs"]) == 1
    run = payload["runs"][0]
    assert run["report"]["retrieval_mode"] == "filesystem"
    assert run["report"]["case_results"][0]["task_success"] is True

    console_output = capsys.readouterr().out
    assert "LIVE-MODEL AGENT NAVIGATION" in console_output
    assert "sk-test-not-real" not in console_output
    assert "sk-test-not-real" not in output_path.read_text(encoding="utf-8")


def test_indexed_mode_run_uses_case_scoped_retrieval_for_an_arbitrary_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    _patch_settings(monkeypatch, openai_api_key="sk-test-not-real")

    # A real model may issue a query that has nothing to do with the
    # benchmark's canonical scripted query text; the case must still be
    # graded correctly rather than treated as a retrieval miss.
    class _ArbitraryQueryClient:
        model = "fake-model"

        def __init__(self) -> None:
            self._calls = 0

        def generate_structured(self, prompt, response_model, *, system_prompt=None, temperature=None):
            self._calls += 1
            if self._calls == 1:
                return response_model.model_validate(
                    {
                        "action": "tool",
                        "tool_name": "indexed_code_search",
                        "tool_arguments": {"query": "an arbitrary phrasing of the model's own choosing"},
                    }
                )
            if self._calls == 2:
                return response_model.model_validate(
                    {
                        "action": "tool",
                        "tool_name": "read_file",
                        "tool_arguments": {"path": "src/jobs/store.py"},
                    }
                )
            return response_model.model_validate(
                {
                    "action": "final",
                    "final_answer": (
                        "JobStore.claim atomically claims one queued job for this worker "
                        "using SKIP LOCKED."
                    ),
                }
            )

    import repomind.llm as repomind_llm

    monkeypatch.setattr(repomind_llm, "OpenAILLMClient", lambda *a, **k: _ArbitraryQueryClient())

    output_path = tmp_path / "results.json"
    exit_code = agent_live_eval.main(
        [
            "--confirm-live",
            "--retrieval-mode",
            "indexed",
            "--case",
            "exact-symbol",
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    run = payload["runs"][0]
    case_result = run["report"]["case_results"][0]
    assert case_result["case_id"] == "exact-symbol"
    assert case_result["indexed_search_calls"] == 1
    assert case_result["verified_retrieval_followup"] is True
    assert case_result["task_success"] is True

    console_output = capsys.readouterr().out
    assert "an arbitrary phrasing of the model's own choosing" in console_output


def test_indexed_mode_index_miss_case_still_returns_zero_indexed_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch, openai_api_key="sk-test-not-real")

    class _AlwaysSearchThenFinalClient:
        model = "fake-model"

        def __init__(self) -> None:
            self._calls = 0

        def generate_structured(self, prompt, response_model, *, system_prompt=None, temperature=None):
            self._calls += 1
            if self._calls == 1:
                return response_model.model_validate(
                    {
                        "action": "tool",
                        "tool_name": "indexed_code_search",
                        "tool_arguments": {"query": "retry backoff multiplier configuration"},
                    }
                )
            if self._calls == 2:
                return response_model.model_validate(
                    {
                        "action": "tool",
                        "tool_name": "read_file",
                        "tool_arguments": {"path": "src/config/retry.py"},
                    }
                )
            return response_model.model_validate(
                {
                    "action": "final",
                    "final_answer": "RETRY_BACKOFF_MULTIPLIER = 2.0 is defined in src/config/retry.py.",
                }
            )

    import repomind.llm as repomind_llm

    monkeypatch.setattr(repomind_llm, "OpenAILLMClient", lambda *a, **k: _AlwaysSearchThenFinalClient())

    captured: dict[str, object] = {}

    def _capture(result):
        captured["result"] = result

    monkeypatch.setattr(agent_live_eval, "_print_console_summary", _capture)

    exit_code = agent_live_eval.main(
        [
            "--confirm-live",
            "--retrieval-mode",
            "indexed",
            "--case",
            "index-miss-filesystem-fallback",
        ]
    )
    assert exit_code == 0

    result = captured["result"]
    run = result.runs[0]
    case_result = run.report.case_results[0]
    assert case_result.indexed_search_calls == 1
    assert case_result.task_success is True
    # The index-miss case is configured with zero fixture chunks, so a
    # zero-result indexed search never marks anything "pending" to verify.
    assert case_result.verified_retrieval_followup is True
    # The attempted query is still persisted, despite the zero-result search.
    assert run.indexed_search_queries_by_case == {
        "index-miss-filesystem-fallback": ("retry backoff multiplier configuration",)
    }


def test_indexed_mode_multi_case_run_aggregates_metrics_and_queries_for_both_cases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_settings(monkeypatch, openai_api_key="sk-test-not-real")

    # _select_cases preserves benchmark_cases() definition order regardless
    # of --case argument order, and _run_indexed_mode evaluates cases one at
    # a time, in that order, reusing this same client - so a single flat
    # script covering both cases in sequence is enough; no per-prompt
    # detection needed. exact-symbol runs first, then literal-error-string.
    class _FakeClient:
        model = "fake-model"

        def __init__(self) -> None:
            self._calls = 0
            self._script = [
                {
                    "action": "tool",
                    "tool_name": "indexed_code_search",
                    "tool_arguments": {"query": "atomic job claiming"},
                },
                {
                    "action": "tool",
                    "tool_name": "read_file",
                    "tool_arguments": {"path": "src/jobs/store.py"},
                },
                {
                    "action": "final",
                    "final_answer": (
                        "JobStore.claim atomically claims one queued job for this "
                        "worker using SKIP LOCKED."
                    ),
                },
                {
                    "action": "tool",
                    "tool_name": "indexed_code_search",
                    "tool_arguments": {"query": "rate limit error constant"},
                },
                {
                    "action": "tool",
                    "tool_name": "read_file",
                    "tool_arguments": {"path": "src/errors.py"},
                },
                {
                    "action": "final",
                    "final_answer": "The rate-limit error code is ERR_RATE_LIMIT_EXCEEDED_42.",
                },
            ]

        def generate_structured(self, prompt, response_model, *, system_prompt=None, temperature=None):
            response = self._script[self._calls]
            self._calls += 1
            return response_model.model_validate(response)

    import repomind.llm as repomind_llm

    monkeypatch.setattr(repomind_llm, "OpenAILLMClient", lambda *a, **k: _FakeClient())

    output_path = tmp_path / "results.json"
    exit_code = agent_live_eval.main(
        [
            "--confirm-live",
            "--retrieval-mode",
            "indexed",
            "--case",
            "exact-symbol",
            "--case",
            "literal-error-string",
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["case_ids"] == ["exact-symbol", "literal-error-string"]
    run = payload["runs"][0]
    assert run["report"]["case_count"] == 2
    assert {c["case_id"] for c in run["report"]["case_results"]} == {
        "exact-symbol",
        "literal-error-string",
    }
    # Aggregate llm_calls must reflect BOTH cases (3 each), not just one.
    assert run["metrics"]["llm_calls"] == 6
    queries = run["indexed_search_queries_by_case"]
    assert queries["exact-symbol"] == ["atomic job claiming"]
    assert queries["literal-error-string"] == ["rate limit error constant"]
