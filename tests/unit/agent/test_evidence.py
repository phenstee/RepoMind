"""Deterministic tests for observed-source evidence extraction."""

from pathlib import Path
from typing import Any

import pytest

from repomind.agent import (
    DEFAULT_MAX_OBSERVED_LOCATIONS,
    AgentDecision,
    AgentStep,
    ToolObservation,
    extract_observed_source_evidence,
)


def _step(
    iteration: int,
    tool_name: str,
    output: dict[str, Any] | None,
    *,
    success: bool = True,
    arguments: dict[str, Any] | None = None,
) -> AgentStep:
    observation = (
        ToolObservation(
            tool_name=tool_name,
            arguments=arguments or {},
            success=True,
            output=output,
        )
        if success
        else ToolObservation(
            tool_name=tool_name,
            arguments=arguments or {},
            success=False,
            error="tool failed",
        )
    )
    return AgentStep(
        iteration=iteration,
        decision=AgentDecision(
            action="tool", tool_name=tool_name, tool_arguments=arguments or {}
        ),
        observation=observation,
    )


def _read_file_output(
    path: str,
    start: int,
    end: int,
    *,
    requested_start: int | None = None,
    requested_end: int | None = None,
) -> dict[str, Any]:
    return {
        "path": path,
        "requested_start_line": requested_start if requested_start is not None else start,
        "requested_end_line": requested_end if requested_end is not None else end,
        "start_line": start,
        "end_line": end,
        "content": "secret source text",
        "total_lines": end,
        "sha256": "0" * 64,
    }


def _search_output(*matches: tuple[str, int]) -> dict[str, Any]:
    return {
        "query": "anything",
        "matches": [
            {"path": path, "line_number": line, "line": "secret source line"}
            for path, line in matches
        ],
        "truncated": False,
    }


def _symbol_output(*matches: tuple[str, int]) -> dict[str, Any]:
    return {
        "symbol": "Thing",
        "matches": [
            {"path": path, "line_number": line, "line": "def Thing():", "kind": "function"}
            for path, line in matches
        ],
        "truncated": False,
    }


def _indexed_output(*locations: tuple[str, int, int]) -> dict[str, Any]:
    return {
        "query": "anything",
        "locations": [
            {
                "relative_path": path,
                "start_line": start,
                "end_line": end,
                "rank": rank,
                "chunk_kind": "function",
                "qualified_symbol_name": "Thing.method",
                "sources": ["semantic"],
            }
            for rank, (path, start, end) in enumerate(locations, start=1)
        ],
        "result_count": len(locations),
    }


def test_read_file_uses_actual_returned_range() -> None:
    # The model requested 1-500; the tool clamped to the real file length
    # (10-42). If the extractor ever read requested_start_line/
    # requested_end_line instead of start_line/end_line, this would assert
    # 1-500 and the test below would fail.
    output = _read_file_output("src/app.py", 10, 42, requested_start=1, requested_end=500)
    assert output["requested_start_line"] == 1
    assert output["requested_end_line"] == 500
    steps = (_step(1, "read_file", output),)

    evidence = extract_observed_source_evidence(steps)

    assert len(evidence.locations) == 1
    location = evidence.locations[0]
    assert location.relative_path == Path("src/app.py")
    assert location.start_line == 10
    assert location.end_line == 42
    assert location.observed_via == "read_file"
    assert evidence.truncated is False


def test_search_code_matches_become_single_line_evidence() -> None:
    steps = (_step(1, "search_code", _search_output(("src/a.py", 7), ("src/b.py", 9))),)

    evidence = extract_observed_source_evidence(steps)

    assert [(loc.relative_path.as_posix(), loc.start_line, loc.end_line) for loc in evidence.locations] == [
        ("src/a.py", 7, 7),
        ("src/b.py", 9, 9),
    ]
    assert {loc.observed_via for loc in evidence.locations} == {"search_code"}


def test_find_symbol_matches_become_evidence() -> None:
    steps = (_step(1, "find_symbol", _symbol_output(("src/models.py", 15))),)

    evidence = extract_observed_source_evidence(steps)

    assert len(evidence.locations) == 1
    assert evidence.locations[0].observed_via == "find_symbol"
    assert evidence.locations[0].start_line == 15


def test_failed_observations_produce_no_evidence() -> None:
    steps = (
        _step(1, "read_file", None, success=False),
        _step(2, "search_code", None, success=False),
        _step(3, "find_symbol", None, success=False),
    )

    assert extract_observed_source_evidence(steps).locations == ()


def test_indexed_search_alone_is_never_evidence() -> None:
    """A persisted index hit may be stale; it is navigation, not observation."""

    steps = (_step(1, "indexed_code_search", _indexed_output(("src/stale.py", 1, 20))),)

    evidence = extract_observed_source_evidence(steps)

    assert evidence.locations == ()
    assert evidence.truncated is False


def test_indexed_hit_followed_by_read_file_reports_only_the_current_read() -> None:
    steps = (
        _step(1, "indexed_code_search", _indexed_output(("src/app.py", 1, 200))),
        _step(2, "read_file", _read_file_output("src/app.py", 30, 48)),
    )

    evidence = extract_observed_source_evidence(steps)

    assert len(evidence.locations) == 1
    location = evidence.locations[0]
    # The stale 1-200 index range must not appear; only the current read does.
    assert (location.start_line, location.end_line) == (30, 48)
    assert location.observed_via == "read_file"


def test_non_source_tools_produce_no_evidence() -> None:
    steps = (
        _step(1, "list_directory", {"path": "src", "recursive": False, "entries": [], "truncated": False}),
        _step(2, "git_status", {"branch": "main", "changed_files": [], "clean": True}),
        _step(3, "git_diff", {"content": "", "truncated": False, "staged": False, "path": None}),
    )

    assert extract_observed_source_evidence(steps).locations == ()


def test_ordering_follows_observation_order_deterministically() -> None:
    steps = (
        _step(1, "search_code", _search_output(("src/z.py", 3))),
        _step(2, "read_file", _read_file_output("src/a.py", 1, 5)),
        _step(3, "find_symbol", _symbol_output(("src/m.py", 8))),
    )

    first = extract_observed_source_evidence(steps)
    second = extract_observed_source_evidence(steps)

    identities = [(loc.relative_path.as_posix(), loc.start_line) for loc in first.locations]
    assert identities == [("src/z.py", 3), ("src/a.py", 1), ("src/m.py", 8)]
    assert first == second


def test_exact_duplicate_locations_are_deduplicated() -> None:
    steps = (
        _step(1, "read_file", _read_file_output("src/app.py", 4, 9)),
        _step(2, "read_file", _read_file_output("src/app.py", 4, 9)),
        # Same line reached by a different tool is still the same location.
        _step(3, "search_code", _search_output(("src/other.py", 12))),
        _step(4, "find_symbol", _symbol_output(("src/other.py", 12))),
    )

    evidence = extract_observed_source_evidence(steps)

    assert len(evidence.locations) == 2
    assert evidence.locations[0].observed_via == "read_file"
    # Deduplication keeps the provenance that observed the location first.
    assert evidence.locations[1].observed_via == "search_code"
    assert evidence.truncated is False


def test_evidence_is_hard_bounded_and_truncation_is_reported() -> None:
    many = _search_output(*[(f"src/f{index}.py", 1) for index in range(60)])
    steps = (_step(1, "search_code", many),)

    evidence = extract_observed_source_evidence(steps)

    assert len(evidence.locations) == DEFAULT_MAX_OBSERVED_LOCATIONS
    assert evidence.truncated is True


def test_truncation_is_not_reported_when_everything_fits() -> None:
    exact = _search_output(
        *[(f"src/f{index}.py", 1) for index in range(DEFAULT_MAX_OBSERVED_LOCATIONS)]
    )
    steps = (_step(1, "search_code", exact),)

    evidence = extract_observed_source_evidence(steps)

    assert len(evidence.locations) == DEFAULT_MAX_OBSERVED_LOCATIONS
    assert evidence.truncated is False


def test_custom_limit_is_honored_and_must_be_positive() -> None:
    steps = (_step(1, "search_code", _search_output(("src/a.py", 1), ("src/b.py", 2))),)

    bounded = extract_observed_source_evidence(steps, limit=1)
    assert len(bounded.locations) == 1
    assert bounded.truncated is True

    with pytest.raises(ValueError, match="limit must be positive"):
        extract_observed_source_evidence(steps, limit=0)


@pytest.mark.parametrize(
    "output",
    [
        {"path": "/etc/passwd", "start_line": 1, "end_line": 2},
        {"path": "../outside.py", "start_line": 1, "end_line": 2},
        {"path": "src/app.py", "start_line": 0, "end_line": 0},
        {"path": "src/app.py", "start_line": 9, "end_line": 4},
        {"path": "src/app.py", "start_line": "1", "end_line": 2},
        {"path": "src/app.py", "start_line": True, "end_line": 2},
        {"path": None, "start_line": 1, "end_line": 2},
        {"start_line": 1, "end_line": 2},
        {},
    ],
)
def test_malformed_read_file_payloads_fail_closed(output: dict[str, Any]) -> None:
    steps = (_step(1, "read_file", output),)

    assert extract_observed_source_evidence(steps).locations == ()


@pytest.mark.parametrize(
    "matches",
    [
        "not-a-list",
        [{"path": "/abs/x.py", "line_number": 3}],
        [{"path": "src/a.py", "line_number": 0}],
        [{"path": "src/a.py"}],
        [{"line_number": 3}],
        ["not-a-dict"],
    ],
)
def test_malformed_match_payloads_fail_closed(matches: Any) -> None:
    steps = (_step(1, "search_code", {"query": "q", "matches": matches, "truncated": False}),)

    assert extract_observed_source_evidence(steps).locations == ()


def test_final_steps_without_observations_are_ignored() -> None:
    steps = (
        AgentStep(
            iteration=1,
            decision=AgentDecision(action="final", final_answer="src/app.py lines 1-9 say so"),
        ),
    )

    # A model-written path reference in the answer is not an observation.
    assert extract_observed_source_evidence(steps).locations == ()


def test_evidence_never_carries_source_content_or_absolute_paths() -> None:
    steps = (
        _step(1, "read_file", _read_file_output("src/app.py", 1, 3)),
        _step(2, "search_code", _search_output(("src/b.py", 4))),
    )

    payload = extract_observed_source_evidence(steps).model_dump(mode="json")

    serialized = str(payload)
    assert "secret source" not in serialized
    assert "sha256" not in serialized
    for location in payload["locations"]:
        assert set(location) == {"relative_path", "start_line", "end_line", "observed_via"}
        assert not Path(location["relative_path"]).is_absolute()
