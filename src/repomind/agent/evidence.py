"""Deterministic observed-source evidence derived from a finished agent run.

This module answers one narrow question: *which current working-tree source
locations did RepoMind actually observe while investigating?* It deliberately
does **not** claim that every statement in ``final_answer`` is proved by these
locations - the model writes the answer, while this extractor only reports the
successful filesystem observations the run really performed.

Evidence therefore comes from successful tool observations, never from
model-authored path/line references. Persisted index hits
(``indexed_code_search``) are navigation hints that may be stale, so they never
become evidence on their own; only a later successful ``read_file`` of the
current file does. Enforcing *whether* indexed finalization is safe stays with
``agent.indexed._IndexedGroundingPolicy``; this extractor merely reports what
was observed.

The extractor is pure: no filesystem, model, database, or embedding access. It
reads only validated observations already present in the run, and ignores
malformed or unexpected payloads rather than failing an otherwise successful
investigation.
"""

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from repomind.agent.models import AgentStep
from repomind.ingestion import validate_repository_relative_path

DEFAULT_MAX_OBSERVED_LOCATIONS = 25

ObservedVia = Literal["read_file", "search_code", "find_symbol"]

_READ_FILE_TOOL = "read_file"
_MATCH_TOOLS: dict[str, ObservedVia] = {
    "search_code": "search_code",
    "find_symbol": "find_symbol",
}


class ObservedSourceLocation(BaseModel):
    """One current-worktree location a successful tool observation reported."""

    model_config = ConfigDict(frozen=True)

    relative_path: Path
    start_line: int = Field(ge=1, strict=True)
    end_line: int = Field(ge=1, strict=True)
    observed_via: ObservedVia

    @model_validator(mode="after")
    def _validate_range(self) -> "ObservedSourceLocation":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class ObservedSourceEvidence(BaseModel):
    """Bounded, deduplicated locations observed during one investigation."""

    model_config = ConfigDict(frozen=True)

    locations: tuple[ObservedSourceLocation, ...] = ()
    truncated: bool = False


def _relative_path(value: Any) -> Path | None:
    if not isinstance(value, (str, Path)):
        return None
    try:
        return validate_repository_relative_path(Path(value))
    except (TypeError, ValueError):
        return None


def _line_number(value: Any) -> int | None:
    # bool is an int subclass; a JSON true must never become line 1.
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _read_file_location(output: dict[str, Any]) -> ObservedSourceLocation | None:
    path = _relative_path(output.get("path"))
    start = _line_number(output.get("start_line"))
    end = _line_number(output.get("end_line"))
    # An empty file reports the 0/0 range; it was observed, but there is no
    # citable line range, so it is skipped rather than coerced.
    if path is None or start is None or end is None or end < start:
        return None
    return ObservedSourceLocation(
        relative_path=path, start_line=start, end_line=end, observed_via=_READ_FILE_TOOL
    )


def _match_locations(
    output: dict[str, Any], observed_via: ObservedVia
) -> Iterator[ObservedSourceLocation]:
    matches = output.get("matches")
    if not isinstance(matches, list):
        return
    for match in matches:
        if not isinstance(match, dict):
            continue
        path = _relative_path(match.get("path"))
        line = _line_number(match.get("line_number"))
        if path is None or line is None:
            continue
        yield ObservedSourceLocation(
            relative_path=path, start_line=line, end_line=line, observed_via=observed_via
        )


def _observed_locations(steps: Sequence[AgentStep]) -> Iterator[ObservedSourceLocation]:
    """Yield locations in the order the run observed them."""

    for step in steps:
        observation = step.observation
        if observation is None or not observation.success:
            continue
        output = observation.output
        if not isinstance(output, dict):
            continue

        if observation.tool_name == _READ_FILE_TOOL:
            location = _read_file_location(output)
            if location is not None:
                yield location
            continue

        observed_via = _MATCH_TOOLS.get(observation.tool_name)
        if observed_via is not None:
            yield from _match_locations(output, observed_via)


def extract_observed_source_evidence(
    steps: Sequence[AgentStep],
    *,
    limit: int = DEFAULT_MAX_OBSERVED_LOCATIONS,
) -> ObservedSourceEvidence:
    """Collect bounded, deduplicated current-source locations from ``steps``.

    Ordering follows the run's own observation order, so the result reads as
    the investigation's trail. Two observations of the identical location
    collapse to the first one, which keeps the provenance that discovered it.
    Truncation is reported rather than applied silently.
    """

    if limit < 1:
        raise ValueError("limit must be positive")

    locations: list[ObservedSourceLocation] = []
    seen: set[tuple[str, int, int]] = set()
    truncated = False
    for location in _observed_locations(steps):
        identity = (location.relative_path.as_posix(), location.start_line, location.end_line)
        if identity in seen:
            continue
        if len(locations) >= limit:
            truncated = True
            break
        seen.add(identity)
        locations.append(location)
    return ObservedSourceEvidence(locations=tuple(locations), truncated=truncated)
