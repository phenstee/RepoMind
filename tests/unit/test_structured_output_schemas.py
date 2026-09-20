"""Strict Structured Output compatibility for every production response model.

A live request failed with::

    Invalid schema for response_format 'AgentDecision':
    In context=('properties', 'tool_arguments', 'anyOf', '0'),
    'additionalProperties' is required to be supplied and to be false.

These tests reproduce that class of defect offline, so the next incompatible
schema is caught here instead of by a paid request. Everything is
NETWORK-FREE: only the installed OpenAI SDK's local schema-conversion helper
is used, and no client is constructed.

``openai.lib._pydantic.to_strict_json_schema`` is a private SDK internal. It
is imported by this audit only, deliberately never by production code, so
RepoMind carries no runtime dependency on an OpenAI private helper.
"""

import ast
from pathlib import Path
from typing import Any

import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, ValidationError

from repomind.agent.models import AgentDecision, AgentDecisionResponse
from repomind.coding.models import CodingPlan, CodingReview
from repomind.rag.models import GroundedLLMResponse
from repomind.retrieval.reranking import RerankLLMResponse

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "repomind"

# Every Pydantic model RepoMind passes to generate_structured() in production.
# test_every_production_structured_call_site_is_audited keeps this dict in step
# with the real call sites, so a newly added structured call cannot quietly
# skip the audit below.
PRODUCTION_RESPONSE_MODELS: dict[str, type[BaseModel]] = {
    "AgentDecisionResponse": AgentDecisionResponse,
    "CodingPlan": CodingPlan,
    "CodingReview": CodingReview,
    "GroundedLLMResponse": GroundedLLMResponse,
    "RerankLLMResponse": RerankLLMResponse,
}

# Strict Structured Outputs accept only this set of string ``format`` values.
# There is deliberately NO allowlist of known-bad formats: a production
# response model that needs an unsupported one (as CodingPlanStep.likely_paths
# once did with "path") is fixed at the model, not excused here.
_SUPPORTED_STRING_FORMATS = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "ipv4", "ipv6", "uuid"}
)

# JSON Schema constructs OpenAI's strict Structured Outputs do not support, per
# https://platform.openai.com/docs/guides/structured-outputs (root-level and
# per-object keyword restrictions) as implemented by the installed SDK's local
# `to_strict_json_schema` conversion. This is not a complete JSON Schema
# vocabulary - only the subset worth pinning because RepoMind's own models
# could plausibly grow one of these (a `Literal` union naturally avoids
# `allOf`/`not`, but a future field could reach for `uniqueItems`,
# `minProperties`, or a conditional schema without anyone remembering this
# constraint). Supported/structural keywords such as type, properties,
# required, additionalProperties, items, anyOf, enum, const, $defs, $ref,
# title, description, default, and the numeric/length constraints RepoMind
# already uses (minLength, maxLength, minimum, maximum, minItems, maxItems)
# are deliberately absent from this set.
_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "allOf",
        "not",
        "dependentRequired",
        "dependentSchemas",
        "if",
        "then",
        "else",
        "unevaluatedProperties",
        "propertyNames",
        "minProperties",
        "maxProperties",
        "unevaluatedItems",
        "contains",
        "minContains",
        "maxContains",
        "uniqueItems",
        "patternProperties",
    }
)


def _open_objects(node: Any, path: tuple[str | int, ...] = ()) -> list[tuple[str, str]]:
    """Return every object branch that does not close with additionalProperties=false."""

    problems: list[tuple[str, str]] = []
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            declared = node.get("additionalProperties", "<missing>")
            if declared is not False:
                problems.append((str(path), repr(declared)))
        for key, value in node.items():
            problems.extend(_open_objects(value, (*path, key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            problems.extend(_open_objects(value, (*path, index)))
    return problems


def _required_gaps(node: Any, path: tuple[str | int, ...] = ()) -> list[tuple[str, list[str]]]:
    """Return object branches whose properties are not all listed in ``required``."""

    gaps: list[tuple[str, list[str]]] = []
    if isinstance(node, dict):
        if "properties" in node:
            missing = sorted(set(node["properties"]) - set(node.get("required", [])))
            if missing:
                gaps.append((str(path), missing))
        for key, value in node.items():
            gaps.extend(_required_gaps(value, (*path, key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            gaps.extend(_required_gaps(value, (*path, index)))
    return gaps


def _string_formats(node: Any) -> set[str]:
    formats: set[str] = set()
    if isinstance(node, dict):
        if isinstance(node.get("format"), str):
            formats.add(node["format"])
        for value in node.values():
            formats |= _string_formats(value)
    elif isinstance(node, list):
        for value in node:
            formats |= _string_formats(value)
    return formats


def _unsupported_keywords(node: Any) -> set[str]:
    """Return every JSON Schema keyword present that strict mode does not support."""

    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _UNSUPPORTED_SCHEMA_KEYWORDS:
                found.add(key)
            found |= _unsupported_keywords(value)
    elif isinstance(node, list):
        for value in node:
            found |= _unsupported_keywords(value)
    return found


def _call_target_name(func: ast.expr) -> str | None:
    """Return the called name for either ``name(...)`` or ``x.attr(...)``."""

    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _resolve_model_expr(expr: ast.expr) -> str:
    """Resolve a response-model AST expression to its simple class name.

    Handles a bare name (``CodingPlan``) and one level of dotted access
    (``models.CodingPlan``), which covers every way RepoMind references an
    imported model. Anything else - a call, a subscript, a conditional
    expression - cannot be resolved statically, and the caller must fail
    loudly rather than silently drop the call site.
    """

    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    raise ValueError(
        f"cannot statically resolve response model expression: {ast.dump(expr)}"
    )


def _response_model_expr(call: ast.Call, *, is_bound_method: bool) -> ast.expr | None:
    """Return the AST node passed as ``response_model`` to one matched call.

    The free-function wrapper is ``generate_structured(provider, prompt,
    response_model, ...)``: the model is the third positional argument. A
    direct bound-method call, ``provider.generate_structured(prompt,
    response_model, ...)``, has no explicit ``self``/``provider`` argument in
    the AST, so the model is the second positional argument there. Both shapes
    also accept ``response_model=`` as a keyword.
    """

    positional_index = 1 if is_bound_method else 2
    if len(call.args) > positional_index:
        return call.args[positional_index]
    for keyword in call.keywords:
        if keyword.arg == "response_model":
            return keyword.value
    return None


def _extract_structured_call_models(source: str, *, filename: str = "<snippet>") -> set[str]:
    """Return every response-model name used in a ``generate_structured`` call.

    Recognizes, at minimum:

    * the free-function wrapper, positional or ``response_model=`` keyword:
      ``generate_structured(provider, prompt, Model)`` /
      ``generate_structured(provider, prompt, response_model=Model)``
    * a direct bound-method call, positional or keyword:
      ``provider.generate_structured(prompt, Model)`` /
      ``provider.generate_structured(prompt, response_model=Model)``
    * one level of dotted model access: ``models.CodingPlan``

    A call whose target name is ``generate_structured`` but whose
    response-model argument cannot be resolved this way raises, rather than
    being silently skipped - a call shape this helper does not yet recognize
    must be a loud test failure, not a quiet gap in the audit.
    """

    found: set[str] = set()
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_target_name(node.func)
        if name != "generate_structured":
            continue
        is_bound_method = isinstance(node.func, ast.Attribute)
        model_expr = _response_model_expr(node, is_bound_method=is_bound_method)
        if model_expr is None:
            raise AssertionError(
                f"{filename}:{node.lineno}: a generate_structured call has no resolvable "
                "response-model argument; update _response_model_expr to handle this call shape"
            )
        found.add(_resolve_model_expr(model_expr))
    return found


def _structured_call_site_models() -> set[str]:
    """Parse src/ for the response model named in every generate_structured call."""

    found: set[str] = set()
    for source_path in sorted(_SRC_ROOT.rglob("*.py")):
        found |= _extract_structured_call_models(
            source_path.read_text(encoding="utf-8"), filename=str(source_path)
        )
    return found


@pytest.mark.parametrize("model_name", sorted(PRODUCTION_RESPONSE_MODELS))
def test_production_response_model_has_no_open_object(model_name: str) -> None:
    schema = to_strict_json_schema(PRODUCTION_RESPONSE_MODELS[model_name])

    problems = _open_objects(schema)

    assert problems == [], (
        f"{model_name} contains an object that strict Structured Outputs would reject; "
        f"every object must declare additionalProperties=false: {problems}"
    )


@pytest.mark.parametrize("model_name", sorted(PRODUCTION_RESPONSE_MODELS))
def test_production_response_model_marks_every_property_required(model_name: str) -> None:
    # Strict mode also requires each object to list all of its properties in
    # `required`; optional fields must be expressed as nullable instead.
    schema = to_strict_json_schema(PRODUCTION_RESPONSE_MODELS[model_name])

    assert _required_gaps(schema) == []


@pytest.mark.parametrize("model_name", sorted(PRODUCTION_RESPONSE_MODELS))
def test_production_response_model_uses_only_supported_string_formats(model_name: str) -> None:
    schema = to_strict_json_schema(PRODUCTION_RESPONSE_MODELS[model_name])

    unsupported = sorted(
        value for value in _string_formats(schema) if value not in _SUPPORTED_STRING_FORMATS
    )

    assert unsupported == [], (
        f"{model_name} uses a string format strict Structured Outputs reject: {unsupported}. "
        "Annotate the field with WithJsonSchema instead of allowlisting it here."
    )


@pytest.mark.parametrize("model_name", sorted(PRODUCTION_RESPONSE_MODELS))
def test_production_response_model_uses_no_unsupported_schema_keywords(model_name: str) -> None:
    schema = to_strict_json_schema(PRODUCTION_RESPONSE_MODELS[model_name])

    found = sorted(_unsupported_keywords(schema))

    assert found == [], (
        f"{model_name} uses a JSON Schema construct strict Structured Outputs do not support: "
        f"{found}"
    )


def test_every_production_structured_call_site_is_audited() -> None:
    # Drift guard: the audited set must equal the models actually requested in
    # src/, so a new generate_structured call cannot bypass these checks.
    assert _structured_call_site_models() == set(PRODUCTION_RESPONSE_MODELS)


def test_unsupported_keyword_detector_flags_a_synthetic_bad_construct() -> None:
    schema = {"type": "object", "properties": {"x": {"allOf": [{"type": "string"}]}}}

    assert _unsupported_keywords(schema) == {"allOf"}


def test_unsupported_keyword_detector_does_not_flag_supported_constructs() -> None:
    # A schema shaped like RepoMind's real production schemas: structural
    # keywords, anyOf-nullable fields, and numeric/length constraints.
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "x": {
                "anyOf": [
                    {"type": "string", "minLength": 1, "maxLength": 10},
                    {"type": "null"},
                ]
            },
            "y": {"type": "integer", "minimum": 0, "maximum": 10},
            "z": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 8},
        },
        "required": ["x", "y", "z"],
        "$defs": {},
        "title": "X",
        "description": "d",
        "default": None,
        "enum": ["a", "b"],
        "const": "a",
    }

    assert _unsupported_keywords(schema) == set()


def test_ast_helper_detects_free_function_positional_call() -> None:
    source = "generate_structured(provider, prompt, CodingPlan, trace=trace)"

    assert _extract_structured_call_models(source) == {"CodingPlan"}


def test_ast_helper_detects_free_function_keyword_call() -> None:
    source = "generate_structured(provider, prompt, response_model=CodingPlan)"

    assert _extract_structured_call_models(source) == {"CodingPlan"}


def test_ast_helper_detects_bound_method_positional_call() -> None:
    source = "provider.generate_structured(prompt, CodingPlan)"

    assert _extract_structured_call_models(source) == {"CodingPlan"}


def test_ast_helper_detects_bound_method_keyword_call() -> None:
    source = "provider.generate_structured(prompt, response_model=CodingPlan)"

    assert _extract_structured_call_models(source) == {"CodingPlan"}


def test_ast_helper_resolves_one_level_dotted_model_reference() -> None:
    source = "generate_structured(provider, prompt, models.CodingPlan)"

    assert _extract_structured_call_models(source) == {"CodingPlan"}


def test_ast_helper_collects_every_call_site_across_shapes() -> None:
    source = (
        "generate_structured(provider, prompt, CodingPlan)\n"
        "provider.generate_structured(prompt, response_model=CodingReview)\n"
    )

    assert _extract_structured_call_models(source) == {"CodingPlan", "CodingReview"}


def test_ast_helper_fails_loudly_on_a_statically_unresolvable_response_model() -> None:
    # A call target this helper cannot resolve (e.g. a computed model) must
    # not be silently ignored - it must fail the audit.
    source = "generate_structured(provider, prompt, _pick_model())"

    with pytest.raises(ValueError, match="cannot statically resolve"):
        _extract_structured_call_models(source)


def test_ast_helper_fails_loudly_when_response_model_argument_is_missing() -> None:
    source = "generate_structured(provider, prompt)"

    with pytest.raises(AssertionError, match="no resolvable response-model argument"):
        _extract_structured_call_models(source)


def test_provider_facing_decision_encodes_tool_arguments_as_a_nullable_string() -> None:
    schema = to_strict_json_schema(AgentDecisionResponse)
    properties = schema["properties"]

    assert "tool_arguments" not in properties
    assert properties["tool_arguments_json"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]
    # Nothing in the envelope is an object, open or closed, besides the root.
    assert [key for key, value in properties.items() if value.get("type") == "object"] == []


def test_coding_plan_no_longer_declares_the_unsupported_path_format() -> None:
    # The specific construct that made CodingPlan incompatible: a bare
    # pathlib.Path renders as {"type": "string", "format": "path"}.
    schema = to_strict_json_schema(CodingPlan)

    assert "path" not in _string_formats(schema)
    assert schema["$defs"]["CodingPlanStep"]["properties"]["likely_paths"]["items"] == {
        "type": "string"
    }


def test_coding_plan_step_paths_still_parse_into_pathlib_objects() -> None:
    # Schema-only annotation: JSON strings in, real Path objects out.
    plan = CodingPlan.model_validate(
        {
            "task_summary": "Inspect the agent loop.",
            "steps": [
                {
                    "step_id": 1,
                    "action": "Read the decision loop.",
                    "likely_paths": ["src/repomind/agent/loop.py"],
                }
            ],
        }
    )

    likely_path = plan.steps[0].likely_paths[0]
    assert isinstance(likely_path, Path)
    assert likely_path == Path("src/repomind/agent/loop.py")


@pytest.mark.parametrize("unsafe", ["../escape.py", "/etc/passwd"])
def test_coding_plan_step_paths_still_reject_unsafe_repository_paths(unsafe: str) -> None:
    # The repository-relative validation is untouched by the annotation.
    with pytest.raises(ValidationError):
        CodingPlan.model_validate(
            {
                "task_summary": "Inspect something outside the repository.",
                "steps": [
                    {"step_id": 1, "action": "Read a file.", "likely_paths": [unsafe]}
                ],
            }
        )


def test_coding_plan_still_serializes_paths_as_json_strings() -> None:
    plan = CodingPlan.model_validate(
        {
            "task_summary": "Inspect the agent loop.",
            "steps": [
                {
                    "step_id": 1,
                    "action": "Read the decision loop.",
                    "likely_paths": ["src/repomind/agent/loop.py"],
                }
            ],
        }
    )

    dumped = plan.model_dump(mode="json")["steps"][0]["likely_paths"]

    assert [isinstance(value, str) for value in dumped] == [True]
    assert Path(dumped[0]) == Path("src/repomind/agent/loop.py")


def test_internal_agent_decision_is_why_the_envelope_exists() -> None:
    # Pins the root cause: the internal domain model is deliberately an open
    # mapping and is NOT strict-schema compatible. If this ever starts passing,
    # AgentDecision.tool_arguments stopped being an open dict and the envelope
    # should be revisited.
    problems = _open_objects(to_strict_json_schema(AgentDecision))

    assert [path for path, _ in problems] == ["('properties', 'tool_arguments', 'anyOf', 0)"]
