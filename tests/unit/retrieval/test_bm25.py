"""Tests for the explicit BM25 lexical-retrieval baseline."""

from math import log

import pytest
from pydantic import ValidationError

from repomind.ingestion import ChunkingStrategy, ChunkKind, CodeChunk
from repomind.retrieval import LEGACY_BM25_CONFIG, BM25Config, BM25Error, BM25Index, is_test_path


def _chunk(path: str, content: str, index: int = 0) -> CodeChunk:
    return CodeChunk(
        relative_path=path,
        language="python",
        start_line=index + 1,
        end_line=index + max(1, len(content.splitlines())),
        content=content,
        chunk_index=index,
    )


def test_obvious_relevant_document_ranks_first() -> None:
    index = BM25Index.from_chunks(
        [
            _chunk("auth.py", "JWT authentication validates access tokens"),
            _chunk("database.py", "PostgreSQL stores repository chunks"),
            _chunk("styles.css", "CSS controls page layout"),
        ]
    )

    results = index.search("authentication token", top_k=5)

    assert [result.chunk.relative_path.as_posix() for result in results] == ["auth.py"]
    assert results[0].rank == 1
    assert results[0].score > 0


def test_exact_identifier_outweighs_partial_concept_matches() -> None:
    index = BM25Index.from_chunks(
        [
            _chunk("settings.py", 'OPENAI_EMBEDDING_MODEL = "test-model"'),
            _chunk("config.py", "OpenAI model configuration"),
            _chunk("search.py", "embedding search implementation"),
        ]
    )

    results = index.search("OPENAI_EMBEDDING_MODEL", top_k=3)

    assert results[0].chunk.relative_path.as_posix() == "settings.py"
    assert results[0].score > results[1].score


def test_structural_qualified_symbol_improves_exact_method_lookup() -> None:
    method = _chunk("service.py", "def login(self): pass", 0).model_copy(
        update={
            "chunking_strategy": ChunkingStrategy.STRUCTURAL,
            "chunk_kind": ChunkKind.METHOD,
            "qualified_symbol_name": "UserService.login",
        }
    )
    class_context = _chunk("service.py", "class UserService:", 1).model_copy(
        update={
            "chunking_strategy": ChunkingStrategy.STRUCTURAL,
            "chunk_kind": ChunkKind.CLASS,
            "qualified_symbol_name": "UserService",
        }
    )

    results = BM25Index.from_chunks([class_context, method]).search("UserService.login")

    assert results[0].chunk == method


@pytest.mark.parametrize("query", ["persist_embedded_chunks", "persist embedded chunks"])
def test_snake_case_identifier_matches_exact_and_component_queries(query: str) -> None:
    index = BM25Index.from_chunks(
        [
            _chunk("repositories.py", "def persist_embedded_chunks(session): pass"),
            _chunk("other.py", "persist files in small chunks"),
        ]
    )

    assert index.search(query, top_k=2)[0].chunk.relative_path.as_posix() == ("repositories.py")


@pytest.mark.parametrize(
    ("identifier", "query"),
    [
        ("OpenAILLMClient", "LLM client"),
        ("RepositoryIngestionError", "repository ingestion error"),
    ],
)
def test_camel_case_components_are_searchable(identifier: str, query: str) -> None:
    index = BM25Index.from_chunks([_chunk("target.py", identifier)])

    assert index.search(query)[0].chunk.relative_path.as_posix() == "target.py"


def test_known_bm25_score_matches_documented_formula() -> None:
    # The legacy configuration scores content alone, so the arithmetic below
    # is not shifted by path tokens.
    index = BM25Index.from_chunks(
        [_chunk("repeated.py", "token token"), _chunk("other.py", "other")],
        LEGACY_BM25_CONFIG,
    )

    result = index.search("token", top_k=1)[0]
    expected_idf = log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    expected_saturation = (2 * (1.5 + 1)) / (2 + 1.5 * (1 - 0.75 + 0.75 * 2 / 1.5))

    assert result.score == pytest.approx(expected_idf * expected_saturation)


def test_repeated_query_term_contributes_repeatedly() -> None:
    index = BM25Index.from_chunks([_chunk("target.py", "token")])

    once = index.search("token")[0].score
    twice = index.search("token token")[0].score

    assert twice == pytest.approx(once * 2)


def test_shorter_matching_document_benefits_from_length_normalization() -> None:
    index = BM25Index.from_chunks(
        [
            _chunk("short.py", "rare"),
            _chunk("long.py", "rare filler filler filler filler filler"),
        ]
    )

    results = index.search("rare", top_k=2)

    assert [result.chunk.relative_path.as_posix() for result in results] == [
        "short.py",
        "long.py",
    ]


def test_rare_term_has_higher_idf_than_common_term() -> None:
    index = BM25Index.from_chunks(
        [
            _chunk("one.py", "common rare"),
            _chunk("two.py", "common"),
            _chunk("three.py", "common"),
        ]
    )

    assert index.inverse_document_frequency("rare") > index.inverse_document_frequency("common")


def test_exact_score_ties_preserve_corpus_order() -> None:
    index = BM25Index.from_chunks([_chunk("z.py", "match"), _chunk("a.py", "match")])

    results = index.search("match", top_k=5)

    assert [result.chunk.relative_path.as_posix() for result in results] == [
        "z.py",
        "a.py",
    ]


@pytest.mark.parametrize("query", ["missing", "...", "   "])
def test_no_useful_matching_terms_returns_no_results(query: str) -> None:
    index = BM25Index.from_chunks([_chunk("one.py", "present")])
    assert index.search(query) == []


def test_empty_corpus_returns_no_results() -> None:
    assert BM25Index.from_chunks([]).search("anything") == []


def test_top_k_above_match_count_returns_all_matches() -> None:
    index = BM25Index.from_chunks([_chunk("one.py", "match"), _chunk("two.py", "match")])
    assert len(index.search("match", top_k=10)) == 2


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_is_rejected(top_k: object) -> None:
    with pytest.raises(BM25Error, match="positive integer"):
        BM25Index.from_chunks([]).search("query", top_k=top_k)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "config",
    [
        {"k1": 0},
        {"k1": -1},
        {"k1": float("inf")},
        {"k1": True},
        {"b": -0.01},
        {"b": 1.01},
        {"b": True},
    ],
)
def test_invalid_bm25_configuration_is_rejected(config: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        BM25Config(**config)


def test_tokenless_corpus_returns_no_results_instead_of_dividing_by_zero() -> None:
    # Paths always contribute tokens, so only the content-only legacy
    # configuration can produce a corpus without a single token.
    index = BM25Index.from_chunks(
        [
            _chunk("blank.py", "   \n", 0),
            _chunk("punctuation.css", "{}();\n", 1),
        ],
        LEGACY_BM25_CONFIG,
    )

    assert index.average_document_length == 0.0
    assert index.search("authentication", top_k=5) == []


def test_file_path_is_searchable_by_default() -> None:
    chunks = [
        _chunk("src/jobs/worker.py", "def run_once(self): return self.store.claim_next()"),
        _chunk("src/ui/layout.py", "def render(self): return self.template"),
    ]

    assert [
        r.chunk.relative_path.as_posix() for r in BM25Index.from_chunks(chunks).search("job worker")
    ] == ["src/jobs/worker.py"]
    assert BM25Index.from_chunks(chunks, LEGACY_BM25_CONFIG).search("job worker") == []


def test_query_stopwords_do_not_let_prose_outrank_the_topic() -> None:
    chunks = [
        _chunk("cache.py", "cache eviction policy"),
        _chunk("notes.py", "how does the code do what it does when the code is the code"),
    ]

    default = BM25Index.from_chunks(chunks).search("how does the cache work")
    legacy = BM25Index.from_chunks(chunks, LEGACY_BM25_CONFIG).search("how does the cache work")

    assert [r.chunk.relative_path.as_posix() for r in default] == ["cache.py"]
    assert {r.chunk.relative_path.as_posix() for r in legacy} == {"cache.py", "notes.py"}


def test_query_made_only_of_stopwords_still_searches_for_them() -> None:
    index = BM25Index.from_chunks([_chunk("guard.py", "if not ready: return")])

    assert [r.chunk.relative_path.as_posix() for r in index.search("if not")] == ["guard.py"]


def test_test_files_rank_below_identical_implementation_matches() -> None:
    content = "def claim_next(self): lease the oldest queued job"
    index = BM25Index.from_chunks(
        [_chunk("tests/test_store.py", content, 0), _chunk("src/store.py", content, 1)]
    )

    results = index.search("claim queued job lease")

    assert [r.chunk.relative_path.as_posix() for r in results] == [
        "src/store.py",
        "tests/test_store.py",
    ]
    assert results[1].score == pytest.approx(results[0].score * 0.5, rel=0.2)


def test_queries_about_tests_do_not_demote_test_files() -> None:
    content = "def test_claim_next(): assert store.claim_next() is None"
    index = BM25Index.from_chunks(
        [_chunk("tests/test_store.py", content, 0), _chunk("src/store.py", content, 1)]
    )

    results = index.search("which test covers claim_next")

    # Undemoted, the test file's own path ("tests/test_store.py") matches "test".
    assert [r.chunk.relative_path.as_posix() for r in results] == [
        "tests/test_store.py",
        "src/store.py",
    ]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tests/unit/test_api.py", True),
        ("src/pkg/tests/helpers.py", True),
        ("test_cli.py", True),
        ("pkg/cli_test.py", True),
        ("conftest.py", True),
        ("frontend/lib/api.test.ts", True),
        ("web/components/button.spec.tsx", True),
        ("src/__tests__/app.js", True),
        ("src/repomind/evaluation/metrics.py", False),
        ("src/testing_utils.py", False),
        ("src/contest.py", False),
        ("docs/testing.md", False),
    ],
)
def test_is_test_path_matches_conventional_test_locations(path: str, expected: bool) -> None:
    assert is_test_path(path) is expected


@pytest.mark.parametrize("weight", [0, -0.5, 1.5, True])
def test_test_file_weight_must_be_a_fraction(weight: object) -> None:
    with pytest.raises(ValidationError):
        BM25Config(test_file_weight=weight)
