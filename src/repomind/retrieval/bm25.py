"""Small in-memory BM25 implementation over RepoMind code chunks."""

import re
from collections import Counter
from collections.abc import Sequence
from math import log
from pathlib import PurePosixPath

from repomind.ingestion import CodeChunk
from repomind.retrieval.models import BM25Config, BM25SearchResult
from repomind.retrieval.similarity import RetrievalError
from repomind.retrieval.tokenization import tokenize_code

# English function words that carry no topic in a question. Only the query is
# filtered: documents keep every token, so identifiers such as ``is_valid``
# still index their ``is`` component and IDF statistics stay unchanged.
_STOPWORD_TEXT = """
a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further
had has have having here how i if in into is it its itself just me more most my no nor not of
off on once only or other our out over own same should so some such than that the their them
then there these they this those through to too under until up very was we were what when
where which while who whom why will with would you your
"""
QUERY_STOPWORDS = frozenset(_STOPWORD_TEXT.split())

# A query containing any of these is about tests, so test files are not demoted.
_TEST_QUERY_TERMS = frozenset(
    {
        "test",
        "tests",
        "testing",
        "tested",
        "pytest",
        "vitest",
        "unittest",
        "spec",
        "specs",
        "fixture",
        "fixtures",
        "conftest",
    }
)
_TEST_DIRECTORIES = frozenset({"test", "tests", "__tests__", "spec", "specs"})
_TEST_FILE_RE = re.compile(
    r"^(?:test_.+\.py|.+_test\.py|conftest\.py|.+[._-](?:test|spec)\.[cm]?[jt]sx?)$"
)


class BM25Error(RetrievalError):
    """Raised when BM25 search inputs are invalid."""


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise BM25Error("top_k must be a positive integer")


def is_test_path(path: PurePosixPath | str) -> bool:
    """Whether a repository-relative path looks like a test file.

    Conventional locations only: a ``tests``/``test``/``__tests__``/``spec``
    directory, ``test_*.py``/``*_test.py``/``conftest.py``, or a JavaScript
    and TypeScript ``*.test.*``/``*.spec.*`` file.
    """

    posix = PurePosixPath(path)
    if any(part.casefold() in _TEST_DIRECTORIES for part in posix.parts[:-1]):
        return True
    return _TEST_FILE_RE.match(posix.name.casefold()) is not None


def _lexical_text(chunk: CodeChunk, config: BM25Config) -> str:
    """Index the path and structural symbol name alongside the content."""

    fields: list[str] = []
    if config.include_path:
        fields.append(chunk.relative_path.as_posix())
    if chunk.qualified_symbol_name is not None:
        fields.append(chunk.qualified_symbol_name)
    fields.append(chunk.content)
    return "\n".join(fields)


def query_terms(query: str, config: BM25Config) -> Counter[str]:
    """Normalize a query into weighted BM25 terms.

    With ``query_stopwords`` enabled, English function words are dropped
    unless that would leave nothing, so a query made only of such words
    still searches for them.
    """

    tokens = tokenize_code(query)
    if config.query_stopwords:
        filtered = tuple(token for token in tokens if token not in QUERY_STOPWORDS)
        if filtered:
            tokens = filtered
    return Counter(tokens)


class BM25Index:
    """A deterministic derived lexical index for a fixed chunk sequence."""

    def __init__(
        self,
        chunks: Sequence[CodeChunk],
        config: BM25Config | None = None,
    ) -> None:
        self.config = config or BM25Config()
        self.chunks = tuple(chunks)
        self._term_frequencies = tuple(
            Counter(tokenize_code(_lexical_text(chunk, self.config))) for chunk in self.chunks
        )
        self._is_test = tuple(is_test_path(chunk.relative_path.as_posix()) for chunk in self.chunks)
        self.document_lengths = tuple(
            sum(term_frequency.values()) for term_frequency in self._term_frequencies
        )
        self.document_frequencies: Counter[str] = Counter()
        for term_frequency in self._term_frequencies:
            self.document_frequencies.update(term_frequency.keys())
        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths)
            if self.document_lengths
            else 0.0
        )

    @classmethod
    def from_chunks(
        cls,
        chunks: Sequence[CodeChunk],
        config: BM25Config | None = None,
    ) -> "BM25Index":
        """Build term and document frequencies once for the supplied chunks."""

        return cls(chunks, config)

    def inverse_document_frequency(self, token: str) -> float:
        """Return positive Robertson-style IDF for an already normalized token."""

        document_frequency = self.document_frequencies.get(token, 0)
        document_count = len(self.chunks)
        return log(1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))

    def search(self, query: str, *, top_k: int = 5) -> list[BM25SearchResult]:
        """Rank matching chunks, preserving corpus order for exact score ties."""

        _validate_top_k(top_k)
        if not isinstance(query, str):
            raise BM25Error("query must be a string")
        terms = query_terms(query, self.config)
        if not self.chunks or not terms or self.average_document_length <= 0.0:
            # A zero average length means no chunk has a single token, so
            # nothing can match and the length normalization is undefined.
            return []
        test_weight = (
            1.0 if not _TEST_QUERY_TERMS.isdisjoint(terms) else self.config.test_file_weight
        )

        scored: list[tuple[int, float]] = []
        for index, term_frequency in enumerate(self._term_frequencies):
            score = 0.0
            document_length = self.document_lengths[index]
            length_ratio = document_length / self.average_document_length
            for term, query_frequency in terms.items():
                frequency = term_frequency.get(term, 0)
                if frequency == 0:
                    continue
                denominator = frequency + self.config.k1 * (
                    1.0 - self.config.b + self.config.b * length_ratio
                )
                score += (
                    query_frequency
                    * self.inverse_document_frequency(term)
                    * (frequency * (self.config.k1 + 1.0) / denominator)
                )
            if score > 0.0:
                if self._is_test[index]:
                    score *= test_weight
                scored.append((index, score))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            BM25SearchResult(
                chunk=self.chunks[index],
                score=score,
                rank=rank,
            )
            for rank, (index, score) in enumerate(scored[:top_k], start=1)
        ]
