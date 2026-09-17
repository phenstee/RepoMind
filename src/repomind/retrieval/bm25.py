"""Small in-memory BM25 implementation over RepoMind code chunks."""

from collections import Counter
from collections.abc import Sequence
from math import log

from repomind.ingestion import CodeChunk
from repomind.retrieval.models import BM25Config, BM25SearchResult
from repomind.retrieval.similarity import RetrievalError
from repomind.retrieval.tokenization import tokenize_code


class BM25Error(RetrievalError):
    """Raised when BM25 search inputs are invalid."""


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise BM25Error("top_k must be a positive integer")


def _lexical_text(chunk: CodeChunk) -> str:
    """Expose structural symbol context without changing line-v1 scoring."""

    if chunk.qualified_symbol_name is None:
        return chunk.content
    return f"{chunk.qualified_symbol_name}\n{chunk.content}"


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
            Counter(tokenize_code(_lexical_text(chunk))) for chunk in self.chunks
        )
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
        query_terms = Counter(tokenize_code(query))
        if not self.chunks or not query_terms:
            return []

        scored: list[tuple[int, float]] = []
        for index, term_frequency in enumerate(self._term_frequencies):
            score = 0.0
            document_length = self.document_lengths[index]
            length_ratio = document_length / self.average_document_length
            for term, query_frequency in query_terms.items():
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
