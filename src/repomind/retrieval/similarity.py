"""Small, explicit vector-similarity functions for in-memory retrieval."""

from collections.abc import Sequence
from math import isfinite

import numpy as np


class RetrievalError(ValueError):
    """Base exception for expected retrieval failures."""


class SimilarityError(RetrievalError):
    """Raised when cosine similarity cannot be calculated safely."""


def _as_vector(values: Sequence[float], name: str) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise SimilarityError(f"{name} vector cannot be converted to a NumPy array") from exc

    if raw.ndim != 1:
        raise SimilarityError(f"{name} vector must be one-dimensional")
    if raw.size == 0:
        raise SimilarityError(f"{name} vector cannot be empty")
    if raw.dtype.kind not in {"f", "i", "u"}:
        raise SimilarityError(f"{name} vector must contain only real numerical values")

    vector = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise SimilarityError(f"{name} vector must contain only finite values")
    return vector


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity after validating both one-dimensional vectors.

    Cosine similarity is the dot product divided by the product of vector
    magnitudes. It compares direction rather than raw magnitude.
    """

    left_vector = _as_vector(left, "left")
    right_vector = _as_vector(right, "right")
    if left_vector.shape != right_vector.shape:
        raise SimilarityError(
            "Cosine similarity requires vectors with matching dimensions: "
            f"{left_vector.size} != {right_vector.size}"
        )

    left_magnitude = float(np.linalg.norm(left_vector))
    right_magnitude = float(np.linalg.norm(right_vector))
    if left_magnitude == 0.0 or right_magnitude == 0.0:
        raise SimilarityError("Cosine similarity is undefined for a zero-magnitude vector")
    if not isfinite(left_magnitude) or not isfinite(right_magnitude):
        raise SimilarityError("Vector magnitude is not finite")

    score = float(np.dot(left_vector, right_vector) / (left_magnitude * right_magnitude))
    if not isfinite(score):
        raise SimilarityError("Cosine similarity result is not finite")
    return float(np.clip(score, -1.0, 1.0))
