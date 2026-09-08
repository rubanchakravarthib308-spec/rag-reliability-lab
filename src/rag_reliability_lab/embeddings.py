from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import sqrt
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Provider-neutral embedding boundary used by semantic retrieval."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text in the same order."""


@dataclass(frozen=True)
class StaticEmbeddingProvider:
    """Deterministic fixture provider for offline tests and demos."""

    vectors: dict[str, tuple[float, ...]]

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [text for text in texts if text not in self.vectors]
        if missing:
            raise KeyError(f"No deterministic embedding configured for: {missing[0]}")
        return [list(self.vectors[text]) for text in texts]


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = tuple(float(value) for value in left)
    right_values = tuple(float(value) for value in right)
    if len(left_values) != len(right_values):
        raise ValueError("Embedding vectors must have the same dimensions")
    if not left_values:
        raise ValueError("Embedding vectors must not be empty")

    numerator = sum(a * b for a, b in zip(left_values, right_values, strict=True))
    left_norm = sqrt(sum(value * value for value in left_values))
    right_norm = sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)
