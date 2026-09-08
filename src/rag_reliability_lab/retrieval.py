from __future__ import annotations

import math
import re
from collections import Counter

from .embeddings import EmbeddingProvider, cosine_similarity
from .types import Document, RetrievalResult

_TOKEN = re.compile(r"[a-zA-Z0-9]+")


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text)]


def _lexical_cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    common = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _validate_top_k(top_k: int) -> None:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")


def retrieve(query: str, documents: list[Document], top_k: int = 3) -> list[RetrievalResult]:
    """Deterministic lexical retrieval retained as the offline default."""
    _validate_top_k(top_k)
    query_vector = Counter(_tokens(query))
    ranked = [
        RetrievalResult(
            document=document,
            score=_lexical_cosine_similarity(query_vector, Counter(_tokens(document.text))),
        )
        for document in documents
    ]
    ranked.sort(key=lambda item: (-item.score, item.document.id))
    return [item for item in ranked if item.score > 0][:top_k]


def retrieve_embeddings(
    query: str,
    documents: list[Document],
    provider: EmbeddingProvider,
    *,
    top_k: int = 3,
) -> list[RetrievalResult]:
    """Rank documents by cosine similarity in provider-supplied embedding space."""
    _validate_top_k(top_k)
    if not documents:
        return []

    texts = [query, *(document.text for document in documents)]
    vectors = provider.embed(texts)
    if len(vectors) != len(texts):
        raise ValueError("Embedding provider must return one vector per input text")

    query_vector = vectors[0]
    ranked = [
        RetrievalResult(document=document, score=cosine_similarity(query_vector, vector))
        for document, vector in zip(documents, vectors[1:], strict=True)
    ]
    ranked.sort(key=lambda item: (-item.score, item.document.id))
    return [item for item in ranked if item.score > 0][:top_k]
