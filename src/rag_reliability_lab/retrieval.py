from __future__ import annotations

import math
import re
from collections import Counter

from .types import Document, RetrievalResult

_TOKEN = re.compile(r"[a-zA-Z0-9]+")


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text)]


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    common = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def retrieve(query: str, documents: list[Document], top_k: int = 3) -> list[RetrievalResult]:
    query_vector = Counter(_tokens(query))
    ranked = [
        RetrievalResult(document=document, score=_cosine_similarity(query_vector, Counter(_tokens(document.text))))
        for document in documents
    ]
    ranked.sort(key=lambda item: (-item.score, item.document.id))
    return [item for item in ranked if item.score > 0][:top_k]
