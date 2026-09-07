from __future__ import annotations

from collections.abc import Callable

from .evaluation import evaluate
from .retrieval import retrieve
from .types import AnswerResult, Document, EvaluationResult, RetrievalResult

Answerer = Callable[[str, list[RetrievalResult]], AnswerResult]


class ReliabilityPipeline:
    def __init__(self, documents: list[Document], answerer: Answerer, *, threshold: float = 0.65) -> None:
        self._documents = documents
        self._answerer = answerer
        self._threshold = threshold

    def run(self, question: str, *, top_k: int = 3) -> tuple[list[RetrievalResult], AnswerResult, EvaluationResult]:
        evidence = retrieve(question, self._documents, top_k=top_k)
        answer = self._answerer(question, evidence)
        result = evaluate(answer, evidence, threshold=self._threshold)
        return evidence, answer, result
