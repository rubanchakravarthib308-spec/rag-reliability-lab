from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class RetrievalResult:
    document: Document
    score: float


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    citations: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EvaluationResult:
    groundedness: float
    citation_precision: float
    hallucination_risk: float
    passed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
