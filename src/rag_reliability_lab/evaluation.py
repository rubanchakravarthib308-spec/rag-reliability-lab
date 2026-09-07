from __future__ import annotations

import re

from .types import AnswerResult, EvaluationResult, RetrievalResult

_TOKEN = re.compile(r"[a-zA-Z0-9]+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in", "is", "it",
    "of", "on", "or", "that", "the", "to", "was", "were", "will", "with",
}


def _content_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN.findall(text)
        if len(token) > 2 and token.lower() not in _STOP
    }


def evaluate(answer: AnswerResult, evidence: list[RetrievalResult], threshold: float = 0.65) -> EvaluationResult:
    evidence_by_id = {item.document.id: item.document for item in evidence}
    answer_tokens = _content_tokens(answer.answer)
    cited_ids = set(answer.citations)
    valid_citations = cited_ids & set(evidence_by_id)

    supported_tokens: set[str] = set()
    for doc_id in valid_citations:
        supported_tokens |= _content_tokens(evidence_by_id[doc_id].text)

    groundedness = 1.0 if not answer_tokens else len(answer_tokens & supported_tokens) / len(answer_tokens)
    citation_precision = 1.0 if not cited_ids else len(valid_citations) / len(cited_ids)
    hallucination_risk = round(1.0 - groundedness, 4)

    reasons: list[str] = []
    if not answer.citations:
        reasons.append("answer has no citations")
    if citation_precision < 1.0:
        reasons.append("one or more citations do not map to retrieved evidence")
    if groundedness < threshold:
        reasons.append("answer contains too much unsupported content")

    passed = bool(answer.citations) and citation_precision == 1.0 and groundedness >= threshold
    return EvaluationResult(
        groundedness=round(groundedness, 4),
        citation_precision=round(citation_precision, 4),
        hallucination_risk=hallucination_risk,
        passed=passed,
        reasons=tuple(reasons),
    )
