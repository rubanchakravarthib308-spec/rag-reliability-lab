from __future__ import annotations

import re

from .types import AnswerResult, ClaimEvaluation, EvaluationResult, RetrievalResult

_TOKEN = re.compile(r"[a-zA-Z0-9]+")
_CLAIM_SPLIT = re.compile(r"(?<=[.!?])\s+")
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


def _split_claims(text: str) -> list[str]:
    return [claim.strip() for claim in _CLAIM_SPLIT.split(text.strip()) if claim.strip()]


def _claim_support(claim: str, evidence_by_id: dict[str, object], cited_ids: set[str], threshold: float) -> ClaimEvaluation:
    claim_tokens = _content_tokens(claim)
    supporting_ids: list[str] = []
    supported_tokens: set[str] = set()

    for doc_id in sorted(cited_ids):
        document = evidence_by_id.get(doc_id)
        if document is None:
            continue
        document_tokens = _content_tokens(document.text)  # type: ignore[attr-defined]
        overlap = claim_tokens & document_tokens
        if overlap:
            supporting_ids.append(doc_id)
            supported_tokens |= overlap

    support_score = 1.0 if not claim_tokens else len(supported_tokens) / len(claim_tokens)
    return ClaimEvaluation(
        claim=claim,
        support_score=round(support_score, 4),
        supported=support_score >= threshold,
        supporting_evidence=tuple(supporting_ids),
    )


def evaluate(answer: AnswerResult, evidence: list[RetrievalResult], threshold: float = 0.65) -> EvaluationResult:
    evidence_by_id = {item.document.id: item.document for item in evidence}
    cited_ids = set(answer.citations)
    valid_citations = cited_ids & set(evidence_by_id)

    claims = tuple(
        _claim_support(claim, evidence_by_id, valid_citations, threshold)
        for claim in _split_claims(answer.answer)
    )
    groundedness = 1.0 if not claims else sum(claim.support_score for claim in claims) / len(claims)
    citation_precision = 1.0 if not cited_ids else len(valid_citations) / len(cited_ids)
    hallucination_risk = round(1.0 - groundedness, 4)

    unsupported_claims = [claim.claim for claim in claims if not claim.supported]
    reasons: list[str] = []
    if not answer.citations:
        reasons.append("answer has no citations")
    if citation_precision < 1.0:
        reasons.append("one or more citations do not map to retrieved evidence")
    if unsupported_claims:
        reasons.append(f"unsupported claims detected: {len(unsupported_claims)}")
    if groundedness < threshold:
        reasons.append("answer contains too much unsupported content")

    passed = (
        bool(answer.citations)
        and citation_precision == 1.0
        and bool(claims)
        and not unsupported_claims
        and groundedness >= threshold
    )
    return EvaluationResult(
        groundedness=round(groundedness, 4),
        citation_precision=round(citation_precision, 4),
        hallucination_risk=hallucination_risk,
        passed=passed,
        reasons=tuple(reasons),
        claims=claims,
    )
