from __future__ import annotations

import json

from .pipeline import ReliabilityPipeline
from .types import AnswerResult, Document, RetrievalResult


def answerer(_question: str, evidence: list[RetrievalResult]) -> AnswerResult:
    if not evidence:
        return AnswerResult(answer="I do not have enough evidence to answer.")

    best = evidence[0].document
    return AnswerResult(
        answer="High-risk agent actions should require human approval before execution.",
        citations=(best.id,),
    )


def main() -> None:
    documents = [
        Document(
            id="doc-approval",
            text="High-risk agent actions should require human approval before execution and should be verified afterward.",
        ),
        Document(
            id="doc-audit",
            text="Audit logs should record important decisions, tool calls, and verification outcomes.",
        ),
    ]

    pipeline = ReliabilityPipeline(documents, answerer)
    evidence, answer, evaluation = pipeline.run("How should high-risk agent actions be handled?")

    print(json.dumps({
        "retrieved": [
            {"id": item.document.id, "score": round(item.score, 4)} for item in evidence
        ],
        "answer": answer.answer,
        "citations": answer.citations,
        "evaluation": {
            "groundedness": evaluation.groundedness,
            "citation_precision": evaluation.citation_precision,
            "hallucination_risk": evaluation.hallucination_risk,
            "passed": evaluation.passed,
            "reasons": evaluation.reasons,
        },
    }, indent=2))


if __name__ == "__main__":
    main()
