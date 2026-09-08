from rag_reliability_lab.evaluation import evaluate
from rag_reliability_lab.types import AnswerResult, Document, RetrievalResult


def _evidence(*documents: Document) -> list[RetrievalResult]:
    return [RetrievalResult(document=document, score=1.0) for document in documents]


def test_partially_supported_answer_flags_only_unsupported_claim() -> None:
    evidence = _evidence(
        Document(id="approval", text="High-risk agent actions require human approval before execution."),
    )
    answer = AnswerResult(
        answer=(
            "High-risk agent actions require human approval before execution. "
            "Every agent action must also be approved by a blockchain validator."
        ),
        citations=("approval",),
    )

    result = evaluate(answer, evidence, threshold=0.65)

    assert len(result.claims) == 2
    assert result.claims[0].supported is True
    assert result.claims[0].supporting_evidence == ("approval",)
    assert result.claims[1].supported is False
    assert result.groundedness < 1.0
    assert result.passed is False
    assert "unsupported claims detected: 1" in result.reasons


def test_claim_can_associate_with_specific_supporting_evidence() -> None:
    evidence = _evidence(
        Document(id="approval", text="High-risk actions require human approval."),
        Document(id="replay", text="Idempotency keys prevent duplicate tool execution."),
    )
    answer = AnswerResult(
        answer=(
            "High-risk actions require human approval. "
            "Idempotency keys prevent duplicate tool execution."
        ),
        citations=("approval", "replay"),
    )

    result = evaluate(answer, evidence)

    assert result.passed is True
    assert result.groundedness == 1.0
    assert result.claims[0].supporting_evidence == ("approval",)
    assert result.claims[1].supporting_evidence == ("replay",)


def test_fully_unsupported_claim_is_explicitly_flagged() -> None:
    evidence = _evidence(Document(id="cache", text="Redis can cache frequently accessed application data."))
    answer = AnswerResult(
        answer="Quantum teleportation is mandatory for all production deployments.",
        citations=("cache",),
    )

    result = evaluate(answer, evidence)

    assert len(result.claims) == 1
    assert result.claims[0].support_score == 0.0
    assert result.claims[0].supporting_evidence == ()
    assert result.claims[0].supported is False
    assert result.hallucination_risk == 1.0
    assert result.passed is False
    assert "unsupported claims detected: 1" in result.reasons
