from rag_reliability_lab.evaluation import evaluate
from rag_reliability_lab.retrieval import retrieve
from rag_reliability_lab.types import AnswerResult, Document


def test_retriever_ranks_relevant_document_first() -> None:
    documents = [
        Document(id="a", text="Human approval is required for high-risk agent actions."),
        Document(id="b", text="Redis can cache frequently accessed application data."),
    ]

    results = retrieve("approval for high-risk agent actions", documents)

    assert results
    assert results[0].document.id == "a"
    assert results[0].score > 0


def test_grounded_answer_with_valid_citation_passes() -> None:
    documents = [Document(id="a", text="High-risk agent actions require human approval before execution.")]
    evidence = retrieve("high-risk agent approval", documents)
    answer = AnswerResult(
        answer="High-risk agent actions require human approval before execution.",
        citations=("a",),
    )

    result = evaluate(answer, evidence)

    assert result.passed is True
    assert result.citation_precision == 1.0
    assert result.hallucination_risk == 0.0


def test_unknown_citation_is_rejected() -> None:
    documents = [Document(id="a", text="High-risk actions require human approval.")]
    evidence = retrieve("high-risk approval", documents)
    answer = AnswerResult(answer="High-risk actions require human approval.", citations=("missing",))

    result = evaluate(answer, evidence)

    assert result.passed is False
    assert result.citation_precision == 0.0
    assert "one or more citations do not map to retrieved evidence" in result.reasons


def test_unsupported_claim_triggers_hallucination_risk() -> None:
    documents = [Document(id="a", text="High-risk actions require human approval.")]
    evidence = retrieve("high-risk approval", documents)
    answer = AnswerResult(
        answer="High-risk actions require human approval and must always use blockchain consensus.",
        citations=("a",),
    )

    result = evaluate(answer, evidence, threshold=0.9)

    assert result.passed is False
    assert result.hallucination_risk > 0
    assert "answer contains too much unsupported content" in result.reasons
