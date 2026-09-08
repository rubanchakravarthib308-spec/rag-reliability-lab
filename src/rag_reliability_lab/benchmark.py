from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from .evaluation import evaluate
from .retrieval import retrieve
from .types import AnswerResult, Document


@dataclass(frozen=True)
class BenchmarkCaseResult:
    case_id: str
    retrieved_ids: tuple[str, ...]
    expected_evidence_ids: tuple[str, ...]
    retrieval_precision: float
    retrieval_coverage: float
    groundedness: float
    hallucination_risk: float
    reliability_passed: bool


@dataclass(frozen=True)
class BenchmarkSummary:
    version: str
    cases: tuple[BenchmarkCaseResult, ...]
    retrieval_precision: float
    retrieval_coverage: float
    groundedness: float
    hallucination_risk: float
    reliability_pass_rate: float
    passed: bool
    failures: tuple[str, ...]


def _dataset_path(version: str) -> Path:
    return Path(__file__).parent / "benchmarks" / f"{version}.json"


def load_benchmark(version: str = "v1") -> dict[str, object]:
    path = _dataset_path(version)
    if not path.exists():
        raise ValueError(f"Unknown benchmark version: {version}")
    return json.loads(path.read_text(encoding="utf-8"))


def _ratio(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else numerator / denominator


def run_benchmark(version: str = "v1") -> BenchmarkSummary:
    dataset = load_benchmark(version)
    documents = [Document(id=item["id"], text=item["text"]) for item in dataset["documents"]]
    top_k = int(dataset["top_k"])
    evaluation_threshold = float(dataset["evaluation_threshold"])

    case_results: list[BenchmarkCaseResult] = []
    for case in dataset["cases"]:
        evidence = retrieve(case["question"], documents, top_k=top_k)
        retrieved_ids = tuple(item.document.id for item in evidence)
        expected_ids = tuple(case["expected_evidence_ids"])
        expected = set(expected_ids)
        retrieved = set(retrieved_ids)

        retrieval_precision = _ratio(len(retrieved & expected), len(retrieved))
        retrieval_coverage = _ratio(len(retrieved & expected), len(expected))
        evaluation = evaluate(
            AnswerResult(answer=case["answer"], citations=tuple(case["citations"])),
            evidence,
            threshold=evaluation_threshold,
        )

        case_results.append(
            BenchmarkCaseResult(
                case_id=case["id"],
                retrieved_ids=retrieved_ids,
                expected_evidence_ids=expected_ids,
                retrieval_precision=round(retrieval_precision, 4),
                retrieval_coverage=round(retrieval_coverage, 4),
                groundedness=evaluation.groundedness,
                hallucination_risk=evaluation.hallucination_risk,
                reliability_passed=evaluation.passed,
            )
        )

    retrieval_precision = mean(item.retrieval_precision for item in case_results)
    retrieval_coverage = mean(item.retrieval_coverage for item in case_results)
    groundedness = mean(item.groundedness for item in case_results)
    hallucination_risk = mean(item.hallucination_risk for item in case_results)
    reliability_pass_rate = mean(1.0 if item.reliability_passed else 0.0 for item in case_results)

    thresholds = dataset["thresholds"]
    checks = {
        "retrieval precision": retrieval_precision >= float(thresholds["min_retrieval_precision"]),
        "retrieval coverage": retrieval_coverage >= float(thresholds["min_retrieval_coverage"]),
        "groundedness": groundedness >= float(thresholds["min_groundedness"]),
        "hallucination risk": hallucination_risk <= float(thresholds["max_hallucination_risk"]),
        "reliability pass rate": reliability_pass_rate >= float(thresholds["min_reliability_pass_rate"]),
    }
    failures = tuple(name for name, passed in checks.items() if not passed)

    return BenchmarkSummary(
        version=str(dataset["version"]),
        cases=tuple(case_results),
        retrieval_precision=round(retrieval_precision, 4),
        retrieval_coverage=round(retrieval_coverage, 4),
        groundedness=round(groundedness, 4),
        hallucination_risk=round(hallucination_risk, 4),
        reliability_pass_rate=round(reliability_pass_rate, 4),
        passed=not failures,
        failures=failures,
    )


def to_json(summary: BenchmarkSummary) -> str:
    return json.dumps(asdict(summary), indent=2, sort_keys=True)


def to_markdown(summary: BenchmarkSummary) -> str:
    lines = [
        f"# RAG Reliability Benchmark — {summary.version}",
        "",
        f"**Result:** {'PASS' if summary.passed else 'FAIL'}",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Score |",
        "| --- | ---: |",
        f"| Retrieval precision | {summary.retrieval_precision:.4f} |",
        f"| Retrieval coverage | {summary.retrieval_coverage:.4f} |",
        f"| Groundedness | {summary.groundedness:.4f} |",
        f"| Hallucination risk | {summary.hallucination_risk:.4f} |",
        f"| Reliability pass rate | {summary.reliability_pass_rate:.4f} |",
        "",
        "## Cases",
        "",
        "| Case | Precision | Coverage | Groundedness | Hallucination risk | Reliability |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summary.cases:
        lines.append(
            f"| {item.case_id} | {item.retrieval_precision:.4f} | {item.retrieval_coverage:.4f} | "
            f"{item.groundedness:.4f} | {item.hallucination_risk:.4f} | "
            f"{'PASS' if item.reliability_passed else 'FAIL'} |"
        )
    if summary.failures:
        lines.extend(["", "## Regression failures", "", *[f"- {failure}" for failure in summary.failures]])
    return "\n".join(lines) + "\n"
