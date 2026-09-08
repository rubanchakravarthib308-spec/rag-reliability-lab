from __future__ import annotations

import json

from rag_reliability_lab.benchmark import run_benchmark, to_json, to_markdown


def test_v1_benchmark_passes_regression_thresholds() -> None:
    summary = run_benchmark("v1")

    assert summary.passed is True
    assert summary.failures == ()
    assert summary.retrieval_coverage == 1.0
    assert summary.groundedness >= 0.9
    assert summary.hallucination_risk <= 0.1
    assert summary.reliability_pass_rate == 1.0


def test_json_report_is_machine_readable() -> None:
    payload = json.loads(to_json(run_benchmark("v1")))

    assert payload["version"] == "v1"
    assert payload["passed"] is True
    assert len(payload["cases"]) == 3
    assert "retrieval_precision" in payload
    assert "groundedness" in payload


def test_markdown_report_is_human_readable() -> None:
    report = to_markdown(run_benchmark("v1"))

    assert "# RAG Reliability Benchmark — v1" in report
    assert "**Result:** PASS" in report
    assert "| Retrieval precision |" in report
    assert "| approval-gate |" in report
