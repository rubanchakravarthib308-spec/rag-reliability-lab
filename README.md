# RAG Reliability Lab

> **A transparent testbed for retrieval quality, citation validity, groundedness, and hallucination risk in RAG systems.**

Many RAG demos stop when the model returns an answer. This project focuses on the harder question:

**Can we measure whether the answer is actually supported by the evidence the system retrieved?**

## What this project demonstrates

```text
Question
  ↓
Retrieval
  ↓
Ranked evidence
  ↓
Answer + citations
  ↓
Citation validation
  ↓
Groundedness scoring
  ↓
Hallucination-risk check
  ↓
Pass / fail with reasons
```

The current implementation is deliberately deterministic and API-key free, so reviewers can inspect and test the reliability boundary itself rather than trusting a black-box model call.

## Why reliability matters

A RAG system can retrieve relevant documents and still produce an unsupported answer. It can also attach citations that do not actually belong to the evidence used.

This lab makes those failure modes explicit by checking:

- **Retrieval relevance** — rank evidence against the question
- **Citation validity** — confirm cited document IDs were actually retrieved
- **Groundedness** — estimate how much answer content is supported by cited evidence
- **Hallucination risk** — surface unsupported content instead of hiding it
- **Failure reasons** — return machine-readable reasons when an answer should not pass

## Architecture

```text
Question
   │
   ▼
Retriever
   │
   ▼
Ranked Evidence
   │
   ▼
Answerer / LLM Adapter
   │
   ▼
Answer + Citations
   │
   ▼
Reliability Evaluator
   ├── Citation validity
   ├── Groundedness
   └── Hallucination risk
   │
   ▼
Pass / Fail + Reasons
```

See [`docs/architecture.md`](docs/architecture.md) for the design rationale.

## Run locally

Requirements: Python 3.11+

```bash
python -m pip install -e . pytest
pytest -q
python -m rag_reliability_lab.demo
```

No paid API, model key, vector database, or external service is required for the baseline demo.

## Example behavior

A supported answer with a valid citation can pass:

```text
Answer: High-risk agent actions should require human approval before execution.
Citation: doc-approval
Groundedness: high
Citation precision: 1.0
Hallucination risk: low
Result: PASS
```

An answer that invents unsupported requirements is rejected, even when it includes a valid citation.

## Project structure

```text
src/rag_reliability_lab/
  retrieval.py      # deterministic evidence ranking
  evaluation.py     # citation + groundedness + hallucination checks
  pipeline.py       # retrieval → answer → evaluation workflow
  types.py          # domain models
  demo.py           # deterministic end-to-end example

tests/
  test_reliability.py

docs/
  architecture.md

.github/workflows/
  ci.yml
```

## Current maturity

**v0.1 — transparent reliability baseline**

Implemented:

- deterministic lexical retrieval
- explicit evidence objects
- machine-checkable citations
- groundedness score
- hallucination-risk score
- pass/fail reliability gate
- safety-focused tests
- CI validation
- deterministic demo

Next milestones:

- embedding-based semantic retrieval
- claim-level evidence alignment
- precision@k / recall@k retrieval metrics
- LLM provider adapter behind a strict interface
- benchmark dataset + experiment runner
- semantic groundedness / entailment checks
- evaluation reports and regression tracking

## Engineering principle

> **A RAG answer is not trustworthy because it has citations. It is trustworthy only when the citations and claims can be verified against the evidence.**

## About this project

Built by **Ruban Chakravarthi** as a public AI-engineering portfolio project while transitioning from enterprise technology into hands-on AI engineering.

The goal is not to hide complexity behind a framework. It is to make reliability behavior visible, testable, and progressively more rigorous.
