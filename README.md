# RAG Reliability Lab

> **A transparent testbed for retrieval quality, citation validity, claim-level groundedness, hallucination risk, and RAG regressions.**

Many RAG demos stop when the model returns an answer. This project focuses on a harder question:

**Can we measure whether retrieval and generated claims remain supported over time?**

## What this project demonstrates

```text
Question
  ↓
Retrieval (lexical or embedding-based)
  ↓
Ranked evidence
  ↓
Answer + citations
  ↓
Claim-level evidence checks
  ↓
Groundedness + hallucination scoring
  ↓
Benchmark regression gate
  ↓
JSON + Markdown report
```

The baseline remains deterministic and API-key free. Semantic retrieval is exposed through a provider-neutral embedding interface, while the benchmark runner makes reliability changes measurable in CI.

## Reliability checks

The lab evaluates:

- **Retrieval relevance** — rank evidence against the question
- **Citation validity** — confirm cited document IDs were actually retrieved
- **Claim support** — evaluate each answer claim against cited evidence
- **Groundedness** — aggregate support scores across claims
- **Hallucination risk** — surface unsupported content instead of hiding it
- **Regression thresholds** — fail CI when reliability metrics fall below the benchmark contract

## Retrieval boundary

Two retrieval paths are available:

```text
Question ──→ Lexical Retriever ──→ token-vector cosine similarity
    │
    └──────→ Embedding Retriever ──→ EmbeddingProvider ──→ vector cosine similarity
```

`retrieve(...)` preserves the deterministic lexical path used by the demo and benchmark suite.

`retrieve_embeddings(...)` accepts any object implementing the provider-neutral `EmbeddingProvider` protocol. `StaticEmbeddingProvider` keeps semantic-ranking tests fully offline.

## Claim-level groundedness

Answers are split into evaluable claims. Each claim records:

- a support score
- whether it passed the support threshold
- the retrieved evidence IDs that support it

The answer passes only when citations are valid and every evaluable claim is sufficiently supported.

## Benchmark + regression reporting

The repository includes a versioned benchmark dataset at:

```text
src/rag_reliability_lab/benchmarks/v1.json
```

Each benchmark case defines a question, expected evidence, a reference answer, and citations. The runner measures:

- retrieval precision
- retrieval coverage
- groundedness
- hallucination risk
- reliability pass rate

Run it locally:

```bash
python -m rag_reliability_lab.benchmark_cli \
  --json benchmark-report.json \
  --markdown benchmark-report.md
```

The command writes both a **machine-readable JSON report** and a **human-readable Markdown report**. It exits with code `1` if benchmark thresholds regress, making it suitable as a CI quality gate.

## Architecture

```text
Question
   │
   ▼
Retriever
   ├── deterministic lexical path
   └── provider-neutral embedding path
   │
   ▼
Ranked Evidence
   │
   ▼
Answer + Citations
   │
   ▼
Claim-level Reliability Evaluator
   ├── Citation validity
   ├── Per-claim support
   ├── Groundedness
   └── Hallucination risk
   │
   ▼
Benchmark Runner
   ├── Retrieval precision / coverage
   ├── Groundedness summary
   ├── Hallucination summary
   └── Regression thresholds
   │
   ▼
JSON report + Markdown report + CI exit status
```

See [`docs/architecture.md`](docs/architecture.md) for the design rationale.

## Run locally

Requirements: Python 3.11+

```bash
python -m pip install -e . pytest
pytest -q
python -m rag_reliability_lab.demo
python -m rag_reliability_lab.benchmark_cli
```

No paid API, model key, vector database, or external service is required for the demo, tests, or benchmark regression suite.

## Project structure

```text
src/rag_reliability_lab/
  embeddings.py       # provider-neutral embedding interface + deterministic fixture
  retrieval.py        # lexical and embedding-based evidence ranking
  evaluation.py       # citation + claim-level groundedness + hallucination checks
  benchmark.py        # benchmark metrics + JSON / Markdown report generation
  benchmark_cli.py    # CI-friendly benchmark command
  benchmarks/v1.json  # versioned questions, evidence expectations, thresholds
  pipeline.py         # retrieval → answer → evaluation workflow
  types.py            # domain models
  demo.py             # deterministic end-to-end example

tests/
  test_reliability.py
  test_embedding_retrieval.py
  test_claim_groundedness.py
  test_benchmark.py

docs/
  architecture.md

.github/workflows/
  ci.yml
```

## Current maturity

**v0.3 — measurable RAG reliability baseline**

Implemented:

- deterministic lexical retrieval
- provider-neutral embedding interface
- cosine-similarity semantic retrieval
- deterministic embedding fixtures for offline tests
- claim-level evidence alignment
- explicit unsupported-claim detection
- machine-checkable citations
- groundedness + hallucination-risk scoring
- versioned benchmark dataset
- retrieval precision / coverage metrics
- benchmark JSON + Markdown reports
- CI-friendly regression thresholds
- deterministic demo and tests

Next milestones:

- larger benchmark coverage
- precision@k / recall@k by retrieval mode
- LLM provider adapter behind a strict interface
- semantic groundedness / entailment checks
- historical report comparison and trend tracking

## Engineering principle

> **A RAG answer is not trustworthy because it has citations. It is trustworthy only when the citations and claims can be verified against evidence — and that reliability should be measurable over time.**

## About this project

Built by **Ruban Chakravarthi** as a public AI-engineering portfolio project while transitioning from enterprise technology into hands-on AI engineering.

The goal is not to hide complexity behind a framework. It is to make reliability behavior visible, testable, measurable, and progressively more rigorous.
