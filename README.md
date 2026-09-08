# RAG Reliability Lab

> **A transparent testbed for retrieval quality, citation validity, groundedness, and hallucination risk in RAG systems.**

Many RAG demos stop when the model returns an answer. This project focuses on the harder question:

**Can we measure whether the answer is actually supported by the evidence the system retrieved?**

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
Citation validation
  ↓
Groundedness scoring
  ↓
Hallucination-risk check
  ↓
Pass / fail with reasons
```

The baseline remains deterministic and API-key free. Semantic retrieval is exposed through a provider-neutral embedding interface, so a real embedding service can be plugged in later without coupling the reliability pipeline to one vendor.

## Why reliability matters

A RAG system can retrieve relevant documents and still produce an unsupported answer. It can also attach citations that do not actually belong to the evidence used.

This lab makes those failure modes explicit by checking:

- **Retrieval relevance** — rank evidence against the question
- **Citation validity** — confirm cited document IDs were actually retrieved
- **Groundedness** — estimate how much answer content is supported by cited evidence
- **Hallucination risk** — surface unsupported content instead of hiding it
- **Failure reasons** — return machine-readable reasons when an answer should not pass

## Retrieval boundary

Two retrieval paths are now available:

```text
Question ──→ Lexical Retriever ──→ token-vector cosine similarity
    │
    └──────→ Embedding Retriever ──→ EmbeddingProvider ──→ vector cosine similarity
```

`retrieve(...)` preserves the deterministic lexical path used by the original demo and offline tests.

`retrieve_embeddings(...)` accepts any object implementing the provider-neutral `EmbeddingProvider` protocol. The repository includes `StaticEmbeddingProvider` for deterministic fixtures, so semantic-ranking behavior can be tested without an API key or network call.

Example:

```python
from rag_reliability_lab import Document, StaticEmbeddingProvider, retrieve_embeddings

query = "prevent duplicate agent actions"
documents = [
    Document(id="a", text="Idempotency keys stop the same tool call from running twice."),
    Document(id="b", text="Redis can cache frequently accessed application data."),
]

provider = StaticEmbeddingProvider(
    vectors={
        query: (1.0, 0.0),
        documents[0].text: (0.99, 0.01),
        documents[1].text: (0.0, 1.0),
    }
)

results = retrieve_embeddings(query, documents, provider, top_k=1)
```

A production adapter only needs to implement:

```python
class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

The retriever validates vector counts and dimensions before ranking documents.

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

No paid API, model key, vector database, or external service is required for the baseline demo or embedding-retrieval tests.

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
  embeddings.py     # provider-neutral embedding interface + deterministic fixture
  retrieval.py      # lexical and embedding-based evidence ranking
  evaluation.py     # citation + groundedness + hallucination checks
  pipeline.py       # retrieval → answer → evaluation workflow
  types.py          # domain models
  demo.py           # deterministic end-to-end example

tests/
  test_reliability.py
  test_embedding_retrieval.py

docs/
  architecture.md

.github/workflows/
  ci.yml
```

## Current maturity

**v0.2 — pluggable retrieval boundary**

Implemented:

- deterministic lexical retrieval
- provider-neutral embedding interface
- cosine-similarity semantic retrieval
- deterministic embedding fixtures for offline tests
- configurable top-k retrieval
- explicit evidence objects
- machine-checkable citations
- groundedness score
- hallucination-risk score
- pass/fail reliability gate
- safety-focused tests
- CI validation
- deterministic demo

Next milestones:

- claim-level evidence alignment
- precision@k / recall@k retrieval metrics
- benchmark dataset + experiment runner
- LLM provider adapter behind a strict interface
- semantic groundedness / entailment checks
- evaluation reports and regression tracking

## Engineering principle

> **A RAG answer is not trustworthy because it has citations. It is trustworthy only when the citations and claims can be verified against the evidence.**

## About this project

Built by **Ruban Chakravarthi** as a public AI-engineering portfolio project while transitioning from enterprise technology into hands-on AI engineering.

The goal is not to hide complexity behind a framework. It is to make reliability behavior visible, testable, and progressively more rigorous.
