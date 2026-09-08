# Architecture

The lab separates retrieval, answering, and evaluation so reliability checks remain testable independently from any specific LLM or embedding provider.

```text
Question
  ↓
Retriever
  ├─ lexical path
  └─ embedding path → EmbeddingProvider
  ↓
Ranked evidence
  ↓
Answerer / LLM adapter
  ↓
Answer + citations
  ↓
Reliability evaluator
  ├─ citation validity
  ├─ groundedness
  └─ hallucination risk
  ↓
Pass / fail + reasons
```

## Design principles

1. **Evidence before generation** — an answer is evaluated only against retrieved evidence.
2. **Citations are machine-checkable** — cited document IDs must exist in the retrieval set.
3. **Unsupported content is measurable** — groundedness compares answer content with cited evidence.
4. **Failure is explicit** — the evaluator returns pass/fail plus reasons rather than silently accepting an answer.
5. **Provider independence** — embedding generation sits behind `EmbeddingProvider`, so retrieval logic is not tied to a vendor SDK.
6. **Offline determinism remains available** — lexical retrieval and `StaticEmbeddingProvider` keep CI and demos reproducible without network access.

## Retrieval boundary

The original lexical retriever converts text to token-frequency vectors and ranks documents with cosine similarity.

The semantic path keeps the ranking contract but delegates vector generation:

```text
query + document texts
        ↓
EmbeddingProvider.embed(...)
        ↓
query vector + document vectors
        ↓
cosine similarity
        ↓
stable score ordering + top-k
```

The provider must return exactly one non-empty vector per input text. Vector dimensions must match. The retrieval layer rejects malformed provider output rather than silently ranking invalid data.

A production adapter can call a hosted embedding API, local model, or internal service while the rest of the code continues to consume the same `RetrievalResult` objects.

## Current scope

The reliability evaluator still uses lightweight transparent token-overlap logic. Retrieval now supports both deterministic lexical ranking and provider-neutral embedding ranking.

Future versions can add claim-level evidence alignment, benchmark datasets, retrieval metrics, semantic entailment, LLM-as-judge evaluation, and experiment tracking while preserving the same interface boundaries.
