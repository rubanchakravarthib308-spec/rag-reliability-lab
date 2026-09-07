# Architecture

The lab separates retrieval, answering, and evaluation so reliability checks remain testable independently from any specific LLM provider.

```text
Question
  ↓
Retriever
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
5. **Model independence** — the deterministic demo can be replaced later with a real LLM adapter without changing the reliability boundary.

## Current scope

The first version intentionally uses deterministic lexical retrieval and lightweight token-overlap evaluation. This keeps the core behavior transparent and testable without API keys.

Future versions can add embeddings, semantic similarity, claim-level entailment, LLM-as-judge evaluation, benchmark datasets, and experiment tracking while preserving the same interface boundaries.
