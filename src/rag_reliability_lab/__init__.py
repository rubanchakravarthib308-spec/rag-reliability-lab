"""Reliability-focused RAG evaluation primitives."""

from .embeddings import EmbeddingProvider, StaticEmbeddingProvider, cosine_similarity
from .pipeline import ReliabilityPipeline
from .retrieval import retrieve, retrieve_embeddings
from .types import AnswerResult, ClaimEvaluation, Document, EvaluationResult, RetrievalResult

__all__ = [
    "ReliabilityPipeline",
    "AnswerResult",
    "ClaimEvaluation",
    "Document",
    "EvaluationResult",
    "RetrievalResult",
    "EmbeddingProvider",
    "StaticEmbeddingProvider",
    "cosine_similarity",
    "retrieve",
    "retrieve_embeddings",
]
