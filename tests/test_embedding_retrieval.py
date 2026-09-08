import pytest

from rag_reliability_lab.embeddings import StaticEmbeddingProvider, cosine_similarity
from rag_reliability_lab.retrieval import retrieve, retrieve_embeddings
from rag_reliability_lab.types import Document


def test_embedding_retrieval_finds_semantic_match_without_token_overlap() -> None:
    query = "prevent duplicate agent actions"
    documents = [
        Document(id="a", text="Idempotency keys stop the same tool call from running twice."),
        Document(id="b", text="Redis can cache frequently accessed application data."),
    ]
    provider = StaticEmbeddingProvider(
        vectors={
            query: (1.0, 0.0, 0.0),
            documents[0].text: (0.98, 0.05, 0.0),
            documents[1].text: (0.0, 1.0, 0.0),
        }
    )

    lexical = retrieve(query, documents)
    semantic = retrieve_embeddings(query, documents, provider)

    assert lexical == []
    assert semantic[0].document.id == "a"
    assert semantic[0].score > 0.9


def test_embedding_retrieval_respects_top_k_and_stable_tie_breaking() -> None:
    query = "agent safety"
    documents = [
        Document(id="b", text="second"),
        Document(id="a", text="first"),
        Document(id="c", text="third"),
    ]
    provider = StaticEmbeddingProvider(
        vectors={
            query: (1.0, 0.0),
            "second": (1.0, 0.0),
            "first": (1.0, 0.0),
            "third": (0.5, 0.5),
        }
    )

    results = retrieve_embeddings(query, documents, provider, top_k=2)

    assert [result.document.id for result in results] == ["a", "b"]


def test_embedding_provider_must_return_one_vector_per_input() -> None:
    class BrokenProvider:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0]]

    documents = [Document(id="a", text="content")]

    with pytest.raises(ValueError, match="one vector per input"):
        retrieve_embeddings("query", documents, BrokenProvider())


def test_cosine_similarity_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="same dimensions"):
        cosine_similarity([1.0, 0.0], [1.0])


def test_top_k_must_be_positive_for_both_retrievers() -> None:
    documents = [Document(id="a", text="content")]
    provider = StaticEmbeddingProvider(vectors={"query": (1.0,), "content": (1.0,)})

    with pytest.raises(ValueError, match="top_k"):
        retrieve("query", documents, top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        retrieve_embeddings("query", documents, provider, top_k=0)
