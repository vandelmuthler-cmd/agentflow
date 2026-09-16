from __future__ import annotations

from agentflow.schemas import Evidence


def reciprocal_rank_fusion(result_sets: list[list[Evidence]], top_k: int = 5, k: int = 60) -> list[Evidence]:
    """Fuse ranked retrieval results with Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    docs: dict[str, Evidence] = {}

    for results in result_sets:
        for rank, doc in enumerate(results, 1):
            docs[doc.id] = doc
            scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (k + rank)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [
        docs[doc_id].model_copy(update={"score": scores[doc_id], "rank": rank})
        for rank, doc_id in enumerate(ranked_ids, 1)
    ]


class HybridRetriever:
    """Hybrid retriever that can combine keyword and vector result sets."""

    def __init__(self, keyword_retriever, vector_retriever=None) -> None:
        self.keyword_retriever = keyword_retriever
        self.vector_retriever = vector_retriever

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        keyword_results = self.keyword_retriever.search(query, top_k=top_k)
        result_sets = [keyword_results]
        if self.vector_retriever is not None:
            result_sets.append(self.vector_retriever.search(query, top_k=top_k))
        return reciprocal_rank_fusion(result_sets, top_k=top_k)
