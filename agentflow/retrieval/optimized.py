from __future__ import annotations

from agentflow.retrieval.query_expansion import expand_query
from agentflow.retrieval.rerank import WeightedVectorReranker
from agentflow.schemas import Evidence


class OptimizedResearchRetriever:
    """Production entry retriever based on the best current evaluation result.

    The pipeline is:
    query -> domain-term expansion -> BM25 top-N candidates -> weighted rerank -> top-k evidence.
    """

    def __init__(
        self,
        keyword_retriever,
        reranker: WeightedVectorReranker,
        candidate_top_k: int = 20,
    ) -> None:
        self.keyword_retriever = keyword_retriever
        self.reranker = reranker
        self.candidate_top_k = candidate_top_k

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        expanded_query = expand_query(query)
        candidates = self.keyword_retriever.search(
            expanded_query, top_k=self.candidate_top_k
        )
        reranked = self.reranker.rerank(expanded_query, candidates, top_k=top_k)
        return [
            evidence.model_copy(update={"rank": rank})
            for rank, evidence in enumerate(reranked, 1)
        ]
