from __future__ import annotations

import numpy as np

from agentflow.retrieval.embedding import embed_texts
from agentflow.retrieval.vector import load_vector_index
from agentflow.schemas import Evidence


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


class WeightedVectorReranker:
    """Rerank a candidate pool with original retriever score plus vector similarity.

    This is a lightweight, reproducible baseline. It is not a neural cross-encoder
    reranker, but it lets us validate the reranking evaluation pipeline before
    adding a heavier model.
    """

    def __init__(self, vector_weight: float = 0.3) -> None:
        self.vector_weight = vector_weight
        self.embeddings, self.ids = load_vector_index()
        self.id_to_index = {doc_id: idx for idx, doc_id in enumerate(self.ids)}

    def rerank(self, query: str, candidates: list[Evidence], top_k: int | None = None) -> list[Evidence]:
        if not candidates:
            return []
        lexical_scores = minmax([candidate.score for candidate in candidates])
        vector_scores = self._vector_scores(query, candidates)
        combined: list[tuple[float, Evidence]] = []

        for candidate, lexical_score, vector_score in zip(candidates, lexical_scores, vector_scores):
            score = (1.0 - self.vector_weight) * lexical_score + self.vector_weight * vector_score
            combined.append((score, candidate))

        ranked = sorted(combined, key=lambda item: item[0], reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return [
            candidate.model_copy(update={"score": score, "rank": rank})
            for rank, (score, candidate) in enumerate(ranked, 1)
        ]

    def _vector_scores(self, query: str, candidates: list[Evidence]) -> list[float]:
        if self.embeddings.size == 0:
            return [0.0 for _ in candidates]

        query_embedding = embed_texts([query])[0]
        scores: list[float] = []
        for candidate in candidates:
            idx = self.id_to_index.get(candidate.id)
            if idx is None:
                scores.append(0.0)
                continue
            scores.append(float(np.dot(self.embeddings[idx], query_embedding)))
        return minmax(scores)
