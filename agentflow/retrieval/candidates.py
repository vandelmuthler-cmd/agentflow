from __future__ import annotations

from agentflow.schemas import Evidence


def dedupe_ranked_results(result_sets: list[list[Evidence]], top_k: int | None = None) -> list[Evidence]:
    """Merge ranked result lists while keeping the first occurrence of each chunk."""
    merged: list[Evidence] = []
    seen: set[str] = set()

    for results in result_sets:
        for result in results:
            if result.id in seen:
                continue
            seen.add(result.id)
            merged.append(result)
            if top_k is not None and len(merged) >= top_k:
                return _rerank_sequentially(merged)

    return _rerank_sequentially(merged)


def _rerank_sequentially(results: list[Evidence]) -> list[Evidence]:
    return [
        result.model_copy(update={"rank": rank})
        for rank, result in enumerate(results, 1)
    ]
