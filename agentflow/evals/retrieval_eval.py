from __future__ import annotations

import json
import sys
from pathlib import Path

from agentflow.config import DEFAULT_CANDIDATE_TOP_K, RAW_DIR
from agentflow.retrieval.candidates import dedupe_ranked_results
from agentflow.retrieval.index import build_hybrid_retriever, build_keyword_retriever, build_vector_retriever
from agentflow.retrieval.query_expansion import expand_query

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def build_strategy(name: str):
    if name == "bm25":
        return build_keyword_retriever(RAW_DIR)
    if name == "vector":
        return build_vector_retriever(RAW_DIR)
    if name == "hybrid":
        return build_hybrid_retriever(RAW_DIR)
    raise ValueError(f"unknown strategy: {name}")


def reciprocal_rank(ranked_ids: list[str], gold_id: str) -> float:
    try:
        return 1.0 / (ranked_ids.index(gold_id) + 1)
    except ValueError:
        return 0.0


def search_strategy(strategy: str, query: str, top_k: int):
    if strategy == "candidate_union":
        bm25 = build_strategy("bm25")
        vector = build_strategy("vector")
        return dedupe_ranked_results(
            [
                bm25.search(expand_query(query), top_k=top_k),
                vector.search(expand_query(query), top_k=top_k),
                bm25.search(query, top_k=top_k),
                vector.search(query, top_k=top_k),
            ],
            top_k=top_k,
        )
    return build_strategy(strategy).search(query, top_k=top_k)


def evaluate(eval_path: Path, ks: list[int], strategy: str = "bm25", expand: bool = False) -> dict:
    cases = json.loads(eval_path.read_text(encoding="utf-8"))
    hits_at = {k: 0 for k in ks}
    misses: list[dict] = []
    max_k = max(ks)
    retrieval_depth = max(max_k, DEFAULT_CANDIDATE_TOP_K)
    mrr_total = 0.0

    for case in cases:
        query = expand_query(case["question"]) if expand else case["question"]
        # Evaluate every K from one fixed candidate ranking. RRF can otherwise
        # change its order when each source is asked for top-5 versus top-20.
        results = search_strategy(strategy, query, top_k=retrieval_depth)
        ranked_ids = [item.id for item in results]
        mrr_total += reciprocal_rank(ranked_ids, case["gold_id"])
        for k in ks:
            if case["gold_id"] in ranked_ids[:k]:
                hits_at[k] += 1
        if case["gold_id"] not in ranked_ids[:max_k]:
            misses.append(
                {
                    "question": case["question"],
                    "gold_id": case["gold_id"],
                    "retrieved_ids": ranked_ids,
                }
            )

    total = len(cases)
    return {
        "total": total,
        "strategy": strategy,
        "query_mode": "expanded" if expand else "original",
        "recall": {f"Recall@{k}": hits_at[k] / total if total else 0.0 for k in ks},
        "hits": {f"Recall@{k}": hits_at[k] for k in ks},
        "mrr": mrr_total / total if total else 0.0,
        "misses": misses,
    }


if __name__ == "__main__":
    path = Path("data/eval_set.json")
    if not path.exists():
        raise SystemExit("Missing data/eval_set.json. Add fixed evaluation cases first.")
    for strategy in ["bm25", "vector", "hybrid", "candidate_union"]:
        expand_modes = [False, True] if strategy != "candidate_union" else [False]
        for expand in expand_modes:
            report = evaluate(path, [1, 3, 5], strategy=strategy, expand=expand)
            print(f"\nStrategy: {strategy} ({report['query_mode']})")
            print(f"Total: {report['total']}")
            for name, value in report["recall"].items():
                print(f"{name}: {value:.3f} ({report['hits'][name]}/{report['total']})")
            print(f"MRR: {report['mrr']:.3f}")
            print(f"Misses@5: {len(report['misses'])}")
