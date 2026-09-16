from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentflow.config import EVAL_RUNS_DIR, RAW_DIR
from agentflow.evals.retrieval_eval import reciprocal_rank
from agentflow.retrieval.index import build_keyword_retriever
from agentflow.retrieval.query_expansion import expand_query
from agentflow.retrieval.rerank import WeightedVectorReranker

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


EVAL_PATH = PROJECT_ROOT / "data" / "eval_set.json"
VECTOR_WEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5]


def find_rank(ranked_ids: list[str], gold_id: str) -> int | None:
    try:
        return ranked_ids.index(gold_id) + 1
    except ValueError:
        return None


def summarize(rows: list[dict], total: int) -> dict:
    hits_at = {1: 0, 3: 0, 5: 0, 20: 0}
    mrr = 0.0
    for row in rows:
        rank = row["gold_rank"]
        if rank is None:
            continue
        for k in hits_at:
            if rank <= k:
                hits_at[k] += 1
        mrr += row["rr"]
    return {
        "recall_at_1": hits_at[1] / total if total else 0.0,
        "recall_at_3": hits_at[3] / total if total else 0.0,
        "recall_at_5": hits_at[5] / total if total else 0.0,
        "candidate_recall_at_20": hits_at[20] / total if total else 0.0,
        "mrr_at_20": mrr / total if total else 0.0,
    }


def evaluate_weight(weight: float) -> tuple[dict, list[dict]]:
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    retriever = build_keyword_retriever(RAW_DIR)
    reranker = WeightedVectorReranker(vector_weight=weight)
    detail_rows: list[dict] = []

    for case in cases:
        expanded_query = expand_query(case["question"])
        candidates = retriever.search(expanded_query, top_k=20)
        reranked = reranker.rerank(expanded_query, candidates, top_k=20)
        ranked_ids = [item.id for item in reranked]
        gold_rank = find_rank(ranked_ids, case["gold_id"])
        detail_rows.append(
            {
                "weight": weight,
                "question": case["question"],
                "gold_id": case["gold_id"],
                "gold_rank": gold_rank,
                "rr": reciprocal_rank(ranked_ids, case["gold_id"]),
                "hit5": case["gold_id"] in ranked_ids[:5],
                "hit20": case["gold_id"] in ranked_ids[:20],
                "top5": " | ".join(ranked_ids[:5]),
            }
        )

    summary = summarize(detail_rows, total=len(cases))
    summary["reranker"] = "weighted_vector"
    summary["vector_weight"] = weight
    return summary, detail_rows


def write_outputs(summary_rows: list[dict], detail_rows: list[dict]) -> None:
    EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    summary_json = EVAL_RUNS_DIR / "reranker_eval_summary.json"
    summary_csv = EVAL_RUNS_DIR / "reranker_eval_summary.csv"
    detail_csv = EVAL_RUNS_DIR / "reranker_eval_details.csv"

    summary_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    with summary_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    with detail_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    print(f"summary_json: {summary_json}")
    print(f"summary_csv: {summary_csv}")
    print(f"detail_csv: {detail_csv}")


def main() -> None:
    summary_rows: list[dict] = []
    all_details: list[dict] = []
    for weight in VECTOR_WEIGHTS:
        summary, details = evaluate_weight(weight)
        summary_rows.append(summary)
        all_details.extend(details)

    write_outputs(summary_rows, all_details)
    for row in summary_rows:
        print(
            f"weighted_vector w={row['vector_weight']}: "
            f"R@1={row['recall_at_1']:.3f}, "
            f"R@3={row['recall_at_3']:.3f}, "
            f"R@5={row['recall_at_5']:.3f}, "
            f"CandidateR@20={row['candidate_recall_at_20']:.3f}, "
            f"MRR@20={row['mrr_at_20']:.3f}"
        )


if __name__ == "__main__":
    main()
