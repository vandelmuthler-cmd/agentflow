from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentflow.config import EVAL_RUNS_DIR
from agentflow.evals.retrieval_eval import evaluate, reciprocal_rank, search_strategy
from agentflow.retrieval.query_expansion import expand_query

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


EVAL_PATH = PROJECT_ROOT / "data" / "eval_set.json"
STRATEGY_MODES = [
    ("bm25", False),
    ("bm25", True),
    ("vector", False),
    ("vector", True),
    ("hybrid", False),
    ("hybrid", True),
    ("candidate_union", False),
]


def find_rank(ranked_ids: list[str], gold_id: str) -> int | None:
    try:
        return ranked_ids.index(gold_id) + 1
    except ValueError:
        return None


def run_reports() -> tuple[list[dict], list[dict]]:
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    summary_rows: list[dict] = []
    detail_rows: list[dict] = []

    for strategy, use_expansion in STRATEGY_MODES:
        report_top5 = evaluate(EVAL_PATH, [1, 3, 5], strategy=strategy, expand=use_expansion)
        report_top20 = evaluate(EVAL_PATH, [20], strategy=strategy, expand=use_expansion)
        mode = "expanded" if use_expansion else "original"
        summary_rows.append(
            {
                "strategy": strategy,
                "query_mode": mode,
                "recall_at_1": report_top5["recall"].get("Recall@1", 0.0),
                "recall_at_3": report_top5["recall"].get("Recall@3", 0.0),
                "recall_at_5": report_top5["recall"].get("Recall@5", 0.0),
                "candidate_recall_at_20": report_top20["recall"].get("Recall@20", 0.0),
                "mrr_at_20": report_top20["mrr"],
            }
        )

        for case in cases:
            query = expand_query(case["question"]) if use_expansion else case["question"]
            results = search_strategy(strategy, query, top_k=20)
            ranked_ids = [result.id for result in results]
            detail_rows.append(
                {
                    "strategy": strategy,
                    "query_mode": mode,
                    "question": case["question"],
                    "gold_id": case["gold_id"],
                    "gold_rank": find_rank(ranked_ids, case["gold_id"]),
                    "rr": reciprocal_rank(ranked_ids, case["gold_id"]),
                    "hit5": case["gold_id"] in ranked_ids[:5],
                    "hit20": case["gold_id"] in ranked_ids[:20],
                    "top5": " | ".join(ranked_ids[:5]),
                }
            )

    return summary_rows, detail_rows


def write_outputs(summary_rows: list[dict], detail_rows: list[dict]) -> None:
    EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    summary_json = EVAL_RUNS_DIR / "candidate_eval_summary.json"
    summary_csv = EVAL_RUNS_DIR / "candidate_eval_summary.csv"
    detail_csv = EVAL_RUNS_DIR / "candidate_eval_details.csv"

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
    summary_rows, detail_rows = run_reports()
    write_outputs(summary_rows, detail_rows)
    for row in summary_rows:
        print(
            f"{row['strategy']} ({row['query_mode']}): "
            f"R@1={row['recall_at_1']:.3f}, "
            f"R@3={row['recall_at_3']:.3f}, "
            f"R@5={row['recall_at_5']:.3f}, "
            f"CandidateR@20={row['candidate_recall_at_20']:.3f}, "
            f"MRR@20={row['mrr_at_20']:.3f}"
        )


if __name__ == "__main__":
    main()
