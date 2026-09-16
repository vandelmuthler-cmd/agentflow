from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentflow.config import EVAL_RUNS_DIR
from agentflow.evals.retrieval_eval import build_strategy
from agentflow.retrieval.query_expansion import make_query_variants

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


STRATEGIES = ["bm25", "vector", "hybrid"]
QUERY_MODES = ["original", "expanded"]


def hit_at(ranked_ids: list[str], gold_id: str, k: int) -> bool:
    return gold_id in ranked_ids[:k]


def classify_failure(record: dict) -> str:
    if record["hybrid_expanded_hit5"]:
        if not record["hybrid_original_hit5"]:
            return "fixed_by_expansion"
        return "hit"
    if record["bm25_expanded_hit5"] and not record["hybrid_expanded_hit5"]:
        return "fusion_ranking_regression"
    if record["hybrid_expanded_hit20"] and not record["hybrid_original_hit20"]:
        return "query_expansion_helped"
    if record["hybrid_expanded_hit20"] and not record["hybrid_expanded_hit5"]:
        return "ranking_issue"
    if record["bm25_expanded_hit20"] or record["vector_expanded_hit20"]:
        return "fusion_candidate_loss"
    if not record["hybrid_expanded_hit20"]:
        return "candidate_recall_miss"
    return "unknown"


def evaluate_records(eval_path: Path) -> list[dict]:
    cases = json.loads(eval_path.read_text(encoding="utf-8"))
    retrievers = {strategy: build_strategy(strategy) for strategy in STRATEGIES}
    records: list[dict] = []

    for idx, case in enumerate(cases, 1):
        variants = make_query_variants(case["question"])
        record = {
            "case_id": idx,
            "question": case["question"],
            "expanded_question": variants["expanded"],
            "gold_id": case["gold_id"],
        }

        for strategy, retriever in retrievers.items():
            for mode in QUERY_MODES:
                ranked_ids = [
                    item.id
                    for item in retriever.search(variants[mode], top_k=20)
                ]
                prefix = f"{strategy}_{mode}"
                record[f"{prefix}_top5"] = ranked_ids[:5]
                record[f"{prefix}_top20"] = ranked_ids
                record[f"{prefix}_hit5"] = hit_at(ranked_ids, case["gold_id"], 5)
                record[f"{prefix}_hit20"] = hit_at(ranked_ids, case["gold_id"], 20)

        record["failure_label"] = classify_failure(record)
        records.append(record)
    return records


def summarize(records: list[dict]) -> dict:
    total = len(records)
    summary: dict[str, dict[str, float | int]] = {}
    for strategy in STRATEGIES:
        for mode in QUERY_MODES:
            prefix = f"{strategy}_{mode}"
            hit5 = sum(1 for record in records if record[f"{prefix}_hit5"])
            hit20 = sum(1 for record in records if record[f"{prefix}_hit20"])
            summary[prefix] = {
                "hit5": hit5,
                "recall_at_5": hit5 / total if total else 0.0,
                "hit20": hit20,
                "candidate_recall_at_20": hit20 / total if total else 0.0,
            }
    return {"total": total, "summary": summary}


def write_outputs(records: list[dict], summary: dict) -> None:
    EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = EVAL_RUNS_DIR / "retrieval_audit_top20.json"
    csv_path = EVAL_RUNS_DIR / "retrieval_audit_top20.csv"
    summary_path = EVAL_RUNS_DIR / "retrieval_audit_summary.json"

    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "case_id",
        "question",
        "expanded_question",
        "gold_id",
        "failure_label",
    ]
    for strategy in STRATEGIES:
        for mode in QUERY_MODES:
            prefix = f"{strategy}_{mode}"
            fieldnames.extend([f"{prefix}_hit5", f"{prefix}_hit20", f"{prefix}_top5"])

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field, "") for field in fieldnames}
            for strategy in STRATEGIES:
                for mode in QUERY_MODES:
                    prefix = f"{strategy}_{mode}"
                    row[f"{prefix}_top5"] = " | ".join(record[f"{prefix}_top5"])
            writer.writerow(row)

    print(f"json: {json_path}")
    print(f"csv: {csv_path}")
    print(f"summary: {summary_path}")


def main() -> None:
    eval_path = PROJECT_ROOT / "data" / "eval_set.json"
    records = evaluate_records(eval_path)
    summary = summarize(records)
    write_outputs(records, summary)

    print(f"Total: {summary['total']}")
    for name, stats in summary["summary"].items():
        print(
            f"{name}: Recall@5={stats['recall_at_5']:.3f} "
            f"({stats['hit5']}/{summary['total']}), "
            f"CandidateRecall@20={stats['candidate_recall_at_20']:.3f} "
            f"({stats['hit20']}/{summary['total']})"
        )

    labels: dict[str, int] = {}
    for record in records:
        labels[record["failure_label"]] = labels.get(record["failure_label"], 0) + 1
    print("Failure labels:")
    for label, count in sorted(labels.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
