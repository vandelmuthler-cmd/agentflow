from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    source = PROJECT_ROOT / "reports/v2/matrix/report.json"
    if not source.exists():
        raise SystemExit("Missing matrix report. Run scripts/evaluate_v2_retrieval.py --mode matrix first.")
    report = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for experiment in report["experiments"]:
        metrics = experiment["methods"]["vector"]
        rows.append({
            "strategy": experiment["strategy"],
            "model": experiment["model"],
            "all_gold_hit@5": metrics["all_gold_hit@5"],
            "mrr@10": metrics["mrr@10"],
            "average_query_pipeline_ms": metrics["average_query_pipeline_ms"],
            "model_parameter_bytes": experiment["index_metadata"]["model_parameter_bytes"],
            "index_bytes": experiment["index_metadata"]["vectors_bytes"],
        })
    best_recall = max(row["all_gold_hit@5"] for row in rows)
    # Twenty development intents each have two language variants. A 0.05 gap
    # therefore corresponds to at most one bilingual intent.
    finalists = [row for row in rows if row["all_gold_hit@5"] >= best_recall - 0.05]
    selected = sorted(
        finalists,
        key=lambda row: (
            -row["mrr@10"],
            row["average_query_pipeline_ms"],
            row["model_parameter_bytes"],
            row["index_bytes"],
        ),
    )[0]
    output = {
        "status": "provisional" if report.get("provisional") else "frozen",
        "dataset_sha256": report["dataset_sha256"],
        "selection_rules": [
            "maximize development all_gold_hit@5",
            "within one bilingual intent, maximize mrr@10",
            "then minimize query latency, model memory, and index size",
        ],
        "selected": selected,
        "all_configurations": sorted(rows, key=lambda row: (-row["all_gold_hit@5"], -row["mrr@10"])),
    }
    target = PROJECT_ROOT / "reports/v2/selected_config.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
