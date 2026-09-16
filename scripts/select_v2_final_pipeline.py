from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    source = PROJECT_ROOT / "reports/v2/ablation/report.json"
    selected_source = PROJECT_ROOT / "reports/v2/selected_config.json"
    if not source.exists() or not selected_source.exists():
        raise SystemExit("Missing development ablation or selected embedding configuration.")
    report = json.loads(source.read_text(encoding="utf-8"))
    experiment = report["experiments"][0]
    rows = [
        {"method": method, **metrics}
        for method, metrics in experiment["methods"].items()
        if method != "candidate_union"
    ]
    best_recall = max(row["all_gold_hit@5"] for row in rows)
    finalists = [row for row in rows if row["all_gold_hit@5"] >= best_recall - 0.05]
    selected_method = sorted(
        finalists,
        key=lambda row: (-row["mrr@10"], row["average_query_pipeline_ms"]),
    )[0]
    base = json.loads(selected_source.read_text(encoding="utf-8"))["selected"]
    output = {
        "status": "provisional" if report.get("provisional") else "frozen",
        "dataset_sha256": report["dataset_sha256"],
        "strategy": base["strategy"],
        "model": base["model"],
        "method": selected_method["method"],
        "development_metrics": selected_method,
        "all_methods": sorted(rows, key=lambda row: (-row["all_gold_hit@5"], -row["mrr@10"])),
    }
    target = PROJECT_ROOT / "reports/v2/final_config.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
