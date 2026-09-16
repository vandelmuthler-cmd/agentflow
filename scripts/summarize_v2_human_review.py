from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_statement_review import statement_review_rows


STATEMENT_LABELS = {"supported", "partial", "unsupported", "not_applicable"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def summarize_review(
    details: list[dict], answer_rows: list[dict[str, str]], statement_rows: list[dict[str, str]],
) -> dict:
    cases = {case["id"]: case for case in details}
    if len(cases) != len(details):
        raise ValueError("Duplicate case IDs in details")
    if len({row["id"] for row in answer_rows}) != len(answer_rows):
        raise ValueError("Duplicate case IDs in answer review")
    if {row["id"] for row in answer_rows} != set(cases):
        raise ValueError("Answer review case IDs do not match details")

    answer_scores = Counter()
    unscored_case_ids = []
    failed_case_ids = []
    for row in answer_rows:
        case = cases[row["id"]]
        failed = bool(case.get("workflow_error")) or case.get("writer_mode") == "skipped_after_verifier_error"
        if failed:
            failed_case_ids.append(row["id"])
        score = row.get("score_0_1_2", "").strip()
        if not score:
            if not failed:
                unscored_case_ids.append(row["id"])
            continue
        if score not in {"0", "1", "2"}:
            raise ValueError(f"Invalid answer score for {row['id']}: {score}")
        if failed:
            raise ValueError(f"Failed-closed case cannot receive an answer score: {row['id']}")
        if not row.get("reason", "").strip():
            raise ValueError(f"Missing answer review reason: {row['id']}")
        answer_scores[score] += 1

    expected_statements = {
        (case["id"], row["statement_index"]): row
        for case in details for row in statement_review_rows(case, {})
    }
    seen_statements = set()
    statement_labels = Counter()
    pending_statements = 0
    for row in statement_rows:
        case_id = row["case_id"]
        if case_id not in cases:
            raise ValueError(f"Unknown statement case ID: {case_id}")
        if case_id in failed_case_ids:
            raise ValueError(f"Statement review includes failed-closed case: {case_id}")
        key = (case_id, row["statement_index"])
        if key in seen_statements:
            raise ValueError(f"Duplicate statement review row: {key}")
        seen_statements.add(key)
        expected = expected_statements.get(key)
        if expected is None or row.get("statement") != expected["statement"] or row.get("cited_ids") != expected["cited_ids"]:
            raise ValueError(f"Statement review is stale or changed: {key}")
        label = row.get("review_label", "").strip()
        if not label:
            pending_statements += 1
            continue
        if label not in STATEMENT_LABELS:
            raise ValueError(f"Invalid statement label for {key}: {label}")
        if not row.get("review_reason", "").strip():
            raise ValueError(f"Missing statement review reason: {key}")
        statement_labels[label] += 1

    if seen_statements != set(expected_statements):
        raise ValueError("Statement review rows do not match answer lines")

    return {
        "case_count": len(cases),
        "failed_closed_case_ids": sorted(failed_case_ids),
        "answer_scores": {score: answer_scores[score] for score in ("0", "1", "2")},
        "unscored_case_ids": sorted(unscored_case_ids),
        "statement_count": len(statement_rows),
        "statement_labels": {label: statement_labels[label] for label in sorted(STATEMENT_LABELS)},
        "pending_statement_count": pending_statements,
        "human_review_complete": not unscored_case_ids and not pending_statements and bool(cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize independent human answer and citation reviews.")
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--answer-scores", type=Path, required=True)
    parser.add_argument("--statement-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_review(
        json.loads(args.details.read_text(encoding="utf-8")),
        read_csv(args.answer_scores), read_csv(args.statement_review),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
