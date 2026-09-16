from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentflow.retrieval.store import load_documents
from agentflow.schemas import Evidence


CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")
REVIEW_FIELDS = [
    "case_id", "statement_index", "statement", "cited_ids", "auto_flag",
    "evidence_excerpts", "review_label", "review_reason",
]


def statement_review_rows(case: dict, documents: dict[str, Evidence]) -> list[dict[str, str]]:
    if case.get("writer_mode") == "skipped_after_verifier_error":
        return []
    evidence_ids = set(case.get("evidence_ids", []))
    rows = []
    for raw_line in case.get("answer", "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line in {"---", "***"}:
            continue
        statement = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", line).strip()
        if len(statement) < 8:
            continue
        cited_ids = [
            match for match in CITATION_PATTERN.findall(statement)
            if "__" in match or re.fullmatch(r"E\d+", match)
        ]
        unknown_ids = [
            item_id for item_id in cited_ids
            if item_id not in evidence_ids or item_id not in documents
        ]
        if unknown_ids:
            auto_flag = "unknown_citation_id"
        elif not cited_ids:
            auto_flag = "missing_citation"
        else:
            auto_flag = "needs_human_support_review"
        excerpts = [
            {"id": item_id, "source": documents[item_id].source, "text": documents[item_id].text}
            for item_id in cited_ids if item_id in evidence_ids and item_id in documents
        ]
        rows.append({
            "case_id": case["id"],
            "statement_index": str(len(rows) + 1),
            "statement": statement,
            "cited_ids": json.dumps(cited_ids, ensure_ascii=False),
            "auto_flag": auto_flag,
            "evidence_excerpts": json.dumps(excerpts, ensure_ascii=False),
            "review_label": "",
            "review_reason": "",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export answer statements for human citation-support review.")
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    details = json.loads(args.details.read_text(encoding="utf-8"))
    documents = {item.id: item for item in load_documents(args.index)}
    rows = [
        statement
        for case in details
        for statement in statement_review_rows(case, documents)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"cases={len(details)} review_statements={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
