from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Export V2 answer review worksheet.")
    parser.add_argument("--dataset", type=Path, default=Path("data/v2_eval_frozen.json"))
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--document-index", type=Path, default=Path("data/index/v2/section_aware/documents.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = json.loads((PROJECT_ROOT / args.dataset).read_text(encoding="utf-8"))
    details = json.loads((PROJECT_ROOT / args.details).read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in dataset["cases"]}
    with (PROJECT_ROOT / args.document_index).open(encoding="utf-8") as file:
        documents = {item["id"]: item for line in file if (item := json.loads(line))}
    fields = [
        "id", "split", "category", "answerable", "question", "reference_answer",
        "required_facts", "gold_evidence", "model_answer", "retrieved_evidence_ids",
        "retrieved_evidence_text",
        "all_gold_retrieved_at_5", "abstention_detected", "fallback",
        "score_0_1_2", "citation_support", "unsupported_claims", "reason",
    ]
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in sorted(details, key=lambda item: item["id"]):
            case = cases[result["id"]]
            language = case["primary_language"]
            writer.writerow({
                "id": case["id"],
                "split": case["split"],
                "category": case["category"],
                "answerable": case["answerable"],
                "question": case[f"question_{language}"],
                "reference_answer": case[f"reference_answer_{language}"],
                "required_facts": " | ".join(case.get("required_facts", [])),
                "gold_evidence": " | ".join(
                    f"{item['document_id']} p.{item['page']}: {item['evidence_quote']}"
                    for item in case.get("gold_evidence", [])
                ),
                "model_answer": result["answer"],
                "retrieved_evidence_ids": " | ".join(result["evidence_ids"]),
                "retrieved_evidence_text": "\n\n".join(
                    f"[{evidence_id}] {documents[evidence_id].get('document_id', '')} "
                    f"p.{documents[evidence_id].get('page', '')}: {documents[evidence_id]['text']}"
                    for evidence_id in result["evidence_ids"] if evidence_id in documents
                ),
                "all_gold_retrieved_at_5": result["all_gold_retrieved_at_5"],
                "abstention_detected": result["abstention_detected"],
                "fallback": result["fallback"],
                "score_0_1_2": "",
                "citation_support": "",
                "unsupported_claims": "",
                "reason": "",
            })
    print(f"Exported {len(details)} cases: {output}")


if __name__ == "__main__":
    main()
