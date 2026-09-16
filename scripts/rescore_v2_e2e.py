from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentflow.evals.v2_eval import bootstrap_interval, load_v2_dataset, map_gold_spans
from agentflow.retrieval.store import load_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescore unchanged V2 end-to-end outputs against a reviewed dataset."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/v2_eval_frozen.json"))
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--answer-review", type=Path)
    parser.add_argument(
        "--document-index",
        type=Path,
        default=Path("data/index/v2/section_aware/documents.jsonl"),
    )
    return parser.parse_args()


def claim_map(case_id: str, mappings: list) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for mapping in mappings:
        if mapping.case_id == case_id:
            result[mapping.claim_id].update(mapping.chunk_ids)
    return dict(result)


def citation_metrics(answer: str, evidence_ids: list[str], claims: dict[str, set[str]]) -> dict:
    cited_ids = set(
        re.findall(
            r"\[([a-z0-9][a-z0-9-]*__(?:fixed_char|recursive_token|section_aware)__\d+)\]",
            answer,
            re.IGNORECASE,
        )
    )
    valid_gold_ids = set().union(*claims.values()) if claims else set()
    return {
        "citation_id_validity": len(cited_ids & set(evidence_ids)) / len(cited_ids)
        if cited_ids
        else None,
        "gold_citation_precision": len(cited_ids & valid_gold_ids) / len(cited_ids)
        if cited_ids and claims
        else None,
        "gold_claim_citation_coverage": sum(bool(ids & cited_ids) for ids in claims.values())
        / len(claims)
        if claims
        else None,
    }


def mean_present(rows: list[dict], field: str) -> float | None:
    values = [row[field] for row in rows if row.get(field) is not None]
    return mean(values) if values else None


def main() -> None:
    args = parse_args()
    dataset = load_v2_dataset(PROJECT_ROOT / args.dataset)
    cases = {
        case.id: case
        for case in dataset.cases
        if case.split in {"frozen", "stress"}
    }
    source_dir = PROJECT_ROOT / args.source_dir
    source_manifest = json.loads((source_dir / "run_manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((source_dir / "details.json").read_text(encoding="utf-8"))
    if set(source_manifest["case_ids"]) != set(cases) or {row["id"] for row in rows} != set(cases):
        raise ValueError("Source run and reviewed dataset do not contain the same 60 cases.")
    for row in rows:
        case = cases[row["id"]]
        expected_question = case.question_zh if case.primary_language == "zh" else case.question_en
        if row["question"] != expected_question:
            raise ValueError(f"Question text changed for {case.id}; regeneration is required.")

    review = None
    review_scores: dict[str, dict] = {}
    if args.answer_review:
        review = json.loads((PROJECT_ROOT / args.answer_review).read_text(encoding="utf-8"))
        if review.get("dataset_sha256") != dataset.sha256:
            raise ValueError("Answer review targets a different frozen dataset.")
        full_credit = set(review.get("full_credit_case_ids", []))
        exceptions = {item["id"]: item for item in review.get("exceptions", [])}
        if full_credit & exceptions.keys() or full_credit | exceptions.keys() != set(cases):
            raise ValueError("Answer review must cover every case exactly once.")
        review_scores = {
            case_id: {"score": 2, "reason": "All required answer facts are present."}
            for case_id in full_credit
        }
        review_scores.update(exceptions)

    chunks = load_documents(PROJECT_ROOT / args.document_index)
    mappings = map_gold_spans(list(cases.values()), chunks)
    rescored = []
    for source_row in rows:
        row = dict(source_row)
        claims = claim_map(row["id"], mappings)
        evidence_ids = row.get("evidence_ids", [])
        top_five = evidence_ids[:5]
        hits_at_five = [bool(ids.intersection(top_five)) for ids in claims.values()]
        hits_in_context = [bool(ids.intersection(evidence_ids)) for ids in claims.values()]
        row["gold_claim_count"] = len(claims)
        row["all_gold_retrieved_at_5"] = bool(hits_at_five) and all(hits_at_five)
        row["claim_recall_at_5"] = mean(hits_at_five) if hits_at_five else 0.0
        row["all_gold_in_final_context"] = bool(hits_in_context) and all(hits_in_context)
        row["claim_recall_final_context"] = mean(hits_in_context) if hits_in_context else 0.0
        row.update(citation_metrics(row.get("answer", ""), evidence_ids, claims))
        if row["id"] in review_scores:
            row["review_score_0_1_2"] = review_scores[row["id"]]["score"]
            row["review_reason"] = review_scores[row["id"]]["reason"]
        rescored.append(row)

    answerable = [row for row in rescored if row["answerable"]]
    unanswerable = [row for row in rescored if not row["answerable"]]
    conflict = [row for row in rescored if row["category"] == "numerical_conflict"]
    all_gold = [float(row["all_gold_retrieved_at_5"]) for row in answerable]
    in_context = [float(row["all_gold_in_final_context"]) for row in answerable]
    low, high = bootstrap_interval(all_gold)
    latencies = [row["latency_ms"] for row in rescored]
    p95 = sorted(latencies)[min(len(latencies) - 1, int(0.95 * len(latencies)))]
    scored = [row for row in rescored if row.get("review_score_0_1_2") is not None]
    summary = {
        "status": "frozen",
        "generation_dataset_sha256": source_manifest["dataset_sha256"],
        "evaluation_dataset_sha256": dataset.sha256,
        "generation_reused": True,
        "reuse_reason": "Case IDs and primary-language question text are unchanged; only reviewed references and Gold spans changed.",
        "configuration": source_manifest["configuration"],
        "case_count": len(rescored),
        "answerable_case_count": len(answerable),
        "primary_language_counts": dict(Counter(row["primary_language"] for row in rescored)),
        "answer_review": {
            "review_type": review.get("review_type") if review else "not_reviewed",
            "independent_human_review": review.get("independent_human_review") if review else False,
            "complete": sum(row["review_score_0_1_2"] == 2 for row in scored),
            "partial": sum(row["review_score_0_1_2"] == 1 for row in scored),
            "failed": sum(row["review_score_0_1_2"] == 0 for row in scored),
            "unscored": len(rescored) - len(scored),
        },
        "all_gold_hit_at_5": mean(all_gold),
        "all_gold_in_final_context": mean(in_context),
        "all_gold_hit_at_5_ci95": [low, high],
        "citation_id_validity": mean_present(rescored, "citation_id_validity"),
        "gold_citation_precision": mean_present(rescored, "gold_citation_precision"),
        "gold_claim_citation_coverage": mean_present(rescored, "gold_claim_citation_coverage"),
        "groundedness_proxy": mean(row["groundedness"] for row in rescored),
        "unanswerable_abstention_keyword_rate": mean(
            row["abstention_detected"] for row in unanswerable
        ),
        "numerical_conflict_phrase_rate": mean(
            bool(row["numerical_conflict_recognized"]) for row in conflict
        ),
        "average_latency_ms": mean(latencies),
        "p95_latency_ms": p95,
        "planner_calls": sum(row["planner_calls"] for row in rescored),
        "verifier_calls": sum(row["verifier_calls"] for row in rescored),
        "writer_calls": sum(row["writer_calls"] for row in rescored),
        "total_tokens": sum(row["total_tokens"] for row in rescored),
        "logical_retrieval_calls": sum(row["logical_retrieval_calls"] for row in rescored),
        "fallback_case_ids": [row["id"] for row in rescored if row["fallback"]],
        "failed_case_ids": [row["id"] for row in rescored if row["workflow_error"]],
    }
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "details.json").write_text(
        json.dumps(sorted(rescored, key=lambda row: row["id"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "source_run": str(args.source_dir).replace("\\", "/"),
                "source_manifest": source_manifest,
                "scoring_dataset": str(args.dataset).replace("\\", "/"),
                "question_identity_verified": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
