from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentflow.agents.context import ContextManager
from agentflow.agents.evidence_verifier import StructuredEvidenceVerifier
from agentflow.agents.langgraph_workflow import LangGraphResearchWorkflow
from agentflow.agents.planner import StructuredPlanner
from agentflow.agents.state import AgentState
from agentflow.config import RAW_DIR
from agentflow.generation.llm_writer import LLMWriter
from agentflow.retrieval.index import build_retriever
from agentflow.retrieval.query_expansion import expand_query
from agentflow.storage.traces import InMemoryTraceStore
from agentflow.tools.document_search import DocumentSearchTool
from agentflow.tools.registry import ToolRegistry


ABSTENTION_PATTERNS = (
    r"^(?:基于|根据).{0,80}(?:无法|不能|证据不足|只能部分回答)",
    r"^(?:无法|不能|证据不足|没有检索到|只能部分回答)",
    r"^(?:based on|according to).{0,80}(?:cannot|insufficient|unable)",
    r"^(?:insufficient evidence|cannot|unable|not enough evidence)",
)


def term_coverage(text: str, terms: list[str]) -> float:
    if not terms:
        return 1.0
    lowered = text.lower()
    return sum(term.lower() in lowered for term in terms) / len(terms)


def detect_abstention(answer: str, structural_abstention: bool) -> bool:
    if structural_abstention:
        return True
    body = "\n".join(
        line for line in answer.splitlines() if not line.lstrip().startswith("#")
    ).strip()
    first_paragraph = re.split(r"\n\s*\n", body, maxsplit=1)[0]
    normalized = re.sub(r"[*_`]", "", first_paragraph).strip()
    return any(
        re.search(pattern, normalized, flags=re.IGNORECASE | re.DOTALL)
        for pattern in ABSTENTION_PATTERNS
    )


def mean(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row[field]) for row in rows) / len(rows), 4)


def evaluate(
    dataset_path: Path,
    *,
    online: bool = False,
    llm_verifier: bool = True,
    case_limit: int | None = None,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_bytes = dataset_path.read_bytes()
    cases = json.loads(dataset_bytes.decode("utf-8"))
    evaluated_at = datetime.now(timezone.utc).isoformat()
    if case_limit is not None:
        cases = cases[:case_limit]

    retriever = build_retriever(RAW_DIR)
    registry = ToolRegistry()
    registry.register(DocumentSearchTool(retriever))
    rows: list[dict[str, Any]] = []

    with TemporaryDirectory() as temporary_dir:
        workflow_options: dict[str, Any] = {}
        if not online:
            workflow_options = {
                "planner": StructuredPlanner("", "", ""),
                "evidence_verifier": StructuredEvidenceVerifier("", "", ""),
                "llm_writer": LLMWriter("", "", ""),
            }
        elif not llm_verifier:
            workflow_options = {
                "evidence_verifier": StructuredEvidenceVerifier("", "", ""),
            }
        workflow = LangGraphResearchWorkflow(
            registry,
            InMemoryTraceStore(),
            context_manager=ContextManager(Path(temporary_dir)),
            **workflow_options,
        )

        for index, case in enumerate(cases, start=1):
            started = time.perf_counter()
            run_id = f"heldout-{uuid.uuid4()}"
            workflow_error = ""
            report = None
            try:
                report = workflow.run(
                    AgentState(run_id=run_id, question=case["question"], top_k=5)
                )
            except Exception as error:  # Keep completed cases when one request fails unexpectedly.
                workflow_error = f"{type(error).__name__}: {error}"
            latency_ms = (time.perf_counter() - started) * 1000

            gold_ids = set(case["gold_ids"])
            final_ids = [item.id for item in report.evidence] if report else []
            expanded = expand_query(case["question"])
            candidate_ids = [
                item.id
                for item in retriever.keyword_retriever.search(expanded, top_k=20)
            ]
            ranks = [final_ids.index(item) + 1 for item in gold_ids if item in final_ids]
            is_unanswerable = case["category"] == "unanswerable"
            any_hit = bool(gold_ids.intersection(final_ids)) if gold_ids else False
            all_hit = gold_ids.issubset(final_ids) if gold_ids else False
            candidate_all_hit = gold_ids.issubset(candidate_ids) if gold_ids else False

            answer = report.answer if report else ""
            coverage = term_coverage(answer, case["required_terms"])
            structural_abstention = bool(report and report.generation_quality.abstention)
            abstention_detected = detect_abstention(answer, structural_abstention)
            citation_id_validity = (
                report.generation_quality.citation_correctness if report else 0.0
            )
            citation_completeness = (
                report.generation_quality.citation_completeness if report else 0.0
            )

            planning_mode = report.planning_mode if report else "failed"
            writer_mode = report.writer_mode if report else "failed"
            verifier_mode = report.verifier_mode if report else "failed"
            verifier_expected = bool(final_ids)
            writer_expected = bool(final_ids)
            planner_online = planning_mode == "llm"
            verifier_online = verifier_mode == "llm"
            writer_online = writer_mode == "llm"
            intended_no_evidence_abstention = not writer_expected and structural_abstention
            online_execution_valid = bool(
                not workflow_error
                and planner_online
                and (
                    verifier_online
                    if llm_verifier and verifier_expected
                    else True
                )
                and (writer_online if writer_expected else intended_no_evidence_abstention)
            ) if online else True

            automated_success = (
                abstention_detected
                if is_unanswerable
                else all_hit and coverage == 1.0 and citation_id_validity == 1.0
            )
            usage = report.model_usage if report else []
            row = {
                "id": case["id"],
                "run_id": run_id,
                "category": case["category"],
                "question": case["question"],
                "gold_ids": sorted(gold_ids),
                "required_terms": case["required_terms"],
                "candidate_all_hit_at_20": candidate_all_hit,
                "any_gold_hit_at_5": any_hit,
                "all_gold_hit_at_5": all_hit,
                "reciprocal_rank_at_5": round(1 / min(ranks), 4) if ranks else 0.0,
                "required_term_coverage": round(coverage, 4),
                "abstention_detected": abstention_detected,
                "structural_abstention": structural_abstention,
                "citation_id_validity": citation_id_validity,
                "citation_completeness": citation_completeness,
                "planning_mode": planning_mode,
                "planning_error": report.planning_error if report else "",
                "verifier_expected": verifier_expected,
                "verifier_mode": verifier_mode,
                "verifier_decision": report.verifier_decision if report else "failed",
                "verifier_error": report.verifier_error if report else "",
                "verifier_retries": report.verifier_history[-1]["retry_index"] if report and report.verifier_history else 0,
                "verifier_history": report.verifier_history if report else [],
                "verifier_missing_aspects": report.verifier_missing_aspects if report else [],
                "verifier_follow_up_queries": report.verifier_follow_up_queries if report else [],
                "retrieval_queries": report.retrieval_queries if report else [],
                "retrieval_rounds": report.retrieval_rounds if report else [],
                "retrieval_tool_calls": report.retrieval_tool_calls if report else 0,
                "retrieval_tool_attempts": report.retrieval_tool_attempts if report else 0,
                "retrieval_duration_ms": report.retrieval_duration_ms if report else 0.0,
                "retrieval_no_progress_rounds": report.retrieval_no_progress_rounds if report else 0,
                "retrieval_stop_reason": report.retrieval_stop_reason if report else "workflow_failed",
                "writer_expected": writer_expected,
                "writer_mode": writer_mode,
                "writer_error": report.writer_error if report else "",
                "online_execution_valid": online_execution_valid,
                "workflow_error": workflow_error,
                "model_calls": len(usage),
                "models": sorted({item.model for item in usage if item.model}),
                "prompt_tokens": sum(item.prompt_tokens for item in usage),
                "completion_tokens": sum(item.completion_tokens for item in usage),
                "total_tokens": sum(item.total_tokens for item in usage),
                "estimated_cost_usd": round(
                    sum(item.estimated_cost_usd for item in usage), 8
                ),
                "latency_ms": round(latency_ms, 3),
                "automated_e2e_success": automated_success,
                "top5_ids": final_ids,
                "plan": report.plan if report else [],
                "answer": answer,
            }
            rows.append(row)
            if verbose:
                print(
                    f"[{index}/{len(cases)}] {case['id']} "
                    f"plan={planning_mode} verifier={verifier_mode}/{row['verifier_decision']} "
                    f"writer={writer_mode} retries={row['verifier_retries']} "
                    f"online_valid={online_execution_valid} tokens={row['total_tokens']}",
                    flush=True,
                )

    answerable = [row for row in rows if row["category"] != "unanswerable"]
    unanswerable = [row for row in rows if row["category"] == "unanswerable"]
    valid_online = [row for row in rows if row["online_execution_valid"]]
    fallback_rows = [row for row in rows if online and not row["online_execution_valid"]]
    cost_configured = any(row["estimated_cost_usd"] > 0 for row in rows)
    summary = {
        "dataset": dataset_path.name,
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "evaluated_at_utc": evaluated_at,
        "evaluation_mode": "online_external_llm" if online else "offline_deterministic",
        "case_count": len(rows),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "candidate_all_recall_at_20": mean(answerable, "candidate_all_hit_at_20"),
        "query_hit_rate_at_5": mean(answerable, "any_gold_hit_at_5"),
        "all_gold_recall_at_5": mean(answerable, "all_gold_hit_at_5"),
        "mrr_at_5": mean(answerable, "reciprocal_rank_at_5"),
        "required_term_coverage": mean(answerable, "required_term_coverage"),
        "abstention_accuracy": mean(unanswerable, "abstention_detected"),
        "citation_id_validity": mean(rows, "citation_id_validity"),
        "automated_e2e_success_rate": mean(rows, "automated_e2e_success"),
        "online_valid_case_count": len(valid_online) if online else 0,
        "online_execution_success_rate": round(len(valid_online) / len(rows), 4) if online and rows else 0.0,
        "planner_llm_case_count": sum(row["planning_mode"] == "llm" for row in rows),
        "llm_verifier_enabled": bool(online and llm_verifier),
        "verifier_llm_case_count": sum(
            sum(item["mode"] == "llm" for item in row["verifier_history"])
            for row in rows
        ),
        "verifier_retry_case_count": sum(row["verifier_retries"] > 0 for row in rows),
        "verifier_final_insufficient_count": sum(
            row["verifier_decision"] == "insufficient" for row in rows
        ),
        "retrieval_tool_calls": sum(row["retrieval_tool_calls"] for row in rows),
        "retrieval_tool_attempts": sum(row["retrieval_tool_attempts"] for row in rows),
        "mean_retrieval_tool_calls": mean(rows, "retrieval_tool_calls"),
        "mean_retrieval_duration_ms": mean(rows, "retrieval_duration_ms"),
        "retrieval_stop_reasons": dict(
            sorted(Counter(row["retrieval_stop_reason"] for row in rows).items())
        ),
        "writer_llm_case_count": sum(row["writer_mode"] == "llm" for row in rows),
        "deterministic_no_evidence_abstention_count": sum(
            not row["writer_expected"] and row["structural_abstention"] for row in rows
        ),
        "fallback_or_error_case_ids": [row["id"] for row in fallback_rows],
        "mean_latency_ms": mean(rows, "latency_ms"),
        "model_calls": sum(row["model_calls"] for row in rows),
        "models": sorted({model for row in rows for model in row["models"]}),
        "planner_temperature": 0.0 if online else None,
        "writer_temperature": 0.2 if online else None,
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "completion_tokens": sum(row["completion_tokens"] for row in rows),
        "total_tokens": sum(row["total_tokens"] for row in rows),
        "estimated_cost_usd": round(sum(row["estimated_cost_usd"] for row in rows), 8),
        "cost_configured": cost_configured,
        "semantic_answer_quality_evaluated": False,
        "limitations": [
            "The frozen set uses the same three source PDFs as the retrieval development set.",
            "Required-term coverage is lexical and does not detect equivalent paraphrases.",
            "Citation ID validity checks identifier provenance, not semantic support for a claim.",
            "Abstention detection combines structural state with a phrase-based heuristic.",
            "Automated end-to-end success is not a human-rated answer-accuracy metric.",
        ],
    }
    return rows, summary


def write_results(
    rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "heldout_e2e_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "heldout_e2e_details.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not rows:
        return

    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
        )
    with (output_dir / "heldout_e2e_details.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    review_fields = [
        "id",
        "category",
        "question",
        "required_terms",
        "top5_ids",
        "answer",
        "answer_correctness_0_to_2",
        "citation_support_0_to_2",
        "abstention_correct",
        "review_notes",
    ]
    with (output_dir / "human_review_template.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=review_fields)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "category": row["category"],
                    "question": row["question"],
                    "required_terms": row["required_terms"],
                    "top5_ids": row["top5_ids"],
                    "answer": row["answer"],
                    "answer_correctness_0_to_2": "",
                    "citation_support_0_to_2": "",
                    "abstention_correct": "",
                    "review_notes": "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/heldout_final_set.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, choices=range(1, 21))
    parser.add_argument(
        "--online",
        action="store_true",
        help="Use the configured external model for planning and evidence-based writing.",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Return success even if an online case falls back or fails. Results still record it.",
    )
    parser.add_argument(
        "--disable-llm-verifier",
        action="store_true",
        help="Run the historical non-empty-evidence verifier for an online baseline.",
    )
    args = parser.parse_args()

    if args.online:
        if not StructuredPlanner.from_env().is_configured or not LLMWriter.from_env().is_configured:
            parser.error("--online requires an API key, base URL, and model in the environment")
        if not args.disable_llm_verifier and not StructuredEvidenceVerifier.from_env().is_configured:
            parser.error("the online LLM verifier requires an API key, base URL, and model")

    output_dir = args.output_dir or Path(
        "data/eval_runs/heldout_online" if args.online else "data/eval_runs/heldout_offline"
    )
    rows, summary = evaluate(
        args.dataset,
        online=args.online,
        llm_verifier=not args.disable_llm_verifier,
        case_limit=args.limit,
        verbose=True,
    )
    write_results(rows, summary, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.online and summary["fallback_or_error_case_ids"] and not args.allow_fallback:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
