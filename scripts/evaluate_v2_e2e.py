from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentflow.agents.context import ContextManager
from agentflow.agents.langgraph_workflow import LangGraphResearchWorkflow
from agentflow.agents.state import AgentState
from agentflow.evals.v2_eval import bootstrap_interval, load_v2_dataset, map_gold_spans, validate_all_gold_mapped
from agentflow.retrieval.store import load_documents
from agentflow.retrieval.v2_retriever import build_v2_retriever
from agentflow.storage.traces import InMemoryTraceStore
from agentflow.tools.document_search import DocumentSearchTool
from agentflow.tools.document_context import DocumentContextTool
from agentflow.tools.registry import ToolRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 60-case AgentFlow V2 online end-to-end evaluation.")
    parser.add_argument("--dataset", type=Path, default=Path("data/v2_eval_frozen.json"))
    parser.add_argument("--strategy")
    parser.add_argument("--model")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--method", default="cross_encoder")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-ids", nargs="+", help="Run only these case IDs, including development cases.")
    parser.add_argument("--tool-calling", action="store_true", help="Enable bounded native document tool calls.")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--human-scores", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/v2/e2e"))
    return parser.parse_args()


def selected_config(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.strategy and args.model:
        return args.strategy, args.model, args.method
    path = PROJECT_ROOT / "reports/v2/final_config.json"
    if not path.exists():
        raise SystemExit("Missing final config. Supply --strategy and --model or run both selection steps first.")
    selected = json.loads(path.read_text(encoding="utf-8"))
    return selected["strategy"], selected["model"], selected["method"]


def claim_map(case_id: str, mappings: list) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for mapping in mappings:
        if mapping.case_id == case_id:
            result[mapping.claim_id].update(mapping.chunk_ids)
    return dict(result)


def detect_abstention(answer: str, structural: bool) -> bool:
    if structural:
        return True
    return bool(re.search(
        r"证据不足|无法回答|不能回答|insufficient evidence|cannot answer|unable to answer",
        answer,
        re.IGNORECASE,
    ))


def citation_metrics(answer: str, evidence_ids: list[str], claims: dict[str, set[str]]) -> dict:
    cited_ids = set(re.findall(
        r"\[([a-z0-9][a-z0-9-]*__(?:fixed_char|recursive_token|section_aware)__\d+)\]",
        answer,
        re.IGNORECASE,
    ))
    valid_gold_ids = set().union(*claims.values()) if claims else set()
    return {
        "citation_id_validity": round(len(cited_ids & set(evidence_ids)) / len(cited_ids), 4)
        if cited_ids else None,
        "gold_citation_precision": round(len(cited_ids & valid_gold_ids) / len(cited_ids), 4)
        if cited_ids and claims else None,
        "gold_claim_citation_coverage": round(
            sum(bool(ids & cited_ids) for ids in claims.values()) / len(claims), 4
        ) if claims else None,
    }


def load_human_scores(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        return {row["id"]: row for row in csv.DictReader(file)}


def save_progress(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def mean_present(rows: list[dict], field: str) -> float | None:
    values = [row[field] for row in rows if row[field] is not None]
    return mean(values) if values else None


def main() -> None:
    args = parse_args()
    dataset = load_v2_dataset(PROJECT_ROOT / args.dataset)
    provisional = any(
        case.review_status not in {"source_verified", "human_verified"}
        for case in dataset.cases
    )
    if provisional and not args.allow_draft:
        raise SystemExit("Dataset has not passed source or human review. Use --allow-draft only for a provisional run.")
    strategy, model, method = selected_config(args)
    index_dir = PROJECT_ROOT / "data/index/v2" / strategy
    chunks = load_documents(index_dir / "documents.jsonl")
    mappings = map_gold_spans(dataset.cases, chunks)
    validate_all_gold_mapped(mappings)
    retriever = build_v2_retriever(
        PROJECT_ROOT / "data/index/v2",
        strategy=strategy,
        model_key=model,
        model_source=args.model_path or model,
        method=method,
        reranker_model=args.reranker_model,
    )
    if args.case_ids:
        if len(args.case_ids) != len(set(args.case_ids)):
            raise SystemExit("Duplicate case IDs are not allowed.")
        by_id = {case.id: case for case in dataset.cases}
        unknown = set(args.case_ids) - by_id.keys()
        if unknown:
            raise SystemExit(f"Unknown case IDs: {sorted(unknown)}")
        cases = [by_id[case_id] for case_id in args.case_ids]
    else:
        cases = [case for case in dataset.cases if case.split in {"frozen", "stress"}]
    if args.limit:
        cases = cases[: args.limit]
    language_counts = Counter(case.primary_language for case in cases)
    if args.limit is None and not args.case_ids and language_counts != Counter({"zh": 40, "en": 20}):
        raise RuntimeError(f"unexpected primary-language counts: {dict(language_counts)}")

    output_dir = PROJECT_ROOT / args.output_dir
    details_path = output_dir / "details.json"
    run_manifest_path = output_dir / "run_manifest.json"
    run_manifest = {
        "dataset_sha256": dataset.sha256,
        "configuration": {"strategy": strategy, "model": model, "method": method},
        "case_ids": [case.id for case in cases],
    }
    if args.tool_calling:
        run_manifest["tool_calling"] = True
    existing = []
    if details_path.exists() and args.retry_failures:
        if not run_manifest_path.exists():
            raise RuntimeError("Cannot resume without run_manifest.json.")
        previous = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if previous != run_manifest:
            raise RuntimeError("Existing run uses a different dataset or retrieval configuration.")
        existing = json.loads(details_path.read_text(encoding="utf-8"))
    elif details_path.exists():
        raise RuntimeError("Output already contains a run. Use --retry-failures or another output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    completed = {
        row["id"]: row for row in existing if not row.get("workflow_error")
    }
    rows = list(completed.values())
    score_rows = load_human_scores(PROJECT_ROOT / args.human_scores if args.human_scores else None)
    for row in rows:
        row.setdefault("all_gold_in_final_context", row["all_gold_retrieved_at_5"])
        row.setdefault("claim_recall_final_context", row["claim_recall_at_5"])
        row.setdefault("native_tool_calls", [])
        row.setdefault("tool_decision_stop_reason", "")
        row.setdefault("planning_mode", "unknown_legacy_run")
        row.setdefault("verifier_mode", "unknown_legacy_run")
        row.setdefault("writer_mode", "unknown_legacy_run")
        row.pop("citation_correctness", None)
        row.pop("citation_completeness", None)
        row.update(citation_metrics(row["answer"], row["evidence_ids"], claim_map(row["id"], mappings)))
        if row["id"] in score_rows:
            score = score_rows[row["id"]]
            value = score.get("score_0_1_2", "")
            row["human_score_0_1_2"] = int(value) if str(value).strip() else None
            row["human_reason"] = score.get("reason", "")
            row["human_citation_support"] = score.get("citation_support", "")

    traces = InMemoryTraceStore()
    registry = ToolRegistry()
    registry.register(DocumentSearchTool(retriever))
    if args.tool_calling:
        registry.register(DocumentContextTool(retriever.keyword_retriever.documents))
    with TemporaryDirectory() as temporary:
        workflow = LangGraphResearchWorkflow(
            registry,
            traces,
            context_manager=ContextManager(Path(temporary)),
            enable_tool_calling=args.tool_calling,
        )
        for position, case in enumerate(cases, 1):
            if case.id in completed:
                continue
            query = case.question_zh if case.primary_language == "zh" else case.question_en
            started = time.perf_counter()
            error_text = ""
            attempt_errors = []
            report = None
            attempts = 0
            for attempt in range(2):
                attempts += 1
                try:
                    report = workflow.run(
                        AgentState(run_id=f"v2-{case.id}-{uuid.uuid4()}", question=query, top_k=5)
                    )
                    break
                except Exception as error:
                    error_text = f"{type(error).__name__}: {error}"
                    attempt_errors.append(error_text)
                    if attempt == 0:
                        time.sleep(1)
            latency_ms = (time.perf_counter() - started) * 1000
            answer = report.answer if report else ""
            evidence_ids = [item.id for item in report.evidence] if report else []
            top_five_ids = evidence_ids[:5]
            claims = claim_map(case.id, mappings)
            claim_hits_at_five = {
                claim_id: bool(ids.intersection(top_five_ids)) for claim_id, ids in claims.items()
            }
            claim_hits_in_context = {
                claim_id: bool(ids.intersection(evidence_ids)) for claim_id, ids in claims.items()
            }
            abstention = detect_abstention(
                answer,
                bool(report and report.generation_quality.abstention),
            )
            conflict_recognized = bool(
                re.search(r"不能直接比较|不可直接比较|not directly comparable|different (?:metric|task|dataset)", answer, re.IGNORECASE)
            ) if case.category == "numerical_conflict" else None
            trace_events = traces.read(report.run_id) if report else []
            component_calls = Counter(
                event["node"].split(".", 1)[1]
                for event in trace_events
                if event.get("node", "").startswith("model.") and event.get("status") == "completed"
            )
            score = score_rows.get(case.id, {})
            human_score = score.get("score_0_1_2", "")
            row = {
                "id": case.id,
                "category": case.category,
                "primary_language": case.primary_language,
                "question": query,
                "answerable": case.answerable,
                "gold_claim_count": len(claims),
                "all_gold_retrieved_at_5": bool(claim_hits_at_five) and all(claim_hits_at_five.values()),
                "claim_recall_at_5": sum(claim_hits_at_five.values()) / len(claim_hits_at_five) if claim_hits_at_five else 0.0,
                "all_gold_in_final_context": bool(claim_hits_in_context) and all(claim_hits_in_context.values()),
                "claim_recall_final_context": sum(claim_hits_in_context.values()) / len(claim_hits_in_context) if claim_hits_in_context else 0.0,
                **citation_metrics(answer, evidence_ids, claims),
                "groundedness": report.generation_quality.groundedness if report else 0.0,
                "abstention_detected": abstention,
                "abstention_correct": abstention if not case.answerable else not abstention,
                "numerical_conflict_recognized": conflict_recognized,
                "human_score_0_1_2": int(human_score) if str(human_score).strip() else None,
                "human_reason": score.get("reason", ""),
                "human_citation_support": score.get("citation_support", ""),
                "planner_calls": component_calls["planner"],
                "verifier_calls": component_calls["verifier"],
                "writer_calls": component_calls["writer"],
                "planning_mode": report.planning_mode if report else "workflow_failed",
                "verifier_mode": report.verifier_mode if report else "workflow_failed",
                "verifier_error_class": report.verifier_error.split(":", 1)[0] if report and report.verifier_error else "",
                "verifier_round_modes": [
                    item["mode"] for item in report.verifier_history
                ] if report else [],
                "writer_mode": report.writer_mode if report else "workflow_failed",
                "model_calls": len(report.model_usage) if report else 0,
                "prompt_tokens": sum(item.prompt_tokens for item in report.model_usage) if report else 0,
                "completion_tokens": sum(item.completion_tokens for item in report.model_usage) if report else 0,
                "total_tokens": sum(item.total_tokens for item in report.model_usage) if report else 0,
                "logical_retrieval_calls": report.retrieval_tool_calls if report else 0,
                "native_tool_calls": report.tool_call_history if report else [],
                "tool_decision_stop_reason": report.tool_decision_stop_reason if report else "",
                "retrieval_attempts": report.retrieval_tool_attempts if report else 0,
                "stop_reason": report.retrieval_stop_reason if report else "workflow_failed",
                "fallback": bool(
                    report
                    and (
                        report.planning_mode != "llm"
                        or report.writer_mode != "llm"
                        or (report.evidence and report.verifier_mode != "llm")
                        or (args.tool_calling and report.tool_decision_stop_reason in {
                            "tool_calling_unavailable", "model_error", "invalid_tool_call"
                        })
                    )
                ),
                "attempts": attempts,
                "attempt_errors": attempt_errors,
                "latency_ms": round(latency_ms, 3),
                "workflow_error": error_text if report is None else "",
                "evidence_ids": evidence_ids,
                "answer": answer,
            }
            rows = [item for item in rows if item["id"] != case.id] + [row]
            save_progress(details_path, rows)
            print(f"[{position}/{len(cases)}] {case.id} error={bool(row['workflow_error'])} tokens={row['total_tokens']}", flush=True)

    ordered = sorted(rows, key=lambda item: item["id"])
    save_progress(details_path, ordered)
    scored = [row for row in ordered if row["human_score_0_1_2"] is not None]
    latencies = [row["latency_ms"] for row in ordered]
    p95_latency = sorted(latencies)[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else 0.0
    all_gold_values = [float(row["all_gold_retrieved_at_5"]) for row in ordered if row["answerable"]]
    final_context_values = [float(row["all_gold_in_final_context"]) for row in ordered if row["answerable"]]
    ci_low, ci_high = bootstrap_interval(all_gold_values)
    conflict_rows = [row for row in ordered if row["category"] == "numerical_conflict"]
    unanswerable_rows = [row for row in ordered if not row["answerable"]]
    summary = {
        "status": "provisional" if provisional else "frozen",
        "dataset_sha256": dataset.sha256,
        "configuration": {"strategy": strategy, "model": model, "method": method},
        "case_count": len(ordered),
        "primary_language_counts": dict(Counter(row["primary_language"] for row in ordered)),
        "human_scores": {
            "complete": sum(row["human_score_0_1_2"] == 2 for row in scored),
            "partial": sum(row["human_score_0_1_2"] == 1 for row in scored),
            "failed": sum(row["human_score_0_1_2"] == 0 for row in scored),
            "unscored": len(ordered) - len(scored),
        },
        "all_gold_recall_at_5": mean(all_gold_values) if all_gold_values else 0.0,
        "all_gold_in_final_context": mean(final_context_values) if final_context_values else 0.0,
        "all_gold_recall_at_5_ci95": [ci_low, ci_high],
        "citation_id_validity": mean_present(ordered, "citation_id_validity"),
        "gold_citation_precision": mean_present(ordered, "gold_citation_precision"),
        "gold_claim_citation_coverage": mean_present(ordered, "gold_claim_citation_coverage"),
        "groundedness_proxy": mean(row["groundedness"] for row in ordered) if ordered else 0.0,
        "unanswerable_abstention_keyword_rate": mean(row["abstention_detected"] for row in unanswerable_rows) if unanswerable_rows else 0.0,
        "numerical_conflict_phrase_rate": mean(bool(row["numerical_conflict_recognized"]) for row in conflict_rows) if conflict_rows else 0.0,
        "human_citation_support": dict(Counter(row.get("human_citation_support", "") for row in scored if row.get("human_citation_support"))),
        "average_latency_ms": mean(latencies) if latencies else 0.0,
        "p95_latency_ms": p95_latency,
        "planner_calls": sum(row["planner_calls"] for row in ordered),
        "verifier_calls": sum(row["verifier_calls"] for row in ordered),
        "writer_calls": sum(row["writer_calls"] for row in ordered),
        "total_tokens": sum(row["total_tokens"] for row in ordered),
        "logical_retrieval_calls": sum(row["logical_retrieval_calls"] for row in ordered),
        "native_tool_actions": sum(
            sum(bool(call.get("executed")) for call in row.get("native_tool_calls", []))
            for row in ordered
        ),
        "native_tool_action_types": dict(Counter(
            call.get("tool", "")
            for row in ordered for call in row.get("native_tool_calls", [])
            if call.get("executed")
        )),
        "tool_decision_stop_reasons": dict(Counter(
            row.get("tool_decision_stop_reason", "") for row in ordered
            if row.get("tool_decision_stop_reason")
        )),
        "stop_reasons": dict(Counter(row["stop_reason"] for row in ordered)),
        "fallback_case_ids": [row["id"] for row in ordered if row["fallback"]],
        "failed_case_ids": [row["id"] for row in ordered if row["workflow_error"]],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    template = output_dir / "human_scores_template.csv"
    with template.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "score_0_1_2", "citation_support", "reason"])
        writer.writeheader()
        for row in ordered:
            writer.writerow({"id": row["id"], "score_0_1_2": row["human_score_0_1_2"] if row["human_score_0_1_2"] is not None else "", "citation_support": row.get("human_citation_support", ""), "reason": row["human_reason"]})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
