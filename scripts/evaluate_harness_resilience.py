from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_conditional_workflow import test_empty_evidence_triggers_bounded_query_rewrite
from tests.test_context_manager import (
    test_context_manager_deduplicates_and_stays_inside_budget,
    test_context_manager_keeps_small_context_without_artifact,
)
from tests.test_persistent_checkpoint import (
    test_resume_unknown_run_returns_none,
    test_sqlite_checkpoint_survives_reopen_and_resume_is_idempotent,
)
from tests.test_trace_spans import test_parent_child_spans_and_summary
from tests.test_tool_runtime import test_tool_validation_retry_timeout_and_empty_results


CHECKS = {
    "tool_retry_timeout_validation": test_tool_validation_retry_timeout_and_empty_results,
    "conditional_retry_branch": test_empty_evidence_triggers_bounded_query_rewrite,
    "context_budget_and_deduplication": test_context_manager_deduplicates_and_stays_inside_budget,
    "small_context_no_overflow": test_context_manager_keeps_small_context_without_artifact,
    "durable_checkpoint_and_idempotent_resume": test_sqlite_checkpoint_survives_reopen_and_resume_is_idempotent,
    "unknown_checkpoint_handling": test_resume_unknown_run_returns_none,
    "trace_parent_child_spans": test_parent_child_spans_and_summary,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval_runs/harness_resilience.json"),
    )
    args = parser.parse_args()
    details = []
    for name, check in CHECKS.items():
        started = time.perf_counter()
        try:
            check()
            passed, error = True, ""
        except Exception as exc:
            passed, error = False, f"{type(exc).__name__}: {exc}"
        details.append(
            {
                "check": name,
                "passed": passed,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "error": error,
            }
        )
    report = {
        "check_count": len(details),
        "passed_count": sum(item["passed"] for item in details),
        "pass_rate": round(sum(item["passed"] for item in details) / len(details), 4),
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["passed_count"] != report["check_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
