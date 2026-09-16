from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentflow.agents.langgraph_workflow import LangGraphResearchWorkflow
from agentflow.agents.evidence_verifier import StructuredEvidenceVerifier
from agentflow.agents.planner import StructuredPlanner
from agentflow.agents.state import AgentState
from agentflow.config import RAW_DIR
from agentflow.generation.llm_writer import LLMWriter
from agentflow.retrieval.index import build_retriever
from agentflow.storage.traces import InMemoryTraceStore
from agentflow.tools.document_search import DocumentSearchTool
from agentflow.tools.document_context import DocumentContextTool
from agentflow.tools.registry import ToolRegistry


DEFAULT_QUESTION = "DEFMap 在预测阶段不需要什么计算？"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run one complete AgentFlow research task.")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Question to research.")
    parser.add_argument("--top-k", type=int, default=5, choices=range(1, 21))
    parser.add_argument(
        "--online",
        action="store_true",
        help="Use the configured external language model. Without this flag the demo is offline.",
    )
    parser.add_argument(
        "--tool-calling",
        action="store_true",
        help="Enable bounded native model tool calls; requires --online.",
    )
    args = parser.parse_args()
    if args.tool_calling and not args.online:
        parser.error("--tool-calling requires --online")

    retriever = build_retriever(RAW_DIR)
    registry = ToolRegistry()
    registry.register(DocumentSearchTool(retriever))
    registry.register(DocumentContextTool(retriever.keyword_retriever.documents))
    traces = InMemoryTraceStore()
    workflow_options = {}
    if not args.online:
        workflow_options = {
            "planner": StructuredPlanner("", "", ""),
            "evidence_verifier": StructuredEvidenceVerifier("", "", ""),
            "llm_writer": LLMWriter("", "", ""),
        }

    report = LangGraphResearchWorkflow(
        registry, traces, enable_tool_calling=args.tool_calling, **workflow_options
    ).run(
        AgentState(run_id=str(uuid.uuid4()), question=args.question, top_k=args.top_k)
    )
    usage = report.model_usage
    print(f"run_id: {report.run_id}")
    print(f"status: {report.status}")
    print(f"planning_mode: {report.planning_mode}")
    print(f"verifier_mode: {report.verifier_mode}")
    print(f"verifier_decision: {report.verifier_decision}")
    print(f"writer_mode: {report.writer_mode}")
    print(f"evidence_count: {len(report.evidence)}")
    print(f"citation_correctness: {report.generation_quality.citation_correctness}")
    print(f"groundedness: {report.generation_quality.groundedness}")
    print(f"model_calls: {len(usage)}")
    print(f"native_tool_calls: {sum(item.get('executed', False) for item in report.tool_call_history)}")
    print(f"tool_decision_stop_reason: {report.tool_decision_stop_reason}")
    print(f"total_tokens: {sum(item.total_tokens for item in usage)}")
    print("nodes:", " -> ".join(event["node"] for event in traces.read(report.run_id) if event["status"] == "started"))
    print("\n--- answer ---\n")
    print(report.answer)

    if report.status == "failed":
        raise SystemExit("Research workflow failed; inspect the run trace for tool errors.")
    if args.tool_calling and report.tool_decision_stop_reason in {
        "tool_calling_unavailable", "model_error", "invalid_tool_call"
    }:
        raise SystemExit(
            "Native tool decision did not complete; inspect tool_decision_stop_reason and the run trace."
        )

    verifier_valid = report.verifier_mode == "llm" if report.evidence else True
    if args.online and (
        report.planning_mode != "llm"
        or not verifier_valid
        or (report.evidence and report.writer_mode != "llm")
    ):
        raise SystemExit("Online mode was requested, but at least one model call used fallback. Check .env, network, and proxy settings.")


if __name__ == "__main__":
    main()
