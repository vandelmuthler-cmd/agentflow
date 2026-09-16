from __future__ import annotations

from agentflow.agents.langgraph_workflow import LangGraphResearchWorkflow
from agentflow.agents.evidence_verifier import StructuredEvidenceVerifier
from agentflow.agents.planner import StructuredPlanner
from agentflow.agents.state import AgentState
from agentflow.generation.llm_writer import LLMWriter
from agentflow.schemas import Evidence
from agentflow.storage.traces import InMemoryTraceStore
from agentflow.tools.base import BaseTool, ToolResult
from agentflow.tools.registry import ToolRegistry


class RetrySearchTool(BaseTool):
    name = "document_search"
    description = "Return evidence only after the query-rewrite branch runs."

    def __init__(self) -> None:
        self.queries: list[str] = []

    def run(self, **kwargs) -> ToolResult:
        self.queries.append(kwargs.get("query", ""))
        if len(self.queries) == 1:
            return ToolResult(ok=True, data=[])
        return ToolResult(
            ok=True,
            data=[
                Evidence(
                    id="mock_doc__0",
                    source="mock_doc.md",
                    text="The method does not require molecular dynamics simulation at prediction time.",
                    score=1.0,
                    rank=1,
                )
            ],
        )


def test_empty_evidence_triggers_bounded_query_rewrite() -> None:
    search_tool = RetrySearchTool()
    registry = ToolRegistry()
    registry.register(search_tool)
    traces = InMemoryTraceStore()
    workflow = LangGraphResearchWorkflow(
        registry,
        traces,
        planner=StructuredPlanner("", "", ""),
        evidence_verifier=StructuredEvidenceVerifier("", "", ""),
        llm_writer=LLMWriter("", "", ""),
    )
    state = AgentState(run_id="conditional-workflow-test", question="What is required at prediction time?")

    report = workflow.run(state)
    checkpoint = workflow.get_checkpoint_state(state.run_id)
    started_nodes = [
        event["node"] for event in traces.read(state.run_id) if event["status"] == "started"
    ]

    assert report.status == "completed"
    assert len(report.evidence) == 1
    assert checkpoint is not None and checkpoint.retries == 1
    assert checkpoint.verifier_decision == "sufficient"
    assert len(search_tool.queries) == 2
    assert started_nodes.count("retriever") == 2
    assert "query_rewriter" in started_nodes
