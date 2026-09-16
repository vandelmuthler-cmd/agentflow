from __future__ import annotations

from agentflow.agents.langgraph_workflow import LangGraphResearchWorkflow
from agentflow.agents.evidence_verifier import StructuredEvidenceVerifier
from agentflow.agents.planner import PlanningOutcome, StructuredPlan, StructuredPlanner
from agentflow.agents.state import AgentState
from agentflow.generation.llm_writer import LLMWriter
from agentflow.schemas import Evidence
from agentflow.storage.traces import InMemoryTraceStore
from agentflow.tools.document_search import DocumentSearchTool
from agentflow.tools.registry import ToolRegistry


TOOLS = [{"name": "document_search", "description": "Search local documents."}]


class StubPlanner:
    is_configured = True
    model = "stub-planner"

    def create_plan(self, question: str, tools: list[dict]) -> PlanningOutcome:
        return PlanningOutcome(
            StructuredPlan.model_validate(
                {
                    "task_type": "complex",
                    "rationale": "问题要求比较两个方面",
                    "steps": [
                        {
                            "id": "step_1",
                            "objective": "检索方法信息",
                            "tool": "document_search",
                            "query": "方法信息",
                            "success_criterion": "找到方法证据",
                        },
                        {
                            "id": "step_2",
                            "objective": "检索结果信息",
                            "tool": "document_search",
                            "query": "结果信息",
                            "success_criterion": "找到结果证据",
                        },
                    ],
                }
            ),
            mode="llm",
        )


class RecordingRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        self.queries.append(query)
        common = Evidence(id="shared", source="doc.pdf", text="共同证据")
        specific = Evidence(id=f"result_{len(self.queries)}", source="doc.pdf", text=query)
        return [specific, common][:top_k]


def test_unconfigured_planner_uses_fixed_fallback() -> None:
    planner = StructuredPlanner(api_key="", base_url="https://api.example.com", model="model")

    outcome = planner.create_plan("什么是检索增强生成？", TOOLS)

    assert outcome.mode == "fixed_fallback"
    assert outcome.plan.task_type == "simple"
    assert len(outcome.plan.steps) == 1
    assert outcome.plan.steps[0].query == "什么是检索增强生成？"


def test_planner_parses_fenced_structured_json() -> None:
    planner = StructuredPlanner(api_key="key", base_url="https://api.example.com", model="model")
    content = """```json
    {
      "task_type": "simple",
      "rationale": "单一事实问题",
      "steps": [{
        "id": "step_1",
        "objective": "检索定义",
        "tool": "document_search",
        "query": "检索增强生成 定义",
        "success_criterion": "找到定义证据"
      }]
    }
    ```"""

    plan = planner._parse_plan(content)

    assert plan.task_type == "simple"
    assert plan.steps[0].tool == "document_search"


def test_langgraph_executes_each_planned_query_and_deduplicates_evidence() -> None:
    retriever = RecordingRetriever()
    registry = ToolRegistry()
    registry.register(DocumentSearchTool(retriever))
    traces = InMemoryTraceStore()
    workflow = LangGraphResearchWorkflow(
        registry,
        traces,
        planner=StubPlanner(),
        evidence_verifier=StructuredEvidenceVerifier("", "", ""),
        llm_writer=LLMWriter("", "", ""),
    )
    state = AgentState(run_id="planner-test", question="比较方法和结果", top_k=3)

    report = workflow.run(state)

    assert retriever.queries == ["方法信息", "结果信息"]
    assert report.task_type == "complex"
    assert report.planning_mode == "llm"
    assert len(report.plan_steps) == 2
    assert {item.id for item in report.evidence} == {"result_1", "result_2", "shared"}
    events = traces.read(state.run_id)
    planner_span = next(
        event["span_id"]
        for event in events
        if event["node"] == "planner" and event["status"] == "started"
    )
    retriever_span = next(
        event["span_id"]
        for event in events
        if event["node"] == "retriever" and event["status"] == "started"
    )
    model_event = next(
        event for event in events if event["node"] == "model.planner" and event["status"] == "started"
    )
    tool_events = [
        event for event in events if event["node"] == "tool.document_search" and event["status"] == "started"
    ]
    context_event = next(
        event for event in events if event["node"] == "context_manager" and event["status"] == "started"
    )
    assert model_event["parent_span_id"] == planner_span
    assert all(event["parent_span_id"] == retriever_span for event in tool_events)
    assert context_event["parent_span_id"] == retriever_span
    summary = traces.summarize(state.run_id)
    assert summary.model_calls == 1
    assert summary.tool_calls == 2
