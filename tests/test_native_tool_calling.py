from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agentflow.agents.context import ContextManager, ContextPolicy
from agentflow.agents.evidence_verifier import (
    EvidenceAssessment, StructuredEvidenceVerifier, VerificationOutcome,
)
from agentflow.agents.langgraph_workflow import LangGraphResearchWorkflow
from agentflow.agents.planner import StructuredPlanner
from agentflow.agents.state import AgentState
from agentflow.agents.tool_caller import BoundedToolCaller
from agentflow.generation.llm_writer import LLMWriter
from agentflow.schemas import Evidence
from agentflow.storage.traces import InMemoryTraceStore
from agentflow.tools.document_context import DocumentContextTool
from agentflow.tools.document_search import DocumentSearchTool
from agentflow.tools.registry import ToolRegistry


DOCUMENTS = [
    Evidence(id="manual__0", source="manual.md", text="The first section introduces a method."),
    Evidence(id="manual__1", source="manual.md", text="The next section gives a measured result."),
    Evidence(id="other__0", source="other.md", text="An unrelated document."),
]


class SearchStub:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        self.queries.append(query)
        if "result" in query:
            return [DOCUMENTS[1]][:top_k]
        return [DOCUMENTS[0]][:top_k]


class ScriptedCaller(BoundedToolCaller):
    def __init__(self, replies: list[dict]) -> None:
        super().__init__("test-key", "https://example.com", "mock-model")
        self.replies = list(replies)
        self.sent: list[list[dict]] = []

    def _request(self, messages: list[dict], tools: list[dict]) -> dict:
        self.sent.append([dict(item) for item in messages])
        assert {item["function"]["name"] for item in tools} == {
            "document_search", "document_context"
        }
        return self.replies.pop(0)


def _reply(*calls: tuple[str, str, dict]) -> dict:
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None if calls else "Enough evidence.",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                    for call_id, name, arguments in calls
                ],
            }
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }


def _registry(retriever: SearchStub) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(DocumentSearchTool(retriever))
    registry.register(DocumentContextTool(DOCUMENTS))
    return registry


def test_document_context_reads_only_nearby_chunks_in_document_order() -> None:
    tool = DocumentContextTool([DOCUMENTS[1], DOCUMENTS[2], DOCUMENTS[0]])
    result = tool.run(evidence_id="manual__1", window=1)
    assert [item.id for item in result.data] == ["manual__0", "manual__1"]
    assert not tool.run(evidence_id="missing", window=1).ok


def test_native_tool_call_returns_observation_to_model() -> None:
    caller = ScriptedCaller([
        _reply(("call-1", "document_context", {"evidence_id": "manual__0", "window": 1})),
        _reply(),
    ])
    retriever = SearchStub()
    outcome = caller.run(
        question="What is the result?",
        evidence=[DOCUMENTS[0]],
        registry=_registry(retriever),
        run_id="native-1",
        top_k=2,
        seen_queries={"What is the result?"},
        remaining_searches=2,
        remaining_duration_ms=5000,
    )
    assert outcome.stop_reason == "model_finished"
    assert outcome.calls[0]["executed"]
    assert outcome.calls[0]["tool"] == "document_context"
    assert [item.id for item in outcome.context_evidence_lists[0][1]] == [
        "manual__0", "manual__1"
    ]
    assert outcome.evidence_lists == []
    assert len(outcome.usage) == 2
    observation = caller.sent[1][-1]
    assert observation["role"] == "tool"
    assert observation["tool_call_id"] == "call-1"
    assert "manual__1" in observation["content"]


def test_native_tool_call_rejects_unseen_id_duplicate_query_and_excess_actions() -> None:
    caller = ScriptedCaller([
        _reply(
            ("call-1", "document_context", {"evidence_id": "other__0"}),
            ("call-2", "document_search", {"query": "Original query"}),
            ("call-3", "document_search", {"query": "result evidence"}),
            ("call-4", "document_search", {"query": "another result"}),
        ),
        _reply(),
    ])
    retriever = SearchStub()
    outcome = caller.run(
        question="Original query",
        evidence=[DOCUMENTS[0]],
        registry=_registry(retriever),
        run_id="native-2",
        top_k=2,
        seen_queries={"Original query"},
        remaining_searches=1,
        remaining_duration_ms=5000,
    )
    assert [item["executed"] for item in outcome.calls] == [False, False, True, False]
    assert retriever.queries == ["result evidence"]
    assert outcome.calls[0]["error"].startswith("evidence_id")
    assert outcome.calls[1]["error"] == "duplicate search query"
    assert outcome.calls[3]["error"] == "search query budget reached"
    assert len([item for item in caller.sent[1] if item["role"] == "tool"]) == 4


def test_native_tool_call_caps_executed_actions_at_two() -> None:
    caller = ScriptedCaller([
        _reply(
            ("call-1", "document_context", {"evidence_id": "manual__0"}),
            ("call-2", "document_search", {"query": "result evidence"}),
            ("call-3", "document_search", {"query": "another result"}),
        ),
    ])
    outcome = caller.run(
        question="Explain the method",
        evidence=[DOCUMENTS[0]],
        registry=_registry(SearchStub()),
        run_id="native-cap",
        top_k=2,
        seen_queries={"Explain the method"},
        remaining_searches=2,
        remaining_duration_ms=5000,
    )
    assert [item["executed"] for item in outcome.calls] == [True, True, False]
    assert outcome.calls[2]["error"] == "tool budget reached"


def test_native_request_uses_provider_tool_call_schema() -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return json.dumps(_reply()).encode("utf-8")

    captured = []

    def fake_urlopen(request, timeout):
        captured.append((request, timeout))
        return FakeResponse()

    caller = BoundedToolCaller("test-key", "https://example.com", "mock-model")
    with patch("agentflow.agents.tool_caller.urllib.request.urlopen", fake_urlopen):
        outcome = caller.run(
            question="Explain the method",
            evidence=[DOCUMENTS[0]],
            registry=_registry(SearchStub()),
            run_id="native-http",
            top_k=2,
            seen_queries={"Explain the method"},
            remaining_searches=2,
            remaining_duration_ms=5000,
        )
    payload = json.loads(captured[0][0].data.decode("utf-8"))
    assert outcome.stop_reason == "model_finished"
    assert payload["tool_choice"] == "auto"
    assert {item["function"]["name"] for item in payload["tools"]} == {
        "document_search", "document_context"
    }
    assert payload["messages"][0]["role"] == "system"


def test_native_tool_calling_integrates_with_graph_without_skipping_verifier() -> None:
    caller = ScriptedCaller([
        _reply(("call-1", "document_context", {"evidence_id": "manual__0"})),
        _reply(),
    ])
    retriever = SearchStub()
    traces = InMemoryTraceStore()
    with TemporaryDirectory() as directory:
        workflow = LangGraphResearchWorkflow(
            _registry(retriever),
            traces,
            planner=StructuredPlanner("", "", ""),
            evidence_verifier=StructuredEvidenceVerifier("", "", ""),
            llm_writer=LLMWriter("", "", ""),
            context_manager=ContextManager(Path(directory)),
            tool_caller=caller,
            enable_tool_calling=True,
        )
        report = workflow.run(AgentState(run_id="native-graph", question="Explain the method"))
    assert {item.id for item in report.evidence} == {"manual__0", "manual__1"}
    assert report.tool_call_history[0]["tool"] == "document_context"
    assert report.tool_call_history[0]["retained_evidence_ids"] == [
        "manual__0", "manual__1"
    ]
    assert report.retrieval_tool_calls == 1
    assert report.tool_decision_stop_reason == "model_finished"
    nodes = [event["node"] for event in traces.read(report.run_id)]
    assert "tool.document_context" in nodes
    assert "model.tool_router" in nodes
    assert "verifier" in nodes and "writer" in nodes


def test_context_neighbor_survives_full_search_top_k() -> None:
    documents = [
        Evidence(id="guide__0", source="guide.md", text="The method begins here."),
        Evidence(id="guide__1", source="guide.md", text="The missing condition is here."),
        *[
            Evidence(id=f"other-{index}__0", source=f"other-{index}.md", text=f"Less useful {index}.")
            for index in range(4)
        ],
    ]

    class CrowdedSearch:
        def search(self, query: str, top_k: int = 5) -> list[Evidence]:
            return [documents[0], *documents[2:]][:top_k]

    registry = ToolRegistry()
    registry.register(DocumentSearchTool(CrowdedSearch()))
    registry.register(DocumentContextTool(documents))
    caller = ScriptedCaller([
        _reply(("call-1", "document_context", {"evidence_id": "guide__0"})),
        _reply(),
    ])
    with TemporaryDirectory() as directory:
        workflow = LangGraphResearchWorkflow(
            registry, InMemoryTraceStore(),
            planner=StructuredPlanner("", "", ""),
            evidence_verifier=StructuredEvidenceVerifier("", "", ""),
            llm_writer=LLMWriter("", "", ""),
            context_manager=ContextManager(Path(directory)),
            tool_caller=caller, enable_tool_calling=True,
        )
        report = workflow.run(AgentState(run_id="crowded", question="Explain the method", top_k=5))
    assert len(report.evidence) == 6
    assert "guide__1" in {item.id for item in report.evidence}
    assert "guide__1" in report.tool_call_history[0]["retained_evidence_ids"]


def test_context_neighbor_survives_verifier_follow_up() -> None:
    class FollowUpVerifier:
        is_configured = False
        model = ""

        def __init__(self) -> None:
            self.calls = 0

        def verify(self, question, plan, evidence):
            self.calls += 1
            if self.calls == 1:
                return VerificationOutcome(
                    assessment=EvidenceAssessment(
                        decision="insufficient", missing_aspects=["one more fact"],
                        follow_up_queries=["another fact"], rationale="A fact is missing.",
                    ), mode="stub",
                )
            return VerificationOutcome(
                assessment=EvidenceAssessment(
                    decision="sufficient", rationale="Enough evidence is available."
                ), mode="stub",
            )

    documents = [
        Evidence(id="guide__0", source="guide.md", text="Initial fact."),
        Evidence(id="guide__1", source="guide.md", text="Context fact."),
        Evidence(id="other__0", source="other.md", text="Additional fact."),
    ]

    class FollowUpSearch:
        def search(self, query: str, top_k: int = 5) -> list[Evidence]:
            return [documents[2]] if query == "another fact" else [documents[0]]

    registry = ToolRegistry()
    registry.register(DocumentSearchTool(FollowUpSearch()))
    registry.register(DocumentContextTool(documents))
    with TemporaryDirectory() as directory:
        workflow = LangGraphResearchWorkflow(
            registry, InMemoryTraceStore(),
            planner=StructuredPlanner("", "", ""),
            evidence_verifier=FollowUpVerifier(), llm_writer=LLMWriter("", "", ""),
            context_manager=ContextManager(Path(directory)),
            tool_caller=ScriptedCaller([
                _reply(("call-1", "document_context", {"evidence_id": "guide__0"})),
                _reply(),
            ]), enable_tool_calling=True,
        )
        report = workflow.run(AgentState(run_id="context-follow-up", question="Compare facts", top_k=2))
    assert len(report.retrieval_rounds) == 2
    assert {item.id for item in report.evidence} == {"guide__0", "guide__1", "other__0"}
    assert "guide__1" in report.tool_call_history[0]["retained_evidence_ids"]


def test_context_history_distinguishes_returned_from_budget_retained() -> None:
    documents = [
        Evidence(id="guide__0", source="guide.md", text="A" * 800),
        Evidence(id="guide__1", source="guide.md", text="B" * 800),
        Evidence(id="other__0", source="other.md", text="C" * 800),
    ]

    class SearchTwo:
        def search(self, query: str, top_k: int = 5) -> list[Evidence]:
            return [documents[0], documents[2]][:top_k]

    registry = ToolRegistry()
    registry.register(DocumentSearchTool(SearchTwo()))
    registry.register(DocumentContextTool(documents))
    with TemporaryDirectory() as directory:
        workflow = LangGraphResearchWorkflow(
            registry, InMemoryTraceStore(),
            planner=StructuredPlanner("", "", ""),
            evidence_verifier=StructuredEvidenceVerifier("", "", ""),
            llm_writer=LLMWriter("", "", ""),
            context_manager=ContextManager(
                Path(directory), ContextPolicy(
                    max_context_tokens=512, reserved_output_tokens=128,
                    prompt_overhead_tokens=0, max_tokens_per_evidence=180,
                ),
            ),
            tool_caller=ScriptedCaller([
                _reply(("call-1", "document_context", {"evidence_id": "guide__0"})),
                _reply(),
            ]), enable_tool_calling=True,
        )
        report = workflow.run(AgentState(run_id="context-budget", question="Explain the method", top_k=2))
    record = report.tool_call_history[0]
    assert record["returned_evidence_ids"] == ["guide__0", "guide__1"]
    assert "guide__1" not in {item.id for item in report.evidence}
    assert record["retained_evidence_ids"] == ["guide__0"]
    assert report.context_stats.dropped_count >= 1


def test_native_tool_model_failure_keeps_original_evidence() -> None:
    caller = ScriptedCaller([])
    retriever = SearchStub()
    traces = InMemoryTraceStore()
    with TemporaryDirectory() as directory:
        workflow = LangGraphResearchWorkflow(
            _registry(retriever), traces,
            planner=StructuredPlanner("", "", ""),
            evidence_verifier=StructuredEvidenceVerifier("", "", ""),
            llm_writer=LLMWriter("", "", ""),
            context_manager=ContextManager(Path(directory)),
            tool_caller=caller, enable_tool_calling=True,
        )
        report = workflow.run(AgentState(run_id="native-error", question="Explain the method"))
    assert report.tool_decision_stop_reason == "model_error"
    assert [item.id for item in report.evidence] == ["manual__0"]
    assert report.answer
