from __future__ import annotations

from agentflow.agents.evidence_verifier import (
    EvidenceAssessment,
    EvidenceVerifierError,
    StructuredEvidenceVerifier,
    VerificationOutcome,
)
from agentflow.agents.langgraph_workflow import LangGraphResearchWorkflow
from agentflow.agents.planner import StructuredPlanner
from agentflow.agents.state import AgentState
from agentflow.model_usage import ModelUsage
from agentflow.generation.llm_writer import LLMWriter
from agentflow.schemas import Evidence
from agentflow.storage.traces import InMemoryTraceStore
from agentflow.tools.base import BaseTool, ToolResult
from agentflow.tools.registry import ToolRegistry


def test_verifier_parses_evidence_aliases_into_chunk_ids() -> None:
    verifier = StructuredEvidenceVerifier("key", "https://api.example.com", "model")
    evidence = [
        Evidence(id="doc__1", source="doc.pdf", text="Fact A"),
        Evidence(id="doc__2", source="doc.pdf", text="Fact B"),
    ]
    content = """
    {
      "decision": "sufficient",
      "required_aspects": ["A", "B"],
      "covered_aspects": ["A", "B"],
      "missing_aspects": [],
      "follow_up_queries": [],
      "evidence_mapping": {"A": ["E1"], "B": ["E2"]},
      "rationale": "Both facts are supported."
    }
    """

    assessment = verifier._parse_assessment(content, evidence)

    assert assessment.evidence_mapping == {"A": ["doc__1"], "B": ["doc__2"]}


def test_verifier_rejects_unknown_evidence_alias() -> None:
    verifier = StructuredEvidenceVerifier("key", "https://api.example.com", "model")
    evidence = [Evidence(id="doc__1", source="doc.pdf", text="Fact A")]
    content = """
    {
      "decision": "sufficient",
      "required_aspects": ["A"],
      "covered_aspects": ["A"],
      "missing_aspects": [],
      "follow_up_queries": [],
      "evidence_mapping": {"A": ["E9"]},
      "rationale": "The fact is supported."
    }
    """

    try:
        verifier._parse_assessment(content, evidence)
    except EvidenceVerifierError as error:
        assert "unknown evidence aliases" in str(error)
    else:
        raise AssertionError("unknown evidence aliases must be rejected")


def test_verifier_keeps_api_usage_when_output_validation_fails() -> None:
    verifier = StructuredEvidenceVerifier("key", "https://api.example.com", "model")
    usage = ModelUsage(model="model", prompt_tokens=20, completion_tokens=10, total_tokens=30, source="api")
    verifier._request_assessment = lambda *_args: ("not valid JSON", usage)
    outcome = verifier.verify("Question", ["Find evidence"], [
        Evidence(id="doc__1", source="doc.pdf", text="Evidence")
    ])
    assert outcome.mode == "deterministic_fallback"
    assert outcome.assessment.decision == "abstain"
    assert outcome.usage.total_tokens == 30
    assert "output validation failed" in outcome.error


def test_online_verifier_validation_error_fails_closed_before_writer() -> None:
    verifier = StructuredEvidenceVerifier("key", "https://api.example.com", "model")
    usage = ModelUsage(model="model", prompt_tokens=20, completion_tokens=10, total_tokens=30, source="api")
    verifier._request_assessment = lambda *_args: ("not valid JSON", usage)
    registry = ToolRegistry()
    registry.register(FollowUpSearchTool())
    workflow = LangGraphResearchWorkflow(
        registry, InMemoryTraceStore(),
        planner=StructuredPlanner("", "", ""),
        evidence_verifier=verifier,
        llm_writer=LLMWriter("", "", ""),
    )
    report = workflow.run(AgentState(run_id="fail-closed-verifier", question="What is fact A?"))
    assert report.status == "completed_with_insufficient_evidence"
    assert report.verifier_mode == "deterministic_fallback"
    assert report.writer_mode == "skipped_after_verifier_error"
    assert report.retrieval_stop_reason == "verifier_error"
    assert "无法可靠确认" in report.answer
    assert sum(item.total_tokens for item in report.model_usage) == 30


class FollowUpSearchTool(BaseTool):
    name = "document_search"
    description = "Return a different fact for the verifier follow-up query."

    def __init__(self) -> None:
        self.queries: list[str] = []

    def run(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        self.queries.append(query)
        if query == "fact B targeted query":
            return ToolResult(
                ok=True,
                data=[Evidence(id="doc__2", source="doc.pdf", text="Fact B", rank=1)],
            )
        return ToolResult(
            ok=True,
            data=[Evidence(id="doc__1", source="doc.pdf", text="Fact A", rank=1)],
        )


class SequencedVerifier:
    is_configured = True
    model = "stub-verifier"

    def __init__(self) -> None:
        self.calls = 0

    def verify(self, question, plan, evidence) -> VerificationOutcome:
        self.calls += 1
        if self.calls == 1:
            return VerificationOutcome(
                assessment=EvidenceAssessment(
                    decision="insufficient",
                    required_aspects=["A", "B"],
                    covered_aspects=["A"],
                    missing_aspects=["B"],
                    follow_up_queries=["fact B targeted query"],
                    evidence_mapping={"A": ["doc__1"]},
                    rationale="Fact B is missing.",
                ),
                mode="llm",
            )
        return VerificationOutcome(
            assessment=EvidenceAssessment(
                decision="sufficient",
                required_aspects=["A", "B"],
                covered_aspects=["A", "B"],
                evidence_mapping={"A": ["doc__1"], "B": ["doc__2"]},
                rationale="Both facts are now supported.",
            ),
            mode="llm",
        )


def test_llm_verifier_drives_targeted_retrieval_and_preserves_evidence() -> None:
    search_tool = FollowUpSearchTool()
    verifier = SequencedVerifier()
    registry = ToolRegistry()
    registry.register(search_tool)
    traces = InMemoryTraceStore()
    workflow = LangGraphResearchWorkflow(
        registry,
        traces,
        planner=StructuredPlanner("", "", ""),
        evidence_verifier=verifier,
        llm_writer=LLMWriter("", "", ""),
    )

    report = workflow.run(
        AgentState(run_id="verifier-loop-test", question="Compare fact A and fact B", top_k=3)
    )

    assert search_tool.queries == ["Compare fact A and fact B", "fact B targeted query"]
    assert {item.id for item in report.evidence} == {"doc__1", "doc__2"}
    assert report.verifier_decision == "sufficient"
    assert report.verifier_mode == "llm"
    assert len(report.verifier_history) == 2
    assert report.verifier_history[0]["missing_aspects"] == ["B"]
    assert report.verifier_history[1]["decision"] == "sufficient"
    assert report.status == "completed"
    assert verifier.calls == 2
    model_events = [
        event
        for event in traces.read(report.run_id)
        if event["node"] == "model.verifier" and event["status"] == "started"
    ]
    assert len(model_events) == 2
