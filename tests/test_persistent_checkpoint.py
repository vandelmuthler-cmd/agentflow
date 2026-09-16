from __future__ import annotations

from tempfile import TemporaryDirectory
from pathlib import Path

from agentflow.agents.context import ContextManager
from agentflow.agents.langgraph_workflow import LangGraphResearchWorkflow
from agentflow.agents.evidence_verifier import StructuredEvidenceVerifier
from agentflow.agents.planner import StructuredPlanner
from agentflow.agents.state import AgentState
from agentflow.generation.llm_writer import LLMWriter
from agentflow.schemas import Evidence
from agentflow.storage.checkpoints import SQLiteCheckpointBackend
from agentflow.storage.traces import InMemoryTraceStore
from agentflow.tools.document_search import DocumentSearchTool
from agentflow.tools.registry import ToolRegistry


class RecordingRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        self.queries.append(query)
        return [Evidence(id="doc__1", source="doc.pdf", text="持久化测试证据", rank=1)]


def _workflow(
    database_path: Path,
    retriever: RecordingRetriever,
    traces: InMemoryTraceStore,
    workspace_root: Path,
    *,
    interrupt_before: list[str] | None = None,
) -> LangGraphResearchWorkflow:
    registry = ToolRegistry()
    registry.register(DocumentSearchTool(retriever))
    return LangGraphResearchWorkflow(
        registry,
        traces,
        planner=StructuredPlanner(api_key="", base_url="https://api.example.com", model="model"),
        evidence_verifier=StructuredEvidenceVerifier("", "", ""),
        llm_writer=LLMWriter("", "", ""),
        context_manager=ContextManager(workspace_root),
        checkpoint_backend=SQLiteCheckpointBackend(database_path),
        interrupt_before=interrupt_before,
    )


def test_sqlite_checkpoint_survives_reopen_and_resume_is_idempotent() -> None:
    with TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        database_path = root / "checkpoints.sqlite3"
        retriever = RecordingRetriever()
        traces = InMemoryTraceStore()
        run_id = "persistent-run"

        first = _workflow(
            database_path,
            retriever,
            traces,
            root / "runs",
            interrupt_before=["writer"],
        )
        partial_report = first.run(
            AgentState(run_id=run_id, question="测试 SQLite 恢复", top_k=1)
        )
        assert partial_report.status == "running"
        assert first.get_checkpoint_state(run_id) is not None
        assert first.graph.get_state(first._graph_config(run_id)).next == ("writer",)
        first.close()

        second = _workflow(database_path, retriever, traces, root / "runs")
        restored = second.get_checkpoint_state(run_id)
        assert restored is not None
        assert restored.answer == ""

        completed = second.resume(run_id)
        assert completed is not None
        assert completed.status == "completed"
        assert completed.answer
        assert retriever.queries == ["测试 SQLite 恢复"]

        writer_events_before = len(
            [event for event in traces.read(run_id) if event["node"] == "writer" and event["status"] == "started"]
        )
        repeated = second.resume(run_id)
        writer_events_after = len(
            [event for event in traces.read(run_id) if event["node"] == "writer" and event["status"] == "started"]
        )
        assert repeated is not None
        assert repeated.answer == completed.answer
        assert writer_events_after == writer_events_before == 1
        second.close()


def test_resume_unknown_run_returns_none() -> None:
    with TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        workflow = _workflow(
            root / "checkpoints.sqlite3",
            RecordingRetriever(),
            InMemoryTraceStore(),
            root / "runs",
        )
        assert workflow.resume("missing-run") is None
        workflow.close()
