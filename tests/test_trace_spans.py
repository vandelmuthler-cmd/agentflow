from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from agentflow.schemas import NodeStatus, TraceEvent
from agentflow.storage.traces import InMemoryTraceStore, SQLiteTraceStore


def _event(run_id: str, node: str, status: NodeStatus, parent_node: str = "") -> TraceEvent:
    payload = {"parent_node": parent_node} if parent_node else {}
    return TraceEvent(
        run_id=run_id,
        node=node,
        status=status,
        message=f"{node} {status.value}",
        payload=payload,
    )


def test_parent_child_spans_and_summary() -> None:
    store = InMemoryTraceStore()
    run_id = "span-run"
    store.append(_event(run_id, "retriever", NodeStatus.STARTED))
    store.append(_event(run_id, "tool.document_search", NodeStatus.STARTED, "retriever"))
    store.append(_event(run_id, "tool.document_search", NodeStatus.COMPLETED, "retriever"))
    store.append(_event(run_id, "retriever", NodeStatus.COMPLETED))

    events = store.read(run_id)
    retriever_start, tool_start, tool_end, retriever_end = events
    assert retriever_start["span_id"] == retriever_end["span_id"]
    assert tool_start["span_id"] == tool_end["span_id"]
    assert tool_start["parent_span_id"] == retriever_start["span_id"]
    assert tool_end["parent_span_id"] == retriever_start["span_id"]
    assert tool_end["duration_ms"] >= 0

    summary = store.summarize(run_id)
    assert summary.span_count == 2
    assert summary.completed_span_count == 2
    assert summary.tool_calls == 1
    assert summary.model_calls == 0


def test_sqlite_trace_survives_reopen() -> None:
    with TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        database_path = root / "trace.sqlite3"
        mirror_dir = root / "jsonl"
        run_id = "sqlite-trace"
        first = SQLiteTraceStore(database_path, mirror_jsonl_dir=mirror_dir)
        first.append(_event(run_id, "planner", NodeStatus.STARTED))
        first.append(_event(run_id, "planner", NodeStatus.COMPLETED))
        first.close()

        second = SQLiteTraceStore(database_path)
        events = second.read(run_id)
        summary = second.summarize(run_id)
        assert len(events) == 2
        assert events[0]["span_id"] == events[1]["span_id"]
        assert summary.span_count == 1
        assert summary.completed_span_count == 1
        assert (mirror_dir / f"{run_id}.jsonl").exists()
        second.close()
