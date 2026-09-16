from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agentflow.schemas import NodeStatus, TraceEvent, TraceSummary


class SpanTrackingStore:
    """Pair started/completed events and add stable span metadata."""

    def __init__(self) -> None:
        self._active: dict[tuple[str, str], list[tuple[str, str, float]]] = {}
        self._span_lock = threading.Lock()

    def _normalize(self, event: TraceEvent) -> TraceEvent:
        now = datetime.now(timezone.utc)
        trace_id = event.trace_id or event.run_id
        component_type = self._component_type(event.node)
        key = (event.run_id, event.node)

        with self._span_lock:
            if event.status == NodeStatus.STARTED:
                span_id = event.span_id or str(uuid.uuid4())
                parent_span_id = event.parent_span_id or self._find_parent_span(event)
                self._active.setdefault(key, []).append((span_id, parent_span_id, time.perf_counter()))
                return event.model_copy(
                    update={
                        "trace_id": trace_id,
                        "span_id": span_id,
                        "parent_span_id": parent_span_id,
                        "component_type": component_type,
                        "timestamp": now,
                    }
                )

            active = self._active.get(key, [])
            if active:
                span_id, parent_span_id, started = active.pop()
                if not active:
                    self._active.pop(key, None)
                measured_duration = (time.perf_counter() - started) * 1000.0
            else:
                span_id = event.span_id or str(uuid.uuid4())
                parent_span_id = event.parent_span_id or self._find_parent_span(event)
                measured_duration = event.duration_ms

            return event.model_copy(
                update={
                    "trace_id": trace_id,
                    "span_id": span_id,
                    "parent_span_id": parent_span_id,
                    "component_type": component_type,
                    "timestamp": now,
                    "duration_ms": event.duration_ms or measured_duration,
                    "error_type": event.error_type or str(event.payload.get("error_type") or ""),
                }
            )

    def _find_parent_span(self, event: TraceEvent) -> str:
        parent_node = event.payload.get("parent_node")
        if not parent_node:
            return ""
        active = self._active.get((event.run_id, str(parent_node)), [])
        return active[-1][0] if active else ""

    def _component_type(self, node: str) -> str:
        if node.startswith("tool."):
            return "tool"
        if node.startswith("model."):
            return "model"
        return "node"


class JsonlTraceStore(SpanTrackingStore):
    def __init__(self, trace_dir: Path) -> None:
        super().__init__()
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def append(self, event: TraceEvent) -> None:
        normalized = self._normalize(event)
        path = self.trace_dir / f"{normalized.run_id}.jsonl"
        with path.open("a", encoding="utf-8") as file:
            file.write(normalized.model_dump_json(ensure_ascii=False) + "\n")

    def read(self, run_id: str) -> list[dict]:
        path = self.trace_dir / f"{run_id}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def summarize(self, run_id: str) -> TraceSummary:
        return summarize_events(run_id, self.read(run_id))


class SQLiteTraceStore(SpanTrackingStore):
    def __init__(self, database_path: Path, mirror_jsonl_dir: Path | None = None) -> None:
        super().__init__()
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(database_path), check_same_thread=False)
        self.database_lock = threading.Lock()
        self._closed = False
        self.mirror_jsonl_dir = mirror_jsonl_dir
        if mirror_jsonl_dir is not None:
            mirror_jsonl_dir.mkdir(parents=True, exist_ok=True)
        self._setup()

    def _setup(self) -> None:
        with self.database_lock:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agentflow_trace_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    parent_span_id TEXT NOT NULL,
                    node TEXT NOT NULL,
                    component_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    error_type TEXT NOT NULL,
                    event_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_agentflow_trace_run
                    ON agentflow_trace_events(run_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_agentflow_trace_span
                    ON agentflow_trace_events(span_id);
                """
            )
            self.connection.commit()

    def append(self, event: TraceEvent) -> None:
        normalized = self._normalize(event)
        event_json = normalized.model_dump_json(ensure_ascii=False)
        with self.database_lock:
            self.connection.execute(
                """
                INSERT INTO agentflow_trace_events (
                    event_id, run_id, trace_id, span_id, parent_span_id, node,
                    component_type, status, timestamp, duration_ms, error_type, event_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized.event_id,
                    normalized.run_id,
                    normalized.trace_id,
                    normalized.span_id,
                    normalized.parent_span_id,
                    normalized.node,
                    normalized.component_type,
                    normalized.status.value,
                    normalized.timestamp.isoformat(),
                    normalized.duration_ms,
                    normalized.error_type,
                    event_json,
                ),
            )
            self.connection.commit()
        self._mirror_jsonl(normalized, event_json)

    def read(self, run_id: str) -> list[dict]:
        with self.database_lock:
            rows = self.connection.execute(
                "SELECT event_json FROM agentflow_trace_events WHERE run_id = ? ORDER BY timestamp, rowid",
                (run_id,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def summarize(self, run_id: str) -> TraceSummary:
        return summarize_events(run_id, self.read(run_id))

    def close(self) -> None:
        with self.database_lock:
            if self._closed:
                return
            self.connection.close()
            self._closed = True

    def _mirror_jsonl(self, event: TraceEvent, event_json: str) -> None:
        if self.mirror_jsonl_dir is None:
            return
        try:
            path = self.mirror_jsonl_dir / f"{event.run_id}.jsonl"
            with path.open("a", encoding="utf-8") as file:
                file.write(event_json + "\n")
        except OSError:
            pass


class InMemoryTraceStore(SpanTrackingStore):
    """Trace store for local smoke tests when file writes are unavailable."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[TraceEvent] = []

    def append(self, event: TraceEvent) -> None:
        self.events.append(self._normalize(event))

    def read(self, run_id: str) -> list[dict]:
        return [event.model_dump(mode="json") for event in self.events if event.run_id == run_id]

    def summarize(self, run_id: str) -> TraceSummary:
        return summarize_events(run_id, self.read(run_id))


def summarize_events(run_id: str, events: list[dict]) -> TraceSummary:
    if not events:
        return TraceSummary(run_id=run_id)

    span_ids = {event.get("span_id", "") for event in events if event.get("span_id")}
    terminal_events = [event for event in events if event.get("status") != NodeStatus.STARTED.value]
    completed = [event for event in terminal_events if event.get("status") == NodeStatus.COMPLETED.value]
    failed = [event for event in terminal_events if event.get("status") == NodeStatus.FAILED.value]
    component_duration_ms: dict[str, float] = {}
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    estimated_cost_usd = 0.0
    for event in terminal_events:
        component = str(event.get("component_type") or "node")
        component_duration_ms[component] = component_duration_ms.get(component, 0.0) + float(
            event.get("duration_ms") or 0.0
        )
        if component == "model":
            payload = event.get("payload") or {}
            prompt_tokens += int(payload.get("prompt_tokens") or 0)
            completion_tokens += int(payload.get("completion_tokens") or 0)
            total_tokens += int(payload.get("total_tokens") or 0)
            estimated_cost_usd += float(payload.get("estimated_cost_usd") or 0.0)

    timestamps = [datetime.fromisoformat(str(event["timestamp"])) for event in events if event.get("timestamp")]
    wall_clock_ms = 0.0
    if len(timestamps) >= 2:
        wall_clock_ms = (max(timestamps) - min(timestamps)).total_seconds() * 1000.0

    return TraceSummary(
        run_id=run_id,
        event_count=len(events),
        span_count=len(span_ids),
        completed_span_count=len(completed),
        failed_span_count=len(failed),
        tool_calls=len({event.get("span_id") for event in events if event.get("component_type") == "tool"}),
        model_calls=len({event.get("span_id") for event in events if event.get("component_type") == "model"}),
        wall_clock_ms=wall_clock_ms,
        component_duration_ms={key: round(value, 3) for key, value in component_duration_ms.items()},
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=round(estimated_cost_usd, 8),
    )
