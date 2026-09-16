from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any

from pydantic import ValidationError

from agentflow.schemas import NodeStatus, TraceEvent
from agentflow.tools.base import (
    BaseTool,
    ToolErrorType,
    ToolExecutionContext,
    ToolExecutionPolicy,
    ToolOutcome,
    ToolResult,
)


class ToolRegistry:
    def __init__(self, trace_store=None, default_policy: ToolExecutionPolicy | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        self.trace_store = trace_store
        self.default_policy = default_policy or ToolExecutionPolicy()

    def set_trace_store(self, trace_store) -> None:
        self.trace_store = trace_store

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def run(
        self,
        name: str,
        *,
        context: ToolExecutionContext | None = None,
        policy: ToolExecutionPolicy | None = None,
        **kwargs,
    ) -> ToolResult:
        tool_call_id = str(uuid.uuid4())
        started = time.perf_counter()
        tool = self._tools.get(name)
        if tool is None:
            result = ToolResult(
                ok=False,
                outcome=ToolOutcome.FAILED,
                error=f"unknown tool: {name}",
                error_type=ToolErrorType.UNKNOWN_TOOL,
                tool_call_id=tool_call_id,
            )
            return self._finish(name, context, result, started)

        self._emit(
            context,
            name,
            NodeStatus.STARTED,
            "工具调用开始",
            {"tool_call_id": tool_call_id},
        )
        try:
            arguments = tool.validate_input(kwargs)
        except ValidationError as exc:
            result = ToolResult(
                ok=False,
                outcome=ToolOutcome.FAILED,
                error=str(exc),
                error_type=ToolErrorType.VALIDATION,
                tool_call_id=tool_call_id,
            )
            return self._finish(name, context, result, started)

        execution_policy = policy or self.default_policy
        last_result: ToolResult | None = None
        for attempt in range(1, execution_policy.max_attempts + 1):
            try:
                result = self._run_with_timeout(tool, arguments, execution_policy.timeout_seconds)
                if not isinstance(result, ToolResult):
                    raise TypeError(f"tool {name} must return ToolResult")
                if result.ok:
                    result.data = tool.validate_output(result.data)
                    result.outcome = ToolOutcome.EMPTY if self._is_empty(result.data) else ToolOutcome.SUCCESS
                    result.error_type = None
                    result.tool_call_id = tool_call_id
                    result.attempts = attempt
                    return self._finish(name, context, result, started)

                result.outcome = ToolOutcome.FAILED
                result.tool_call_id = tool_call_id
                result.attempts = attempt
                result.error_type = result.error_type or ToolErrorType.PERMANENT
                last_result = result
                if result.error_type != ToolErrorType.TRANSIENT:
                    return self._finish(name, context, result, started)
            except FutureTimeoutError:
                last_result = ToolResult(
                    ok=False,
                    outcome=ToolOutcome.FAILED,
                    error=f"tool timed out after {execution_policy.timeout_seconds:.3f}s",
                    error_type=ToolErrorType.TIMEOUT,
                    tool_call_id=tool_call_id,
                    attempts=attempt,
                )
            except (ConnectionError, OSError) as exc:
                last_result = ToolResult(
                    ok=False,
                    outcome=ToolOutcome.FAILED,
                    error=str(exc),
                    error_type=ToolErrorType.TRANSIENT,
                    tool_call_id=tool_call_id,
                    attempts=attempt,
                )
            except (ValidationError, TypeError, ValueError) as exc:
                last_result = ToolResult(
                    ok=False,
                    outcome=ToolOutcome.FAILED,
                    error=str(exc),
                    error_type=ToolErrorType.PERMANENT,
                    tool_call_id=tool_call_id,
                    attempts=attempt,
                )
                return self._finish(name, context, last_result, started)
            except Exception as exc:
                last_result = ToolResult(
                    ok=False,
                    outcome=ToolOutcome.FAILED,
                    error=str(exc),
                    error_type=ToolErrorType.PERMANENT,
                    tool_call_id=tool_call_id,
                    attempts=attempt,
                )
                return self._finish(name, context, last_result, started)

            if attempt < execution_policy.max_attempts and execution_policy.backoff_seconds:
                time.sleep(execution_policy.backoff_seconds * (2 ** (attempt - 1)))

        return self._finish(name, context, last_result or ToolResult(ok=False), started)

    def _run_with_timeout(
        self,
        tool: BaseTool,
        arguments: dict[str, Any],
        timeout_seconds: float,
    ) -> ToolResult:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{tool.name}")
        future = executor.submit(tool.run, **arguments)
        try:
            return future.result(timeout=timeout_seconds)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _finish(
        self,
        name: str,
        context: ToolExecutionContext | None,
        result: ToolResult,
        started: float,
    ) -> ToolResult:
        result.duration_ms = (time.perf_counter() - started) * 1000.0
        status = NodeStatus.COMPLETED if result.ok else NodeStatus.FAILED
        self._emit(
            context,
            name,
            status,
            "工具调用完成" if result.ok else "工具调用失败",
            {
                "tool_call_id": result.tool_call_id,
                "outcome": result.outcome.value,
                "error_type": result.error_type.value if result.error_type else None,
                "attempts": result.attempts,
                "duration_ms": round(result.duration_ms, 3),
                "error": result.error,
            },
        )
        return result

    def _emit(
        self,
        context: ToolExecutionContext | None,
        name: str,
        status: NodeStatus,
        message: str,
        payload: dict[str, Any],
    ) -> None:
        if self.trace_store is None or context is None:
            return
        self.trace_store.append(
            TraceEvent(
                run_id=context.run_id,
                node=f"tool.{name}",
                status=status,
                message=message,
                payload={"parent_node": context.node, **payload},
            )
        )

    def _is_empty(self, data: Any) -> bool:
        return data is None or data == "" or data == [] or data == {}

    def list_tools(self, names: set[str] | None = None) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_model.model_json_schema(),
                "output_schema": tool.output_model.model_json_schema() if tool.output_model else None,
            }
            for tool in self._tools.values()
            if names is None or tool.name in names
        ]
