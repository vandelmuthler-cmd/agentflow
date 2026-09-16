from __future__ import annotations

import time

from pydantic import BaseModel, Field, RootModel

from agentflow.storage.traces import InMemoryTraceStore
from agentflow.tools.base import (
    BaseTool,
    ToolErrorType,
    ToolExecutionContext,
    ToolExecutionPolicy,
    ToolOutcome,
    ToolResult,
)
from agentflow.tools.registry import ToolRegistry


class ValueInput(BaseModel):
    value: int = Field(ge=0)


class ValueOutput(RootModel[list[int]]):
    pass


class SuccessTool(BaseTool):
    name = "success"
    description = "Return one validated value."
    input_model = ValueInput
    output_model = ValueOutput

    def run(self, **kwargs) -> ToolResult:
        return ToolResult(ok=True, data=[kwargs["value"]])


class FlakyTool(SuccessTool):
    name = "flaky"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, **kwargs) -> ToolResult:
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("temporary connection failure")
        return super().run(**kwargs)


class SlowTool(SuccessTool):
    name = "slow"

    def run(self, **kwargs) -> ToolResult:
        time.sleep(0.05)
        return super().run(**kwargs)


class EmptyTool(SuccessTool):
    name = "empty"

    def run(self, **kwargs) -> ToolResult:
        return ToolResult(ok=True, data=[])


def test_tool_validation_retry_timeout_and_empty_results() -> None:
    traces = InMemoryTraceStore()
    registry = ToolRegistry(trace_store=traces)
    for tool in [SuccessTool(), FlakyTool(), SlowTool(), EmptyTool()]:
        registry.register(tool)

    context = ToolExecutionContext(run_id="tool-runtime-test", node="test")
    success = registry.run("success", context=context, value=3)
    invalid = registry.run("success", context=context, value=-1)
    flaky = registry.run(
        "flaky",
        context=context,
        policy=ToolExecutionPolicy(timeout_seconds=1, max_attempts=2, backoff_seconds=0),
        value=5,
    )
    timeout = registry.run(
        "slow",
        context=context,
        policy=ToolExecutionPolicy(timeout_seconds=0.01, max_attempts=2, backoff_seconds=0),
        value=7,
    )
    empty = registry.run("empty", context=context, value=0)

    assert success.ok and success.data == [3] and success.attempts == 1
    assert not invalid.ok and invalid.error_type == ToolErrorType.VALIDATION
    assert flaky.ok and flaky.attempts == 2
    assert not timeout.ok and timeout.error_type == ToolErrorType.TIMEOUT
    assert timeout.attempts == 2
    assert empty.ok and empty.outcome == ToolOutcome.EMPTY
    assert registry.list_tools()[0]["input_schema"]["properties"]["value"]["minimum"] == 0
    assert len(traces.read(context.run_id)) == 10
