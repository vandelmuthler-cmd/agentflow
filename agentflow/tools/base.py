from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field


class ToolOutcome(str, Enum):
    SUCCESS = "success"
    EMPTY = "empty_result"
    FAILED = "failed"


class ToolErrorType(str, Enum):
    VALIDATION = "validation_error"
    TIMEOUT = "timeout"
    TRANSIENT = "transient_error"
    PERMANENT = "permanent_error"
    UNKNOWN_TOOL = "unknown_tool"


class ToolExecutionPolicy(BaseModel):
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_attempts: int = Field(default=2, ge=1, le=5)
    backoff_seconds: float = Field(default=0.1, ge=0)


class ToolExecutionContext(BaseModel):
    run_id: str
    node: str


class LooseToolInput(BaseModel):
    model_config = ConfigDict(extra="allow")


class ToolResult(BaseModel):
    ok: bool
    data: Any = None
    error: str | None = None
    outcome: ToolOutcome = ToolOutcome.SUCCESS
    error_type: ToolErrorType | None = None
    tool_call_id: str = ""
    attempts: int = 0
    duration_ms: float = 0.0


class BaseTool(ABC):
    name: str
    description: str
    input_model: ClassVar[type[BaseModel]] = LooseToolInput
    output_model: ClassVar[type[BaseModel] | None] = None

    def validate_input(self, arguments: dict[str, Any]) -> dict[str, Any]:
        validated = self.input_model.model_validate(arguments)
        return validated.model_dump(mode="python")

    def validate_output(self, data: Any) -> Any:
        if self.output_model is None:
            return data
        validated = self.output_model.model_validate(data)
        if hasattr(validated, "root"):
            return validated.root
        return validated

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        raise NotImplementedError
