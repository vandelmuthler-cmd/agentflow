from __future__ import annotations

import re
from collections import defaultdict

from pydantic import BaseModel, Field, RootModel

from agentflow.schemas import Evidence
from agentflow.tools.base import BaseTool, ToolErrorType, ToolResult


class DocumentContextInput(BaseModel):
    evidence_id: str = Field(min_length=1)
    window: int = Field(default=1, ge=0, le=2)


class DocumentContextOutput(RootModel[list[Evidence]]):
    pass


def _chunk_order(item: Evidence) -> tuple[int, int]:
    if item.start_offset is not None:
        return 0, item.start_offset
    match = re.search(r"__(\d+)$", item.id)
    return 1, int(match.group(1)) if match else 0


class DocumentContextTool(BaseTool):
    name = "document_context"
    description = "Read a retrieved evidence chunk with nearby chunks from the same document."
    input_model = DocumentContextInput
    output_model = DocumentContextOutput

    def __init__(self, documents: list[Evidence]) -> None:
        groups: dict[tuple[str, str], list[Evidence]] = defaultdict(list)
        for item in documents:
            groups[(item.document_id, item.source)].append(item)
        self._neighborhoods: dict[str, list[Evidence]] = {}
        self._positions: dict[str, int] = {}
        for group in groups.values():
            ordered = sorted(group, key=_chunk_order)
            for position, item in enumerate(ordered):
                self._neighborhoods[item.id] = ordered
                self._positions[item.id] = position

    def run(self, **kwargs) -> ToolResult:
        evidence_id = kwargs["evidence_id"]
        group = self._neighborhoods.get(evidence_id)
        if group is None:
            return ToolResult(
                ok=False,
                error="evidence_id was not found in the active document index",
                error_type=ToolErrorType.PERMANENT,
            )
        position = self._positions[evidence_id]
        window = kwargs["window"]
        return ToolResult(ok=True, data=group[max(0, position - window) : position + window + 1])
