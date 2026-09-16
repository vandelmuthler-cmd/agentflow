from __future__ import annotations

from pydantic import BaseModel, Field, RootModel

from agentflow.schemas import Evidence
from agentflow.tools.base import BaseTool, ToolResult


class DocumentSearchInput(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class DocumentSearchOutput(RootModel[list[Evidence]]):
    pass


class DocumentSearchTool(BaseTool):
    name = "document_search"
    description = "Search local documents and return evidence chunks."
    input_model = DocumentSearchInput
    output_model = DocumentSearchOutput

    def __init__(self, retriever) -> None:
        self.retriever = retriever

    def run(self, **kwargs) -> ToolResult:
        query = kwargs["query"]
        top_k = kwargs["top_k"]
        evidence = self.retriever.search(query, top_k=top_k)
        return ToolResult(ok=True, data=evidence)
