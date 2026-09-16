from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
import uuid

from pydantic import BaseModel, Field, model_validator

from agentflow.agents.planner import PlanStep
from agentflow.model_usage import ModelUsage


class NodeStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class Evidence(BaseModel):
    id: str
    source: str
    text: str
    score: float = 0.0
    rank: int = 0
    document_id: str = ""
    language: str = ""
    page: int | None = None
    section: str = ""
    start_offset: int | None = None
    end_offset: int | None = None
    chunking_strategy: str = ""


class CitationCheck(BaseModel):
    evidence_id: str
    present_in_answer: bool
    source: str


class GenerationQuality(BaseModel):
    citation_correctness: float = 0.0
    citation_completeness: float = 0.0
    groundedness: float = 0.0
    abstention: bool = False
    unsupported_citations: list[str] = Field(default_factory=list)
    missing_citations: list[str] = Field(default_factory=list)
    checks: list[CitationCheck] = Field(default_factory=list)


class ContextStats(BaseModel):
    max_context_tokens: int = 0
    reserved_output_tokens: int = 0
    estimated_question_tokens: int = 0
    estimated_plan_tokens: int = 0
    evidence_budget_tokens: int = 0
    evidence_tokens: int = 0
    estimated_input_tokens: int = 0
    evidence_before: int = 0
    evidence_after: int = 0
    duplicate_count: int = 0
    truncated_count: int = 0
    dropped_count: int = 0
    overflowed: bool = False
    artifact_path: str = ""
    artifact_error: str = ""


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=2)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalSearchRequest(BaseModel):
    query: str = Field(min_length=2)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalSearchResponse(BaseModel):
    query: str
    backend: str
    latency_ms: float
    evidence: list[Evidence]


class DocumentIndexRequest(BaseModel):
    document_id: str = Field(pattern=r"^[A-Za-z0-9._-]{1,128}$")
    source: str = Field(min_length=1, max_length=255)
    text: str = Field(min_length=1, max_length=5_000_000)
    chunk_size: int = Field(default=500, ge=100, le=4000)
    overlap: int = Field(default=50, ge=0, le=1000)
    language: Literal["en", "zh"] | None = None

    @model_validator(mode="after")
    def validate_chunk_window(self) -> "DocumentIndexRequest":
        if self.chunk_size <= self.overlap:
            raise ValueError("chunk_size must be larger than overlap")
        return self


class DocumentMutationResponse(BaseModel):
    document_id: str
    backend: str
    affected_chunks: int


class ResearchReport(BaseModel):
    run_id: str
    question: str
    plan: list[str]
    plan_steps: list[PlanStep] = Field(default_factory=list)
    task_type: str = "simple"
    planning_mode: str = "fixed_fallback"
    planning_error: str = ""
    context_stats: ContextStats = Field(default_factory=ContextStats)
    evidence: list[Evidence]
    answer: str
    status: str
    writer_mode: str = "offline"
    writer_error: str = ""
    generation_quality: GenerationQuality = Field(default_factory=GenerationQuality)
    model_usage: list[ModelUsage] = Field(default_factory=list)
    verifier_decision: str = ""
    verifier_mode: str = "deterministic"
    verifier_error: str = ""
    verifier_required_aspects: list[str] = Field(default_factory=list)
    verifier_covered_aspects: list[str] = Field(default_factory=list)
    verifier_missing_aspects: list[str] = Field(default_factory=list)
    verifier_follow_up_queries: list[str] = Field(default_factory=list)
    verifier_evidence_mapping: dict[str, list[str]] = Field(default_factory=dict)
    verifier_history: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_queries: list[str] = Field(default_factory=list)
    retrieval_rounds: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_tool_calls: int = 0
    retrieval_tool_attempts: int = 0
    retrieval_duration_ms: float = 0.0
    retrieval_no_progress_rounds: int = 0
    retrieval_stop_reason: str = ""
    tool_call_history: list[dict[str, Any]] = Field(default_factory=list)
    tool_decision_stop_reason: str = ""


class TraceEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""
    node: str
    component_type: str = "node"
    status: NodeStatus
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0
    error_type: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class TraceSummary(BaseModel):
    run_id: str
    event_count: int = 0
    span_count: int = 0
    completed_span_count: int = 0
    failed_span_count: int = 0
    tool_calls: int = 0
    model_calls: int = 0
    wall_clock_ms: float = 0.0
    component_duration_ms: dict[str, float] = Field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
