from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentflow.agents.planner import PlanStep
from agentflow.model_usage import ModelUsage
from agentflow.schemas import ContextStats, Evidence, GenerationQuality


class AgentState(BaseModel):
    run_id: str
    question: str
    search_query: str = ""
    top_k: int = 5
    plan: list[str] = Field(default_factory=list)
    plan_steps: list[PlanStep] = Field(default_factory=list)
    task_type: str = "simple"
    planning_mode: str = "fixed_fallback"
    planning_error: str = ""
    context_stats: ContextStats = Field(default_factory=ContextStats)
    evidence: list[Evidence] = Field(default_factory=list)
    answer: str = ""
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
    retries: int = 0
    status: str = "running"
