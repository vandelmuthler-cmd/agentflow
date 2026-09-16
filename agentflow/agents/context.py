from __future__ import annotations

import json
import math
import re
from pathlib import Path

from pydantic import BaseModel, Field

from agentflow.agents.planner import PlanStep
from agentflow.schemas import ContextStats, Evidence


class ContextPolicy(BaseModel):
    max_context_tokens: int = Field(default=6000, ge=512)
    reserved_output_tokens: int = Field(default=1200, ge=128)
    prompt_overhead_tokens: int = Field(default=400, ge=0)
    per_evidence_overhead_tokens: int = Field(default=24, ge=0)
    max_tokens_per_evidence: int = Field(default=900, ge=32)
    min_evidence_text_tokens: int = Field(default=48, ge=8)


class PreparedContext(BaseModel):
    evidence: list[Evidence]
    stats: ContextStats


class ContextManager:
    """Build a bounded model context while retaining overflow evidence as an artifact."""

    def __init__(self, workspace_root: Path, policy: ContextPolicy | None = None) -> None:
        self.workspace_root = workspace_root
        self.policy = policy or ContextPolicy()

    def prepare(
        self,
        *,
        run_id: str,
        question: str,
        plan_steps: list[PlanStep],
        evidence: list[Evidence],
    ) -> PreparedContext:
        deduplicated = self._deduplicate(evidence)
        question_tokens = self.estimate_tokens(question)
        plan_tokens = self.estimate_tokens(self._serialize_plan(plan_steps))
        evidence_budget = max(
            0,
            self.policy.max_context_tokens
            - self.policy.reserved_output_tokens
            - self.policy.prompt_overhead_tokens
            - question_tokens
            - plan_tokens,
        )

        prepared: list[Evidence] = []
        evidence_tokens = 0
        truncated_count = 0
        for item in deduplicated:
            remaining = evidence_budget - evidence_tokens
            if remaining <= self.policy.per_evidence_overhead_tokens:
                break

            text_budget = min(
                self.policy.max_tokens_per_evidence,
                remaining - self.policy.per_evidence_overhead_tokens,
            )
            if text_budget < self.policy.min_evidence_text_tokens:
                break

            original_tokens = self.estimate_tokens(item.text)
            text = item.text
            if original_tokens > text_budget:
                text = self._truncate_to_tokens(item.text, text_budget)
                truncated_count += 1

            item_tokens = self.policy.per_evidence_overhead_tokens + self.estimate_tokens(text)
            if item_tokens > remaining:
                break
            prepared.append(item.model_copy(update={"text": text}))
            evidence_tokens += item_tokens

        dropped_count = len(deduplicated) - len(prepared)
        overflowed = truncated_count > 0 or dropped_count > 0
        artifact_path = ""
        artifact_error = ""
        if overflowed:
            try:
                artifact_path = self._write_overflow_artifact(
                    run_id=run_id,
                    evidence=evidence,
                    evidence_budget=evidence_budget,
                )
            except OSError as error:
                artifact_error = f"{type(error).__name__}: {error}"

        stats = ContextStats(
            max_context_tokens=self.policy.max_context_tokens,
            reserved_output_tokens=self.policy.reserved_output_tokens,
            estimated_question_tokens=question_tokens,
            estimated_plan_tokens=plan_tokens,
            evidence_budget_tokens=evidence_budget,
            evidence_tokens=evidence_tokens,
            estimated_input_tokens=(
                self.policy.prompt_overhead_tokens + question_tokens + plan_tokens + evidence_tokens
            ),
            evidence_before=len(evidence),
            evidence_after=len(prepared),
            duplicate_count=len(evidence) - len(deduplicated),
            truncated_count=truncated_count,
            dropped_count=dropped_count,
            overflowed=overflowed,
            artifact_path=artifact_path,
            artifact_error=artifact_error,
        )
        return PreparedContext(evidence=prepared, stats=stats)

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        cjk_count = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
        non_cjk = re.sub(r"[\u3400-\u4dbf\u4e00-\u9fff]", "", text)
        return max(1, cjk_count + math.ceil(len(non_cjk) / 4))

    def _deduplicate(self, evidence: list[Evidence]) -> list[Evidence]:
        seen_ids: set[str] = set()
        seen_text: set[str] = set()
        result = []
        for item in evidence:
            normalized_text = " ".join(item.text.split()).casefold()
            if item.id in seen_ids or normalized_text in seen_text:
                continue
            seen_ids.add(item.id)
            seen_text.add(normalized_text)
            result.append(item)
        return result

    def _serialize_plan(self, plan_steps: list[PlanStep]) -> str:
        return "\n".join(
            f"{step.objective}\n{step.query}\n{step.success_criterion}" for step in plan_steps
        )

    def _truncate_to_tokens(self, text: str, token_budget: int) -> str:
        low, high = 0, len(text)
        while low < high:
            midpoint = (low + high + 1) // 2
            if self.estimate_tokens(text[:midpoint]) <= token_budget:
                low = midpoint
            else:
                high = midpoint - 1
        return text[:low].rstrip()

    def _write_overflow_artifact(
        self,
        *,
        run_id: str,
        evidence: list[Evidence],
        evidence_budget: int,
    ) -> str:
        safe_run_id = re.sub(r"[^A-Za-z0-9_-]", "_", run_id)
        artifact_dir = self.workspace_root / safe_run_id / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "retrieval_evidence.json"
        temporary_path = artifact_path.with_suffix(".json.tmp")
        payload = {
            "run_id": run_id,
            "evidence_budget_tokens": evidence_budget,
            "evidence": [item.model_dump(mode="json") for item in evidence],
        }
        temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_path.replace(artifact_path)
        return str(artifact_path.relative_to(self.workspace_root.parent.parent))
