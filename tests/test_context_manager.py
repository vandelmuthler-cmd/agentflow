from __future__ import annotations

import json
from tempfile import TemporaryDirectory

from agentflow.agents.context import ContextManager, ContextPolicy
from agentflow.agents.planner import PlanStep
from agentflow.schemas import Evidence


def _plan() -> list[PlanStep]:
    return [
        PlanStep(
            id="step_1",
            objective="检索方法证据",
            tool="document_search",
            query="方法证据",
            success_criterion="至少找到一条证据",
        )
    ]


def test_context_manager_deduplicates_and_stays_inside_budget() -> None:
    policy = ContextPolicy(
        max_context_tokens=512,
        reserved_output_tokens=128,
        prompt_overhead_tokens=64,
        per_evidence_overhead_tokens=8,
        max_tokens_per_evidence=80,
        min_evidence_text_tokens=16,
    )
    evidence = [
        Evidence(id="a", source="a.pdf", text="方法结果 " * 120, rank=1),
        Evidence(id="b", source="b.pdf", text="方法结果 " * 120, rank=2),
        Evidence(id="c", source="c.pdf", text="另一个结论 " * 120, rank=3),
        Evidence(id="d", source="d.pdf", text="补充数据 " * 120, rank=4),
    ]

    with TemporaryDirectory() as temporary_dir:
        from pathlib import Path

        project_root = Path(temporary_dir)
        manager = ContextManager(project_root / "data" / "runs", policy)
        result = manager.prepare(
            run_id="context-test",
            question="比较方法和结果",
            plan_steps=_plan(),
            evidence=evidence,
        )

        assert result.stats.duplicate_count == 1
        assert result.stats.overflowed is True
        assert result.stats.truncated_count > 0 or result.stats.dropped_count > 0
        assert result.stats.estimated_input_tokens + policy.reserved_output_tokens <= policy.max_context_tokens
        assert result.stats.evidence_after == len(result.evidence)
        artifact_path = project_root / result.stats.artifact_path
        assert artifact_path.exists()
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert len(payload["evidence"]) == 4


def test_context_manager_keeps_small_context_without_artifact() -> None:
    with TemporaryDirectory() as temporary_dir:
        from pathlib import Path

        project_root = Path(temporary_dir)
        manager = ContextManager(project_root / "data" / "runs")
        evidence = [Evidence(id="a", source="a.pdf", text="短证据", rank=1)]
        result = manager.prepare(
            run_id="small-context",
            question="问题",
            plan_steps=_plan(),
            evidence=evidence,
        )

        assert result.evidence == evidence
        assert result.stats.overflowed is False
        assert result.stats.artifact_path == ""
