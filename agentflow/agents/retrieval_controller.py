from __future__ import annotations

import re
from difflib import SequenceMatcher

from pydantic import BaseModel, Field


class RetrievalBudgetPolicy(BaseModel):
    max_follow_up_rounds: int = Field(default=2, ge=0, le=10)
    max_total_queries: int = Field(default=7, ge=1, le=50)
    max_follow_up_queries: int = Field(default=2, ge=1, le=10)
    max_retrieval_duration_ms: float = Field(default=30_000.0, gt=0)
    max_no_progress_rounds: int = Field(default=1, ge=1, le=5)
    query_similarity_threshold: float = Field(default=0.95, ge=0.5, le=1.0)


class RetrievalControlDecision(BaseModel):
    allowed: bool
    queries: list[str] = Field(default_factory=list)
    reason: str
    rejected_queries: list[str] = Field(default_factory=list)


class AdaptiveRetrievalController:
    """Apply deterministic budgets to model-proposed retrieval actions."""

    def __init__(self, policy: RetrievalBudgetPolicy | None = None) -> None:
        self.policy = policy or RetrievalBudgetPolicy()

    def select_initial_queries(self, queries: list[str]) -> RetrievalControlDecision:
        selected, rejected = self._deduplicate(queries, [])
        selected = selected[: self.policy.max_total_queries]
        if not selected:
            return RetrievalControlDecision(
                allowed=False,
                reason="no_unique_initial_queries",
                rejected_queries=rejected,
            )
        return RetrievalControlDecision(
            allowed=True,
            queries=selected,
            reason="initial_queries_approved",
            rejected_queries=rejected,
        )

    def decide_follow_up(
        self,
        *,
        proposed_queries: list[str],
        executed_queries: list[str],
        follow_up_rounds: int,
        total_queries: int,
        retrieval_duration_ms: float,
        no_progress_rounds: int,
        missing_aspects: list[str],
        previous_missing_aspects: list[str],
        verifier_decision: str,
    ) -> RetrievalControlDecision:
        if verifier_decision == "abstain":
            return RetrievalControlDecision(allowed=False, reason="verifier_abstained")
        if follow_up_rounds >= self.policy.max_follow_up_rounds:
            return RetrievalControlDecision(allowed=False, reason="max_follow_up_rounds")
        if total_queries >= self.policy.max_total_queries:
            return RetrievalControlDecision(allowed=False, reason="max_total_queries")
        if retrieval_duration_ms >= self.policy.max_retrieval_duration_ms:
            return RetrievalControlDecision(allowed=False, reason="retrieval_time_budget")
        if no_progress_rounds >= self.policy.max_no_progress_rounds:
            return RetrievalControlDecision(allowed=False, reason="no_new_evidence")
        if (
            follow_up_rounds > 0
            and previous_missing_aspects
            and self._aspect_signature(missing_aspects)
            == self._aspect_signature(previous_missing_aspects)
        ):
            return RetrievalControlDecision(
                allowed=False,
                reason="missing_aspects_unchanged",
            )

        selected, rejected = self._deduplicate(proposed_queries, executed_queries)
        remaining = self.policy.max_total_queries - total_queries
        selected = selected[: min(self.policy.max_follow_up_queries, remaining)]
        if not selected:
            return RetrievalControlDecision(
                allowed=False,
                reason="no_novel_queries",
                rejected_queries=rejected,
            )
        return RetrievalControlDecision(
            allowed=True,
            queries=selected,
            reason="follow_up_approved",
            rejected_queries=rejected,
        )

    def _deduplicate(
        self, candidates: list[str], executed_queries: list[str]
    ) -> tuple[list[str], list[str]]:
        selected: list[str] = []
        rejected: list[str] = []
        comparison = [query for query in executed_queries if query.strip()]
        for raw_query in candidates:
            query = " ".join(raw_query.split()).strip()
            if not query or any(
                self._similarity(query, previous)
                >= self.policy.query_similarity_threshold
                for previous in [*comparison, *selected]
            ):
                rejected.append(raw_query)
                continue
            selected.append(query)
        return selected, rejected

    def _similarity(self, left: str, right: str) -> float:
        normalized_left = self._normalize(left)
        normalized_right = self._normalize(right)
        if not normalized_left or not normalized_right:
            return 0.0
        return SequenceMatcher(None, normalized_left, normalized_right).ratio()

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"[^0-9a-z\u3400-\u4dbf\u4e00-\u9fff]+", "", value.casefold())

    @classmethod
    def _aspect_signature(cls, aspects: list[str]) -> tuple[str, ...]:
        return tuple(sorted(cls._normalize(aspect) for aspect in aspects if aspect.strip()))
