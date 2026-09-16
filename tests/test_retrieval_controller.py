from __future__ import annotations

from agentflow.agents.retrieval_controller import (
    AdaptiveRetrievalController,
    RetrievalBudgetPolicy,
)


def test_controller_rejects_duplicate_queries_but_allows_meaningful_rewrite() -> None:
    controller = AdaptiveRetrievalController()
    original = "What is required at prediction time?"

    duplicate = controller.decide_follow_up(
        proposed_queries=[original],
        executed_queries=[original],
        follow_up_rounds=0,
        total_queries=1,
        retrieval_duration_ms=10,
        no_progress_rounds=0,
        missing_aspects=["requirement"],
        previous_missing_aspects=[],
        verifier_decision="insufficient",
    )
    rewritten = controller.decide_follow_up(
        proposed_queries=[f"{original} 关键事实 证据"],
        executed_queries=[original],
        follow_up_rounds=0,
        total_queries=1,
        retrieval_duration_ms=10,
        no_progress_rounds=0,
        missing_aspects=["requirement"],
        previous_missing_aspects=[],
        verifier_decision="insufficient",
    )

    assert not duplicate.allowed and duplicate.reason == "no_novel_queries"
    assert rewritten.allowed and len(rewritten.queries) == 1


def test_controller_stops_after_no_progress_or_unchanged_missing_aspects() -> None:
    controller = AdaptiveRetrievalController()
    common = {
        "proposed_queries": ["new targeted query"],
        "executed_queries": ["original query"],
        "follow_up_rounds": 1,
        "total_queries": 2,
        "retrieval_duration_ms": 10,
        "verifier_decision": "insufficient",
    }

    no_progress = controller.decide_follow_up(
        **common,
        no_progress_rounds=1,
        missing_aspects=["B"],
        previous_missing_aspects=["A"],
    )
    stagnant = controller.decide_follow_up(
        **common,
        no_progress_rounds=0,
        missing_aspects=["B"],
        previous_missing_aspects=["B"],
    )

    assert not no_progress.allowed and no_progress.reason == "no_new_evidence"
    assert not stagnant.allowed and stagnant.reason == "missing_aspects_unchanged"


def test_controller_enforces_query_and_time_budgets() -> None:
    controller = AdaptiveRetrievalController(
        RetrievalBudgetPolicy(
            max_total_queries=3,
            max_retrieval_duration_ms=100,
        )
    )
    common = {
        "proposed_queries": ["new query"],
        "executed_queries": ["old query"],
        "follow_up_rounds": 0,
        "no_progress_rounds": 0,
        "missing_aspects": ["B"],
        "previous_missing_aspects": [],
        "verifier_decision": "insufficient",
    }

    query_limited = controller.decide_follow_up(
        **common,
        total_queries=3,
        retrieval_duration_ms=10,
    )
    time_limited = controller.decide_follow_up(
        **common,
        total_queries=1,
        retrieval_duration_ms=100,
    )

    assert not query_limited.allowed and query_limited.reason == "max_total_queries"
    assert not time_limited.allowed and time_limited.reason == "retrieval_time_budget"
