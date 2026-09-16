from __future__ import annotations

from agentflow.evals.v2_eval import (
    GoldEvidenceSpan,
    V2EvalCase,
    bootstrap_interval,
    evaluate_ranking,
    is_verified_review_status,
    map_gold_spans,
)
from agentflow.retrieval.v2_retriever import V2ResearchRetriever
from agentflow.schemas import Evidence
from scripts.evaluate_v2_e2e import citation_metrics


def test_lexical_only_expansion_leaves_vector_query_unchanged() -> None:
    document = Evidence(id="doc__0", source="doc.pdf", text="flexibility")

    class SpyRetriever:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: str, top_k: int = 5) -> list[Evidence]:
            self.queries.append(query)
            return [document]

    vector = SpyRetriever()
    keyword = SpyRetriever()
    retriever = V2ResearchRetriever([document], vector, method="rrf_lexical_expanded")
    retriever.keyword_retriever = keyword
    results = retriever.search("蛋白质柔性", top_k=1)
    assert [item.id for item in results] == ["doc__0"]
    assert "flexibility" in keyword.queries[0]
    assert vector.queries == ["蛋白质柔性"]


def test_gold_span_maps_across_chunk_ids() -> None:
    case = V2EvalCase(
        id="dev-001",
        split="development",
        category="fact",
        primary_language="zh",
        question_zh="如何测量柔性？",
        question_en="How is flexibility measured?",
        answerable=True,
        gold_evidence=[
            GoldEvidenceSpan(
                document_id="paper-a",
                page=2,
                evidence_quote="Root mean square fluctuation measures residue flexibility.",
                claim_id="c1",
            )
        ],
    )
    chunk = Evidence(
        id="new-chunk-id",
        source="paper.pdf",
        text="The root mean square fluctuation measures residue flexibility in a trajectory.",
        document_id="paper-a",
        page=2,
    )
    mapping = map_gold_spans([case], [chunk])
    assert mapping[0].chunk_ids == ["new-chunk-id"]


def test_all_gold_metric_requires_every_claim() -> None:
    metrics = evaluate_ranking(
        ["a", "x", "b"],
        {"claim-1": {"a"}, "claim-2": {"b"}},
    )
    assert metrics["all_gold_hit@1"] == 0.0
    assert metrics["claim_recall@1"] == 0.5
    assert metrics["all_gold_hit@3"] == 1.0
    assert metrics["mrr@10"] == 1.0


def test_bootstrap_interval_is_deterministic() -> None:
    assert bootstrap_interval([0.0, 1.0, 1.0], samples=100) == bootstrap_interval(
        [0.0, 1.0, 1.0], samples=100
    )


def test_source_and_human_review_states_are_formally_verified() -> None:
    assert is_verified_review_status("source_verified")
    assert is_verified_review_status("human_verified")
    assert not is_verified_review_status("pending_human")


def test_citation_metrics_ignore_paper_reference_numbers() -> None:
    gold = "paper-a__section_aware__29"
    extra = "paper-b__section_aware__3"
    result = citation_metrics(
        f"Answer [{gold}]. Background [{extra}], original study [49].",
        [gold, extra],
        {"claim-1": {gold}},
    )
    assert result == {
        "citation_id_validity": 1.0,
        "gold_citation_precision": 0.5,
        "gold_claim_citation_coverage": 1.0,
    }
