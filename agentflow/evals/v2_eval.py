from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from agentflow.schemas import Evidence


class GoldEvidenceSpan(BaseModel):
    document_id: str
    page: int = Field(ge=1)
    evidence_quote: str = Field(min_length=12)
    claim_id: str = Field(min_length=1)


class V2EvalCase(BaseModel):
    id: str
    split: Literal["development", "frozen", "stress"]
    category: str
    primary_language: Literal["zh", "en"]
    question_zh: str = Field(min_length=2)
    question_en: str = Field(min_length=2)
    answerable: bool
    reference_answer_zh: str = ""
    reference_answer_en: str = ""
    required_facts: list[str] = Field(default_factory=list)
    gold_evidence: list[GoldEvidenceSpan] = Field(default_factory=list)
    review_status: Literal[
        "pending_human", "source_verified", "human_verified", "rejected"
    ] = "pending_human"

    @model_validator(mode="after")
    def validate_evidence(self) -> "V2EvalCase":
        if self.answerable and not self.gold_evidence:
            raise ValueError("answerable cases require at least one gold evidence span")
        if not self.answerable and self.gold_evidence:
            raise ValueError("unanswerable cases cannot have gold evidence spans")
        return self


class V2EvalDataset(BaseModel):
    dataset_id: str = "agentflow-bilingual-eval-v2"
    version: str = "2.0.0-draft"
    corpus_id: str = "agentflow-bilingual-literature-v2"
    frozen: bool = False
    sha256: str = ""
    review_method: str = ""
    review_notes: str = ""
    cases: list[V2EvalCase]

    @model_validator(mode="after")
    def validate_cases(self) -> "V2EvalDataset":
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("evaluation case ids must be unique")
        return self


def is_verified_review_status(status: str) -> bool:
    """Return whether a case passed a traceable source or human review."""
    return status in {"source_verified", "human_verified"}


class GoldMapping(BaseModel):
    case_id: str
    claim_id: str
    document_id: str
    page: int
    chunk_ids: list[str]
    best_overlap: float


def load_v2_dataset(path: Path) -> V2EvalDataset:
    return V2EvalDataset.model_validate_json(path.read_text(encoding="utf-8"))


def canonical_dataset_sha256(dataset: V2EvalDataset) -> str:
    payload = dataset.model_dump(exclude={"sha256"})
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def map_gold_spans(
    cases: list[V2EvalCase],
    chunks: list[Evidence],
    *,
    minimum_overlap: float = 0.72,
) -> list[GoldMapping]:
    by_document_page: dict[tuple[str, int], list[Evidence]] = defaultdict(list)
    by_document: dict[str, list[Evidence]] = defaultdict(list)
    for chunk in chunks:
        by_document[chunk.document_id].append(chunk)
        if chunk.page is not None:
            by_document_page[(chunk.document_id, chunk.page)].append(chunk)

    mappings: list[GoldMapping] = []
    for case in cases:
        for gold in case.gold_evidence:
            # Chunk.page records the page where a chunk starts. Overlapping chunks
            # can carry a quote from the following page, so document identity and
            # quote overlap are authoritative while page remains an audit anchor.
            same_page = by_document_page.get((gold.document_id, gold.page), [])
            document_candidates = by_document.get(gold.document_id, [])
            candidates = [*same_page, *[item for item in document_candidates if item not in same_page]]
            scored = [(quote_coverage(gold.evidence_quote, item.text), item.id) for item in candidates]
            best = max((score for score, _ in scored), default=0.0)
            chunk_ids = [chunk_id for score, chunk_id in scored if score >= minimum_overlap]
            if not chunk_ids and scored:
                top_score, top_id = max(scored)
                if top_score >= 0.5:
                    chunk_ids = [top_id]
            mappings.append(
                GoldMapping(
                    case_id=case.id,
                    claim_id=gold.claim_id,
                    document_id=gold.document_id,
                    page=gold.page,
                    chunk_ids=chunk_ids,
                    best_overlap=round(best, 4),
                )
            )
    return mappings


def validate_all_gold_mapped(mappings: list[GoldMapping]) -> None:
    missing = [f"{item.case_id}:{item.claim_id}" for item in mappings if not item.chunk_ids]
    if missing:
        raise ValueError(f"Gold spans without a matching chunk: {', '.join(missing)}")


def evaluate_ranking(
    ranked_ids: list[str],
    claim_chunk_ids: dict[str, set[str]],
    *,
    ks: tuple[int, ...] = (1, 3, 5),
    mrr_k: int = 10,
    ndcg_k: int = 10,
) -> dict[str, float]:
    if not claim_chunk_ids:
        return {}
    first_ranks: list[int | None] = []
    for ids in claim_chunk_ids.values():
        rank = next((index for index, value in enumerate(ranked_ids, 1) if value in ids), None)
        first_ranks.append(rank)
    metrics: dict[str, float] = {}
    for k in ks:
        hits = sum(rank is not None and rank <= k for rank in first_ranks)
        metrics[f"claim_recall@{k}"] = hits / len(first_ranks)
        metrics[f"all_gold_hit@{k}"] = float(hits == len(first_ranks))
    valid_ranks = [rank for rank in first_ranks if rank is not None and rank <= mrr_k]
    metrics[f"mrr@{mrr_k}"] = 1.0 / min(valid_ranks) if valid_ranks else 0.0

    relevance = []
    for chunk_id in ranked_ids[:ndcg_k]:
        relevance.append(sum(chunk_id in ids for ids in claim_chunk_ids.values()))
    dcg = sum(value / math.log2(index + 2) for index, value in enumerate(relevance))
    ideal = sorted(relevance + [1] * max(0, len(claim_chunk_ids) - sum(value > 0 for value in relevance)), reverse=True)[:ndcg_k]
    idcg = sum(value / math.log2(index + 2) for index, value in enumerate(ideal))
    metrics[f"ndcg@{ndcg_k}"] = dcg / idcg if idcg else 0.0
    return metrics


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: mean(row.get(key, 0.0) for row in rows) for key in keys}


def bootstrap_interval(
    values: list[float], *, samples: int = 2000, seed: int = 20260912
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    generator = random.Random(seed)
    estimates = sorted(
        mean(generator.choice(values) for _ in values) for _ in range(samples)
    )
    return estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]


def quote_coverage(quote: str, chunk: str) -> float:
    quote_norm = _normalize_text(quote)
    chunk_norm = _normalize_text(chunk)
    if not quote_norm:
        return 0.0
    if quote_norm in chunk_norm:
        return 1.0
    quote_tokens = _tokens(quote)
    chunk_tokens = set(_tokens(chunk))
    if not quote_tokens:
        return 0.0
    return sum(token in chunk_tokens for token in quote_tokens) / len(quote_tokens)


def _normalize_text(value: str) -> str:
    return re.sub(r"[^0-9a-z\u3400-\u4dbf\u4e00-\u9fff]+", "", value.casefold())


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]", value.casefold())
