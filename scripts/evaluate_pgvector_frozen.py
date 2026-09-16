from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from agentflow.config import (
    DATABASE_URL,
    DEFAULT_CANDIDATE_TOP_K,
    DEFAULT_RERANK_CANDIDATE_TOP_K,
    RERANKER_MODEL_PATH,
)
from agentflow.evals.v2_eval import (
    aggregate_metrics,
    bootstrap_interval,
    evaluate_ranking,
    load_v2_dataset,
    map_gold_spans,
    validate_all_gold_mapped,
)
from agentflow.retrieval.pgvector import PgVectorResearchRetriever, PgVectorStore
from agentflow.retrieval.store import load_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen bilingual retrieval set through pgvector."
    )
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--vector-ids", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-report", type=Path)
    parser.add_argument("--keep-index", action="store_true")
    return parser.parse_args()


def claim_map_for(case_id: str, mappings: list) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for item in mappings:
        if item.case_id == case_id:
            result[item.claim_id].update(item.chunk_ids)
    return dict(result)


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def load_local_summary(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["experiments"][0]["methods"]["cross_encoder"]


def main() -> None:
    args = parse_args()
    documents = load_documents(args.documents)
    vectors = np.load(args.vectors)
    vector_ids = json.loads(args.vector_ids.read_text(encoding="utf-8"))
    if vector_ids != [item.id for item in documents]:
        raise ValueError("document and vector identifiers are not aligned")
    if vectors.shape != (len(documents), 1024):
        raise ValueError(f"expected a (N, 1024) BGE-M3 index, got {vectors.shape}")

    dataset = load_v2_dataset(args.dataset)
    mappings = map_gold_spans(dataset.cases, documents)
    validate_all_gold_mapped(mappings)
    cases = [case for case in dataset.cases if case.split == "frozen" and case.answerable]
    document_ids = sorted({item.document_id for item in documents})
    store = PgVectorStore(DATABASE_URL)
    existing = store.load_documents()
    if existing:
        raise RuntimeError(
            "pgvector must be empty before the frozen benchmark; "
            f"found {len(existing)} chunks"
        )

    imported_chunks = 0
    evaluation_started = time.perf_counter()
    succeeded = False
    try:
        positions: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(documents):
            positions[item.document_id].append(index)
        for document_id, indices in positions.items():
            chunks = [documents[index] for index in indices]
            imported_chunks += store.replace_document_embeddings(
                document_id,
                chunks[0].source,
                chunks,
                vectors[indices],
            )

        retriever = PgVectorResearchRetriever(
            store,
            method="cross_encoder",
            candidate_top_k=DEFAULT_CANDIDATE_TOP_K,
            rerank_candidate_top_k=DEFAULT_RERANK_CANDIDATE_TOP_K,
            reranker_model=RERANKER_MODEL_PATH,
        )
        rows: list[dict] = []
        latencies: list[float] = []
        total_variants = len(cases) * 2
        completed = 0
        for case in cases:
            claims = claim_map_for(case.id, mappings)
            for language, query in (("zh", case.question_zh), ("en", case.question_en)):
                started = time.perf_counter()
                ranking = retriever.search(query, top_k=30)
                latency_ms = (time.perf_counter() - started) * 1000
                metrics = evaluate_ranking(
                    [item.id for item in ranking], claims, ks=(1, 3, 5, 30)
                )
                rows.append(
                    {
                        "case_id": case.id,
                        "language": language,
                        "latency_ms": latency_ms,
                        **metrics,
                    }
                )
                latencies.append(latency_ms)
                completed += 1
                if completed % 10 == 0 or completed == total_variants:
                    print(f"completed {completed}/{total_variants} query variants", flush=True)

        metric_rows = [
            {
                key: value
                for key, value in row.items()
                if key not in {"case_id", "language", "latency_ms"}
            }
            for row in rows
        ]
        summary = aggregate_metrics(metric_rows)
        summary["candidate_all_gold_recall@30"] = summary["all_gold_hit@30"]
        summary["average_query_pipeline_ms"] = mean(latencies)
        summary["p95_query_pipeline_ms"] = percentile95(latencies)
        low, high = bootstrap_interval([row["all_gold_hit@5"] for row in rows])
        summary["all_gold_hit@5_ci95_low"] = low
        summary["all_gold_hit@5_ci95_high"] = high
        zh = {
            row["case_id"]: row["all_gold_hit@5"]
            for row in rows
            if row["language"] == "zh"
        }
        en = {
            row["case_id"]: row["all_gold_hit@5"]
            for row in rows
            if row["language"] == "en"
        }
        deltas = [zh[key] - en[key] for key in zh.keys() & en.keys()]
        summary["zh_minus_en_all_gold_hit@5"] = mean(deltas)
        summary["mean_absolute_language_gap@5"] = mean(abs(value) for value in deltas)

        local_summary = load_local_summary(args.local_report)
        comparison_keys = (
            "all_gold_hit@1",
            "all_gold_hit@3",
            "all_gold_hit@5",
            "mrr@10",
        )
        report = {
            "backend": "pgvector",
            "method": "cross_encoder",
            "document_count": len(document_ids),
            "chunk_count": imported_chunks,
            "query_variants": len(rows),
            "embedding_dimension": int(vectors.shape[1]),
            "total_seconds": time.perf_counter() - evaluation_started,
            "summary": summary,
            "local_comparison": {
                key: {
                    "local": local_summary.get(key),
                    "pgvector": summary.get(key),
                    "delta": (
                        summary.get(key) - local_summary.get(key)
                        if key in local_summary
                        else None
                    ),
                }
                for key in comparison_keys
            },
            "per_case": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({"output": str(args.output), **summary}, indent=2))
        succeeded = True
    finally:
        if not args.keep_index or not succeeded:
            for document_id in document_ids:
                store.delete_document(document_id)


if __name__ == "__main__":
    main()
