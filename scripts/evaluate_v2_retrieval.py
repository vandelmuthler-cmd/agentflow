from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentflow.evals.v2_eval import (
    aggregate_metrics,
    bootstrap_interval,
    evaluate_ranking,
    load_v2_dataset,
    map_gold_spans,
    validate_all_gold_mapped,
)
from agentflow.retrieval.bm25 import SimpleBM25Retriever
from agentflow.retrieval.embedding import get_named_embedding_model
from agentflow.retrieval.hybrid import reciprocal_rank_fusion
from agentflow.retrieval.query_expansion import expand_query
from agentflow.retrieval.rerank import minmax
from agentflow.retrieval.store import load_documents
from agentflow.retrieval.vector import VectorRetriever


MODELS = ("bge-small-zh-v1.5", "bge-small-en-v1.5", "bge-m3")
STRATEGIES = ("fixed_char", "recursive_token", "section_aware")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible AgentFlow V2 retrieval experiments.")
    parser.add_argument("--mode", choices=("matrix", "ablation", "frozen"), default="matrix")
    parser.add_argument("--dataset", type=Path, default=Path("data/v2_eval_frozen.json"))
    parser.add_argument("--strategy", choices=STRATEGIES)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--model-override", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--with-reranker", action="store_true")
    parser.add_argument("--with-lexical-expansion", action="store_true")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def overrides(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or key not in MODELS:
            raise ValueError(f"invalid --model-override: {value}")
        result[key] = path
    return result


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def claim_map_for(case_id: str, mappings: list) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for item in mappings:
        if item.case_id == case_id:
            result[item.claim_id].update(item.chunk_ids)
    return dict(result)


def union_rankings(*rankings: list) -> list:
    seen = set()
    result = []
    for ranking in rankings:
        for item in ranking:
            if item.id not in seen:
                seen.add(item.id)
                result.append(item.model_copy(update={"rank": len(result) + 1}))
    return result


def weighted_fusion(bm25_results: list, vector_results: list, weight: float = 0.4) -> list:
    documents = {item.id: item for item in [*bm25_results, *vector_results]}
    lexical = dict(zip([item.id for item in bm25_results], minmax([item.score for item in bm25_results])))
    semantic = dict(zip([item.id for item in vector_results], minmax([item.score for item in vector_results])))
    scored = [
        ((1 - weight) * lexical.get(item_id, 0.0) + weight * semantic.get(item_id, 0.0), item)
        for item_id, item in documents.items()
    ]
    return [
        item.model_copy(update={"score": score, "rank": rank})
        for rank, (score, item) in enumerate(sorted(scored, key=lambda pair: pair[0], reverse=True), 1)
    ]


def cross_encoder_rerank(query: str, candidates: list, model) -> list:
    scores = model.predict([(query, item.text) for item in candidates], show_progress_bar=False)
    ranked = sorted(zip(scores, candidates), key=lambda pair: float(pair[0]), reverse=True)
    return [
        item.model_copy(update={"score": float(score), "rank": rank})
        for rank, (score, item) in enumerate(ranked, 1)
    ]


def run_configuration(
    *,
    strategy: str,
    model_key: str,
    model_source: str,
    dataset,
    split: str,
    methods: list[str],
    with_reranker: bool,
    with_lexical_expansion: bool = False,
    reranker_model: str = "BAAI/bge-reranker-base",
) -> dict:
    index_dir = PROJECT_ROOT / "data/index/v2" / strategy
    documents = load_documents(index_dir / "documents.jsonl")
    vector_dir = index_dir / model_key
    metadata = json.loads((vector_dir / "index_metadata.json").read_text(encoding="utf-8"))
    embedding_model = get_named_embedding_model(model_source)
    metadata["model_parameter_bytes"] = sum(
        parameter.numel() * parameter.element_size()
        for parameter in embedding_model.parameters()
    )
    vector = VectorRetriever.from_index(
        documents,
        vector_dir / "vectors.npy",
        vector_dir / "vector_ids.json",
        model_name_or_path=model_source,
        model_key=model_key,
    )
    bm25 = SimpleBM25Retriever()
    bm25.add_documents(documents)
    mappings = map_gold_spans(dataset.cases, documents)
    validate_all_gold_mapped(mappings)
    cases = [case for case in dataset.cases if case.split == split and case.answerable]
    reranker = None
    if with_reranker:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(reranker_model)

    method_rows: dict[str, list[dict]] = defaultdict(list)
    latencies: dict[str, list[float]] = defaultdict(list)
    per_case: list[dict] = []
    for case in cases:
        claims = claim_map_for(case.id, mappings)
        for language, query in (("zh", case.question_zh), ("en", case.question_en)):
            started = time.perf_counter()
            vec = vector.search(query, top_k=50)
            vector_ms = (time.perf_counter() - started) * 1000
            rankings = {"vector": vec}
            timing_ms = {"vector": vector_ms}
            if methods != ["vector"]:
                started = time.perf_counter()
                bm = bm25.search(query, top_k=50)
                bm25_ms = (time.perf_counter() - started) * 1000
                expanded = expand_query(query)
                started = time.perf_counter()
                bm_expanded = bm25.search(expanded, top_k=50)
                bm25_expanded_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                vec_expanded = vector.search(expanded, top_k=50)
                vector_expanded_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                candidate_union = union_rankings(bm, vec)
                union_fusion_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                rrf = reciprocal_rank_fusion([bm, vec], top_k=100)
                rrf_fusion_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                weighted = weighted_fusion(bm, vec)
                weighted_fusion_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                rrf_expanded = reciprocal_rank_fusion([bm_expanded, vec_expanded], top_k=100)
                rrf_expanded_fusion_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                weighted_expanded = weighted_fusion(bm_expanded, vec_expanded)
                weighted_expanded_fusion_ms = (time.perf_counter() - started) * 1000
                if with_lexical_expansion:
                    started = time.perf_counter()
                    rrf_lexical_expanded = reciprocal_rank_fusion([bm_expanded, vec], top_k=100)
                    rrf_lexical_fusion_ms = (time.perf_counter() - started) * 1000
                    started = time.perf_counter()
                    weighted_lexical_expanded = weighted_fusion(bm_expanded, vec)
                    weighted_lexical_fusion_ms = (time.perf_counter() - started) * 1000
                rankings.update({
                    "bm25": bm,
                    "candidate_union": candidate_union,
                    "rrf": rrf,
                    "weighted": weighted,
                    "rrf_expanded": rrf_expanded,
                    "weighted_expanded": weighted_expanded,
                })
                timing_ms.update({
                    "bm25": bm25_ms,
                    "candidate_union": bm25_ms + vector_ms + union_fusion_ms,
                    "rrf": bm25_ms + vector_ms + rrf_fusion_ms,
                    "weighted": bm25_ms + vector_ms + weighted_fusion_ms,
                    "rrf_expanded": bm25_expanded_ms + vector_expanded_ms + rrf_expanded_fusion_ms,
                    "weighted_expanded": bm25_expanded_ms + vector_expanded_ms + weighted_expanded_fusion_ms,
                })
                if with_lexical_expansion:
                    rankings.update({
                        "rrf_lexical_expanded": rrf_lexical_expanded,
                        "weighted_lexical_expanded": weighted_lexical_expanded,
                    })
                    timing_ms.update({
                        "rrf_lexical_expanded": bm25_expanded_ms + vector_ms + rrf_lexical_fusion_ms,
                        "weighted_lexical_expanded": bm25_expanded_ms + vector_ms + weighted_lexical_fusion_ms,
                    })
                if reranker is not None:
                    candidates = rankings["rrf_expanded"][:30]
                    started = time.perf_counter()
                    rankings["cross_encoder"] = cross_encoder_rerank(expanded, candidates, reranker)
                    timing_ms["cross_encoder"] = (
                        timing_ms["rrf_expanded"]
                        + (time.perf_counter() - started) * 1000
                    )
            for method in methods:
                if method not in rankings:
                    continue
                ranking = rankings[method]
                metrics = evaluate_ranking(
                    [item.id for item in ranking], claims, ks=(1, 3, 5, 30, 50)
                )
                if method in {"candidate_union", "rrf", "weighted"}:
                    candidate_ranking = candidate_union
                    candidate_depth = 50
                elif method in {"rrf_expanded", "weighted_expanded"}:
                    candidate_ranking = union_rankings(bm_expanded, vec_expanded)
                    candidate_depth = 50
                elif method in {"rrf_lexical_expanded", "weighted_lexical_expanded"}:
                    candidate_ranking = union_rankings(bm_expanded, vec)
                    candidate_depth = 50
                elif method == "cross_encoder":
                    candidate_ranking = rankings["rrf_expanded"][:30]
                    candidate_depth = 30
                else:
                    candidate_ranking = ranking[:50]
                    candidate_depth = 50
                candidate_metrics = evaluate_ranking(
                    [item.id for item in candidate_ranking],
                    claims,
                    ks=(len(candidate_ranking),),
                )
                metrics["candidate_all_gold_recall"] = candidate_metrics.get(
                    f"all_gold_hit@{len(candidate_ranking)}", 0.0
                )
                metrics["candidate_depth_per_source"] = float(candidate_depth)
                row = {"case_id": case.id, "language": language, "method": method, **metrics}
                method_rows[method].append(metrics)
                latencies[method].append(timing_ms[method])
                per_case.append(row)

    summaries = {}
    for method, rows in method_rows.items():
        summary = aggregate_metrics(rows)
        summary["average_query_pipeline_ms"] = mean(latencies[method])
        summary["p95_query_pipeline_ms"] = p95(latencies[method])
        values = [row.get("all_gold_hit@5", 0.0) for row in rows]
        low, high = bootstrap_interval(values)
        summary["all_gold_hit@5_ci95_low"] = low
        summary["all_gold_hit@5_ci95_high"] = high
        zh = [row for row in per_case if row["method"] == method and row["language"] == "zh"]
        en = [row for row in per_case if row["method"] == method and row["language"] == "en"]
        zh_by_id = {row["case_id"]: row["all_gold_hit@5"] for row in zh}
        en_by_id = {row["case_id"]: row["all_gold_hit@5"] for row in en}
        deltas = [zh_by_id[key] - en_by_id[key] for key in zh_by_id.keys() & en_by_id.keys()]
        summary["zh_minus_en_all_gold_hit@5"] = mean(deltas) if deltas else 0.0
        summary["mean_absolute_language_gap@5"] = mean(abs(value) for value in deltas) if deltas else 0.0
        summaries[method] = summary
    return {
        "strategy": strategy,
        "model": model_key,
        "split": split,
        "query_variants": len(cases) * 2,
        "index_metadata": metadata,
        "methods": summaries,
        "per_case": per_case,
    }


def write_reports(output_dir: Path, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    flat_rows = []
    for experiment in report["experiments"]:
        for method, metrics in experiment["methods"].items():
            flat_rows.append({
                "strategy": experiment["strategy"],
                "model": experiment["model"],
                "split": experiment["split"],
                "method": method,
                **experiment["index_metadata"],
                **metrics,
            })
    if flat_rows:
        with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)
    lines = [
        "# AgentFlow V2 Retrieval Report",
        "",
        f"- Status: `{'PROVISIONAL' if report['provisional'] else 'FROZEN'}`",
        f"- Dataset SHA-256: `{report['dataset_sha256']}`",
        "",
        "| Chunking | Embedding | Method | All-gold Hit@5 | MRR@10 | nDCG@10 | Candidate all-gold |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in flat_rows:
        lines.append(
            f"| {row['strategy']} | {row['model']} | {row['method']} | "
            f"{row.get('all_gold_hit@5', 0):.3f} | {row.get('mrr@10', 0):.3f} | "
            f"{row.get('ndcg@10', 0):.3f} | {row.get('candidate_all_gold_recall', 0):.3f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset = load_v2_dataset(PROJECT_ROOT / args.dataset)
    provisional = any(
        case.review_status not in {"source_verified", "human_verified"}
        for case in dataset.cases
    )
    if provisional and not args.allow_draft:
        raise SystemExit("Dataset has not passed source or human review. Use --allow-draft only for provisional experiments.")
    model_overrides = overrides(args.model_override)
    if args.mode == "matrix":
        configurations = [(strategy, model) for strategy in STRATEGIES for model in MODELS]
        methods = ["vector"]
        split = "development"
    else:
        if not args.strategy or not args.model:
            raise SystemExit("--strategy and --model are required for ablation/frozen modes")
        configurations = [(args.strategy, args.model)]
        methods = ["bm25", "vector", "candidate_union", "rrf", "weighted", "rrf_expanded", "weighted_expanded"]
        if args.with_lexical_expansion:
            methods.extend(["rrf_lexical_expanded", "weighted_lexical_expanded"])
        if args.with_reranker:
            methods.append("cross_encoder")
        split = "development" if args.mode == "ablation" else "frozen"
    experiments = []
    for strategy, model in configurations:
        print(f"running {strategy} x {model} ({split})")
        experiments.append(
            run_configuration(
                strategy=strategy,
                model_key=model,
                model_source=model_overrides.get(model, model),
                dataset=dataset,
                split=split,
                methods=methods,
                with_reranker=args.with_reranker,
                with_lexical_expansion=args.with_lexical_expansion,
                reranker_model=args.reranker_model,
            )
        )
    report = {
        "mode": args.mode,
        "provisional": provisional,
        "dataset_sha256": dataset.sha256,
        "experiments": experiments,
    }
    output_dir = PROJECT_ROOT / "reports/v2" / args.mode
    if args.mode == "frozen":
        output_dir = output_dir / f"{args.strategy}_{args.model}"
    if args.output_dir:
        output_dir = PROJECT_ROOT / args.output_dir
    write_reports(output_dir, report)
    print(f"reports: {output_dir}")


if __name__ == "__main__":
    main()
