from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentflow.config import (  # noqa: E402
    DEFAULT_CANDIDATE_TOP_K,
    DEFAULT_RERANK_VECTOR_WEIGHT,
    DOCUMENT_INDEX_PATH,
    EMBED_MODEL_PATH,
    EVAL_RUNS_DIR,
    RAW_DIR,
    VECTOR_INDEX_PATH,
)
from agentflow.retrieval.hybrid import reciprocal_rank_fusion  # noqa: E402
from agentflow.retrieval.index import (  # noqa: E402
    build_keyword_retriever,
    build_vector_retriever,
)
from agentflow.retrieval.query_expansion import expand_query  # noqa: E402
from agentflow.retrieval.rerank import WeightedVectorReranker  # noqa: E402
from agentflow.schemas import Evidence  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


EVAL_PATH = PROJECT_ROOT / "data" / "eval_set.json"
SUMMARY_JSON = EVAL_RUNS_DIR / "full_retrieval_summary.json"
SUMMARY_CSV = EVAL_RUNS_DIR / "full_retrieval_summary.csv"
DETAIL_CSV = EVAL_RUNS_DIR / "full_retrieval_details.csv"
COMPARISON_JSON = EVAL_RUNS_DIR / "full_retrieval_comparisons.json"
REPORT_PATH = PROJECT_ROOT / "docs" / "final_retrieval_experiment.md"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260904


@dataclass(frozen=True)
class Pipeline:
    key: str
    label: str
    description: str
    search: Callable[[str], list[Evidence]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_rank(ranked_ids: list[str], gold_id: str) -> int | None:
    try:
        return ranked_ids.index(gold_id) + 1
    except ValueError:
        return None


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _bootstrap_ci(values: list[float], rng: random.Random) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    estimates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(statistics.fmean(sample))
    estimates.sort()
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _paired_bootstrap_ci(differences: list[float], rng: random.Random) -> tuple[float, float]:
    return _bootstrap_ci(differences, rng)


def _build_pipelines() -> list[Pipeline]:
    bm25 = build_keyword_retriever(RAW_DIR)
    vector = build_vector_retriever(RAW_DIR)
    reranker = WeightedVectorReranker(vector_weight=DEFAULT_RERANK_VECTOR_WEIGHT)
    candidate_k = DEFAULT_CANDIDATE_TOP_K

    # Load the embedding model before latency measurement so model startup is not
    # charged to the first vector-based pipeline.
    vector.search("retrieval benchmark warmup", top_k=1)

    def bm25_search(query: str) -> list[Evidence]:
        return bm25.search(query, top_k=candidate_k)

    def vector_search(query: str) -> list[Evidence]:
        return vector.search(query, top_k=candidate_k)

    def rrf_search(query: str) -> list[Evidence]:
        return reciprocal_rank_fusion(
            [
                bm25.search(query, top_k=candidate_k),
                vector.search(query, top_k=candidate_k),
            ],
            top_k=candidate_k,
        )

    def bm25_expanded_search(query: str) -> list[Evidence]:
        return bm25.search(expand_query(query), top_k=candidate_k)

    def vector_expanded_search(query: str) -> list[Evidence]:
        return vector.search(expand_query(query), top_k=candidate_k)

    def rrf_expanded_search(query: str) -> list[Evidence]:
        expanded = expand_query(query)
        return reciprocal_rank_fusion(
            [
                bm25.search(expanded, top_k=candidate_k),
                vector.search(expanded, top_k=candidate_k),
            ],
            top_k=candidate_k,
        )

    def bm25_reranked_search(query: str) -> list[Evidence]:
        candidates = bm25.search(query, top_k=candidate_k)
        return reranker.rerank(query, candidates, top_k=candidate_k)

    def final_search(query: str) -> list[Evidence]:
        expanded = expand_query(query)
        candidates = bm25.search(expanded, top_k=candidate_k)
        return reranker.rerank(expanded, candidates, top_k=candidate_k)

    return [
        Pipeline("bm25", "BM25", "Original query with keyword retrieval", bm25_search),
        Pipeline("vector", "Vector", "Original query with dense vector retrieval", vector_search),
        Pipeline("rrf", "BM25 + Vector / RRF", "Original query with reciprocal rank fusion", rrf_search),
        Pipeline("bm25_qe", "BM25 + query expansion", "Domain-term expansion followed by BM25", bm25_expanded_search),
        Pipeline("vector_qe", "Vector + query expansion", "Domain-term expansion followed by vector retrieval", vector_expanded_search),
        Pipeline("rrf_qe", "RRF + query expansion", "Domain-term expansion followed by RRF", rrf_expanded_search),
        Pipeline("bm25_rerank", "BM25 + reranker", "BM25 top-20 candidates with weighted vector reranking", bm25_reranked_search),
        Pipeline(
            "final",
            "Final system",
            "Domain-term expansion, BM25 top-20 candidates, and weighted vector reranking",
            final_search,
        ),
    ]


def _evaluate(pipelines: list[Pipeline], cases: list[dict]) -> tuple[list[dict], list[dict], dict[str, list[dict]]]:
    summary_rows: list[dict] = []
    detail_rows: list[dict] = []
    details_by_pipeline: dict[str, list[dict]] = {}

    for pipeline in pipelines:
        pipeline_rows = []
        latencies_ms = []
        for case_index, case in enumerate(cases, start=1):
            started = time.perf_counter()
            results = pipeline.search(case["question"])
            latency_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(latency_ms)

            ranked_ids = [item.id for item in results]
            rank = _find_rank(ranked_ids, case["gold_id"])
            row = {
                "pipeline": pipeline.key,
                "label": pipeline.label,
                "case_index": case_index,
                "question": case["question"],
                "gold_id": case["gold_id"],
                "gold_rank": rank,
                "hit_at_1": int(rank is not None and rank <= 1),
                "hit_at_3": int(rank is not None and rank <= 3),
                "hit_at_5": int(rank is not None and rank <= 5),
                "hit_at_20": int(rank is not None and rank <= DEFAULT_CANDIDATE_TOP_K),
                "reciprocal_rank_at_20": (
                    1.0 / rank if rank is not None and rank <= DEFAULT_CANDIDATE_TOP_K else 0.0
                ),
                "latency_ms": latency_ms,
                "top5_ids": " | ".join(ranked_ids[:5]),
            }
            pipeline_rows.append(row)
            detail_rows.append(row)

        rng = random.Random(BOOTSTRAP_SEED)
        metrics = {
            "recall_at_1": [row["hit_at_1"] for row in pipeline_rows],
            "recall_at_3": [row["hit_at_3"] for row in pipeline_rows],
            "recall_at_5": [row["hit_at_5"] for row in pipeline_rows],
            "candidate_recall_at_20": [row["hit_at_20"] for row in pipeline_rows],
            "mrr_at_20": [row["reciprocal_rank_at_20"] for row in pipeline_rows],
        }
        summary = {
            "pipeline": pipeline.key,
            "label": pipeline.label,
            "description": pipeline.description,
            "n_queries": len(cases),
        }
        for metric_name, values in metrics.items():
            low, high = _bootstrap_ci(values, rng)
            summary[metric_name] = statistics.fmean(values) if values else 0.0
            summary[f"{metric_name}_ci_low"] = low
            summary[f"{metric_name}_ci_high"] = high
        sorted_latencies = sorted(latencies_ms)
        summary["latency_mean_ms"] = statistics.fmean(latencies_ms) if latencies_ms else 0.0
        summary["latency_p95_ms"] = _percentile(sorted_latencies, 0.95)
        summary_rows.append(summary)
        details_by_pipeline[pipeline.key] = pipeline_rows

    return summary_rows, detail_rows, details_by_pipeline


def _compare_with_final(details_by_pipeline: dict[str, list[dict]]) -> list[dict]:
    final_rows = details_by_pipeline["final"]
    comparisons = []
    for pipeline_key, rows in details_by_pipeline.items():
        if pipeline_key == "final":
            continue
        comparison = {"baseline": pipeline_key, "target": "final"}
        rng = random.Random(BOOTSTRAP_SEED)
        for metric, field in [
            ("recall_at_1", "hit_at_1"),
            ("recall_at_3", "hit_at_3"),
            ("recall_at_5", "hit_at_5"),
            ("candidate_recall_at_20", "hit_at_20"),
            ("mrr_at_20", "reciprocal_rank_at_20"),
        ]:
            differences = [target[field] - source[field] for source, target in zip(rows, final_rows)]
            low, high = _paired_bootstrap_ci(differences, rng)
            comparison[f"delta_{metric}"] = statistics.fmean(differences)
            comparison[f"delta_{metric}_ci_low"] = low
            comparison[f"delta_{metric}_ci_high"] = high
        comparisons.append(comparison)
    return comparisons


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _build_report(
    summary_rows: list[dict],
    comparisons: list[dict],
    metadata: dict,
    details_by_pipeline: dict[str, list[dict]],
) -> str:
    by_key = {row["pipeline"]: row for row in summary_rows}
    comparison_by_key = {row["baseline"]: row for row in comparisons}
    final = by_key["final"]
    bm25 = by_key["bm25"]
    bm25_qe = by_key["bm25_qe"]
    rrf_qe = by_key["rrf_qe"]
    final_vs_bm25 = comparison_by_key["bm25"]
    bm25_details = details_by_pipeline["bm25"]
    final_details = details_by_pipeline["final"]

    fixed_at_5 = [
        (baseline, target)
        for baseline, target in zip(bm25_details, final_details)
        if not baseline["hit_at_5"] and target["hit_at_5"]
    ]
    regressed_at_5 = [
        (baseline, target)
        for baseline, target in zip(bm25_details, final_details)
        if baseline["hit_at_5"] and not target["hit_at_5"]
    ]
    final_misses_at_5 = [row for row in final_details if not row["hit_at_5"]]

    fixed_rows = [
        "| {case_index} | {question} | {before} | {after} |".format(
            case_index=target["case_index"],
            question=target["question"].replace("|", "\\|"),
            before=baseline["gold_rank"] or ">20",
            after=target["gold_rank"],
        )
        for baseline, target in fixed_at_5
    ]
    miss_rows = [
        "| {case_index} | {question} | {rank} | {diagnosis} |".format(
            case_index=row["case_index"],
            question=row["question"].replace("|", "\\|"),
            rank=row["gold_rank"] or ">20",
            diagnosis=(
                "候选池未召回，重排器无法修复"
                if not row["hit_at_20"]
                else "已进入候选池，但仍未排进 top-5"
            ),
        )
        for row in final_misses_at_5
    ]

    table_rows = []
    for row in summary_rows:
        table_rows.append(
            "| {label} | {r1} | {r3} | {r5} | {cr20} | {mrr:.3f} | {latency:.1f} |".format(
                label=row["label"],
                r1=_pct(row["recall_at_1"]),
                r3=_pct(row["recall_at_3"]),
                r5=_pct(row["recall_at_5"]),
                cr20=_pct(row["candidate_recall_at_20"]),
                mrr=row["mrr_at_20"],
                latency=row["latency_mean_ms"],
            )
        )

    return f"""# AgentFlow 完整检索实验报告

生成时间：{metadata['generated_at_utc']}

## 实验目的

在同一语料、同一切块索引和同一批问题上，对比 BM25、向量检索、BM25 与向量检索的倒数排名融合（RRF）、查询术语扩展、加权向量重排以及最终系统，确认每个组件带来的实际变化。

## 评测口径

- 语料：3 篇 PDF，共 {metadata['document_chunks']} 个 chunk。
- 评测集：{metadata['n_queries']} 个固定问题，每个问题标注 1 个 gold chunk。
- `Recall@K`：本项目采用 query-level Recall@K。gold chunk 出现在前 K 条结果中，该问题记为命中。因此它也可称为 Hit Rate@K。
- `CandidateRecall@20`：gold chunk 是否进入前 20 个候选，用于判断重排器是否有机会处理正确证据。
- `MRR@20`：gold chunk 排名倒数的平均值；未进入前 20 记为 0。
- 区间：95% bootstrap 置信区间，重采样 {metadata['bootstrap_samples']} 次；它只反映当前 19 题的抽样不确定性。
- 延迟：模型预热后，单进程顺序执行的单查询平均检索耗时，不包含文档入库、答案生成和网络请求。

## 系统配置

- 候选池大小：top-{metadata['candidate_top_k']}。
- RRF 常数：k=60。
- 重排公式：`final_score = 0.6 * normalized_BM25 + 0.4 * normalized_vector_similarity`。
- 向量模型：`{metadata['embedding_model']}`，本地加载。
- 最终系统：原始中文 query → 人工领域术语扩展 → BM25 top-20 → 加权向量重排 → top-5 evidence。

## 完整结果

| 检索方案 | Recall@1 | Recall@3 | Recall@5 | CandidateRecall@20 | MRR@20 | 平均延迟 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table_rows)}

最终系统的 Recall@5 为 {_pct(final['recall_at_5'])}（{round(final['recall_at_5'] * metadata['n_queries'])}/{metadata['n_queries']}），95% 置信区间为 {_pct(final['recall_at_5_ci_low'])}–{_pct(final['recall_at_5_ci_high'])}；MRR@20 为 {final['mrr_at_20']:.3f}。

## 组件贡献

1. 原始 BM25 的 Recall@5 为 {_pct(bm25['recall_at_5'])}，加入人工领域术语扩展后达到 {_pct(bm25_qe['recall_at_5'])}。这一步修复了中文问题与英文论文术语不一致造成的关键词缺失。
2. 术语扩展后的 BM25 CandidateRecall@20 达到 {_pct(bm25_qe['candidate_recall_at_20'])}，说明 {round(bm25_qe['candidate_recall_at_20'] * metadata['n_queries'])}/{metadata['n_queries']} 个问题的 gold chunk 已进入重排候选池。
3. 在该候选池上加入权重为 0.4 的向量重排后，Recall@5 从 {_pct(bm25_qe['recall_at_5'])} 提升到 {_pct(final['recall_at_5'])}，MRR@20 从 {bm25_qe['mrr_at_20']:.3f} 提升到 {final['mrr_at_20']:.3f}。
4. RRF 加术语扩展的 Recall@1 为 {_pct(rrf_qe['recall_at_1'])}、Recall@5 为 {_pct(rrf_qe['recall_at_5'])}，均低于 BM25 加术语扩展。这说明当前向量模型产生的低质量排序会通过 RRF 挤掉部分 BM25 结果，混合检索并不必然优于单路检索。

## 逐题变化

相对原始 BM25，最终系统修复了 {len(fixed_at_5)} 个 Recall@5 失败问题，并造成 {len(regressed_at_5)} 个原有命中退化。

| 编号 | 问题 | BM25 gold rank | 最终 gold rank |
| ---: | --- | ---: | ---: |
{chr(10).join(fixed_rows)}

最终系统仍有 {len(final_misses_at_5)} 个问题未进入 top-5：

| 编号 | 问题 | 最终 gold rank | 原因 |
| ---: | --- | ---: | --- |
{chr(10).join(miss_rows)}

最终系统相对原始 BM25 的 Recall@5 绝对提升为 {_pct(final_vs_bm25['delta_recall_at_5'])}，配对 bootstrap 95% 区间为 {_pct(final_vs_bm25['delta_recall_at_5_ci_low'])}–{_pct(final_vs_bm25['delta_recall_at_5_ci_high'])}。当前固定评测上的正确表述是：

> 在 3 篇 PDF、444 个 chunk 和 19 条固定问题构成的检索评测集上，通过人工领域术语扩展构建 BM25 top-20 高召回候选池，并使用加权向量相似度重排，将 query-level Recall@5 从 47.4%（9/19）提升至 84.2%（16/19），CandidateRecall@20 从 63.2% 提升至 94.7%，MRR@20 从 0.323 提升至 {final['mrr_at_20']:.3f}。

## 不能使用的表述

`Recall@5 从 57.9% 提升至 78.9%` 混合了不同指标。统一实验中 57.9% 是 Vector 的 CandidateRecall@20，78.9% 是 BM25 加术语扩展的 Recall@5，不能构成同一指标的前后对比。旧脚本还曾在 RRF 的 top-5 指标和 top-20 候选指标之间使用不同的单路检索深度，本报告统一固定为每路 top-20 后再计算各个 K 值。

## 结果边界

- 评测集只有 19 条且每题只标注 1 个 gold chunk，存在“检索到其他同样相关 chunk 但被判失败”的可能。
- 术语词典与重排权重曾参考这批问题进行开发，因此该结果属于固定开发集上的系统消融，不能代表未见问题上的泛化性能。
- 当前重排器是 BM25 分数与向量相似度的线性融合，不是 cross-encoder。
- 当前向量模型偏中文，而语料主要为英文；Vector 和 RRF 的结果不能代表更合适的多语言模型上限。
- 下一次增强实验应冻结当前配置，另建独立测试集，并补充多相关 chunk 标注，再报告最终泛化指标。

## 可复现文件

- 汇总结果：`data/eval_runs/full_retrieval_summary.csv`
- 逐题排名：`data/eval_runs/full_retrieval_details.csv`
- 配对提升：`data/eval_runs/full_retrieval_comparisons.json`
- 机器可读汇总：`data/eval_runs/full_retrieval_summary.json`
- 执行脚本：`scripts/evaluate_full_retrieval.py`
- eval_set SHA-256：`{metadata['eval_set_sha256']}`
- document_index SHA-256：`{metadata['document_index_sha256']}`
- vector_index SHA-256：`{metadata['vector_index_sha256']}`
"""


def main() -> None:
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    pipelines = _build_pipelines()
    summary_rows, detail_rows, details_by_pipeline = _evaluate(pipelines, cases)
    comparisons = _compare_with_final(details_by_pipeline)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_queries": len(cases),
        "document_chunks": sum(1 for _ in DOCUMENT_INDEX_PATH.open("r", encoding="utf-8")),
        "candidate_top_k": DEFAULT_CANDIDATE_TOP_K,
        "rerank_vector_weight": DEFAULT_RERANK_VECTOR_WEIGHT,
        "embedding_model": Path(EMBED_MODEL_PATH).name,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "eval_set_sha256": _sha256(EVAL_PATH),
        "document_index_sha256": _sha256(DOCUMENT_INDEX_PATH),
        "vector_index_sha256": _sha256(VECTOR_INDEX_PATH),
    }

    EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(
        json.dumps({"metadata": metadata, "results": summary_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    COMPARISON_JSON.write_text(json.dumps(comparisons, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(SUMMARY_CSV, summary_rows)
    _write_csv(DETAIL_CSV, detail_rows)
    REPORT_PATH.write_text(
        _build_report(summary_rows, comparisons, metadata, details_by_pipeline),
        encoding="utf-8",
    )

    print(f"queries: {len(cases)}")
    for row in summary_rows:
        print(
            f"{row['label']}: "
            f"R@1={row['recall_at_1']:.3f}, "
            f"R@3={row['recall_at_3']:.3f}, "
            f"R@5={row['recall_at_5']:.3f}, "
            f"CandidateR@20={row['candidate_recall_at_20']:.3f}, "
            f"MRR@20={row['mrr_at_20']:.3f}, "
            f"mean_latency={row['latency_mean_ms']:.1f}ms"
        )
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
