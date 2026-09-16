from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agentflow.retrieval.bm25 import SimpleBM25Retriever
from agentflow.retrieval.hybrid import reciprocal_rank_fusion
from agentflow.retrieval.query_expansion import expand_query
from agentflow.retrieval.rerank import minmax
from agentflow.retrieval.store import load_documents
from agentflow.retrieval.vector import VectorRetriever
from agentflow.schemas import Evidence


def _weighted_fusion(
    keyword_results: list[Evidence],
    vector_results: list[Evidence],
    *,
    vector_weight: float = 0.4,
) -> list[Evidence]:
    documents = {item.id: item for item in [*keyword_results, *vector_results]}
    keyword_scores = dict(
        zip(
            [item.id for item in keyword_results],
            minmax([item.score for item in keyword_results]),
        )
    )
    vector_scores = dict(
        zip(
            [item.id for item in vector_results],
            minmax([item.score for item in vector_results]),
        )
    )
    scored = [
        (
            (1 - vector_weight) * keyword_scores.get(item_id, 0.0)
            + vector_weight * vector_scores.get(item_id, 0.0),
            item,
        )
        for item_id, item in documents.items()
    ]
    return [
        item.model_copy(update={"score": score, "rank": rank})
        for rank, (score, item) in enumerate(
            sorted(scored, key=lambda pair: pair[0], reverse=True), 1
        )
    ]


class V2ResearchRetriever:
    def __init__(
        self,
        documents: list[Evidence],
        vector_retriever: VectorRetriever,
        *,
        method: str = "rrf_expanded",
        candidate_top_k: int = 50,
        reranker_model: str = "BAAI/bge-reranker-base",
    ) -> None:
        self.keyword_retriever = SimpleBM25Retriever()
        self.keyword_retriever.add_documents(documents)
        self.vector_retriever = vector_retriever
        self.method = method
        self.candidate_top_k = candidate_top_k
        self.reranker_model = reranker_model
        self._cross_encoder = None

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        lexical_only = self.method.endswith("_lexical_expanded")
        expanded = expand_query(query) if self.method.endswith("_expanded") or self.method == "cross_encoder" else query
        keyword = self.keyword_retriever.search(expanded, top_k=self.candidate_top_k)
        vector = self.vector_retriever.search(
            query if lexical_only else expanded, top_k=self.candidate_top_k
        )
        if self.method == "bm25":
            return keyword[:top_k]
        if self.method == "vector":
            return vector[:top_k]
        if self.method in {"rrf", "rrf_expanded", "rrf_lexical_expanded", "cross_encoder"}:
            fused = reciprocal_rank_fusion(
                [keyword, vector], top_k=self.candidate_top_k * 2
            )
        elif self.method in {"weighted", "weighted_expanded", "weighted_lexical_expanded"}:
            fused = _weighted_fusion(keyword, vector)
        else:
            raise ValueError(f"unsupported V2 retrieval method: {self.method}")
        if self.method != "cross_encoder":
            return fused[:top_k]
        candidates = fused[:30]
        scores = self._get_cross_encoder().predict(
            [(expanded, item.text) for item in candidates],
            show_progress_bar=False,
        )
        ranked = sorted(zip(scores, candidates), key=lambda pair: float(pair[0]), reverse=True)
        return [
            item.model_copy(update={"score": float(score), "rank": rank})
            for rank, (score, item) in enumerate(ranked[:top_k], 1)
        ]

    def _get_cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.reranker_model)
        return self._cross_encoder


def build_v2_retriever(
    index_root: Path,
    *,
    strategy: str,
    model_key: str,
    model_source: str,
    method: str,
    reranker_model: str,
) -> V2ResearchRetriever:
    strategy_dir = index_root / strategy
    documents = load_documents(strategy_dir / "documents.jsonl")
    if not documents:
        raise FileNotFoundError(f"Missing V2 document index: {strategy_dir / 'documents.jsonl'}")
    model_dir = strategy_dir / model_key
    metadata_path = model_dir / "index_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing V2 vector metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("chunking_strategy") != strategy or metadata.get("model_key") != model_key:
        raise ValueError("V2 vector metadata does not match the requested strategy and model.")
    digest = hashlib.sha256((strategy_dir / "documents.jsonl").read_bytes()).hexdigest()
    if metadata.get("document_index_sha256") != digest:
        raise ValueError("V2 document index SHA-256 does not match its vector index.")
    vector = VectorRetriever.from_index(
        documents,
        model_dir / "vectors.npy",
        model_dir / "vector_ids.json",
        model_name_or_path=model_source,
        model_key=model_key,
    )
    if not vector.ids:
        raise FileNotFoundError(f"Missing V2 vector index: {model_dir}")
    document_ids = [item.id for item in documents]
    if vector.ids != document_ids:
        raise ValueError("V2 vector IDs are not aligned with the document index.")
    return V2ResearchRetriever(
        documents,
        vector,
        method=method,
        reranker_model=reranker_model,
    )
