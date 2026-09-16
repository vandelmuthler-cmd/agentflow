from __future__ import annotations

from collections import Counter
from pathlib import Path

from agentflow.config import (
    CORPUS_VERSION,
    DATABASE_URL,
    DEFAULT_CANDIDATE_TOP_K,
    DEFAULT_RERANK_CANDIDATE_TOP_K,
    DEFAULT_RERANK_VECTOR_WEIGHT,
    DOCUMENT_INDEX_PATH,
    RETRIEVAL_BACKEND,
    RETRIEVAL_METHOD,
    RERANKER_MODEL_PATH,
    V2_CHUNKING_STRATEGY,
    V2_EMBED_MODEL,
    V2_EMBED_MODEL_PATH,
    V2_INDEX_ROOT,
    V2_RERANKER_MODEL,
    V2_RETRIEVAL_METHOD,
)
from agentflow.retrieval.bm25 import SimpleBM25Retriever
from agentflow.retrieval.chunking import chunk_text
from agentflow.retrieval.document_loader import iter_supported_files, load_document_text
from agentflow.retrieval.hybrid import HybridRetriever
from agentflow.retrieval.optimized import OptimizedResearchRetriever
from agentflow.retrieval.pgvector import PgVectorResearchRetriever, PgVectorStore
from agentflow.retrieval.rerank import WeightedVectorReranker
from agentflow.retrieval.serving_index import active_v2_index_root
from agentflow.retrieval.store import load_documents
from agentflow.retrieval.vector import VectorRetriever
from agentflow.retrieval.v2_retriever import V2ResearchRetriever, build_v2_retriever
from agentflow.schemas import Evidence


def build_documents_from_raw(raw_dir: Path) -> list[Evidence]:
    documents: list[Evidence] = []
    for path in iter_supported_files(raw_dir):
        text = load_document_text(path)
        for idx, chunk in enumerate(chunk_text(text)):
            documents.append(
                Evidence(
                    id=f"{path.stem}__{idx}",
                    source=path.name,
                    text=chunk,
                )
            )
    return documents


def summarize_documents(documents: list[Evidence]) -> dict[str, int]:
    return dict(Counter(document.source for document in documents))


def build_retriever(
    raw_dir: Path,
) -> OptimizedResearchRetriever | PgVectorResearchRetriever | V2ResearchRetriever:
    if RETRIEVAL_BACKEND == "pgvector":
        return PgVectorResearchRetriever(
            PgVectorStore(DATABASE_URL),
            method=RETRIEVAL_METHOD,
            candidate_top_k=DEFAULT_CANDIDATE_TOP_K,
            rerank_candidate_top_k=DEFAULT_RERANK_CANDIDATE_TOP_K,
            reranker_model=RERANKER_MODEL_PATH,
        )
    if RETRIEVAL_BACKEND != "local":
        raise ValueError(f"unsupported retrieval backend: {RETRIEVAL_BACKEND}")
    if CORPUS_VERSION == "v2":
        return build_v2_retriever(
            active_v2_index_root(V2_INDEX_ROOT, V2_CHUNKING_STRATEGY, V2_EMBED_MODEL),
            strategy=V2_CHUNKING_STRATEGY,
            model_key=V2_EMBED_MODEL,
            model_source=V2_EMBED_MODEL_PATH or V2_EMBED_MODEL,
            method=V2_RETRIEVAL_METHOD,
            reranker_model=V2_RERANKER_MODEL,
        )
    if CORPUS_VERSION != "v1":
        raise ValueError(f"unsupported corpus version: {CORPUS_VERSION}")
    return build_optimized_retriever(raw_dir)


def load_or_build_documents(raw_dir: Path) -> list[Evidence]:
    documents = load_documents(DOCUMENT_INDEX_PATH)
    if not documents:
        documents = build_documents_from_raw(raw_dir)
    if not documents:
        raise FileNotFoundError(
            f"No indexed or source documents were found. "
            f"Add the corpus described in {raw_dir / 'README.md'} and build the index."
        )
    return documents


def build_keyword_retriever(raw_dir: Path) -> SimpleBM25Retriever:
    documents = load_or_build_documents(raw_dir)
    keyword_retriever = SimpleBM25Retriever()
    keyword_retriever.add_documents(documents)
    return keyword_retriever


def build_vector_retriever(raw_dir: Path) -> VectorRetriever:
    documents = load_or_build_documents(raw_dir)
    return VectorRetriever.from_index(documents)


def build_hybrid_retriever(raw_dir: Path) -> HybridRetriever:
    keyword_retriever = build_keyword_retriever(raw_dir)
    vector_retriever = build_vector_retriever(raw_dir)
    return HybridRetriever(keyword_retriever, vector_retriever)


def build_optimized_retriever(raw_dir: Path) -> OptimizedResearchRetriever:
    documents = load_or_build_documents(raw_dir)
    keyword_retriever = SimpleBM25Retriever()
    keyword_retriever.add_documents(documents)
    vector_retriever = VectorRetriever.from_index(documents)
    reranker = WeightedVectorReranker(vector_weight=DEFAULT_RERANK_VECTOR_WEIGHT)
    return OptimizedResearchRetriever(
        keyword_retriever=keyword_retriever,
        reranker=reranker,
        candidate_top_k=DEFAULT_CANDIDATE_TOP_K,
    )
