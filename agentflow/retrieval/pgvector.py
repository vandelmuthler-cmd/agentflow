from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Protocol

from agentflow.config import RERANKER_MODEL_PATH
from agentflow.retrieval.bm25 import SimpleBM25Retriever
from agentflow.retrieval.embedding import encode_passages, encode_queries
from agentflow.retrieval.hybrid import reciprocal_rank_fusion
from agentflow.retrieval.query_expansion import expand_query
from agentflow.schemas import Evidence


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in values) + "]"


def _postgres_safe(value):
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {key: _postgres_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_postgres_safe(item) for item in value]
    return value


class VectorSearchStore(Protocol):
    def load_documents(self) -> list[Evidence]: ...

    def vector_search(self, query: str, top_k: int) -> list[Evidence]: ...


class PgVectorStore:
    """Persistent chunk and embedding storage backed by PostgreSQL and pgvector."""

    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise ValueError("AGENTFLOW_DATABASE_URL is required for the pgvector backend")
        self.database_url = database_url

    @contextmanager
    def _connection(self):
        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError(
                "The pgvector backend requires `psycopg[binary]`."
            ) from error
        try:
            with psycopg.connect(self.database_url) as connection:
                yield connection
        except psycopg.Error as error:
            raise RuntimeError(f"pgvector operation failed: {error}") from error

    def ensure_schema(self, embedding_dimension: int | None = None) -> None:
        dimension = embedding_dimension or int(
            encode_passages(["dimension probe"]).shape[1]
        )
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute("SELECT to_regclass('public.agentflow_chunks')")
                table_exists = cursor.fetchone()[0] is not None
                if table_exists:
                    cursor.execute(
                        """
                        SELECT format_type(attribute.atttypid, attribute.atttypmod)
                        FROM pg_attribute AS attribute
                        JOIN pg_class AS relation ON relation.oid = attribute.attrelid
                        WHERE relation.relname = 'agentflow_chunks'
                          AND attribute.attname = 'embedding'
                          AND attribute.attnum > 0
                        """
                    )
                    row = cursor.fetchone()
                    stored_type = row[0] if row else ""
                    if stored_type != f"vector({dimension})":
                        cursor.execute("SELECT COUNT(*) FROM agentflow_chunks")
                        row_count = int(cursor.fetchone()[0])
                        if row_count:
                            raise RuntimeError(
                                "embedding dimension changed from "
                                f"{stored_type or 'unknown'} to vector({dimension}); "
                                "export or delete the existing documents before rebuilding the index"
                            )
                        cursor.execute("DROP TABLE agentflow_chunks")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS agentflow_chunks (
                        id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding VECTOR({dimension}) NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cursor.execute(
                    "ALTER TABLE agentflow_chunks ADD COLUMN IF NOT EXISTS "
                    "metadata JSONB NOT NULL DEFAULT '{}'::jsonb"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS agentflow_chunks_document_id_idx "
                    "ON agentflow_chunks (document_id)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS agentflow_chunks_embedding_idx "
                    "ON agentflow_chunks USING hnsw (embedding vector_cosine_ops)"
                )

    def replace_document(
        self, document_id: str, source: str, chunks: list[Evidence]
    ) -> int:
        embeddings = encode_passages([chunk.text for chunk in chunks])
        return self.replace_document_embeddings(document_id, source, chunks, embeddings)

    def replace_document_embeddings(
        self,
        document_id: str,
        source: str,
        chunks: list[Evidence],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunk and embedding counts must match")
        if not chunks:
            return 0
        dimension = len(embeddings[0])
        if dimension <= 0 or any(len(values) != dimension for values in embeddings):
            raise ValueError("all embeddings must have the same positive dimension")
        self.ensure_schema(dimension)
        rows = [
            (
                chunk.id,
                document_id,
                _postgres_safe(source),
                _postgres_safe(chunk.text),
                json.dumps(
                    _postgres_safe(
                        chunk.model_dump(
                            exclude={"id", "source", "text", "score", "rank"}
                        )
                    ),
                    ensure_ascii=False,
                ),
                _vector_literal(embedding),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM agentflow_chunks WHERE document_id = %s",
                    (document_id,),
                )
                cursor.executemany(
                    """
                    INSERT INTO agentflow_chunks
                        (id, document_id, source, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector)
                    """,
                    rows,
                )
        return len(rows)

    def delete_document(self, document_id: str) -> int:
        self.ensure_schema()
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM agentflow_chunks WHERE document_id = %s",
                    (document_id,),
                )
                return cursor.rowcount

    def load_documents(self) -> list[Evidence]:
        self.ensure_schema()
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT id, document_id, source, content, metadata "
                    "FROM agentflow_chunks ORDER BY id"
                )
                return [
                    _evidence_from_row(row)
                    for row in cursor.fetchall()
                ]

    def vector_search(self, query: str, top_k: int) -> list[Evidence]:
        if top_k <= 0:
            return []
        query_embedding_values = encode_queries([query])[0]
        self.ensure_schema(len(query_embedding_values))
        query_embedding = _vector_literal(query_embedding_values)
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, document_id, source, content, metadata,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM agentflow_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, top_k),
                )
                return [
                    _evidence_from_row(row[:5]).model_copy(
                        update={"score": float(row[5]), "rank": rank}
                    )
                    for rank, row in enumerate(cursor.fetchall(), 1)
                ]


def _evidence_from_row(row) -> Evidence:
    metadata = row[4] or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return Evidence.model_validate(
        {
            **metadata,
            "id": row[0],
            "document_id": row[1],
            "source": row[2],
            "text": row[3],
        }
    )


class PgVectorResearchRetriever:
    """Independent BM25/vector recall, RRF fusion, and optional reranking."""

    def __init__(
        self,
        store: VectorSearchStore,
        *,
        method: str = "cross_encoder",
        candidate_top_k: int = 50,
        rerank_candidate_top_k: int = 30,
        reranker_model: str = RERANKER_MODEL_PATH,
    ) -> None:
        self.store = store
        self.candidate_top_k = candidate_top_k
        self.method = method
        self.rerank_candidate_top_k = rerank_candidate_top_k
        self.reranker_model = reranker_model
        self._cross_encoder = None
        self.keyword_retriever = SimpleBM25Retriever()
        self.refresh()

    def refresh(self) -> None:
        self.keyword_retriever.add_documents(self.store.load_documents())

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        if self.method not in {"bm25", "vector", "rrf", "rrf_expanded", "cross_encoder"}:
            raise ValueError(f"unsupported retrieval method: {self.method}")
        expanded_query = (
            expand_query(query)
            if self.method in {"rrf_expanded", "cross_encoder"}
            else query
        )
        keyword_results = self.keyword_retriever.search(
            expanded_query, top_k=self.candidate_top_k
        )
        vector_results = self.store.vector_search(
            expanded_query, top_k=self.candidate_top_k
        )
        if self.method == "bm25":
            return keyword_results[:top_k]
        if self.method == "vector":
            return vector_results[:top_k]
        fused = reciprocal_rank_fusion(
            [keyword_results, vector_results], top_k=self.candidate_top_k * 2
        )
        if self.method != "cross_encoder":
            return fused[:top_k]
        candidates = fused[: self.rerank_candidate_top_k]
        if not candidates:
            return []
        scores = self._get_cross_encoder().predict(
            [(expanded_query, item.text) for item in candidates],
            show_progress_bar=False,
        )
        ranked = sorted(
            zip(scores, candidates), key=lambda pair: float(pair[0]), reverse=True
        )
        return [
            candidate.model_copy(update={"score": float(score), "rank": rank})
            for rank, (score, candidate) in enumerate(ranked[:top_k], 1)
        ]

    def _get_cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.reranker_model)
        return self._cross_encoder
