from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Protocol

from agentflow.retrieval.bm25 import SimpleBM25Retriever
from agentflow.retrieval.embedding import embed_texts
from agentflow.retrieval.query_expansion import expand_query
from agentflow.retrieval.rerank import minmax
from agentflow.schemas import Evidence


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in values) + "]"


class VectorSearchStore(Protocol):
    def load_documents(self) -> list[Evidence]: ...

    def similarity_scores(self, query: str, ids: list[str]) -> dict[str, float]: ...


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
        dimension = embedding_dimension or int(embed_texts(["dimension probe"]).shape[1])
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS agentflow_chunks (
                        id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        content TEXT NOT NULL,
                        embedding VECTOR({dimension}) NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
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
        embeddings = embed_texts([chunk.text for chunk in chunks])
        self.ensure_schema(int(embeddings.shape[1]))
        rows = [
            (
                chunk.id,
                document_id,
                source,
                chunk.text,
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
                        (id, document_id, source, content, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
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
                    "SELECT id, source, content FROM agentflow_chunks ORDER BY id"
                )
                return [
                    Evidence(id=row[0], source=row[1], text=row[2])
                    for row in cursor.fetchall()
                ]

    def similarity_scores(self, query: str, ids: list[str]) -> dict[str, float]:
        if not ids:
            return {}
        query_embedding = _vector_literal(embed_texts([query])[0])
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, 1 - (embedding <=> %s::vector) AS similarity
                    FROM agentflow_chunks
                    WHERE id = ANY(%s)
                    """,
                    (query_embedding, ids),
                )
                return {row[0]: float(row[1]) for row in cursor.fetchall()}


class PgVectorResearchRetriever:
    """BM25 candidate retrieval with pgvector-backed semantic reranking."""

    def __init__(
        self,
        store: VectorSearchStore,
        *,
        candidate_top_k: int = 20,
        vector_weight: float = 0.4,
    ) -> None:
        self.store = store
        self.candidate_top_k = candidate_top_k
        self.vector_weight = vector_weight
        self.keyword_retriever = SimpleBM25Retriever()
        self.refresh()

    def refresh(self) -> None:
        self.keyword_retriever.add_documents(self.store.load_documents())

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        expanded_query = expand_query(query)
        candidates = self.keyword_retriever.search(
            expanded_query, top_k=self.candidate_top_k
        )
        if not candidates:
            return []
        similarities = self.store.similarity_scores(
            expanded_query, [candidate.id for candidate in candidates]
        )
        lexical_scores = minmax([candidate.score for candidate in candidates])
        vector_scores = minmax(
            [similarities.get(candidate.id, 0.0) for candidate in candidates]
        )
        ranked = sorted(
            (
                (
                    (1.0 - self.vector_weight) * lexical_score
                    + self.vector_weight * vector_score,
                    candidate,
                )
                for candidate, lexical_score, vector_score in zip(
                    candidates, lexical_scores, vector_scores
                )
            ),
            key=lambda item: item[0],
            reverse=True,
        )[:top_k]
        return [
            candidate.model_copy(update={"score": score, "rank": rank})
            for rank, (score, candidate) in enumerate(ranked, 1)
        ]
