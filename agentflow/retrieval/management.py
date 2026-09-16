from __future__ import annotations

from pathlib import Path

from agentflow.config import (
    CORPUS_VERSION,
    DATABASE_URL,
    DOCUMENT_INDEX_PATH,
    ONLINE_CHUNKING_STRATEGY,
    RETRIEVAL_BACKEND,
    VECTOR_IDS_PATH,
    VECTOR_INDEX_PATH,
    V2_CHUNKING_STRATEGY,
    V2_EMBED_MODEL,
    V2_EMBED_MODEL_PATH,
    V2_INDEX_ROOT,
)
from agentflow.retrieval.pgvector import PgVectorStore
from agentflow.retrieval.serving_index import V2ServingIndexManager
from agentflow.retrieval.store import load_documents, save_documents
from agentflow.retrieval.vector import save_vector_index
from agentflow.retrieval.v2_corpus import build_text_chunks
from agentflow.schemas import Evidence


def build_document_chunks(
    document_id: str,
    source: str,
    text: str,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
    language: str | None = None,
    chunking_strategy: str = ONLINE_CHUNKING_STRATEGY,
) -> list[Evidence]:
    resolved_language = language or _detect_language(text)
    return build_text_chunks(
        document_id,
        source,
        text,
        chunking_strategy,
        resolved_language,
        fixed_size=chunk_size,
        fixed_overlap=overlap,
    )


def _detect_language(text: str) -> str:
    sample = text[:4000]
    cjk_count = sum("\u4e00" <= character <= "\u9fff" for character in sample)
    return "zh" if cjk_count >= max(1, len(sample) // 20) else "en"


class RetrievalIndexManager:
    def __init__(
        self,
        backend: str = RETRIEVAL_BACKEND,
        *,
        database_url: str = DATABASE_URL,
        document_index_path: Path = DOCUMENT_INDEX_PATH,
        vector_index_path: Path = VECTOR_INDEX_PATH,
        vector_ids_path: Path = VECTOR_IDS_PATH,
        pgvector_store=None,
        corpus_version: str = CORPUS_VERSION,
        v2_index_root: Path = V2_INDEX_ROOT,
        v2_strategy: str = V2_CHUNKING_STRATEGY,
        v2_model_key: str = V2_EMBED_MODEL,
        v2_model_source: str = V2_EMBED_MODEL_PATH or V2_EMBED_MODEL,
    ) -> None:
        self.backend = backend
        self.corpus_version = corpus_version
        self.document_index_path = document_index_path
        self.vector_index_path = vector_index_path
        self.vector_ids_path = vector_ids_path
        self.pgvector_store = pgvector_store
        self.v2_manager = (
            V2ServingIndexManager(
                v2_index_root,
                strategy=v2_strategy,
                model_key=v2_model_key,
                model_source=v2_model_source,
            )
            if corpus_version == "v2" and backend == "local" else None
        )
        if backend == "pgvector" and self.pgvector_store is None:
            self.pgvector_store = PgVectorStore(database_url)
        if backend not in {"local", "pgvector"}:
            raise ValueError(f"unsupported retrieval backend: {backend}")

    def index_text(
        self,
        document_id: str,
        source: str,
        text: str,
        *,
        chunk_size: int = 500,
        overlap: int = 50,
        language: str | None = None,
        chunking_strategy: str = ONLINE_CHUNKING_STRATEGY,
    ) -> int:
        if self.v2_manager is not None:
            return self.v2_manager.index_text(
                document_id, source, text,
                chunk_size=chunk_size, overlap=overlap, language=language,
            )
        chunks = build_document_chunks(
            document_id,
            source,
            text,
            chunk_size=chunk_size,
            overlap=overlap,
            language=language,
            chunking_strategy=chunking_strategy,
        )
        if not chunks:
            raise ValueError("document produced no non-empty chunks")
        if self.backend == "pgvector":
            return self.pgvector_store.replace_document(document_id, source, chunks)

        documents = self._without_document(
            load_documents(self.document_index_path), document_id
        )
        documents.extend(chunks)
        self._save_local(documents)
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        if self.v2_manager is not None:
            return self.v2_manager.delete_document(document_id)
        if self.backend == "pgvector":
            return self.pgvector_store.delete_document(document_id)

        documents = load_documents(self.document_index_path)
        remaining = self._without_document(documents, document_id)
        deleted = len(documents) - len(remaining)
        if deleted:
            self._save_local(remaining)
        return deleted

    def _save_local(self, documents: list[Evidence]) -> None:
        save_documents(self.document_index_path, documents)
        save_vector_index(
            documents,
            vector_path=self.vector_index_path,
            ids_path=self.vector_ids_path,
        )

    @staticmethod
    def _without_document(
        documents: list[Evidence], document_id: str
    ) -> list[Evidence]:
        prefix = f"{document_id}__"
        return [
            document for document in documents if not document.id.startswith(prefix)
        ]
