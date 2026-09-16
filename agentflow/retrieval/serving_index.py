from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from contextlib import closing, contextmanager
from pathlib import Path

import numpy as np

from agentflow.retrieval.embedding import encode_passages
from agentflow.retrieval.store import load_documents, save_documents
from agentflow.retrieval.v2_corpus import ChunkingStrategy, build_text_chunks
from agentflow.retrieval.v2_retriever import build_v2_retriever
from agentflow.schemas import Evidence


class IndexConflictError(ValueError):
    pass


def active_v2_index_root(index_root: Path, strategy: str, model_key: str) -> Path:
    pointer = index_root / "_serving" / strategy / model_key / "active.json"
    if not pointer.exists():
        return index_root
    snapshot_id = json.loads(pointer.read_text(encoding="utf-8")).get("snapshot_id", "")
    if not re.fullmatch(r"[0-9a-f]{32}", snapshot_id):
        raise ValueError("invalid V2 serving index pointer")
    snapshot = pointer.parent / "versions" / snapshot_id
    if not snapshot.is_dir():
        raise FileNotFoundError(f"V2 serving index snapshot is missing: {snapshot_id}")
    return snapshot


class V2ServingIndexManager:
    def __init__(
        self,
        index_root: Path,
        *,
        strategy: ChunkingStrategy,
        model_key: str,
        model_source: str,
    ) -> None:
        self.index_root = index_root
        self.strategy = strategy
        self.model_key = model_key
        self.model_source = model_source

    def index_text(
        self,
        document_id: str,
        source: str,
        text: str,
        *,
        chunk_size: int = 500,
        overlap: int = 50,
        language: str | None = None,
        chunking_strategy: ChunkingStrategy | None = None,
    ) -> int:
        if chunking_strategy is not None and chunking_strategy != self.strategy:
            raise ValueError(
                "chunking_strategy must match the active serving index strategy: "
                f"{self.strategy}"
            )
        if self.strategy != "fixed_char" and (chunk_size, overlap) != (500, 50):
            raise ValueError("chunk_size and overlap apply only to fixed_char V2 indexes")
        base_documents = load_documents(self.index_root / self.strategy / "documents.jsonl")
        if any(item.document_id == document_id for item in base_documents):
            raise IndexConflictError("benchmark corpus documents are immutable")
        detected_language = language or ("zh" if re.search(r"[\u3400-\u9fff]", text) else "en")
        chunks = build_text_chunks(
            document_id,
            source,
            text,
            self.strategy,
            detected_language,
            fixed_size=chunk_size,
            fixed_overlap=overlap,
        )
        if not chunks:
            raise ValueError("document produced no non-empty chunks")
        new_vectors = np.asarray(
            encode_passages([item.text for item in chunks], model_name_or_path=self.model_source),
            dtype=np.float32,
        )
        if new_vectors.ndim != 2 or new_vectors.shape[0] != len(chunks):
            raise ValueError("embedding output does not match the new document chunks")
        if not np.allclose(np.linalg.norm(new_vectors, axis=1), 1.0, atol=1e-3):
            raise ValueError("embedding output must be normalized")

        with self._writer_lock():
            documents, vectors, managed = self._read_current()
            if new_vectors.shape[1] != vectors.shape[1]:
                raise ValueError("embedding dimension does not match the active V2 index")
            self._reject_base_document(document_id, documents, managed)
            keep = [i for i, item in enumerate(documents) if item.document_id != document_id]
            updated_documents = [documents[i] for i in keep] + chunks
            updated_vectors = np.concatenate((vectors[keep], new_vectors), axis=0)
            self._publish(updated_documents, updated_vectors, managed | {document_id})
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        with self._writer_lock():
            documents, vectors, managed = self._read_current()
            self._reject_base_document(document_id, documents, managed)
            if document_id not in managed:
                return 0
            keep = [i for i, item in enumerate(documents) if item.document_id != document_id]
            deleted = len(documents) - len(keep)
            self._publish([documents[i] for i in keep], vectors[keep], managed - {document_id})
        return deleted

    @contextmanager
    def _writer_lock(self):
        serving_dir = self.index_root / "_serving" / self.strategy / self.model_key
        serving_dir.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(serving_dir / "writers.sqlite3", timeout=60)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _read_current(self) -> tuple[list[Evidence], np.ndarray, set[str]]:
        root = active_v2_index_root(self.index_root, self.strategy, self.model_key)
        retriever = build_v2_retriever(
            root,
            strategy=self.strategy,
            model_key=self.model_key,
            model_source=self.model_source,
            method="vector",
            reranker_model="",
        )
        documents = load_documents(root / self.strategy / "documents.jsonl")
        vectors = retriever.vector_retriever.embeddings
        ids = retriever.vector_retriever.ids
        if ids != [item.id for item in documents] or len(vectors) != len(documents):
            raise ValueError("active V2 index is not aligned")
        managed_path = root / "managed_documents.json"
        managed = (
            set(json.loads(managed_path.read_text(encoding="utf-8")))
            if managed_path.exists() else set()
        )
        return documents, vectors, managed

    @staticmethod
    def _reject_base_document(document_id: str, documents: list[Evidence], managed: set[str]) -> None:
        if document_id not in managed and any(item.document_id == document_id for item in documents):
            raise IndexConflictError("benchmark corpus documents are immutable")

    def _publish(self, documents: list[Evidence], vectors: np.ndarray, managed: set[str]) -> None:
        serving_dir = self.index_root / "_serving" / self.strategy / self.model_key
        snapshot_id = uuid.uuid4().hex
        snapshot_root = serving_dir / "versions" / snapshot_id
        strategy_dir = snapshot_root / self.strategy
        model_dir = strategy_dir / self.model_key
        model_dir.mkdir(parents=True)
        document_path = strategy_dir / "documents.jsonl"
        save_documents(document_path, documents)
        np.save(model_dir / "vectors.npy", vectors)
        (model_dir / "vector_ids.json").write_text(
            json.dumps([item.id for item in documents], ensure_ascii=False), encoding="utf-8"
        )
        metadata = {
            "corpus_version": "2.0.0-serving",
            "chunking_strategy": self.strategy,
            "model_key": self.model_key,
            "document_index_sha256": hashlib.sha256(document_path.read_bytes()).hexdigest(),
            "document_count": len(documents),
            "embedding_dimension": int(vectors.shape[1]),
            "normalized": True,
        }
        (model_dir / "index_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (snapshot_root / "managed_documents.json").write_text(
            json.dumps(sorted(managed)), encoding="utf-8"
        )
        build_v2_retriever(
            snapshot_root,
            strategy=self.strategy,
            model_key=self.model_key,
            model_source=self.model_source,
            method="vector",
            reranker_model="",
        )
        temporary_pointer = serving_dir / f"active-{snapshot_id}.json"
        temporary_pointer.write_text(json.dumps({"snapshot_id": snapshot_id}), encoding="utf-8")
        os.replace(temporary_pointer, serving_dir / "active.json")
