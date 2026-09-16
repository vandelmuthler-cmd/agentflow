from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agentflow.config import VECTOR_IDS_PATH, VECTOR_INDEX_PATH
from agentflow.retrieval.embedding import encode_passages, encode_queries
from agentflow.schemas import Evidence


def save_vector_index(
    documents: list[Evidence],
    vector_path: Path = VECTOR_INDEX_PATH,
    ids_path: Path = VECTOR_IDS_PATH,
    *,
    model_name_or_path: str | None = None,
) -> None:
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = encode_passages(
        [document.text for document in documents],
        model_name_or_path=model_name_or_path,
    )
    np.save(vector_path, embeddings)
    ids_path.write_text(
        json.dumps([document.id for document in documents], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_vector_index(vector_path: Path = VECTOR_INDEX_PATH, ids_path: Path = VECTOR_IDS_PATH) -> tuple[np.ndarray, list[str]]:
    if not vector_path.exists() or not ids_path.exists():
        return np.empty((0, 0), dtype=np.float32), []
    embeddings = np.load(vector_path)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    return embeddings, ids


class VectorRetriever:
    def __init__(
        self,
        documents: list[Evidence],
        embeddings: np.ndarray,
        ids: list[str],
        *,
        model_name_or_path: str | None = None,
        model_key: str | None = None,
    ) -> None:
        self.documents = {document.id: document for document in documents}
        self.embeddings = embeddings
        self.ids = ids
        self.model_name_or_path = model_name_or_path
        self.model_key = model_key

    @classmethod
    def from_index(
        cls,
        documents: list[Evidence],
        vector_path: Path = VECTOR_INDEX_PATH,
        ids_path: Path = VECTOR_IDS_PATH,
        *,
        model_name_or_path: str | None = None,
        model_key: str | None = None,
    ) -> "VectorRetriever":
        embeddings, ids = load_vector_index(vector_path, ids_path)
        if len(ids) != len(embeddings):
            return cls(
                documents,
                np.empty((0, 0), dtype=np.float32),
                [],
                model_name_or_path=model_name_or_path,
                model_key=model_key,
            )
        return cls(
            documents,
            embeddings,
            ids,
            model_name_or_path=model_name_or_path,
            model_key=model_key,
        )

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        if self.embeddings.size == 0 or not self.ids:
            return []
        query_embedding = encode_queries(
            [query],
            model_name_or_path=self.model_name_or_path,
            model_key=self.model_key,
        )[0]
        if query_embedding.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                "Query embedding dimension does not match the selected vector index."
            )
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]
        results: list[Evidence] = []
        for rank, idx in enumerate(top_indices, 1):
            doc_id = self.ids[int(idx)]
            document = self.documents.get(doc_id)
            if document is None:
                continue
            results.append(
                document.model_copy(update={"score": float(scores[int(idx)]), "rank": rank})
            )
        return results
