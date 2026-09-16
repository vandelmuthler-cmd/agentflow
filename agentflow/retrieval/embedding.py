from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from agentflow.config import EMBED_MODEL_PATH


MODEL_ALIASES = {
    "bge-small-zh-v1.5": "BAAI/bge-small-zh-v1.5",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "bge-m3": "BAAI/bge-m3",
}

QUERY_INSTRUCTIONS = {
    "bge-small-zh-v1.5": "为这个句子生成表示以用于检索相关文章：",
    "bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "bge-m3": "",
}


@lru_cache(maxsize=1)
def get_embedding_model() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Embedding requires `sentence-transformers`. Install project dependencies first."
        ) from error
    return SentenceTransformer(str(EMBED_MODEL_PATH))


@lru_cache(maxsize=4)
def get_named_embedding_model(model_name_or_path: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Embedding requires `sentence-transformers`. Install project dependencies first."
        ) from error
    resolved = MODEL_ALIASES.get(model_name_or_path, model_name_or_path)
    return SentenceTransformer(str(resolved))


def encode_queries(
    texts: list[str],
    *,
    model_name_or_path: str | None = None,
    model_key: str | None = None,
    language: str = "",
) -> np.ndarray:
    del language
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    if model_name_or_path is None:
        model = get_embedding_model()
    else:
        model = get_named_embedding_model(model_name_or_path)
    prepared = prepare_queries(
        texts, model_name_or_path=model_name_or_path, model_key=model_key
    )
    return _encode(model, prepared)


def prepare_queries(
    texts: list[str],
    *,
    model_name_or_path: str | None = None,
    model_key: str | None = None,
) -> list[str]:
    """Apply a model-specific retrieval instruction without loading the model."""
    instruction = QUERY_INSTRUCTIONS.get(model_key or model_name_or_path or "", "")
    return [f"{instruction}{value}" if instruction else value for value in texts]


def encode_passages(
    texts: list[str],
    *,
    model_name_or_path: str | None = None,
    language: str = "",
) -> np.ndarray:
    del language
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    model = (
        get_embedding_model()
        if model_name_or_path is None
        else get_named_embedding_model(model_name_or_path)
    )
    return _encode(model, texts)


def _encode(model: Any, texts: list[str]) -> np.ndarray:
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Backward-compatible V1 passage encoder."""
    return encode_passages(texts)
