from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import agentflow.api.app as api_module
import agentflow.retrieval.index as index_module
from agentflow.retrieval.management import RetrievalIndexManager
from agentflow.retrieval.serving_index import (
    IndexConflictError,
    V2ServingIndexManager,
    active_v2_index_root,
)
from agentflow.retrieval.store import load_documents, save_documents
from agentflow.retrieval.v2_retriever import build_v2_retriever
from agentflow.schemas import Evidence


STRATEGY = "section_aware"
MODEL = "bge-m3"


def _seed_base(root: Path) -> Path:
    strategy_dir = root / STRATEGY
    model_dir = strategy_dir / MODEL
    model_dir.mkdir(parents=True)
    document_path = strategy_dir / "documents.jsonl"
    save_documents(
        document_path,
        [Evidence(
            id="base__section_aware__0",
            document_id="base",
            source="base.pdf",
            text="alpha baseline reference",
            language="en",
            chunking_strategy=STRATEGY,
        )],
    )
    np.save(model_dir / "vectors.npy", np.array([[1.0, 0.0]], dtype=np.float32))
    (model_dir / "vector_ids.json").write_text(
        json.dumps(["base__section_aware__0"]), encoding="utf-8"
    )
    (model_dir / "index_metadata.json").write_text(
        json.dumps({
            "chunking_strategy": STRATEGY,
            "model_key": MODEL,
            "document_index_sha256": hashlib.sha256(document_path.read_bytes()).hexdigest(),
        }),
        encoding="utf-8",
    )
    return document_path


def _fake_passages(texts: list[str], **_kwargs) -> np.ndarray:
    return np.array(
        [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts],
        dtype=np.float32,
    )


def _fake_queries(texts: list[str], **_kwargs) -> np.ndarray:
    return _fake_passages(texts)


def _manager(root: Path) -> V2ServingIndexManager:
    return V2ServingIndexManager(root, strategy=STRATEGY, model_key=MODEL, model_source=MODEL)


def _retriever(root: Path):
    return build_v2_retriever(
        active_v2_index_root(root, STRATEGY, MODEL),
        strategy=STRATEGY,
        model_key=MODEL,
        model_source=MODEL,
        method="vector",
        reranker_model="",
    )


def test_v2_serving_lifecycle_preserves_benchmark_and_survives_reload() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        base_path = _seed_base(root)
        original_hash = hashlib.sha256(base_path.read_bytes()).hexdigest()
        manager = _manager(root)
        with patch("agentflow.retrieval.serving_index.encode_passages", _fake_passages), patch(
            "agentflow.retrieval.vector.encode_queries", _fake_queries
        ):
            assert manager.index_text("manual", "manual.md", "zeta document content") == 1
            assert _retriever(root).search("zeta", top_k=1)[0].document_id == "manual"
            with patch.object(index_module, "CORPUS_VERSION", "v2"), patch.object(
                index_module, "RETRIEVAL_BACKEND", "local"
            ), patch.object(index_module, "V2_INDEX_ROOT", root), patch.object(
                index_module, "V2_CHUNKING_STRATEGY", STRATEGY
            ), patch.object(index_module, "V2_EMBED_MODEL", MODEL), patch.object(
                index_module, "V2_RETRIEVAL_METHOD", "vector"
            ):
                result = index_module.build_retriever(root).search("zeta", top_k=1)
                assert result[0].document_id == "manual"
            assert _manager(root).index_text("manual", "manual.md", "zeta replacement text") == 1
            documents = load_documents(
                active_v2_index_root(root, STRATEGY, MODEL) / STRATEGY / "documents.jsonl"
            )
            assert len(documents) == 2
            assert any(item.text == "zeta replacement text" for item in documents)
            assert _manager(root).delete_document("manual") == 1
            assert [item.document_id for item in _retriever(root).search("alpha", top_k=2)] == ["base"]
            assert _manager(root).delete_document("manual") == 0
        assert hashlib.sha256(base_path.read_bytes()).hexdigest() == original_hash


def test_v2_serving_rejects_base_mutation_and_keeps_pointer_on_failure() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        _seed_base(root)
        manager = _manager(root)
        with patch("agentflow.retrieval.serving_index.encode_passages", _fake_passages):
            try:
                manager.index_text("base", "base.pdf", "zeta replacement")
            except IndexConflictError:
                pass
            else:
                raise AssertionError("benchmark documents must be immutable")
            assert active_v2_index_root(root, STRATEGY, MODEL) == root
            manager.index_text("manual", "manual.md", "zeta document")
            current = active_v2_index_root(root, STRATEGY, MODEL)
            with patch("agentflow.retrieval.serving_index.os.replace", side_effect=OSError("disk failure")):
                try:
                    manager.index_text("manual", "manual.md", "zeta changed")
                except OSError:
                    pass
                else:
                    raise AssertionError("pointer swap failure should propagate")
            assert active_v2_index_root(root, STRATEGY, MODEL) == current
            current_documents = load_documents(current / STRATEGY / "documents.jsonl")
            assert any(item.text == "zeta document" for item in current_documents)


def test_v2_serving_serializes_concurrent_writers() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        _seed_base(root)
        with patch("agentflow.retrieval.serving_index.encode_passages", _fake_passages):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(_manager(root).index_text, name, f"{name}.md", "zeta content")
                    for name in ("first", "second")
                ]
                assert [future.result() for future in futures] == [1, 1]
        documents = load_documents(
            active_v2_index_root(root, STRATEGY, MODEL) / STRATEGY / "documents.jsonl"
        )
        assert {item.document_id for item in documents} == {"base", "first", "second"}


def test_v2_serving_http_lifecycle() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        _seed_base(root)
        manager = _manager(root)
        api_module.close_workflow()
        api_module.create_retriever.cache_clear()
        with patch("agentflow.retrieval.serving_index.encode_passages", _fake_passages), patch(
            "agentflow.retrieval.vector.encode_queries", _fake_queries
        ), patch.object(api_module, "ADMIN_API_KEY", "test-admin"), patch.object(
            api_module, "RetrievalIndexManager", return_value=manager
        ), patch.object(api_module, "build_retriever", side_effect=lambda _raw: _retriever(root)):
            with TestClient(api_module.app) as client:
                payload = {"document_id": "manual", "source": "manual.md", "text": "zeta content"}
                assert client.post("/retrieval/index", json=payload).status_code == 401
                headers = {"x-agentflow-admin-key": "test-admin"}
                assert client.post("/retrieval/index", json=payload, headers=headers).status_code == 200
                repeated = client.post("/retrieval/index", json=payload, headers=headers)
                assert repeated.json()["affected_chunks"] == 1
                response = client.post("/retrieval/search", json={"query": "zeta query", "top_k": 1})
                assert response.status_code == 200
                assert response.json()["evidence"][0]["document_id"] == "manual"
                payload["text"] = "zeta replacement"
                assert client.post("/retrieval/index", json=payload, headers=headers).status_code == 200
                assert client.delete("/retrieval/documents/base", headers=headers).status_code == 409
                deleted = client.delete("/retrieval/documents/manual", headers=headers)
                assert deleted.json()["affected_chunks"] == 1
                response = client.post("/retrieval/search", json={"query": "alpha query", "top_k": 1})
                assert response.json()["evidence"][0]["document_id"] == "base"
        api_module.create_retriever.cache_clear()


def test_v1_local_document_management_is_unchanged() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        manager = RetrievalIndexManager(
            backend="local",
            corpus_version="v1",
            document_index_path=root / "documents.jsonl",
            vector_index_path=root / "vectors.npy",
            vector_ids_path=root / "ids.json",
        )
        with patch("agentflow.retrieval.management.save_vector_index"):
            assert manager.index_text("manual", "manual.md", "A" * 150, chunk_size=100, overlap=20) == 2
            assert len(load_documents(root / "documents.jsonl")) == 2
            assert manager.delete_document("manual") == 2
            assert load_documents(root / "documents.jsonl") == []


def test_v2_pgvector_configuration_fails_explicitly() -> None:
    try:
        RetrievalIndexManager(backend="pgvector", corpus_version="v2")
    except ValueError as error:
        assert "requires the local serving index" in str(error)
    else:
        raise AssertionError("V2 must not silently write to an unrelated pgvector index")
