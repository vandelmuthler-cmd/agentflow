from __future__ import annotations

from pydantic import ValidationError

import agentflow.api.app as api_module
from agentflow.retrieval.management import RetrievalIndexManager, build_document_chunks
from agentflow.retrieval.pgvector import PgVectorResearchRetriever
from agentflow.schemas import DocumentIndexRequest, Evidence


class FakePgVectorStore:
    def __init__(self) -> None:
        self.documents = [
            Evidence(id="doc-a__0", source="a.txt", text="shared retrieval evidence alpha"),
            Evidence(id="doc-b__0", source="b.txt", text="shared retrieval evidence beta"),
        ]
        self.replaced: tuple[str, str, list[Evidence]] | None = None
        self.deleted = ""

    def load_documents(self) -> list[Evidence]:
        return self.documents

    def vector_search(self, _query: str, top_k: int) -> list[Evidence]:
        return [
            self.documents[1].model_copy(update={"score": 1.0, "rank": 1}),
            self.documents[0].model_copy(update={"score": 0.2, "rank": 2}),
        ][:top_k]

    def replace_document(
        self, document_id: str, source: str, chunks: list[Evidence]
    ) -> int:
        self.replaced = (document_id, source, chunks)
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        self.deleted = document_id
        return 2


def test_document_chunks_have_stable_document_prefix() -> None:
    chunks = build_document_chunks(
        "policy-v2", "policy.md", "A" * 220, chunk_size=100, overlap=20
    )

    assert [chunk.id for chunk in chunks] == ["policy-v2__section_aware__0"]
    assert chunks[0].chunking_strategy == "section_aware"


def test_pgvector_retriever_fuses_independent_keyword_and_vector_recall() -> None:
    store = FakePgVectorStore()
    retriever = PgVectorResearchRetriever(
        store, method="rrf_expanded", candidate_top_k=2
    )

    results = retriever.search("shared retrieval evidence", top_k=2)

    assert {item.id for item in results} == {"doc-a__0", "doc-b__0"}


def test_pgvector_index_manager_replaces_and_deletes_one_document() -> None:
    store = FakePgVectorStore()
    manager = RetrievalIndexManager(backend="pgvector", pgvector_store=store)

    indexed = manager.index_text(
        "manual", "manual.md", "B" * 150, chunk_size=100, overlap=20
    )
    deleted = manager.delete_document("manual")

    assert indexed == 1
    assert store.replaced is not None
    assert store.replaced[0:2] == ("manual", "manual.md")
    assert store.deleted == "manual"
    assert deleted == 2


def test_document_index_request_rejects_invalid_chunk_window() -> None:
    try:
        DocumentIndexRequest(
            document_id="manual",
            source="manual.md",
            text="content",
            chunk_size=100,
            overlap=100,
        )
    except ValidationError:
        return
    raise AssertionError("overlap must be smaller than chunk_size")


def test_document_management_is_disabled_without_admin_key() -> None:
    original = api_module.ADMIN_API_KEY
    api_module.ADMIN_API_KEY = ""
    try:
        try:
            api_module.require_admin_key("candidate")
        except Exception as error:
            assert getattr(error, "status_code", None) == 503
        else:
            raise AssertionError("document management should be disabled")
    finally:
        api_module.ADMIN_API_KEY = original


def test_retrieval_api_routes_are_registered() -> None:
    paths = api_module.app.openapi()["paths"]

    assert "/retrieval/search" in paths
    assert "/retrieval/index" in paths
    assert "/retrieval/documents/{document_id}" in paths
