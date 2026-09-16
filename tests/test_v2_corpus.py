from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pymupdf

from agentflow.retrieval.embedding import prepare_queries
from agentflow.retrieval.v2_corpus import CorpusDocumentSpec, build_document_chunks


def _spec(filename: str) -> CorpusDocumentSpec:
    return CorpusDocumentSpec(
        document_id="test-document",
        filename=filename,
        title="Test Document",
        year=2026,
        language="en",
        topic="test",
        source_url="https://example.com/test",
        sha256="0" * 64,
        document_role="development",
        document_type="article",
        extraction_quality="verified",
    )


def test_v2_chunking_preserves_metadata_and_merges_short_tail() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "sample.pdf"
        document = pymupdf.open()
        for page_number in range(2):
            page = document.new_page()
            body = "Methods\n" + "Protein flexibility evidence is measured here. " * 130
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 790), body, fontsize=9)
        document.save(path)
        document.close()

        chunks, report = build_document_chunks(
            _spec(path.name),
            path,
            strategy="recursive_token",
            target_tokens=120,
            overlap_tokens=20,
            minimum_tokens=80,
        )

    assert chunks
    assert all(chunk.document_id == "test-document" for chunk in chunks)
    assert all(chunk.chunking_strategy == "recursive_token" for chunk in chunks)
    assert all(chunk.page in {1, 2} for chunk in chunks)
    assert report.short_chunk_count == 0
    assert report.covered_characters / report.source_characters > 0.99


def test_embedding_query_instructions_are_model_specific() -> None:
    query = "protein flexibility"
    assert prepare_queries([query], model_name_or_path="bge-small-en-v1.5")[0].endswith(query)
    assert prepare_queries([query], model_name_or_path="bge-small-en-v1.5")[0] != query
    assert prepare_queries([query], model_name_or_path="bge-m3") == [query]
