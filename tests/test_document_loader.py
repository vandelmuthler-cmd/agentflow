from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import agentflow.retrieval.index as retrieval_index
from agentflow.retrieval.document_loader import iter_supported_files


def test_raw_readme_is_not_indexed() -> None:
    with TemporaryDirectory() as temporary_dir:
        raw_dir = Path(temporary_dir)
        (raw_dir / "README.md").write_text("setup instructions", encoding="utf-8")
        (raw_dir / "paper.md").write_text("source content", encoding="utf-8")

        assert [path.name for path in iter_supported_files(raw_dir)] == ["paper.md"]


def test_missing_corpus_fails_with_setup_instruction() -> None:
    with TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        raw_dir = root / "raw"
        raw_dir.mkdir()
        (raw_dir / "README.md").write_text("setup instructions", encoding="utf-8")
        original_index_path = retrieval_index.DOCUMENT_INDEX_PATH
        retrieval_index.DOCUMENT_INDEX_PATH = root / "missing-index.jsonl"
        try:
            try:
                retrieval_index.load_or_build_documents(raw_dir)
            except FileNotFoundError as error:
                assert "README.md" in str(error)
            else:
                raise AssertionError("missing corpus should fail explicitly")
        finally:
            retrieval_index.DOCUMENT_INDEX_PATH = original_index_path
