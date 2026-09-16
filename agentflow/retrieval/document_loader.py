from __future__ import annotations

from pathlib import Path


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}
IGNORED_FILENAMES = {"readme.md"}


def load_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return _load_pdf_text(path)
    raise ValueError(f"unsupported document type: {path.suffix}")


def iter_supported_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    return [
        path
        for path in sorted(raw_dir.iterdir())
        if path.is_file()
        and path.name.lower() not in IGNORED_FILENAMES
        and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]


def _load_pdf_text(path: Path) -> str:
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError(
            "PDF parsing requires PyMuPDF. Install it with `pip install pymupdf`."
        ) from exc

    doc = pymupdf.open(path)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()
