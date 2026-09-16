from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from agentflow.schemas import Evidence


CorpusRole = Literal["development", "heldout"]
DocumentType = Literal["article", "review", "dataset"]
ChunkingStrategy = Literal["fixed_char", "recursive_token", "section_aware"]


class CorpusDocumentSpec(BaseModel):
    document_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,127}$")
    filename: str
    title: str
    year: int | None = None
    doi: str | None = None
    language: Literal["en", "zh"]
    topic: str
    source_url: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_role: CorpusRole
    document_type: DocumentType
    extraction_quality: Literal["verified", "warning", "failed"] = "verified"


class CorpusManifest(BaseModel):
    corpus_id: str
    version: str
    documents: list[CorpusDocumentSpec]
    excluded_files: list[dict[str, str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_documents(self) -> "CorpusManifest":
        for field in ("document_id", "sha256"):
            values = [getattr(item, field) for item in self.documents]
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate corpus {field}")
        normalized_titles = [_normalize_title(item.title) for item in self.documents]
        if len(normalized_titles) != len(set(normalized_titles)):
            raise ValueError("duplicate normalized corpus title")
        dois = [item.doi.casefold() for item in self.documents if item.doi]
        if len(dois) != len(set(dois)):
            raise ValueError("duplicate corpus DOI")
        return self


class PageText(BaseModel):
    page: int
    text: str


class ChunkingReport(BaseModel):
    document_id: str
    strategy: ChunkingStrategy
    chunk_count: int
    average_estimated_tokens: float
    minimum_estimated_tokens: int
    maximum_estimated_tokens: int
    short_chunk_count: int
    source_characters: int
    covered_characters: int


_TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z0-9]+(?:[-._/][A-Za-z0-9]+)*|[^\s]"
)
_SECTION_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?(?:abstract|introduction|background|related work|"
    r"materials and methods|methods?|results?|discussion|conclusions?|references|"
    r"摘要|引言|背景|相关工作|材料与方法|方法|结果|讨论|结论|参考文献)\s*$",
    re.IGNORECASE,
)


def load_manifest(path: Path) -> CorpusManifest:
    return CorpusManifest.model_validate_json(path.read_text(encoding="utf-8"))


def validate_corpus_files(
    manifest: CorpusManifest, corpus_dir: Path
) -> list[dict[str, str | int | bool]]:
    rows: list[dict[str, str | int | bool]] = []
    for spec in manifest.documents:
        path = corpus_dir / spec.filename
        exists = path.is_file()
        actual_hash = sha256_file(path) if exists else ""
        rows.append(
            {
                "document_id": spec.document_id,
                "filename": spec.filename,
                "exists": exists,
                "hash_matches": exists and actual_hash == spec.sha256,
                "bytes": path.stat().st_size if exists else 0,
            }
        )
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_pdf_pages(path: Path) -> list[PageText]:
    try:
        import pymupdf
    except ImportError as error:
        raise RuntimeError("V2 PDF parsing requires PyMuPDF") from error

    document = pymupdf.open(path)
    try:
        raw_pages = [page.get_text("text") for page in document]
    finally:
        document.close()
    return _remove_repeated_headers_and_footers(raw_pages)


def build_document_chunks(
    spec: CorpusDocumentSpec,
    path: Path,
    strategy: ChunkingStrategy,
    *,
    fixed_size: int = 500,
    fixed_overlap: int = 50,
    target_tokens: int = 300,
    overlap_tokens: int = 50,
    minimum_tokens: int = 80,
) -> tuple[list[Evidence], ChunkingReport]:
    pages = extract_pdf_pages(path)
    text, page_ranges = _combine_pages(pages)
    if not text.strip():
        raise ValueError(f"document has no extractable text: {path}")

    if strategy == "fixed_char":
        intervals = _fixed_intervals(len(text), fixed_size, fixed_overlap)
    elif strategy == "recursive_token":
        intervals = _token_intervals(text, target_tokens, overlap_tokens)
    elif strategy == "section_aware":
        intervals = _section_intervals(text, target_tokens, overlap_tokens)
    else:
        raise ValueError(f"unsupported chunking strategy: {strategy}")
    intervals = _merge_short_intervals(text, intervals, minimum_tokens)
    sections = _section_markers(text)

    chunks: list[Evidence] = []
    for index, (start, end) in enumerate(intervals):
        content = " ".join(text[start:end].split())
        if not content:
            continue
        chunks.append(
            Evidence(
                id=f"{spec.document_id}__{strategy}__{index}",
                source=spec.filename,
                text=content,
                document_id=spec.document_id,
                language=spec.language,
                page=_page_for_offset(start, page_ranges),
                section=_section_for_offset(start, sections),
                start_offset=start,
                end_offset=end,
                chunking_strategy=strategy,
            )
        )

    token_counts = [estimate_tokens(item.text) for item in chunks]
    report = ChunkingReport(
        document_id=spec.document_id,
        strategy=strategy,
        chunk_count=len(chunks),
        average_estimated_tokens=(
            round(sum(token_counts) / len(token_counts), 2) if token_counts else 0.0
        ),
        minimum_estimated_tokens=min(token_counts, default=0),
        maximum_estimated_tokens=max(token_counts, default=0),
        short_chunk_count=sum(count < minimum_tokens for count in token_counts),
        source_characters=len(text),
        covered_characters=_covered_characters(intervals),
    )
    return chunks, report


def build_text_chunks(
    document_id: str,
    source: str,
    text: str,
    strategy: ChunkingStrategy,
    language: Literal["en", "zh"],
    *,
    fixed_size: int = 500,
    fixed_overlap: int = 50,
) -> list[Evidence]:
    if not text.strip():
        raise ValueError("document has no text")
    if strategy == "fixed_char":
        intervals = _fixed_intervals(len(text), fixed_size, fixed_overlap)
    elif strategy == "recursive_token":
        intervals = _token_intervals(text, 300, 50)
    elif strategy == "section_aware":
        intervals = _section_intervals(text, 300, 50)
    else:
        raise ValueError(f"unsupported chunking strategy: {strategy}")
    intervals = _merge_short_intervals(text, intervals, 80)
    sections = _section_markers(text)
    return [
        Evidence(
            id=f"{document_id}__{strategy}__{index}",
            source=source,
            text=" ".join(text[start:end].split()),
            document_id=document_id,
            language=language,
            section=_section_for_offset(start, sections),
            start_offset=start,
            end_offset=end,
            chunking_strategy=strategy,
        )
        for index, (start, end) in enumerate(intervals)
        if text[start:end].strip()
    ]


def estimate_tokens(text: str) -> int:
    return len(_TOKEN_PATTERN.findall(text))


def _remove_repeated_headers_and_footers(raw_pages: list[str]) -> list[PageText]:
    page_lines = [[line.strip() for line in page.splitlines() if line.strip()] for page in raw_pages]
    edge_counts: Counter[str] = Counter()
    for lines in page_lines:
        for line in [*lines[:2], *lines[-2:]]:
            key = _normalize_repeated_line(line)
            if key:
                edge_counts[key] += 1
    threshold = max(3, math.ceil(len(page_lines) * 0.5))
    repeated = {line for line, count in edge_counts.items() if count >= threshold}

    pages: list[PageText] = []
    for page_number, lines in enumerate(page_lines, start=1):
        cleaned = [
            line
            for line in lines
            if _normalize_repeated_line(line) not in repeated
            and not re.fullmatch(r"(?:page\s*)?\d+", line, re.IGNORECASE)
        ]
        pages.append(PageText(page=page_number, text="\n".join(cleaned)))
    return pages


def _normalize_repeated_line(line: str) -> str:
    normalized = re.sub(r"\d+", "#", " ".join(line.casefold().split()))
    if len(normalized) < 4 or len(normalized) > 140:
        return ""
    return normalized


def _combine_pages(pages: list[PageText]) -> tuple[str, list[tuple[int, int, int]]]:
    parts: list[str] = []
    ranges: list[tuple[int, int, int]] = []
    cursor = 0
    for page in pages:
        normalized = "\n".join(
            " ".join(line.split()) for line in page.text.splitlines() if line.strip()
        )
        if not normalized:
            continue
        if parts:
            parts.append("\n")
            cursor += 1
        start = cursor
        parts.append(normalized)
        cursor += len(normalized)
        ranges.append((start, cursor, page.page))
    return "".join(parts), ranges


def _fixed_intervals(length: int, size: int, overlap: int) -> list[tuple[int, int]]:
    if size <= overlap:
        raise ValueError("fixed chunk size must be larger than overlap")
    return [(start, min(start + size, length)) for start in range(0, length, size - overlap)]


def _token_intervals(text: str, target: int, overlap: int) -> list[tuple[int, int]]:
    if target <= overlap:
        raise ValueError("target tokens must be larger than overlap")
    tokens = list(_TOKEN_PATTERN.finditer(text))
    if not tokens:
        return []
    intervals: list[tuple[int, int]] = []
    start_token = 0
    while start_token < len(tokens):
        end_token = min(start_token + target, len(tokens))
        if end_token < len(tokens):
            lower = start_token + max(1, int(target * 0.7))
            for candidate in range(end_token - 1, lower - 1, -1):
                if tokens[candidate].group(0) in {".", "!", "?", "。", "！", "？", ";", "；"}:
                    end_token = candidate + 1
                    break
        intervals.append((tokens[start_token].start(), tokens[end_token - 1].end()))
        if end_token == len(tokens):
            break
        start_token = max(start_token + 1, end_token - overlap)
    return intervals


def _section_intervals(text: str, target: int, overlap: int) -> list[tuple[int, int]]:
    markers = _section_markers(text)
    boundaries = [0, *[offset for offset, _ in markers if offset > 0], len(text)]
    boundaries = sorted(set(boundaries))
    intervals: list[tuple[int, int]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        section_text = text[start:end]
        intervals.extend(
            (start + local_start, start + local_end)
            for local_start, local_end in _token_intervals(section_text, target, overlap)
        )
    return intervals


def _section_markers(text: str) -> list[tuple[int, str]]:
    markers: list[tuple[int, str]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        value = line.strip()
        if value and len(value) <= 80 and _SECTION_PATTERN.fullmatch(value):
            markers.append((cursor, value))
        cursor += len(line)
    return markers


def _merge_short_intervals(
    text: str, intervals: list[tuple[int, int]], minimum_tokens: int
) -> list[tuple[int, int]]:
    if len(intervals) < 2:
        return intervals
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if estimate_tokens(text[start:end]) >= minimum_tokens or not merged:
            merged.append((start, end))
            continue
        previous_start, _ = merged[-1]
        merged[-1] = (previous_start, end)
    if len(merged) > 1 and estimate_tokens(text[merged[0][0] : merged[0][1]]) < minimum_tokens:
        first_start, first_end = merged.pop(0)
        _, next_end = merged[0]
        merged[0] = (first_start, max(first_end, next_end))
    return merged


def _page_for_offset(offset: int, ranges: list[tuple[int, int, int]]) -> int | None:
    for start, end, page in ranges:
        if start <= offset < end:
            return page
    return ranges[-1][2] if ranges else None


def _section_for_offset(offset: int, markers: list[tuple[int, str]]) -> str:
    section = ""
    for marker_offset, name in markers:
        if marker_offset > offset:
            break
        section = name
    return section


def _covered_characters(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _normalize_title(value: str) -> str:
    return re.sub(r"[^0-9a-z\u3400-\u4dbf\u4e00-\u9fff]+", "", value.casefold())
