from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks.

    The implementation uses deterministic fixed-size chunks so indexing and
    retrieval benchmarks remain reproducible.
    """
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    step = chunk_size - overlap
    return [
        normalized[i : i + chunk_size]
        for i in range(0, len(normalized), step)
        if normalized[i : i + chunk_size].strip()
    ]
