from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentflow.retrieval.embedding import encode_passages
from agentflow.retrieval.store import load_documents


MODEL_KEYS = ("bge-small-zh-v1.5", "bge-small-en-v1.5", "bge-m3")
STRATEGIES = ("fixed_char", "recursive_token", "section_aware")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one isolated AgentFlow V2 vector index.")
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--model", required=True, choices=MODEL_KEYS)
    parser.add_argument(
        "--model-path",
        help="Optional local model directory. The public model identity remains --model.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source_dir = PROJECT_ROOT / "data" / "index" / "v2" / args.strategy
    documents_path = source_dir / "documents.jsonl"
    output_dir = source_dir / args.model
    vectors_path = output_dir / "vectors.npy"
    ids_path = output_dir / "vector_ids.json"
    metadata_path = output_dir / "index_metadata.json"
    if metadata_path.exists() and not args.force:
        raise SystemExit(f"Index already exists: {metadata_path}. Use --force to rebuild it.")

    documents = load_documents(documents_path)
    if not documents:
        raise SystemExit(f"Missing V2 documents: {documents_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_source = args.model_path or args.model
    started = time.perf_counter()
    vectors = encode_passages(
        [document.text for document in documents],
        model_name_or_path=model_source,
    )
    elapsed = time.perf_counter() - started
    if vectors.ndim != 2 or vectors.shape[0] != len(documents):
        raise RuntimeError("Embedding output shape does not match the document index.")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError("Embedding model returned vectors that are not normalized.")

    np.save(vectors_path, vectors)
    ids_path.write_text(
        json.dumps([document.id for document in documents], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "corpus_version": "2.0.0",
        "chunking_strategy": args.strategy,
        "model_key": args.model,
        "model_source": args.model if not args.model_path else "local_override",
        "document_index_sha256": sha256(documents_path),
        "document_count": len(documents),
        "embedding_dimension": int(vectors.shape[1]),
        "normalized": True,
        "build_seconds": round(elapsed, 3),
        "vectors_bytes": vectors_path.stat().st_size,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
