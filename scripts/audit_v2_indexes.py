from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentflow.retrieval.store import load_documents


MODELS = ("bge-small-zh-v1.5", "bge-small-en-v1.5", "bge-m3")
STRATEGIES = ("fixed_char", "recursive_token", "section_aware")


def main() -> None:
    rows = []
    errors = []
    for strategy in STRATEGIES:
        strategy_dir = PROJECT_ROOT / "data/index/v2" / strategy
        documents_path = strategy_dir / "documents.jsonl"
        documents = load_documents(documents_path)
        digest = hashlib.sha256(documents_path.read_bytes()).hexdigest()
        expected_ids = [item.id for item in documents]
        for model in MODELS:
            model_dir = strategy_dir / model
            required = [model_dir / "vectors.npy", model_dir / "vector_ids.json", model_dir / "index_metadata.json"]
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                errors.append(f"{strategy}/{model}: missing {missing}")
                continue
            vectors = np.load(required[0], mmap_mode="r")
            ids = json.loads(required[1].read_text(encoding="utf-8"))
            metadata = json.loads(required[2].read_text(encoding="utf-8"))
            normalized = bool(np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3))
            valid = (
                ids == expected_ids
                and vectors.shape[0] == len(documents)
                and metadata.get("document_index_sha256") == digest
                and metadata.get("chunking_strategy") == strategy
                and metadata.get("model_key") == model
                and normalized
            )
            if not valid:
                errors.append(f"{strategy}/{model}: metadata, IDs, shape, or normalization mismatch")
            rows.append({
                "strategy": strategy,
                "model": model,
                "documents": len(documents),
                "dimension": int(vectors.shape[1]),
                "normalized": normalized,
                "valid": valid,
            })
    report = {"index_count": len(rows), "expected_index_count": 9, "rows": rows, "errors": errors, "passed": not errors and len(rows) == 9}
    target = PROJECT_ROOT / "reports/v2/index_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
