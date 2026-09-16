from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentflow.config import V2_CORPUS_DIR, V2_INDEX_ROOT, V2_MANIFEST_PATH
from agentflow.retrieval.store import save_documents
from agentflow.retrieval.v2_corpus import (
    build_document_chunks,
    load_manifest,
    validate_corpus_files,
)


STRATEGIES = ("fixed_char", "recursive_token", "section_aware")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and chunk the AgentFlow V2 corpus.")
    parser.add_argument("--manifest", type=Path, default=V2_MANIFEST_PATH)
    parser.add_argument("--corpus-dir", type=Path, default=V2_CORPUS_DIR)
    parser.add_argument("--output-root", type=Path, default=V2_INDEX_ROOT)
    parser.add_argument("--strategy", choices=[*STRATEGIES, "all"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    validation = validate_corpus_files(manifest, args.corpus_dir)
    invalid = [row for row in validation if not row["exists"] or not row["hash_matches"]]
    if invalid:
        raise SystemExit(json.dumps({"invalid_corpus_files": invalid}, ensure_ascii=False, indent=2))

    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)
    summary: dict[str, object] = {
        "corpus_id": manifest.corpus_id,
        "version": manifest.version,
        "document_count": len(manifest.documents),
        "languages": dict(Counter(item.language for item in manifest.documents)),
        "roles": dict(Counter(item.document_role for item in manifest.documents)),
        "validation": validation,
        "strategies": {},
    }
    for strategy in strategies:
        documents = []
        reports = []
        for spec in manifest.documents:
            chunks, report = build_document_chunks(
                spec, args.corpus_dir / spec.filename, strategy
            )
            documents.extend(chunks)
            reports.append(report.model_dump(mode="json"))
        output_dir = args.output_root / strategy
        output_dir.mkdir(parents=True, exist_ok=True)
        save_documents(output_dir / "documents.jsonl", documents)
        (output_dir / "chunking_report.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary["strategies"][strategy] = {
            "chunk_count": len(documents),
            "short_chunk_count": sum(item["short_chunk_count"] for item in reports),
            "documents_path": str(output_dir / "documents.jsonl"),
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "corpus_preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
