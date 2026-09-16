from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentflow.config import V2_CORPUS_DIR
from agentflow.evals.v2_eval import (
    canonical_dataset_sha256,
    is_verified_review_status,
    load_v2_dataset,
    map_gold_spans,
    quote_coverage,
)
from agentflow.retrieval.store import load_documents
from agentflow.retrieval.v2_corpus import CorpusManifest, extract_pdf_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit AgentFlow V2 evaluation data without running retrieval.")
    parser.add_argument("--dataset", type=Path, default=Path("data/v2_eval_frozen.json"))
    parser.add_argument(
        "--require-verified",
        action="store_true",
        help="Require every case to pass source or independent human review.",
    )
    parser.add_argument("--require-human", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def normalized(value: str) -> str:
    return re.sub(r"\W+", "", value.casefold())


def main() -> None:
    args = parse_args()
    dataset_path = PROJECT_ROOT / args.dataset
    dataset = load_v2_dataset(dataset_path)
    manifest = CorpusManifest.model_validate_json(
        (PROJECT_ROOT / "data/v2_corpus_manifest.json").read_text(encoding="utf-8")
    )
    roles = {item.document_id: item.document_role for item in manifest.documents}
    filenames = {item.document_id: item.filename for item in manifest.documents}
    errors: list[str] = []
    warnings: list[str] = []
    split_counts = Counter(case.split for case in dataset.cases)
    if split_counts != Counter({"development": 20, "frozen": 40, "stress": 20}):
        errors.append(f"unexpected split counts: {dict(split_counts)}")
    questions_zh = [normalized(case.question_zh) for case in dataset.cases]
    questions_en = [normalized(case.question_en) for case in dataset.cases]
    if len(questions_zh) != len(set(questions_zh)):
        errors.append("duplicate Chinese questions found")
    if len(questions_en) != len(set(questions_en)):
        errors.append("duplicate English questions found")
    for case in dataset.cases:
        cited_roles = {roles.get(gold.document_id, "missing") for gold in case.gold_evidence}
        if "missing" in cited_roles:
            errors.append(f"{case.id}: unknown document_id")
        if case.split == "development" and "heldout" in cited_roles:
            errors.append(f"{case.id}: development question cites a heldout document")
    frozen_heldout = sum(
        case.split == "frozen"
        and any(roles.get(gold.document_id) == "heldout" for gold in case.gold_evidence)
        for case in dataset.cases
    )
    if frozen_heldout != 20:
        errors.append(f"expected 20 frozen cases from heldout documents, found {frozen_heldout}")
    primary_counts = Counter(
        case.primary_language for case in dataset.cases if case.split in {"frozen", "stress"}
    )
    if primary_counts != Counter({"zh": 40, "en": 20}):
        errors.append(f"unexpected frozen/stress primary languages: {dict(primary_counts)}")

    page_cache = {}
    for case in dataset.cases:
        for gold in case.gold_evidence:
            if gold.document_id not in page_cache:
                path = V2_CORPUS_DIR / filenames[gold.document_id]
                if not path.exists():
                    warnings.append("raw corpus unavailable; PDF page anchors were not independently checked")
                    page_cache[gold.document_id] = []
                else:
                    page_cache[gold.document_id] = extract_pdf_pages(path)
            matching_page = next(
                (page for page in page_cache[gold.document_id] if page.page == gold.page), None
            )
            if matching_page and quote_coverage(gold.evidence_quote, matching_page.text) < 0.5:
                errors.append(f"{case.id}:{gold.claim_id}: evidence quote does not match page {gold.page}")
    if args.require_verified or args.require_human:
        pending = [
            case.id
            for case in dataset.cases
            if not is_verified_review_status(case.review_status)
        ]
        if pending:
            errors.append(f"{len(pending)} cases have not passed a verified review")
    elif any(not is_verified_review_status(case.review_status) for case in dataset.cases):
        warnings.append("dataset is an unverified draft and is not eligible for a formal frozen score")

    mapping_reports = {}
    for strategy in ("fixed_char", "recursive_token", "section_aware"):
        chunks = load_documents(PROJECT_ROOT / f"data/index/v2/{strategy}/documents.jsonl")
        mappings = map_gold_spans(dataset.cases, chunks)
        missing = [item.model_dump() for item in mappings if not item.chunk_ids]
        mapping_reports[strategy] = {
            "gold_span_count": len(mappings),
            "mapped_count": len(mappings) - len(missing),
            "missing": missing,
        }
        if missing:
            errors.append(f"{strategy}: {len(missing)} gold spans do not map to a chunk")

    actual_sha = canonical_dataset_sha256(dataset)
    if dataset.sha256 != actual_sha:
        errors.append("stored dataset SHA-256 does not match its canonical content")
    report = {
        "dataset": str(args.dataset),
        "version": dataset.version,
        "stored_sha256": dataset.sha256,
        "actual_sha256": actual_sha,
        "split_counts": dict(split_counts),
        "review_status": dict(Counter(case.review_status for case in dataset.cases)),
        "mapping": mapping_reports,
        "warnings": warnings,
        "errors": errors,
        "passed": not errors,
    }
    output_dir = PROJECT_ROOT / "data/eval_runs/v2_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# AgentFlow V2 Evaluation Audit",
        "",
        f"- Dataset version: `{dataset.version}`",
        f"- Cases: `{len(dataset.cases)}`",
        f"- Status: `{'PASS' if not errors else 'BLOCKED'}`",
        f"- Review states: `{dict(report['review_status'])}`",
        "",
        "## Gold Mapping",
        "",
        "| Chunking | Mapped | Total |",
        "| --- | ---: | ---: |",
    ]
    for strategy, item in mapping_reports.items():
        lines.append(f"| {strategy} | {item['mapped_count']} | {item['gold_span_count']} |")
    lines.extend(["", "## Warnings", ""] + [f"- {item}" for item in warnings] or ["- None"])
    lines.extend(["", "## Errors", ""] + [f"- {item}" for item in errors] or ["- None"])
    (output_dir / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
