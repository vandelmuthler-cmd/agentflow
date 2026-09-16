from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentflow.config import DOCUMENT_INDEX_PATH, RAW_DIR
from agentflow.retrieval.index import build_documents_from_raw, summarize_documents
from agentflow.retrieval.store import save_documents


def main() -> None:
    documents = build_documents_from_raw(RAW_DIR)
    if not documents:
        raise SystemExit(f"No supported files found in {RAW_DIR}")
    save_documents(DOCUMENT_INDEX_PATH, documents)
    print(f"documents: {len(documents)}")
    print("sources:")
    for source, count in summarize_documents(documents).items():
        print(f"  {source}: {count} chunks")
    print(f"index: {DOCUMENT_INDEX_PATH}")


if __name__ == "__main__":
    main()
