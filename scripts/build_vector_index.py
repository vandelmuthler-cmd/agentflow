from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentflow.config import DOCUMENT_INDEX_PATH, VECTOR_IDS_PATH, VECTOR_INDEX_PATH
from agentflow.retrieval.store import load_documents
from agentflow.retrieval.vector import save_vector_index


def main() -> None:
    documents = load_documents(DOCUMENT_INDEX_PATH)
    if not documents:
        raise SystemExit("Missing document index. Run `python -B scripts/build_index.py` first.")
    save_vector_index(documents)
    print(f"documents: {len(documents)}")
    print(f"vectors: {VECTOR_INDEX_PATH}")
    print(f"ids: {VECTOR_IDS_PATH}")


if __name__ == "__main__":
    main()
