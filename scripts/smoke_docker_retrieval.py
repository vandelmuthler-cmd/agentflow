from __future__ import annotations

import json

from agentflow.config import RAW_DIR, RETRIEVAL_BACKEND
from agentflow.retrieval.index import build_retriever
from agentflow.retrieval.management import RetrievalIndexManager


DOCUMENTS = {
    "smoke-transformer": (
        "transformer-maintenance.md",
        "# Transformer maintenance\n\n"
        "Dissolved gas analysis and insulation temperature trends are used to "
        "trigger preventive maintenance for power transformers.",
    ),
    "smoke-protein": (
        "protein-flexibility.md",
        "# Protein flexibility\n\n"
        "Cryogenic electron microscopy density and molecular simulations can be "
        "combined to study flexible protein regions.",
    ),
}


def main() -> None:
    if RETRIEVAL_BACKEND != "pgvector":
        raise RuntimeError("this smoke test requires AGENTFLOW_RETRIEVAL_BACKEND=pgvector")
    manager = RetrievalIndexManager()
    indexed_document_ids: list[str] = []
    try:
        indexed = {}
        for document_id, (source, text) in DOCUMENTS.items():
            indexed[document_id] = manager.index_text(document_id, source, text)
            indexed_document_ids.append(document_id)
        results = build_retriever(RAW_DIR).search(
            "What evidence triggers preventive maintenance for a power transformer?",
            top_k=2,
        )
        if not results or results[0].document_id != "smoke-transformer":
            raise AssertionError("the relevant transformer document was not ranked first")
        print(
            json.dumps(
                {
                    "backend": RETRIEVAL_BACKEND,
                    "indexed_chunks": indexed,
                    "ranked_document_ids": [item.document_id for item in results],
                    "top_score": results[0].score,
                    "status": "ok",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        for document_id in indexed_document_ids:
            manager.delete_document(document_id)


if __name__ == "__main__":
    main()
