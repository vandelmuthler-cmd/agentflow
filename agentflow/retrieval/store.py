from __future__ import annotations

import json
from pathlib import Path

from agentflow.schemas import Evidence


def save_documents(path: Path, documents: list[Evidence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(document.model_dump_json(ensure_ascii=False) + "\n")


def load_documents(path: Path) -> list[Evidence]:
    if not path.exists():
        return []
    documents: list[Evidence] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            documents.append(Evidence.model_validate(json.loads(line)))
    return documents
