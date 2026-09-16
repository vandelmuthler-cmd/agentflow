from __future__ import annotations

import math
import re
from collections import Counter

from agentflow.schemas import Evidence


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for piece in re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", piece):
            tokens.extend(piece)
            tokens.extend(piece[i : i + 2] for i in range(len(piece) - 1))
        else:
            tokens.append(piece)
    return tokens


class SimpleBM25Retriever:
    """Small dependency-free BM25-like retriever for the project skeleton."""

    def __init__(self) -> None:
        self.documents: list[Evidence] = []
        self.term_freqs: list[Counter[str]] = []
        self.doc_freqs: Counter[str] = Counter()
        self.avg_len = 0.0

    def add_documents(self, documents: list[Evidence]) -> None:
        self.documents = documents
        self.term_freqs = []
        self.doc_freqs = Counter()

        total_len = 0
        for doc in documents:
            terms = tokenize(doc.text)
            freqs = Counter(terms)
            self.term_freqs.append(freqs)
            self.doc_freqs.update(freqs.keys())
            total_len += len(terms)

        self.avg_len = total_len / len(documents) if documents else 0.0

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        if not self.documents:
            return []

        query_terms = tokenize(query)
        scored: list[tuple[float, Evidence]] = []
        n_docs = len(self.documents)
        k1 = 1.5
        b = 0.75

        for doc, freqs in zip(self.documents, self.term_freqs):
            doc_len = sum(freqs.values()) or 1
            score = 0.0
            for term in query_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                df = self.doc_freqs.get(term, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1 - b + b * doc_len / (self.avg_len or 1))
                score += idf * (tf * (k1 + 1)) / denom
            if score > 0:
                scored.append((score, doc))

        ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
        return [
            doc.model_copy(update={"score": score, "rank": rank})
            for rank, (score, doc) in enumerate(ranked, 1)
        ]
