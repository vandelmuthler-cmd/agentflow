from __future__ import annotations

from agentflow.schemas import Evidence
from scripts.export_statement_review import statement_review_rows


def test_statement_review_flags_missing_and_unknown_citations() -> None:
    document = Evidence(id="doc__section_aware__1", source="doc.pdf", text="A measured fact.")
    case = {
        "id": "case-1",
        "writer_mode": "llm",
        "evidence_ids": [document.id],
        "answer": (
            "# Conclusion\n"
            "- Supported claim [doc__section_aware__1]\n"
            "- Claim with no citation\n"
            "- Claim with unknown ID [other__section_aware__9]\n"
        ),
    }
    rows = statement_review_rows(case, {document.id: document})
    assert [row["auto_flag"] for row in rows] == [
        "needs_human_support_review", "missing_citation", "unknown_citation_id"
    ]
    assert "A measured fact" in rows[0]["evidence_excerpts"]
    assert all(not row["review_label"] for row in rows)


def test_statement_review_excludes_failed_closed_answer() -> None:
    assert statement_review_rows({
        "id": "case-2", "writer_mode": "skipped_after_verifier_error",
        "answer": "Please retry.", "evidence_ids": [],
    }, {}) == []


def test_statement_review_requires_index_text_and_keeps_full_chunk() -> None:
    document = Evidence(id="doc__section_aware__1", source="doc.pdf", text="A" * 700 + " decisive fact")
    case = {
        "id": "case-3", "writer_mode": "llm", "evidence_ids": [document.id],
        "answer": f"Claim with late support [{document.id}]",
    }
    row = statement_review_rows(case, {document.id: document})[0]
    assert "decisive fact" in row["evidence_excerpts"]
    assert statement_review_rows(case, {})[0]["auto_flag"] == "unknown_citation_id"
