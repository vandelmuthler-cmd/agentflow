from __future__ import annotations

from scripts.summarize_v2_human_review import summarize_review


def _inputs() -> tuple[list[dict], list[dict], list[dict]]:
    details = [
        {"id": "ok", "writer_mode": "llm", "workflow_error": "", "answer": "A sufficiently long factual claim."},
        {"id": "closed", "writer_mode": "skipped_after_verifier_error", "workflow_error": "", "answer": "Review failed."},
    ]
    scores = [
        {"id": "ok", "score_0_1_2": "2", "reason": "Answer supported"},
        {"id": "closed", "score_0_1_2": "", "reason": ""},
    ]
    statements = [{
        "case_id": "ok", "statement_index": "1", "statement": "A sufficiently long factual claim.",
        "cited_ids": "[]", "review_label": "supported",
        "review_reason": "The source contains the stated fact",
    }]
    return details, scores, statements


def test_human_review_summary_keeps_failed_case_separate() -> None:
    details, scores, statements = _inputs()
    summary = summarize_review(details, scores, statements)
    assert summary["answer_scores"]["2"] == 1
    assert summary["failed_closed_case_ids"] == ["closed"]
    assert summary["human_review_complete"] is True
    assert summary["statement_labels"]["supported"] == 1


def test_human_review_summary_rejects_unjustified_or_invalid_labels() -> None:
    details, scores, statements = _inputs()
    statements[0]["review_reason"] = ""
    try:
        summarize_review(details, scores, statements)
    except ValueError as error:
        assert "Missing statement review reason" in str(error)
    else:
        raise AssertionError("Missing review reason accepted")
    statements[0]["review_reason"] = "checked"
    scores[1]["score_0_1_2"] = "0"
    scores[1]["reason"] = "failed"
    try:
        summarize_review(details, scores, statements)
    except ValueError as error:
        assert "Failed-closed" in str(error)
    else:
        raise AssertionError("Failed-closed answer was scored")


def test_human_review_summary_detects_unscored_claims() -> None:
    details, scores, statements = _inputs()
    statements[0]["review_label"] = ""
    summary = summarize_review(details, scores, statements)
    assert summary["pending_statement_count"] == 1
    assert summary["human_review_complete"] is False


def test_human_review_summary_rejects_missing_statement_row() -> None:
    details, scores, _ = _inputs()
    try:
        summarize_review(details, scores, [])
    except ValueError as error:
        assert "do not match answer lines" in str(error)
    else:
        raise AssertionError("Incomplete statement table accepted")


def test_human_review_summary_rejects_stale_answer_text() -> None:
    details, scores, statements = _inputs()
    statements[0]["statement"] = "A different answer with the same line count."
    try:
        summarize_review(details, scores, statements)
    except ValueError as error:
        assert "stale or changed" in str(error)
    else:
        raise AssertionError("Stale review text accepted")
