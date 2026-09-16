from scripts.evaluate_heldout_e2e import detect_abstention, mean, term_coverage


def test_term_coverage_is_case_insensitive() -> None:
    assert term_coverage("The result uses RMSF and Cryo-EM.", ["rmsf", "cryo-em"]) == 1.0


def test_abstention_detection_uses_state_and_explicit_language() -> None:
    assert detect_abstention("A normal answer.", structural_abstention=True)
    assert detect_abstention("# 结论\n\n基于当前证据，无法确定结果。", structural_abstention=False)
    assert not detect_abstention("The evidence supports this result.", structural_abstention=False)
    assert not detect_abstention(
        "# 结论\n\n结果由证据直接支持。\n\n# 局限\n\n未提供更多细节。",
        structural_abstention=False,
    )


def test_mean_handles_empty_and_boolean_values() -> None:
    assert mean([], "value") == 0.0
    assert mean([{"value": True}, {"value": False}], "value") == 0.5
