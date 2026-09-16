from __future__ import annotations

import re

from agentflow.schemas import CitationCheck, Evidence, GenerationQuality


class CitationVerifier:
    """Check whether generated answers cite retrieved evidence ids."""

    citation_pattern = re.compile(r"\[([^\[\]]+?)\]")

    def verify(self, answer: str, evidence: list[Evidence]) -> GenerationQuality:
        if not evidence:
            return GenerationQuality(
                citation_correctness=1.0,
                citation_completeness=1.0,
                groundedness=1.0,
                abstention=True,
            )

        evidence_ids = {item.id for item in evidence}
        cited_ids = set(self.citation_pattern.findall(answer))
        unsupported = sorted(cited_ids - evidence_ids)
        missing = sorted(evidence_ids - cited_ids)

        checks = [
            CitationCheck(
                evidence_id=item.id,
                present_in_answer=item.id in cited_ids,
                source=item.source,
            )
            for item in evidence
        ]
        supported_count = len(cited_ids & evidence_ids)
        citation_correctness = supported_count / len(cited_ids) if cited_ids else 0.0
        citation_completeness = supported_count / len(evidence_ids) if evidence_ids else 1.0

        return GenerationQuality(
            citation_correctness=round(citation_correctness, 3),
            citation_completeness=round(citation_completeness, 3),
            groundedness=1.0 if not unsupported else 0.0,
            abstention=False,
            unsupported_citations=unsupported,
            missing_citations=missing,
            checks=checks,
        )
