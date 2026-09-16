from __future__ import annotations

import re

from agentflow.schemas import Evidence


class EvidenceWriter:
    """Generate a deterministic evidence-grounded answer.

    This deterministic writer supports offline regression tests and runtime
    fallback while preserving the production citation format.
    """

    def write(self, question: str, evidence: list[Evidence]) -> str:
        if not evidence:
            return (
                f"# 回答：{question}\n\n"
                "没有检索到足够证据，因此当前系统拒绝给出确定结论。"
            )

        top_evidence = evidence[:5]
        lines = [
            f"# 回答：{question}",
            "",
            "## 结论",
            self._build_conclusion(question, top_evidence),
            "",
            "## 证据依据",
        ]
        for item in top_evidence:
            lines.append(
                f"- [{item.id}] {self._clean_snippet(item.text, max_chars=260)} "
                f"(source: {item.source})"
            )
        lines.extend(
            [
                "",
                "## 可信度说明",
                "以上回答只基于当前检索到的本地证据生成；如果证据不足，系统应拒答或继续检索。",
            ]
        )
        return "\n".join(lines)

    def _build_conclusion(self, question: str, evidence: list[Evidence]) -> str:
        citation_list = " ".join(f"[{item.id}]" for item in evidence[:3])
        if "不需要" in question or "without" in question.lower():
            return f"当前证据显示，这一问题需要结合方法描述判断，关键依据见 {citation_list}。"
        if "为什么" in question or "why" in question.lower():
            return f"主要原因需要从方法目标、输入信息和证据约束三方面理解，依据见 {citation_list}。"
        if "区别" in question or "difference" in question.lower():
            return f"二者差异应围绕输入、处理流程和输出目标比较，依据见 {citation_list}。"
        return f"当前检索结果支持从相关文献证据中归纳回答，核心依据见 {citation_list}。"

    def _clean_snippet(self, text: str, max_chars: int) -> str:
        snippet = re.sub(r"\s+", " ", text).strip()
        if len(snippet) <= max_chars:
            return snippet
        return snippet[: max_chars - 3].rstrip() + "..."
