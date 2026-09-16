from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urljoin

from pydantic import BaseModel, Field, ValidationError, model_validator

from agentflow.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
    LLM_VERIFIER_ENABLED,
)
from agentflow.model_usage import ModelUsage, parse_model_usage
from agentflow.schemas import Evidence


class EvidenceAssessment(BaseModel):
    decision: Literal["sufficient", "insufficient", "abstain"]
    required_aspects: list[str] = Field(default_factory=list, max_length=8)
    covered_aspects: list[str] = Field(default_factory=list, max_length=8)
    missing_aspects: list[str] = Field(default_factory=list, max_length=8)
    follow_up_queries: list[str] = Field(default_factory=list, max_length=2)
    evidence_mapping: dict[str, list[str]] = Field(default_factory=dict)
    rationale: str = Field(min_length=2, max_length=600)

    @model_validator(mode="after")
    def validate_decision(self) -> "EvidenceAssessment":
        if self.decision == "sufficient" and self.missing_aspects:
            raise ValueError("a sufficient assessment cannot contain missing aspects")
        if self.decision == "insufficient" and not self.missing_aspects:
            raise ValueError("an insufficient assessment must identify missing aspects")
        return self


@dataclass(frozen=True)
class VerificationOutcome:
    assessment: EvidenceAssessment
    mode: str
    error: str = ""
    usage: ModelUsage = field(default_factory=ModelUsage)


class EvidenceVerifierError(RuntimeError):
    pass


class StructuredEvidenceVerifier:
    """OpenAI-compatible evidence coverage judge with deterministic fallback."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: int = 30,
    ) -> None:
        self.api_key = api_key
        self.base_url = self._normalize_chat_completions_url(base_url)
        self.model = model
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> "StructuredEvidenceVerifier":
        return cls(
            api_key=LLM_API_KEY if LLM_VERIFIER_ENABLED else "",
            base_url=LLM_BASE_URL,
            model=LLM_MODEL,
            timeout_seconds=LLM_TIMEOUT_SECONDS,
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def verify(
        self,
        question: str,
        plan: list[str],
        evidence: list[Evidence],
    ) -> VerificationOutcome:
        if not evidence:
            return VerificationOutcome(
                assessment=self._empty_assessment(question),
                mode="deterministic",
            )
        if not self.is_configured:
            return VerificationOutcome(
                assessment=self._fallback_assessment(evidence),
                mode="deterministic_fallback",
            )

        usage = ModelUsage()
        try:
            content, usage = self._request_assessment(question, plan, evidence)
            assessment = self._parse_assessment(content, evidence)
            return VerificationOutcome(assessment=assessment, mode="llm", usage=usage)
        except EvidenceVerifierError as error:
            return VerificationOutcome(
                assessment=EvidenceAssessment(
                    decision="abstain",
                    required_aspects=["问题所需事实"],
                    missing_aspects=["证据充分性评审不可用"],
                    rationale="模型证据评审失败，无法确认当前证据是否充分。",
                ),
                mode="deterministic_fallback",
                error=str(error),
                usage=usage,
            )

    def _request_assessment(
        self,
        question: str,
        plan: list[str],
        evidence: list[Evidence],
    ) -> tuple[str, ModelUsage]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": self._user_prompt(question, plan, evidence),
                },
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as error:
            message = error.read().decode("utf-8", errors="ignore")
            raise EvidenceVerifierError(
                f"verifier HTTP error {error.code}: {message[:300]}"
            ) from error
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
        ) as error:
            raise EvidenceVerifierError(f"verifier request failed: {error}") from error
        if not isinstance(content, str) or not content.strip():
            raise EvidenceVerifierError("verifier returned empty content")
        return content, parse_model_usage(body, self.model)

    def _parse_assessment(
        self, content: str, evidence: list[Evidence]
    ) -> EvidenceAssessment:
        cleaned = content.strip()
        fenced = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fenced:
            cleaned = fenced.group(1)
        try:
            assessment = EvidenceAssessment.model_validate_json(cleaned)
        except (ValidationError, json.JSONDecodeError) as error:
            raise EvidenceVerifierError(
                f"verifier output validation failed: {error}"
            ) from error

        aliases = {f"E{index}": item.id for index, item in enumerate(evidence, 1)}
        unknown = {
            alias
            for values in assessment.evidence_mapping.values()
            for alias in values
            if alias not in aliases
        }
        if unknown:
            raise EvidenceVerifierError(
                f"verifier referenced unknown evidence aliases: {sorted(unknown)}"
            )
        mapping = {
            aspect: [aliases[alias] for alias in values]
            for aspect, values in assessment.evidence_mapping.items()
        }
        return assessment.model_copy(update={"evidence_mapping": mapping})

    def _system_prompt(self) -> str:
        return (
            "你是证据充分性评审器，只判断给定 evidence 是否足以完整回答问题，不负责写答案。"
            "先把问题拆成最多8个必要回答要点，再逐项检查 evidence。"
            "只有全部必要要点都有直接证据时才能判 sufficient；部分覆盖必须判 insufficient。"
            "问题明显超出当前知识库且证据无法支持时可判 abstain。"
            "evidence_mapping 只能引用提供的 E1、E2 等短编号。"
            "insufficient 时生成1到2条针对缺失要点的简洁检索查询；不要重复原问题。"
            "只输出 JSON 对象，不输出答案或 Markdown。"
        )

    def _user_prompt(
        self, question: str, plan: list[str], evidence: list[Evidence]
    ) -> str:
        blocks = [
            f"[E{index}]\nsource: {item.source}\ntext: {item.text}"
            for index, item in enumerate(evidence, 1)
        ]
        schema = {
            "decision": "sufficient|insufficient|abstain",
            "required_aspects": ["要点"],
            "covered_aspects": ["已覆盖要点"],
            "missing_aspects": ["缺失要点"],
            "follow_up_queries": ["补充检索查询，最多2条"],
            "evidence_mapping": {"已覆盖要点": ["E1"]},
            "rationale": "简短判断依据",
        }
        return (
            f"问题：{question}\n"
            f"原计划：{json.dumps(plan, ensure_ascii=False)}\n\n"
            "当前 evidence：\n\n"
            + "\n\n".join(blocks)
            + "\n\n输出格式："
            + json.dumps(schema, ensure_ascii=False)
        )

    def _empty_assessment(self, question: str) -> EvidenceAssessment:
        return EvidenceAssessment(
            decision="insufficient",
            required_aspects=["问题所需事实"],
            missing_aspects=["未检索到可用证据"],
            follow_up_queries=[f"{question} 关键事实 证据"],
            rationale="当前检索结果为空，无法基于证据回答。",
        )

    def _fallback_assessment(
        self, evidence: list[Evidence]
    ) -> EvidenceAssessment:
        return EvidenceAssessment(
            decision="sufficient",
            required_aspects=["问题所需事实"],
            covered_aspects=["已检索到可引用证据"],
            evidence_mapping={"已检索到可引用证据": [evidence[0].id]},
            rationale="模型评审不可用，沿用非空证据规则。",
        )

    def _normalize_chat_completions_url(self, base_url: str) -> str:
        if base_url.rstrip("/").endswith("/chat/completions"):
            return base_url
        return urljoin(base_url.rstrip("/") + "/", "chat/completions")
