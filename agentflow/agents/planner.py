from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urljoin

from pydantic import BaseModel, Field, ValidationError, field_validator

from agentflow.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_SECONDS
from agentflow.model_usage import ModelUsage, parse_model_usage


class PlanStep(BaseModel):
    id: str = Field(min_length=1)
    objective: str = Field(min_length=2)
    tool: str = Field(min_length=1)
    query: str = Field(min_length=1)
    success_criterion: str = Field(min_length=2)


class StructuredPlan(BaseModel):
    task_type: str = Field(pattern="^(simple|complex)$")
    rationale: str = Field(min_length=2)
    steps: list[PlanStep] = Field(min_length=1, max_length=3)

    @field_validator("steps")
    @classmethod
    def unique_step_ids(cls, steps: list[PlanStep]) -> list[PlanStep]:
        ids = [step.id for step in steps]
        if len(ids) != len(set(ids)):
            raise ValueError("plan step ids must be unique")
        return steps


@dataclass(frozen=True)
class PlanningOutcome:
    plan: StructuredPlan
    mode: str
    error: str = ""
    usage: ModelUsage = field(default_factory=ModelUsage)


class PlannerError(RuntimeError):
    pass


class StructuredPlanner:
    """OpenAI-compatible planner with a deterministic fallback."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout_seconds: int = 30) -> None:
        self.api_key = api_key
        self.base_url = self._normalize_chat_completions_url(base_url)
        self.model = model
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> "StructuredPlanner":
        return cls(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            model=LLM_MODEL,
            timeout_seconds=LLM_TIMEOUT_SECONDS,
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def create_plan(self, question: str, tools: list[dict]) -> PlanningOutcome:
        available_tools = {item["name"] for item in tools}
        if not self.is_configured:
            return PlanningOutcome(self._fallback_plan(question), mode="fixed_fallback")

        try:
            content, usage = self._request_plan(question, tools)
            plan = self._parse_plan(content)
            unknown_tools = {step.tool for step in plan.steps} - available_tools
            if unknown_tools:
                raise PlannerError(f"planner selected unknown tools: {sorted(unknown_tools)}")
            return PlanningOutcome(plan, mode="llm", usage=usage)
        except PlannerError as error:
            return PlanningOutcome(
                self._fallback_plan(question),
                mode="fixed_fallback",
                error=str(error),
            )

    def _request_plan(self, question: str, tools: list[dict]) -> tuple[str, ModelUsage]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._user_prompt(question, tools)},
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
            raise PlannerError(f"planner HTTP error {error.code}: {message[:300]}") from error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise PlannerError(f"planner request failed: {error}") from error

        if not isinstance(content, str) or not content.strip():
            raise PlannerError("planner returned empty content")
        return content, parse_model_usage(body, self.model)

    def _parse_plan(self, content: str) -> StructuredPlan:
        cleaned = content.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            cleaned = fenced.group(1)
        try:
            return StructuredPlan.model_validate_json(cleaned)
        except (ValidationError, json.JSONDecodeError) as error:
            raise PlannerError(f"planner output validation failed: {error}") from error

    def _fallback_plan(self, question: str) -> StructuredPlan:
        return StructuredPlan(
            task_type="simple",
            rationale="模型规划不可用，使用确定性单步检索计划",
            steps=[
                PlanStep(
                    id="step_1",
                    objective="检索与问题直接相关的本地证据",
                    tool="document_search",
                    query=question,
                    success_criterion="至少返回一条可引用证据",
                )
            ],
        )

    def _system_prompt(self) -> str:
        return (
            "你是受约束的研究任务规划器，只负责生成计划，不负责回答问题。"
            "简单事实问题生成1个步骤；只有比较、归纳多个方面或需要分别取证时才生成2到3个步骤。"
            "不得选择未提供的工具，不得生成超过3个步骤，不得添加代码执行或网络搜索。"
            "每个步骤必须有id、objective、tool、query、success_criterion。"
            "只输出JSON对象，task_type只能是simple或complex。"
        )

    def _user_prompt(self, question: str, tools: list[dict]) -> str:
        compact_tools = [
            {"name": item["name"], "description": item.get("description", "")}
            for item in tools
        ]
        return (
            f"用户问题：{question}\n"
            f"可用工具：{json.dumps(compact_tools, ensure_ascii=False)}\n"
            "输出格式："
            '{"task_type":"simple|complex","rationale":"...","steps":'
            '[{"id":"step_1","objective":"...","tool":"...","query":"...",'
            '"success_criterion":"..."}]} '
        )

    def _normalize_chat_completions_url(self, base_url: str) -> str:
        if base_url.rstrip("/").endswith("/chat/completions"):
            return base_url
        return urljoin(base_url.rstrip("/") + "/", "chat/completions")
