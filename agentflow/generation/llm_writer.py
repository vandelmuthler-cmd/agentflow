from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from urllib.parse import urljoin

from pydantic import BaseModel

from agentflow.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_SECONDS
from agentflow.model_usage import ModelUsage, parse_model_usage
from agentflow.schemas import Evidence


class LLMWriterError(RuntimeError):
    pass


class LLMWriteResult(BaseModel):
    content: str
    usage: ModelUsage


class LLMWriter:
    """OpenAI-compatible chat-completions writer for evidence-grounded answers."""

    citation_alias_pattern = re.compile(r"\[(E\d+)\]")

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
    def from_env(cls) -> "LLMWriter":
        return cls(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            model=LLM_MODEL,
            timeout_seconds=LLM_TIMEOUT_SECONDS,
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def write(self, question: str, evidence: list[Evidence]) -> str:
        return self.write_with_usage(question, evidence).content

    def write_with_usage(self, question: str, evidence: list[Evidence]) -> LLMWriteResult:
        if not self.is_configured:
            raise LLMWriterError("LLM writer is not configured")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._user_prompt(question, evidence)},
            ],
            "temperature": 0.2,
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
        except urllib.error.HTTPError as error:
            message = error.read().decode("utf-8", errors="ignore")
            raise LLMWriterError(f"LLM HTTP error {error.code}: {message[:300]}") from error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise LLMWriterError(f"LLM request failed: {error}") from error

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LLMWriterError("LLM response format is not compatible with chat completions") from error

        if not isinstance(content, str) or not content.strip():
            raise LLMWriterError("LLM returned an empty answer")
        content = self._restore_evidence_ids(content.strip(), evidence)
        return LLMWriteResult(content=content, usage=parse_model_usage(body, self.model))

    def _system_prompt(self) -> str:
        return (
            "你是一个证据约束的研究助理。只能基于用户提供的 evidence 回答。"
            "每个关键结论必须使用方括号引用给定的短编号，例如 [E1]。"
            "只能原样复制可用 evidence 中出现的 E 编号，不能缩写或创造编号。"
            "如果证据不足，必须明确说明证据不足，不能编造。"
            "输出 Markdown，包含：结论、证据依据、可信度说明。"
        )

    def _normalize_chat_completions_url(self, base_url: str) -> str:
        if base_url.rstrip("/").endswith("/chat/completions"):
            return base_url
        return urljoin(base_url.rstrip("/") + "/", "chat/completions")

    def _user_prompt(self, question: str, evidence: list[Evidence]) -> str:
        evidence_blocks = []
        for index, item in enumerate(evidence, start=1):
            evidence_blocks.append(
                f"[E{index}]\nsource: {item.source}\ntext: {item.text}"
            )
        return (
            f"问题：{question}\n\n"
            "可用 evidence：\n\n"
            + "\n\n".join(evidence_blocks)
            + "\n\n请严格基于以上 evidence 回答，并仅使用对应的 [E1]、[E2] 等短编号引用。"
        )

    def _restore_evidence_ids(self, content: str, evidence: list[Evidence]) -> str:
        alias_to_id = {f"E{index}": item.id for index, item in enumerate(evidence, start=1)}

        def replace_alias(match: re.Match[str]) -> str:
            evidence_id = alias_to_id.get(match.group(1))
            return f"[{evidence_id}]" if evidence_id else match.group(0)

        return self.citation_alias_pattern.sub(replace_alias, content)
