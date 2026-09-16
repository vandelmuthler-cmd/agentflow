import json
from unittest.mock import patch

from agentflow.agents.planner import StructuredPlanner
from agentflow.generation.llm_writer import LLMWriter
from agentflow.model_usage import parse_model_usage
from agentflow.schemas import Evidence
from agentflow.schemas import NodeStatus, TraceEvent
from agentflow.storage.traces import InMemoryTraceStore


def test_api_usage_is_parsed() -> None:
    usage = parse_model_usage(
        {
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
                "prompt_cache_hit_tokens": 20,
                "prompt_cache_miss_tokens": 100,
            }
        },
        "test-model",
    )
    assert usage.source == "api"
    assert usage.total_tokens == 150
    assert usage.prompt_cache_hit_tokens == 20


def test_trace_summary_aggregates_model_usage() -> None:
    store = InMemoryTraceStore()
    run_id = "usage-run"
    store.append(
        TraceEvent(
            run_id=run_id,
            node="model.writer",
            status=NodeStatus.STARTED,
            message="start",
        )
    )
    store.append(
        TraceEvent(
            run_id=run_id,
            node="model.writer",
            status=NodeStatus.COMPLETED,
            message="done",
            payload={
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
                "estimated_cost_usd": 0.0012,
            },
        )
    )
    summary = store.summarize(run_id)
    assert summary.model_calls == 1
    assert summary.total_tokens == 150
    assert summary.estimated_cost_usd == 0.0012


class FakeResponse:
    def __init__(self, body: dict) -> None:
        self.payload = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_writer_returns_api_usage() -> None:
    body = {
        "choices": [{"message": {"content": "结论 [E1]"}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }
    writer = LLMWriter("test-key", "https://example.test", "test-model")
    with patch(
        "agentflow.generation.llm_writer.urllib.request.urlopen",
        return_value=FakeResponse(body),
    ):
        result = writer.write_with_usage(
            "问题",
            [
                Evidence(
                    id="2021_long_document_name__16",
                    source="doc.pdf",
                    text="证据",
                )
            ],
        )
    assert result.content == "结论 [2021_long_document_name__16]"
    assert result.usage.total_tokens == 25


def test_writer_prompt_uses_short_evidence_aliases() -> None:
    writer = LLMWriter("test-key", "https://example.test", "test-model")
    prompt = writer._user_prompt(
        "问题",
        [Evidence(id="a_very_long_document_id__3", source="doc.pdf", text="证据")],
    )
    assert "[E1]" in prompt
    assert "[a_very_long_document_id__3]" not in prompt


def test_planner_returns_api_usage() -> None:
    plan = {
        "task_type": "simple",
        "rationale": "单步事实检索",
        "steps": [
            {
                "id": "step_1",
                "objective": "查找证据",
                "tool": "document_search",
                "query": "目标事实",
                "success_criterion": "找到一条证据",
            }
        ],
    }
    body = {
        "choices": [{"message": {"content": json.dumps(plan, ensure_ascii=False)}}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40},
    }
    planner = StructuredPlanner("test-key", "https://example.test", "test-model")
    with patch(
        "agentflow.agents.planner.urllib.request.urlopen",
        return_value=FakeResponse(body),
    ):
        outcome = planner.create_plan(
            "问题",
            [{"name": "document_search", "description": "本地检索"}],
        )
    assert outcome.mode == "llm"
    assert outcome.usage.total_tokens == 40
