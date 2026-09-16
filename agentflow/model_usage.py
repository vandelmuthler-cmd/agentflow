from typing import Any

from pydantic import BaseModel

from agentflow.config import INPUT_COST_PER_MILLION_USD, OUTPUT_COST_PER_MILLION_USD


class ModelUsage(BaseModel):
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    estimated_cost_usd: float = 0.0
    cost_configured: bool = False
    source: str = "unavailable"


def parse_model_usage(body: dict[str, Any], model: str) -> ModelUsage:
    raw = body.get("usage") or {}
    prompt_tokens = int(raw.get("prompt_tokens") or 0)
    completion_tokens = int(raw.get("completion_tokens") or 0)
    total_tokens = int(raw.get("total_tokens") or prompt_tokens + completion_tokens)
    input_cost = prompt_tokens * INPUT_COST_PER_MILLION_USD / 1_000_000
    output_cost = completion_tokens * OUTPUT_COST_PER_MILLION_USD / 1_000_000
    return ModelUsage(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_cache_hit_tokens=int(raw.get("prompt_cache_hit_tokens") or 0),
        prompt_cache_miss_tokens=int(raw.get("prompt_cache_miss_tokens") or 0),
        estimated_cost_usd=round(input_cost + output_cost, 8),
        cost_configured=INPUT_COST_PER_MILLION_USD > 0 or OUTPUT_COST_PER_MILLION_USD > 0,
        source="api" if raw else "unavailable",
    )
