"""Persistent per-run OpenAI token and cost accounting for the Shorts pipeline."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Standard API list prices per 1M tokens, checked 2026-07-28.  These are an
# estimate only: dashboard billing remains authoritative and prices can change.
MODEL_PRICING_USD_PER_MILLION = {
    "gpt-5.4": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "gpt-5.6-sol": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    "gpt-4.1-mini": {"input": None, "cached_input": None, "output": None},
}


def usage_log_path() -> Path | None:
    value = os.environ.get("SHORTS_USAGE_LOG_PATH", "").strip()
    return Path(value) if value else None


def _number(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _value(source: Any, name: str, default: Any = None) -> Any:
    return getattr(source, name, default) if source is not None else default


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float | None:
    rates = MODEL_PRICING_USD_PER_MILLION.get(model)
    if not rates or rates.get("input") is None or rates.get("output") is None:
        return None
    uncached_input = max(0, input_tokens - cached_input_tokens)
    return round(
        (uncached_input * float(rates["input"]) + cached_input_tokens * float(rates["cached_input"]) + output_tokens * float(rates["output"])) / 1_000_000,
        6,
    )


def append_chat_usage(*, stage: str, model: str, response: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    usage = _value(response, "usage")
    prompt_tokens = _number(_value(usage, "prompt_tokens"))
    completion_tokens = _number(_value(usage, "completion_tokens"))
    details = _value(usage, "prompt_tokens_details")
    cached_tokens = _number(_value(details, "cached_tokens"))
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "chat_completion",
        "stage": stage,
        "model": model,
        "input_tokens": prompt_tokens,
        "cached_input_tokens": cached_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": _number(_value(usage, "total_tokens")) or prompt_tokens + completion_tokens,
        "estimated_usd": estimate_cost_usd(model, prompt_tokens, completion_tokens, cached_tokens),
        "pricing_basis": "standard_api_list_price_2026-07-28",
        **(extra or {}),
    }
    append_record(record)
    return record


def append_event(*, stage: str, model: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "non_token_event",
        "stage": stage,
        "model": model,
        "estimated_usd": None,
        **(extra or {}),
    }
    append_record(record)
    return record


def append_record(record: dict[str, Any]) -> None:
    path = usage_log_path()
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def summarize_usage(path: Path) -> dict[str, Any]:
    def empty_totals() -> dict[str, Any]:
        return {
            "request_count": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_usd": 0.0,
            "unpriced_events": 0,
        }

    totals = empty_totals()
    by_model: dict[str, dict[str, Any]] = defaultdict(empty_totals)
    by_stage: dict[str, dict[str, Any]] = defaultdict(empty_totals)
    if not path.exists():
        return {"status": "not_recorded", "path": str(path), "totals": totals, "by_model": {}, "by_stage": {}}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        buckets = (totals, by_model[str(record.get("model") or "unknown")], by_stage[str(record.get("stage") or "unknown")])
        for bucket in buckets:
            bucket["request_count"] += 1
            for key in ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"):
                bucket[key] += _number(record.get(key))
            amount = record.get("estimated_usd")
            if isinstance(amount, (int, float)):
                bucket["estimated_usd"] = round(float(bucket["estimated_usd"]) + float(amount), 6)
            else:
                bucket["unpriced_events"] += 1
    return {"status": "completed", "path": str(path), "totals": totals, "by_model": dict(by_model), "by_stage": dict(by_stage)}
