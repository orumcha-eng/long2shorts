"""Continuously measure short-term YouTube view velocity for trend candidates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from discover_trending_sources import discover_candidates, get_api_key, load_config
from env_loader import load_project_env


BASE_DIR = Path(__file__).resolve().parent
TRENDS_DIR = BASE_DIR / "analysis" / "trends"
DEFAULT_CONFIG = BASE_DIR / "templates" / "trend_config.example.json"
DEFAULT_STATE = TRENDS_DIR / "live_trend_observations.json"
DEFAULT_OUTPUT = TRENDS_DIR / "live_hot_candidates.json"


def read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def update_velocity(
    *,
    config_path: Path,
    state_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    load_project_env()
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY or GOOGLE_API_KEY is required for the live trend monitor.")
    config = load_config(config_path)
    snapshot = discover_candidates(config, api_key=api_key, include_processed=True)
    now = datetime.now(timezone.utc)
    observed_at = now.isoformat()
    state = read_json(state_path, {"videos": {}})
    next_state: dict[str, dict[str, Any]] = {}
    ranked: list[dict[str, Any]] = []
    production_candidates: list[dict[str, Any]] = []

    for candidate in snapshot.get("candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        video_id = str(candidate.get("video_id") or "")
        if not video_id:
            continue
        views = int(candidate.get("view_count") or 0)
        item = dict(candidate)
        item["observed_at"] = observed_at
        ranked.append(item)
        next_state[video_id] = {
            "observed_at": observed_at,
            "view_count": views,
            "title": str(candidate.get("title") or ""),
            "channel_title": str(candidate.get("channel_title") or ""),
        }

    for candidate in snapshot.get("production_candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        item = dict(candidate)
        item["observed_at"] = observed_at
        production_candidates.append(item)

    result = {
        "created_at": observed_at,
        "monitor_interval_minutes": int(config.get("monitor_interval_minutes") or 10),
        "candidates": ranked,
        "production_candidates": production_candidates,
        "selected": ranked[0] if ranked else None,
    }
    write_json(state_path, {"updated_at": observed_at, "videos": next_state})
    write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously collect trend-candidate view velocity.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--interval-minutes", type=int, default=None)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    interval = max(5, int(args.interval_minutes or config.get("monitor_interval_minutes") or 10))
    while True:
        try:
            result = update_velocity(
                config_path=args.config,
                state_path=args.state,
                output_path=args.output,
            )
            selected = result.get("selected") or {}
            print(
                "[trend-monitor] "
                f"candidates={len(result.get('candidates', []))} "
                f"selected={selected.get('video_id', '')} "
                f"views={selected.get('view_count', 0)}",
                flush=True,
            )
        except Exception as exc:
            print(f"[trend-monitor] error={exc}", flush=True)
        if args.once:
            return
        time.sleep(interval * 60)


if __name__ == "__main__":
    main()
