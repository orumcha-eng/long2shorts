"""Controlled final-package model benchmark (no render, review, or upload)."""

from __future__ import annotations

import json
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

import generate_shorts_packages as packaging


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis" / "youtube_bEwGLTxqCOI"
MODELS = [
    "gpt-4.1-mini",
    "gpt-5-mini",
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.5",
    "gpt-5.6-sol",
]


def load_candidate() -> tuple[dict, int]:
    selected = json.loads(
        (ANALYSIS_DIR / "shorts_candidates" / "global" / "selected_candidates.json").read_text(encoding="utf-8")
    )["selected"]
    top = selected[0]
    candidates = json.loads(
        (ANALYSIS_DIR / "shorts_candidates" / "local" / "all_candidates.json").read_text(encoding="utf-8")
    )["candidates"]
    candidate = next(item for item in candidates if item["candidate_id"] == top["candidate_id"])
    candidate.update(top)
    return candidate, int(top["global_rank"])


def structural_summary(result: dict) -> dict:
    clips = result.get("source_clips") or []
    return {
        "valid": packaging.validate_final_result(result)[0],
        "duration_sec": packaging.safe_total_clip_duration_sec(clips),
        "clip_count": len(clips),
        "visible_jump_cuts": packaging.visible_jump_cut_count(clips),
        "title": [result.get("title_line1"), result.get("title_line2")],
        "upload_title": result.get("upload_title"),
        "hook_line": result.get("hook_line"),
        "core_event": result.get("core_event"),
        "selection_pitch": result.get("selection_pitch"),
        "captions": len(result.get("point_captions") or []),
        "sound_effects": len(result.get("sound_effects") or []),
        "narration": len(result.get("narration") or []),
    }


def call_once(client, model: str, prompt: str, max_completion_tokens: int) -> tuple[dict | None, dict]:
    started = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            max_completion_tokens=max_completion_tokens,
            messages=[
                {"role": "system", "content": packaging.PACKAGING_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        result = packaging.parse_json_response(response.choices[0].message.content or "{}")
        valid, validation_error = packaging.validate_final_result(result)
        usage = response.usage
        meta = {
            "status": "ok" if valid else "invalid",
            "validation_error": validation_error,
            "elapsed_sec": round(time.monotonic() - started, 2),
            "usage": {
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            },
        }
        return result, meta
    except Exception as exc:  # Keep the whole benchmark useful if a model is unavailable.
        return None, {
            "status": "error",
            "error": str(exc),
            "elapsed_sec": round(time.monotonic() - started, 2),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=packaging.PACKAGING_MAX_COMPLETION_TOKENS,
    )
    parser.add_argument("--label", default="", help="Optional subdirectory for a separate trial.")
    args = parser.parse_args()
    candidate, rank = load_candidate()
    packaging.configure_runtime(
        analysis_dir=ANALYSIS_DIR,
        source_title="악명 높은 영국 음식 먹어보기 (w.명예영국인)",
        movie_info_path=None,
        youtube_context_path=ANALYSIS_DIR / "youtube_context.json",
        benchmark_profile_path=ROOT / "templates" / "benchmark_profiles" / "rescene_gyaru_variety.json",
        learning_rule_path=ROOT / "analysis" / "automation" / "runs" / "daily_20260727T043131Z_883a3a" / "learning_rule.json",
    )
    segments = packaging.load_merged_segments()
    prompt = packaging.build_packaging_prompt(
        candidate,
        packaging.extract_candidate_context(segments, candidate),
        rank,
    )
    client = packaging.load_client()
    output_dir = ANALYSIS_DIR / "model_benchmark"
    if args.label:
        output_dir = output_dir / args.label
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Same source, same candidate, same system and user prompt. No rendering or upload.",
        "candidate_id": candidate["candidate_id"],
        "models": {},
    }
    for model in args.models:
        print(f"[benchmark] {model}", flush=True)
        result, meta = call_once(client, model, prompt, args.max_completion_tokens)
        report["models"][model] = meta
        if result is not None:
            (output_dir / f"{model}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            report["models"][model]["summary"] = structural_summary(result)
        (output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(output_dir / "report.json")


if __name__ == "__main__":
    main()
