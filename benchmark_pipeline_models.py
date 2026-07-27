"""Stage-by-stage model comparison for visual analysis and candidate selection."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import build_visual_event_script as vision
import generate_shorts_packages as packaging


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis" / "youtube_bEwGLTxqCOI"
MODELS = ["gpt-4.1-mini", "gpt-5-mini", "gpt-5.4-mini", "gpt-5.4", "gpt-5.5", "gpt-5.6-sol"]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def configure() -> None:
    packaging.configure_runtime(
        analysis_dir=ANALYSIS_DIR,
        source_title="악명 높은 영국 음식 먹어보기 (w.명예영국인)",
        movie_info_path=None,
        youtube_context_path=ANALYSIS_DIR / "youtube_context.json",
        benchmark_profile_path=ROOT / "templates" / "benchmark_profiles" / "rescene_gyaru_variety.json",
        learning_rule_path=ROOT / "analysis" / "automation" / "runs" / "daily_20260727T043131Z_883a3a" / "learning_rule.json",
    )


def call(stage: str, model: str, fn) -> dict:
    started = time.monotonic()
    try:
        result = fn()
        return {"status": "ok", "elapsed_sec": round(time.monotonic() - started, 2), "result": result}
    except Exception as exc:
        return {"status": "error", "elapsed_sec": round(time.monotonic() - started, 2), "error": str(exc)}


def vision_groups() -> list[list[dict]]:
    manifest = json.loads((ANALYSIS_DIR / "visual_events" / "frames" / "frame_manifest.json").read_text(encoding="utf-8"))
    frames = manifest["frames"]
    groups = []
    for start, end in [(0, 84), (1570, 1655)]:
        group = [item for item in frames if start <= float(item["time_sec"]) <= end][:6]
        if len(group) >= 2:
            groups.append(group)
    return groups


def visual_score(events: list[dict]) -> dict:
    return {
        "event_count": len(events),
        "high_confidence": sum(1 for event in events if event.get("confidence") == "high"),
        "visible_actions": sum(1 for event in events if str(event.get("action") or "").strip()),
        "reactions": sum(1 for event in events if str(event.get("reaction") or "").strip()),
        "payoffs": sum(1 for event in events if str(event.get("payoff") or "").strip()),
        "hooks": [event.get("visual_hook") for event in events],
    }


def main() -> None:
    configure()
    client = packaging.load_client()
    output_dir = ANALYSIS_DIR / "model_benchmark" / "pipeline_stages"
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript = vision.load_transcript(ANALYSIS_DIR)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "models": MODELS, "visual_events": {}, "local_candidates": {}, "global_selection": {}}

    # Stage 3: exact same frames + exact same nearby transcript, twice.
    for model in MODELS:
        print(f"[pipeline-benchmark] visual {model}", flush=True)
        trials = []
        for frames in vision_groups():
            trial = call("visual", model, lambda frames=frames: vision.analyze_batch(client, model, frames, transcript))
            if trial["status"] == "ok":
                trial["summary"] = visual_score(trial["result"])
            trials.append(trial)
        report["visual_events"][model] = trials
        write_json(output_dir / "report.json", report)

    # Stage 4a: exact same six-hundred-second transcript/event window.
    chunk = next(item for item in packaging.load_chunk_transcripts(include_wide_windows=False) if item["chunk_id"] == "chunk_003")
    local_prompt = packaging.build_local_prompt(chunk)
    for model in MODELS:
        print(f"[pipeline-benchmark] local {model}", flush=True)
        trial = call(
            "local",
            model,
            lambda model=model: packaging.model_json_validated(
                client, model, packaging.LOCAL_EXTRACTION_SYSTEM, local_prompt,
                packaging.validate_local_result, max_completion_tokens=packaging.BENCHMARK_LOCAL_MAX_COMPLETION_TOKENS,
            ),
        )
        if trial["status"] == "ok":
            candidates = trial["result"].get("candidates", [])
            trial["summary"] = {
                "candidate_count": len(candidates),
                "scores": [candidate.get("score") for candidate in candidates],
                "hooks": [candidate.get("hook_frame_name") for candidate in candidates],
                "threads": [candidate.get("thread_key") for candidate in candidates],
            }
        report["local_candidates"][model] = trial
        write_json(output_dir / "report.json", report)

    # Stage 4b: same fixed pool, so only the global-ranking judgment changes.
    pool = json.loads((ANALYSIS_DIR / "shorts_candidates" / "local" / "all_candidates.json").read_text(encoding="utf-8"))["candidates"]
    global_prompt = packaging.build_global_prompt(pool)
    for model in MODELS:
        print(f"[pipeline-benchmark] global {model}", flush=True)
        trial = call(
            "global",
            model,
            lambda model=model: packaging.model_json_validated(
                client, model, packaging.GLOBAL_RERANK_SYSTEM, global_prompt,
                packaging.validate_global_result, max_completion_tokens=packaging.GLOBAL_MAX_COMPLETION_TOKENS,
            ),
        )
        if trial["status"] == "ok":
            selected = trial["result"].get("selected", [])
            trial["summary"] = {"selected": [item.get("candidate_id") for item in selected], "scores": [item.get("global_score") for item in selected]}
        report["global_selection"][model] = trial
        write_json(output_dir / "report.json", report)


if __name__ == "__main__":
    main()
