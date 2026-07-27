"""Compare global candidate ranking only for models that passed local discovery."""

from __future__ import annotations

import json
import time
from pathlib import Path

import generate_shorts_packages as packaging


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis" / "youtube_bEwGLTxqCOI"
MODELS = ["gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4"]


def main() -> None:
    packaging.configure_runtime(
        ANALYSIS_DIR,
        "악명 높은 영국 음식 먹어보기 (w.명예영국인)",
        None,
        ANALYSIS_DIR / "youtube_context.json",
        benchmark_profile_path=ROOT / "templates" / "benchmark_profiles" / "rescene_gyaru_variety.json",
    )
    pool = json.loads((ANALYSIS_DIR / "shorts_candidates" / "local" / "all_candidates.json").read_text(encoding="utf-8"))["candidates"]
    prompt = packaging.build_global_prompt(pool)
    client = packaging.load_client()
    report = {"scope": "Same fixed local candidate pool; only global ranking changes.", "models": {}}
    output = ANALYSIS_DIR / "model_benchmark" / "pipeline_stages" / "global_survivors.json"
    for model in MODELS:
        print(f"[global-benchmark] {model}", flush=True)
        started = time.monotonic()
        try:
            result = packaging.model_json_validated(
                client, model, packaging.GLOBAL_RERANK_SYSTEM, prompt,
                packaging.validate_global_result, max_completion_tokens=packaging.GLOBAL_MAX_COMPLETION_TOKENS,
            )
            report["models"][model] = {
                "status": "ok",
                "elapsed_sec": round(time.monotonic() - started, 2),
                "selected": result.get("selected", []),
            }
        except Exception as exc:
            report["models"][model] = {"status": "error", "elapsed_sec": round(time.monotonic() - started, 2), "error": str(exc)}
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
