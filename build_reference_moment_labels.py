from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent


def load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turn reference-video timestamp comments into a manual shorts-scoring label set."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--moments-per-video", type=int, default=12)
    parser.add_argument("--lead-sec", type=float, default=5.0)
    parser.add_argument("--tail-sec", type=float, default=10.0)
    return parser


def label_template(profile: dict[str, Any]) -> dict[str, Any]:
    score = profile.get("score") if isinstance(profile.get("score"), dict) else {}
    dimensions = score.get("dimensions") if isinstance(score.get("dimensions"), list) else []
    return {
        "verdict": "unlabeled",
        "score": None,
        "dimension_scores": {
            str(item.get("id")): None
            for item in dimensions
            if isinstance(item, dict) and item.get("id")
        },
        "penalties": [],
        "observed_trigger": "",
        "observed_reaction": "",
        "observed_payoff": "",
        "crop_risk": "unknown",
        "notes": "",
    }


def moment_record(
    source: dict[str, Any],
    moment: dict[str, Any],
    index: int,
    lead_sec: float,
    tail_sec: float,
    profile: dict[str, Any],
) -> dict[str, Any]:
    time_sec = float(moment.get("representative_time_sec", moment.get("bucket_start_sec", 0)) or 0)
    samples = moment.get("samples") if isinstance(moment.get("samples"), list) else []
    return {
        "reference_id": f"{source.get('label', 'source')}_moment_{index:02d}",
        "source_label": source.get("label", ""),
        "source_url": source.get("source_url", ""),
        "video_id": source.get("video_id", ""),
        "expected_engine": source.get("expected_engine", ""),
        "candidate_window": {
            "start_sec": round(max(0.0, time_sec - lead_sec), 3),
            "anchor_sec": round(time_sec, 3),
            "end_sec": round(time_sec + tail_sec, 3),
        },
        "audience_signal": {
            "comment_count": int(moment.get("comment_count", 0) or 0),
            "like_count": int(moment.get("like_count", 0) or 0),
            "reply_count": int(moment.get("reply_count", 0) or 0),
            "reaction_score": float(moment.get("reaction_score", 0) or 0),
            "samples": samples[:3],
        },
        "label": label_template(profile),
    }


def main() -> None:
    args = build_parser().parse_args()
    summary_path = args.summary.resolve()
    profile_path = args.profile.resolve()
    summary = load_json(summary_path)
    profile = load_json(profile_path)
    output_path = (args.output or summary_path.parent / "reference_moment_labels.json").resolve()
    max_moments = max(1, args.moments_per_video)

    references = []
    for source in summary.get("sources", []):
        if not isinstance(source, dict) or not source.get("context_path"):
            continue
        context_path = Path(str(source["context_path"]))
        if not context_path.exists():
            continue
        context = load_json(context_path)
        insights = context.get("comment_insights") if isinstance(context.get("comment_insights"), dict) else {}
        moments = insights.get("timecode_moments") if isinstance(insights.get("timecode_moments"), list) else []
        labels = [
            moment_record(source, moment, index, args.lead_sec, args.tail_sec, profile)
            for index, moment in enumerate(moments[:max_moments], start=1)
            if isinstance(moment, dict)
        ]
        references.append(
            {
                "source_label": source.get("label", ""),
                "source_url": source.get("source_url", ""),
                "video_id": source.get("video_id", ""),
                "status": source.get("status", ""),
                "candidate_moments": labels,
            }
        )

    payload = {
        "profile_id": profile.get("profile_id", ""),
        "labeling_rule": "Score a candidate only after watching its source window. Timestamp comments are audience-interest signals, not proof that the window is a good standalone short.",
        "references": references,
        "candidate_count": sum(len(item["candidate_moments"]) for item in references),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[labels] candidates={payload['candidate_count']}", flush=True)
    print(f"[labels] output={output_path}", flush=True)


if __name__ == "__main__":
    main()
