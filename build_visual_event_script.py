"""Create a time-aligned visual event script from sampled video frames and dialogue."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import imageio_ffmpeg
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

from env_loader import format_checked_env_paths, load_project_env
from usage_ledger import append_chat_usage, summarize_usage, usage_log_path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_INTERVAL_SEC = 12
DEFAULT_FRAMES_PER_BATCH = 6
DEFAULT_MAX_FRAMES = 180


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample-interval-sec", type=int, default=DEFAULT_INTERVAL_SEC)
    parser.add_argument("--frames-per-batch", type=int, default=DEFAULT_FRAMES_PER_BATCH)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument(
        "--max-estimated-usd",
        type=float,
        default=0.0,
        help="Stop before starting another visual-analysis batch once this per-run estimated budget is exhausted.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def load_client() -> OpenAI:
    load_project_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY not found. Checked: {format_checked_env_paths()}")
    return OpenAI(api_key=api_key, timeout=120.0, max_retries=1)


def extract_frames(source_video: Path, frame_dir: Path, interval_sec: int, max_frames: int) -> list[dict[str, Any]]:
    """Choose one representative per time slot, preferring a real cut-change frame."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = frame_dir / "frame_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest.get("frames", []) if isinstance(manifest, dict) else []
        if records and all(Path(item["path"]).exists() for item in records):
            return records[:max_frames] if max_frames > 0 else records

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    anchor_dir = frame_dir / "anchors"
    scene_dir = frame_dir / "scene_changes"
    anchor_dir.mkdir(exist_ok=True)
    scene_dir.mkdir(exist_ok=True)
    if not list(anchor_dir.glob("*.jpg")):
        subprocess.run(
            [
                ffmpeg, "-y", "-i", str(source_video), "-vf", f"fps=1/{interval_sec},scale=640:-2",
                "-q:v", "4", str(anchor_dir / "anchor_%05d.jpg"),
            ],
            check=True, capture_output=True,
        )
    scene_paths = sorted(scene_dir.glob("scene_*.jpg"))
    scene_times: list[float] = []
    if not scene_paths:
        completed = subprocess.run(
            [
                ffmpeg, "-y", "-i", str(source_video), "-vf",
                "select=gt(scene\\,0.35),scale=640:-2,showinfo", "-vsync", "vfr", "-q:v", "4",
                str(scene_dir / "scene_%05d.jpg"),
            ],
            check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        import re
        scene_times = [float(value) for value in re.findall(r"pts_time:([0-9.]+)", completed.stderr)]
        scene_paths = sorted(scene_dir.glob("scene_*.jpg"))
        (scene_dir / "times.json").write_text(json.dumps(scene_times), encoding="utf-8")
    else:
        times_path = scene_dir / "times.json"
        if times_path.exists():
            scene_times = [float(value) for value in json.loads(times_path.read_text(encoding="utf-8"))]

    anchors = [
        {"time_sec": round(index * interval_sec, 3), "path": path, "kind": "anchor"}
        for index, path in enumerate(sorted(anchor_dir.glob("anchor_*.jpg")))
    ]
    scenes = [
        {"time_sec": round(time_sec, 3), "path": path, "kind": "scene_change"}
        for time_sec, path in zip(scene_times, scene_paths)
    ]
    selected = []
    for anchor in anchors:
        near = [scene for scene in scenes if abs(scene["time_sec"] - anchor["time_sec"]) <= interval_sec / 2]
        selected.append(min(near, key=lambda item: abs(item["time_sec"] - anchor["time_sec"])) if near else anchor)
    if max_frames > 0 and len(selected) > max_frames:
        indices = [round(index * (len(selected) - 1) / (max_frames - 1)) for index in range(max_frames)] if max_frames > 1 else [0]
        selected = [selected[index] for index in sorted(set(indices))]
    records = [
        {"index": index, "time_sec": item["time_sec"], "path": str(item["path"]), "selection": item["kind"]}
        for index, item in enumerate(selected)
    ]
    manifest_path.write_text(json.dumps({"frames": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def load_transcript(analysis_dir: Path) -> list[dict[str, Any]]:
    path = analysis_dir / "merged" / "merged_transcript.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("segments", []) if isinstance(data, dict) else []


def transcript_window(segments: list[dict[str, Any]], start_sec: float, end_sec: float) -> list[dict[str, Any]]:
    return [
        {
            "start": round(float(segment.get("start", 0)), 3),
            "end": round(float(segment.get("end", 0)), 3),
            "text": str(segment.get("text", "")).strip(),
        }
        for segment in segments
        if float(segment.get("end", 0)) >= start_sec - 6 and float(segment.get("start", 0)) <= end_sec + 6
        and str(segment.get("text", "")).strip()
    ]


def font() -> ImageFont.ImageFont:
    for path in [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/gulim.ttc"),
    ]:
        if path.exists():
            return ImageFont.truetype(str(path), 24)
    return ImageFont.load_default()


def contact_sheet(frames: list[dict[str, Any]]) -> str:
    tiles: list[Image.Image] = []
    active_font = font()
    for frame in frames:
        image = Image.open(frame["path"]).convert("RGB")
        image.thumbnail((480, 270))
        tile = Image.new("RGB", (480, 310), "black")
        tile.paste(image, ((480 - image.width) // 2, 0))
        ImageDraw.Draw(tile).text((12, 278), f"{frame['time_sec']:.1f}s", fill="white", font=active_font)
        tiles.append(tile)
    columns = 3
    rows = max(1, (len(tiles) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * 480, rows * 310), "black")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 480, (index // columns) * 310))
    buffer = io.BytesIO()
    sheet.save(buffer, format="JPEG", quality=86)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def batch_prompt(frames: list[dict[str, Any]], segments: list[dict[str, Any]]) -> str:
    start = frames[0]["time_sec"]
    end = frames[-1]["time_sec"]
    return f"""You are creating a Korean visual event script for one long-form YouTube video.
The image is a contact sheet. Each frame has its real source timestamp printed below it.

Only describe what is visibly supported by the frames and what is audibly supported by the timestamped transcript. Do not infer off-screen actions, identities, emotions, or causes that neither source proves.
Find 0 to 3 distinct, reusable event threads in this {start:.1f}s to {end:.1f}s window. A useful event has a visible context, action/change, reaction or consequence, and a possible payoff. A quiet talking-head stretch may correctly produce zero events.

Return JSON only:
{{
  "events": [
    {{
      "event_id": "batch_event_01",
      "start_sec": 0.0,
      "end_sec": 0.0,
      "context": "Korean: who/where/visible setup",
      "visible_setup": "Korean: objects, framing, pose, or situation that can be seen",
      "action": "Korean: visible action or dialogue-backed change",
      "reaction": "Korean: visible/audible reaction; say unknown if not supported",
      "payoff": "Korean: visible or spoken consequence; say unresolved if none",
      "visual_hook": "Korean: concise viewer-facing visual promise",
      "visual_shortability_score": 0,
      "editorial_role": "hook|context|escalation|reaction|payoff|unusable",
      "characters": ["role or confirmed name"],
      "frame_timestamps": [0.0],
      "confidence": "high|medium|low"
    }}
  ]
}}

Timestamped transcript near this contact sheet:
{json.dumps(transcript_window(segments, start, end), ensure_ascii=False)}
"""


def validate_events(payload: Any, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        return []
    lower = frames[0]["time_sec"] - 6
    upper = frames[-1]["time_sec"] + 12
    events = []
    for index, event in enumerate(payload["events"], start=1):
        if not isinstance(event, dict):
            continue
        try:
            start = float(event.get("start_sec"))
            end = float(event.get("end_sec"))
        except (TypeError, ValueError):
            continue
        if end <= start or start < lower or end > upper:
            continue
        if not str(event.get("context") or "").strip() or not str(event.get("visual_hook") or "").strip():
            continue
        event["event_id"] = str(event.get("event_id") or f"event_{index:03d}")
        event["start_sec"] = round(start, 3)
        event["end_sec"] = round(end, 3)
        event["frame_timestamps"] = [round(float(value), 3) for value in event.get("frame_timestamps", []) if isinstance(value, (int, float))]
        event["characters"] = [str(value).strip() for value in event.get("characters", []) if str(value).strip()][:4]
        if event.get("confidence") not in {"high", "medium", "low"}:
            event["confidence"] = "low"
        try:
            event["visual_shortability_score"] = max(0, min(10, int(event.get("visual_shortability_score", 0))))
        except (TypeError, ValueError):
            event["visual_shortability_score"] = 0
        if event.get("editorial_role") not in {"hook", "context", "escalation", "reaction", "payoff", "unusable"}:
            event["editorial_role"] = "unusable"
        events.append(event)
    return events


def analyze_batch(client: OpenAI, model: str, frames: list[dict[str, Any]], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        max_completion_tokens=2200,
        messages=[
            {"role": "system", "content": "Return a conservative evidence-grounded Korean visual event script as JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": batch_prompt(frames, segments)},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + contact_sheet(frames), "detail": "high"}},
                ],
            },
        ],
    )
    append_chat_usage(
        stage="visual_event_analysis",
        model=model,
        response=response,
        extra={"frame_count": len(frames)},
    )
    try:
        payload = json.loads(response.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return []
    return validate_events(payload, frames)


def visual_budget_exhausted(max_estimated_usd: float) -> bool:
    if max_estimated_usd <= 0:
        return False
    path = usage_log_path()
    if not path or not path.exists():
        return False
    summary = summarize_usage(path)
    visual = summary.get("by_stage", {}).get("visual_event_analysis", {})
    return float(visual.get("estimated_usd", 0.0) or 0.0) >= max_estimated_usd


def main() -> None:
    args = build_parser().parse_args()
    source_video = args.source_video.resolve()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = analysis_dir / "visual_events"
    script_path = output_dir / "visual_event_script.json"
    if script_path.exists() and not args.force:
        print(f"[vision] existing_script={script_path}", flush=True)
        return
    if not source_video.exists():
        raise FileNotFoundError(f"Source video not found: {source_video}")
    output_dir.mkdir(parents=True, exist_ok=True)
    interval = max(4, int(args.sample_interval_sec))
    frames_per_batch = max(2, min(12, int(args.frames_per_batch)))
    frames = extract_frames(source_video, output_dir / "frames", interval, max(1, int(args.max_frames)))
    segments = load_transcript(analysis_dir)
    client = load_client()
    events: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    for offset in range(0, len(frames), frames_per_batch):
        group = frames[offset : offset + frames_per_batch]
        if visual_budget_exhausted(float(args.max_estimated_usd or 0.0)):
            batches.append(
                {
                    "status": "budget_capped",
                    "start_sec": group[0]["time_sec"],
                    "end_sec": group[-1]["time_sec"],
                    "max_estimated_usd": float(args.max_estimated_usd),
                }
            )
            print(f"[vision] budget_capped=${float(args.max_estimated_usd):.2f}", flush=True)
            break
        print(f"[vision] batch={offset // frames_per_batch + 1} frames={len(group)}", flush=True)
        try:
            batch_events = analyze_batch(client, args.model, group, segments)
            for index, event in enumerate(batch_events, start=1):
                event["event_id"] = f"visual_{offset // frames_per_batch + 1:03d}_{index:02d}"
            events.extend(batch_events)
            batches.append({"status": "ok", "start_sec": group[0]["time_sec"], "end_sec": group[-1]["time_sec"], "event_count": len(batch_events)})
        except Exception as exc:
            batches.append({"status": "failed", "start_sec": group[0]["time_sec"], "end_sec": group[-1]["time_sec"], "error": str(exc)})
    payload = {
        "version": 2,
        "source_video": str(source_video),
        "model": args.model,
        "sample_interval_sec": interval,
        "frames_per_batch": frames_per_batch,
        "frame_count": len(frames),
        "event_count": len(events),
        "events": events,
        "batches": batches,
    }
    script_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[vision] event_count={len(events)} script={script_path}", flush=True)


if __name__ == "__main__":
    main()
