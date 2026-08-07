"""Verify only the selected Shorts candidates against dense source-video frames.

This is deliberately a narrow second vision pass.  The broad visual-event
script finds possible stories across a full longform; this script checks the
actual candidate cut skeleton before an editorial model is allowed to write a
headline or final package.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import imageio_ffmpeg
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

from env_loader import format_checked_env_paths, load_project_env
from usage_ledger import append_chat_usage


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-video", type=Path, required=True)
    value.add_argument("--analysis-dir", type=Path, required=True)
    value.add_argument("--candidates-json", type=Path, required=True)
    value.add_argument("--selected-json", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--model", default="gpt-5.4")
    value.add_argument("--frames-per-clip", type=int, default=2)
    value.add_argument("--force", action="store_true")
    return value


def load_client() -> OpenAI:
    load_project_env()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY not found. Checked: {format_checked_env_paths()}")
    return OpenAI(api_key=api_key, timeout=120.0, max_retries=1)


def read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def load_transcript(analysis_dir: Path, candidate: dict[str, Any]) -> list[dict[str, Any]]:
    payload = read_json(analysis_dir / "merged" / "merged_transcript.json", {})
    segments = payload.get("segments", []) if isinstance(payload, dict) else []
    try:
        low = float(candidate.get("candidate_start", 0.0)) - 4.0
        high = float(candidate.get("candidate_end", 0.0)) + 4.0
    except (TypeError, ValueError):
        return []
    compact = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        try:
            start = float(segment.get("start"))
            end = float(segment.get("end"))
        except (TypeError, ValueError):
            continue
        text = str(segment.get("text") or "").strip()
        if text and end >= low and start <= high:
            compact.append({"start": round(start, 2), "end": round(end, 2), "text": text})
    return compact[:80]


def candidate_frame_times(candidate: dict[str, Any], frames_per_clip: int) -> list[tuple[float, int]]:
    clips = candidate.get("clip_blueprint", []) if isinstance(candidate.get("clip_blueprint"), list) else []
    values: list[tuple[float, int]] = []
    for clip_index, clip in enumerate(clips, start=1):
        if not isinstance(clip, dict):
            continue
        try:
            start = float(clip.get("source_start"))
            end = float(clip.get("source_end"))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        count = max(1, min(3, frames_per_clip))
        for offset in range(count):
            ratio = (offset + 0.5) / count
            values.append((round(start + (end - start) * ratio, 3), clip_index))
    # A malformed local candidate should still get one visual check, rather
    # than being silently passed through to final packaging.
    if not values:
        try:
            values.append(((float(candidate["candidate_start"]) + float(candidate["candidate_end"])) / 2, 0))
        except (KeyError, TypeError, ValueError):
            pass
    deduplicated: list[tuple[float, int]] = []
    for time_sec, clip_index in sorted(values):
        if not deduplicated or abs(time_sec - deduplicated[-1][0]) >= 0.25:
            deduplicated.append((time_sec, clip_index))
    return deduplicated[:16]


def extract_candidate_frames(source_video: Path, output_dir: Path, candidate: dict[str, Any], frames_per_clip: int) -> list[dict[str, Any]]:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    candidate_id = str(candidate.get("candidate_id") or "candidate")
    frame_dir = output_dir / "frames" / candidate_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    # Some Windows ffmpeg builds receive non-ASCII paths through the legacy
    # process codepage.  The project root contains Korean characters, so feed
    # ffmpeg an ASCII relative source path and an ASCII temporary output; Python
    # then copies the finished image back to the normal project artifact path.
    try:
        relative_source = os.path.relpath(source_video.resolve(), Path.cwd())
    except ValueError:
        relative_source = str(source_video)
    source_arg = relative_source if relative_source.isascii() and not relative_source.startswith("..") else str(source_video)
    staging_dir = Path(tempfile.gettempdir()) / "long2shorts_candidate_frames" / candidate_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (time_sec, clip_index) in enumerate(candidate_frame_times(candidate, frames_per_clip), start=1):
        path = frame_dir / f"frame_{index:02d}_{time_sec:09.3f}.jpg"
        staging_path = staging_dir / f"frame_{index:02d}_{time_sec:09.3f}.jpg"
        if not path.exists():
            if not staging_path.exists():
                subprocess.run(
                    [
                        ffmpeg, "-y", "-ss", f"{time_sec:.3f}", "-i", source_arg,
                        "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "4", str(staging_path),
                    ],
                    check=True,
                    capture_output=True,
                )
            shutil.copy2(staging_path, path)
        records.append({"time_sec": time_sec, "clip_index": clip_index, "path": str(path)})
    return records


def active_font() -> ImageFont.ImageFont:
    for path in (Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/gulim.ttc")):
        if path.exists():
            return ImageFont.truetype(str(path), 22)
    return ImageFont.load_default()


def contact_sheet(frames: list[dict[str, Any]]) -> str:
    tiles: list[Image.Image] = []
    font = active_font()
    for frame in frames:
        image = Image.open(frame["path"]).convert("RGB")
        image.thumbnail((420, 236))
        tile = Image.new("RGB", (420, 274), "black")
        tile.paste(image, ((420 - image.width) // 2, 0))
        label = f"CUT {frame['clip_index']}  |  {frame['time_sec']:.1f}s"
        ImageDraw.Draw(tile).text((10, 244), label, fill="white", font=font)
        tiles.append(tile)
    columns = 3
    rows = max(1, (len(tiles) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * 420, rows * 274), "black")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 420, (index // columns) * 274))
    buffer = io.BytesIO()
    sheet.save(buffer, format="JPEG", quality=86)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def prompt(candidate: dict[str, Any], transcript: list[dict[str, Any]], frames: list[dict[str, Any]]) -> str:
    clip_plan = [
        {
            "cut": index,
            "start": clip.get("source_start"),
            "end": clip.get("source_end"),
            "purpose": clip.get("purpose"),
        }
        for index, clip in enumerate(candidate.get("clip_blueprint", []) or [], start=1)
        if isinstance(clip, dict)
    ]
    return f"""You are the final visual fact-checker for a Korean YouTube Shorts editor.
The attached contact sheet shows dense samples from the *planned candidate cuts*.  Each tile is labeled CUT number and its real source time.  The candidate must be rejected if the visible footage cannot support one clear, watchable story without inventing meaning.

Return JSON only in Korean:
{{
  "verdict": "ready|reject",
  "story_type": "visual_event|conversation_banter|food_reaction",
  "story": "한 문장으로, 화면과 대사로 확인된 사건",
  "grounded_title_facts": ["제목에 안전하게 쓸 수 있는 구체적 사실"],
  "safe_hook": "첫 컷에서 실제로 보이거나 들리는 훅",
  "safe_payoff": "마지막 컷에서 실제로 보이거나 들리는 결말/반응",
  "story_strength": 0,
  "payoff_strength": 0,
  "cut_evidence": [{{"cut": 1, "observed": "보이는 행동/표정/상황", "usable": true, "issue": "없으면 빈 문자열"}}],
  "crop_note": "인물을 놓치지 않는 확대/구도 지시. 불필요한 추적은 금지.",
  "rejection_reason": "reject일 때만 구체적 사유"
}}

Rules:
- Do not identify an unseen person, motive, or consequence from guesswork.
- A usable title needs a subject/role, concrete action or line, and a visible/audible reaction or consequence.
- Reject a candidate that is only continuous ordinary talk, has no understandable trigger, or whose payoff is not actually evidenced.
- Do not confuse a static camera with a weak story. A conversation_banter or food_reaction candidate may remain in one setting when timestamped dialogue plus the sampled faces or food clearly show a concrete trigger, escalation or answer, and reaction/payoff.
- A bridge cut can remain usable when it preserves the same people and situation while the escalation is carried by dialogue. Do not reject it solely because no new visual event occurs in that frame.
- Still reject vague conversation whose dialogue could be moved to any episode, or a supposed punchline not supported by the supplied timestamped dialogue.
- A polite greeting, seating 안내, or the start of another conversation is not a payoff. Reject it unless a concrete joke, reversal, answer, or strong reaction closes the same event.
- Use ready for visual_event only when both story_strength and payoff_strength are 7 or higher out of 10. For conversation_banter and food_reaction, 6 or higher is enough when trigger, escalation, and payoff are explicitly grounded in the transcript and the same situation is visually confirmed.
- Do not write a finished catchy title. Supply only facts the final editor may use.
- Check every numbered cut; do not approve a candidate merely because one still frame looks good.

Candidate skeleton:
{json.dumps({"core_event": candidate.get("core_event"), "viewer_promise": candidate.get("viewer_promise"), "clip_plan": clip_plan}, ensure_ascii=False)}

Transcript around this candidate:
{json.dumps(transcript, ensure_ascii=False)}

Frame labels provided: {json.dumps([{ "cut": item["clip_index"], "time": item["time_sec"] } for item in frames], ensure_ascii=False)}
"""


def validate(payload: Any, candidate: dict[str, Any]) -> dict[str, Any]:
    result = payload if isinstance(payload, dict) else {}
    verdict = str(result.get("verdict") or "reject").strip().lower()
    if verdict not in {"ready", "reject"}:
        verdict = "reject"
    facts = [str(item).strip() for item in result.get("grounded_title_facts", []) if str(item).strip()][:5]
    cut_evidence = [item for item in result.get("cut_evidence", []) if isinstance(item, dict)][:10]
    has_usable = any(bool(item.get("usable")) for item in cut_evidence)
    story_type = str(result.get("story_type") or "visual_event").strip().lower()
    if story_type not in {"visual_event", "conversation_banter", "food_reaction"}:
        story_type = "visual_event"
    try:
        story_strength = int(result.get("story_strength", 0))
        payoff_strength = int(result.get("payoff_strength", 0))
    except (TypeError, ValueError):
        story_strength = payoff_strength = 0
    usable_count = sum(1 for item in cut_evidence if bool(item.get("usable")))
    minimum_strength = 6 if story_type in {"conversation_banter", "food_reaction"} else 7
    minimum_usable_ratio = 0.65 if story_type in {"conversation_banter", "food_reaction"} else 1.0
    usable_ratio = usable_count / len(cut_evidence) if cut_evidence else 0.0
    if verdict == "ready" and (
        not str(result.get("story") or "").strip()
        or not facts
        or not has_usable
        or usable_ratio < minimum_usable_ratio
        or story_strength < minimum_strength
        or payoff_strength < minimum_strength
    ):
        verdict = "reject"
        result["rejection_reason"] = "명확한 사건·결말을 뒷받침할 예정 컷 화면 근거가 부족함"
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "verdict": verdict,
        "story_type": story_type,
        "story": str(result.get("story") or "").strip(),
        "grounded_title_facts": facts,
        "safe_hook": str(result.get("safe_hook") or "").strip(),
        "safe_payoff": str(result.get("safe_payoff") or "").strip(),
        "story_strength": story_strength,
        "payoff_strength": payoff_strength,
        "cut_evidence": cut_evidence,
        "crop_note": str(result.get("crop_note") or "").strip(),
        "rejection_reason": str(result.get("rejection_reason") or "").strip(),
    }


def analyze(client: OpenAI, model: str, candidate: dict[str, Any], transcript: list[dict[str, Any]], frames: list[dict[str, Any]]) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        max_completion_tokens=1600,
        messages=[
            {"role": "system", "content": "Return only conservative Korean JSON grounded in visible frames and timestamped dialogue."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt(candidate, transcript, frames)},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + contact_sheet(frames), "detail": "high"}},
                ],
            },
        ],
    )
    append_chat_usage(
        stage="candidate_visual_refinement",
        model=model,
        response=response,
        extra={"candidate_id": candidate.get("candidate_id"), "frame_count": len(frames)},
    )
    try:
        payload = json.loads(response.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        payload = {}
    return validate(payload, candidate)


def main() -> None:
    args = parser().parse_args()
    output = args.output.resolve()
    source_video = args.source_video.resolve()
    if not source_video.exists():
        raise FileNotFoundError(f"Source video not found: {source_video}")
    candidates_payload = read_json(args.candidates_json.resolve(), {})
    candidates = candidates_payload.get("candidates", []) if isinstance(candidates_payload, dict) else []
    candidate_by_id = {str(item.get("candidate_id")): item for item in candidates if isinstance(item, dict)}
    selected_payload = read_json(args.selected_json.resolve(), {})
    selected = selected_payload.get("selected", []) if isinstance(selected_payload, dict) else []
    existing = read_json(output, {}) if output.exists() and not args.force else {}
    results = [item for item in existing.get("candidates", []) if isinstance(item, dict)] if isinstance(existing, dict) else []
    completed_ids = {str(item.get("candidate_id")) for item in results if item.get("candidate_id")}
    pending = [item for item in selected if isinstance(item, dict) and str(item.get("candidate_id")) not in completed_ids]
    if not pending:
        print(f"[candidate-vision] existing_complete={output}", flush=True)
        return
    client = load_client()
    output.parent.mkdir(parents=True, exist_ok=True)
    def persist() -> None:
        output.write_text(
            json.dumps({"version": 1, "model": args.model, "candidate_count": len(results), "candidates": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    for index, item in enumerate(pending, start=1):
        candidate = candidate_by_id.get(str(item.get("candidate_id"))) if isinstance(item, dict) else None
        if not candidate:
            continue
        print(f"[candidate-vision] {index}/{len(selected)} -> {candidate['candidate_id']}", flush=True)
        try:
            frames = extract_candidate_frames(source_video, output.parent, candidate, int(args.frames_per_clip))
            evidence = analyze(client, args.model, candidate, load_transcript(args.analysis_dir.resolve(), candidate), frames)
        except Exception as exc:
            evidence = {
                "candidate_id": candidate.get("candidate_id"), "verdict": "reject",
                "story": "", "grounded_title_facts": [], "safe_hook": "", "safe_payoff": "",
                "cut_evidence": [], "crop_note": "", "rejection_reason": f"visual verification failed: {exc}",
            }
        results.append(evidence)
        persist()
    ready_count = sum(1 for item in results if item.get("verdict") == "ready")
    print(f"[candidate-vision] ready={ready_count}/{len(results)} output={output}", flush=True)


if __name__ == "__main__":
    main()
