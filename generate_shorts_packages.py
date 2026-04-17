from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import unicodedata
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent / "auto_Youtube" / "shorts" / ".env"

ANALYSIS_DIR = BASE_DIR / "analysis" / "jiunsudaetong1"
TRANSCRIPTS_DIR = ANALYSIS_DIR / "transcripts"
MERGED_TRANSCRIPT_PATH = ANALYSIS_DIR / "merged" / "merged_transcript.json"
OUTPUT_DIR = ANALYSIS_DIR / "shorts_candidates"
SOURCE_TITLE = "지은수대통 1회"
MOVIE_INFO: Optional[dict] = None
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_TEMPERATURE = 0.2
CHUNK_TOP_K = 4
FINAL_TOP_K = 10
MAX_RETRIES = 3
TARGET_DURATION_MIN = 22.0
TARGET_DURATION_MAX = 40.0
ALLOWED_DURATION_MIN = 18.0
ALLOWED_DURATION_MAX = 55.0
MIN_CLIP_COUNT = 5
RECOMMENDED_CLIP_COUNT_MIN = 5
RECOMMENDED_CLIP_COUNT_MAX = 8
MAX_CLIP_COUNT = 10
MIN_CLIP_DURATION_SEC = 0.4
MAX_FINAL_CLIP_DURATION_SEC = 14.0
MAX_LOCAL_CLIP_DURATION_SEC = 20.0
TARGET_DURATION_TOLERANCE_SEC = 9.0
LOCAL_BLUEPRINT_MIN_DURATION_SEC = 12.0
LOCAL_BLUEPRINT_MAX_DURATION_SEC = 60.0
CLIP_PURPOSES = {
    "hook",
    "context",
    "reaction",
    "bridge",
    "payoff",
    "conflict",
    "reveal",
    "aftermath",
}
ENDING_PURPOSES = {"payoff", "reveal", "reaction", "aftermath"}
MIDDLE_PURPOSES = {"context", "bridge", "conflict", "reaction"}

NOISE_PATTERNS = {
    "마이콜과 마이콜의 대화",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate reusable shorts packages from transcript analysis.")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Analysis directory created by analyze_longform.py",
    )
    parser.add_argument(
        "--source-title",
        type=str,
        default="",
        help="Display title used in prompts and output metadata.",
    )
    parser.add_argument(
        "--movie-info",
        type=Path,
        default=None,
        help="Optional movie info JSON file to stabilize plot/context understanding.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true")
    return parser


def configure_runtime(analysis_dir: Path, source_title: str, movie_info_path: Optional[Path]) -> None:
    global ANALYSIS_DIR, TRANSCRIPTS_DIR, MERGED_TRANSCRIPT_PATH, OUTPUT_DIR, SOURCE_TITLE, MOVIE_INFO
    ANALYSIS_DIR = analysis_dir
    TRANSCRIPTS_DIR = ANALYSIS_DIR / "transcripts"
    MERGED_TRANSCRIPT_PATH = ANALYSIS_DIR / "merged" / "merged_transcript.json"
    OUTPUT_DIR = ANALYSIS_DIR / "shorts_candidates"
    SOURCE_TITLE = source_title.strip() or analysis_dir.name
    if movie_info_path and movie_info_path.exists():
        with open(movie_info_path, "r", encoding="utf-8") as f:
            MOVIE_INFO = json.load(f)
    else:
        MOVIE_INFO = None


LOCAL_EXTRACTION_SYSTEM = """You are a senior Korean movie/drama shorts editor.
You find reliable shorts candidates from transcripts.

You must optimize for stop-scroll strength and view potential, not safe coverage.
Always use the same standards:
- hook within first 1 to 3 seconds
- premise should be understandable in one short sentence
- clear conflict, humiliation, money tension, power abuse, scam, corruption, betrayal, reversal, or payoff
- understandable even if character names are incomplete
- editable into at least 5 cuts

Market framing rule:
- every candidate must have a market-facing hook frame
- a hook frame is the viewer-facing promise of the short, not the whole movie genre
- hook frames must be inferred from the scene itself, not forced globally across the movie
- examples of hook frames: "무명 배우 굴욕", "가짜 권력 응징", "돈 때문에 버티는 청춘", "권력 상승과 배신", "부패 시스템 폭로", "오해가 부른 파국"
- prefer candidates where protagonist role, pressure source, and payoff are legible in one short phrase
- avoid candidates that are only emotional but cannot be packaged with a clear hook frame

View-maximizing rule:
- prefer scenes that make a viewer instantly ask "what happened?" or "how does this end?"
- prefer power imbalance, exposure, scam, corruption, humiliation, comeback, authority bluff, betrayal, or money stress over quiet mood pieces
- a strong candidate usually has a clear role label, a clear opponent or target, and a clear payoff
- do not choose scenes that only feel dramatic after long explanation
- do not choose scenes just because the acting is good; choose scenes that are clickable

Editing rules:
- every final short must be reconstructable into at least 5 cuts
- recommended cut count is 5 to 8
- maximum cut count is 10
- source order does not need to be preserved
- a later moment may open the short if it is the strongest hook
- the result must still remain easy to follow

Narration rule:
- default is no narration
- narration is allowed only when a short would otherwise be confusing

Uniqueness rule:
- do not generate multiple candidates from the same core scene just by changing title angle, emotion label, or wording
- if two candidates rely on mostly the same footage, they are duplicates
- if a later hook and an earlier context still lead to the same payoff beat, treat that as one candidate, not two

Ignore obvious ASR junk, repeated hallucinations, and non-dialogue artifacts.
Return JSON only.
"""


GLOBAL_RERANK_SYSTEM = """You are the lead Korean movie/drama shorts editor.
You choose the strongest final shorts from a pool of local candidates.

Priorities:
1. strong opening hook
2. instantly legible stop-scroll premise
3. clear and marketable hook frame
4. clear conflict and payoff
5. editability into a satisfying 5+ cut short
6. emotional clarity

Do not keep weak duplicates.
Prefer authority abuse, scam exposure, corruption, humiliation, money stress, betrayal, reversal, and comeback when available.
Prefer candidates whose clip blueprint already shows a tight hook -> context recovery -> payoff flow.
Prefer candidates with low narration need and compact source duration.
- prefer candidates whose hook_frame_name is instantly legible and title-friendly
- do not confuse "different emotion" with "different hook frame"
- if multiple candidates share the same hook_frame_name, keep only the strongest one unless the source scene and payoff are clearly different
- do not reward variety for its own sake
- different titles on the same scene do not count as different shorts
- if two candidates would use mostly the same footage, keep only the stronger one
- it is acceptable to keep multiple candidates from the same frame family if they come from clearly different scenes and each has strong view potential
Return JSON only.
"""


PACKAGING_SYSTEM = """You package one Korean movie/drama short for CapCut rough-cut generation.

Rules:
- title must be exactly 2 lines
- title should feel clickable, specific, and stop-scroll friendly
- title should reveal the candidate's hook frame quickly when possible
- title should not read like a calm synopsis
- title should usually surface role, target, conflict, or payoff fast
- narration should be omitted unless needed
- maximum 2 narration lines
- maximum 3 point captions
- the short must be reconstructable into at least 5 cuts
- source order may be rearranged for hook and flow
- use the candidate clip blueprint as the starting skeleton, not as a loose suggestion
- the first cut must function as a hook
- if you open with a later hook, restore minimum context in the next 1 to 2 cuts
- the last cut must land on payoff, reveal, reaction, or aftermath
- keep one emotional event only; do not combine unrelated beats
- the summed source_clips duration must stay close to the target duration
- point caption times must be relative to the short timeline
- narration target_start must also be relative to the short timeline
- use natural Korean
- if names are uncertain, use roles or relationship labels

Return JSON only.
"""


def load_client() -> OpenAI:
    load_dotenv(ENV_PATH)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY not found in {ENV_PATH}")
    return OpenAI(api_key=api_key)


def ensure_output_dirs() -> None:
    for path in [
        OUTPUT_DIR,
        OUTPUT_DIR / "local",
        OUTPUT_DIR / "global",
        OUTPUT_DIR / "final",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def parse_json_response(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model did not return valid JSON:\n{text}") from exc


def model_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return parse_json_response(response.choices[0].message.content or "{}")


def clip_duration_sec(clip: dict) -> float:
    return float(clip["source_end"]) - float(clip["source_start"])


def total_clip_duration_sec(clips: list[dict]) -> float:
    return round(sum(clip_duration_sec(clip) for clip in clips), 3)


def validate_clip_plan(
    clips: list[dict],
    *,
    target_duration: Optional[float],
    max_clip_duration: float,
    label: str,
) -> tuple[bool, str]:
    if not isinstance(clips, list) or len(clips) < MIN_CLIP_COUNT or len(clips) > MAX_CLIP_COUNT:
        return False, f"{label} must contain {MIN_CLIP_COUNT} to {MAX_CLIP_COUNT} items."

    purposes: list[str] = []
    total_duration = 0.0
    for idx, clip in enumerate(clips, start=1):
        try:
            start = float(clip.get("source_start"))
            end = float(clip.get("source_end"))
        except Exception:
            return False, f"{label} {idx} has invalid source_start/source_end."
        if end <= start:
            return False, f"{label} {idx} end must be greater than start."

        duration = end - start
        if duration < MIN_CLIP_DURATION_SEC:
            return False, f"{label} {idx} is too short for meaningful editing."
        if duration > max_clip_duration:
            return False, f"{label} {idx} is too long and will weaken short-form rhythm."

        purpose = str(clip.get("purpose") or "").strip().lower()
        if purpose not in CLIP_PURPOSES:
            allowed = ", ".join(sorted(CLIP_PURPOSES))
            return False, f"{label} {idx} purpose must be one of: {allowed}."

        purposes.append(purpose)
        total_duration += duration

    if purposes[0] != "hook":
        return False, f"{label} must open with purpose 'hook'."
    if purposes[-1] not in ENDING_PURPOSES:
        return False, f"{label} must end with payoff, reveal, reaction, or aftermath."
    if not any(purpose in {"payoff", "reveal"} for purpose in purposes):
        return False, f"{label} must contain a payoff or reveal beat."
    if not any(purpose in MIDDLE_PURPOSES for purpose in purposes[1:-1]):
        return False, f"{label} needs middle cuts that restore context or escalate conflict."

    if target_duration is None:
        if total_duration < LOCAL_BLUEPRINT_MIN_DURATION_SEC or total_duration > LOCAL_BLUEPRINT_MAX_DURATION_SEC:
            return False, f"{label} total duration must stay within a plausible shorts range."
    else:
        lower = max(ALLOWED_DURATION_MIN, target_duration - TARGET_DURATION_TOLERANCE_SEC)
        upper = min(ALLOWED_DURATION_MAX, target_duration + TARGET_DURATION_TOLERANCE_SEC)
        if total_duration < lower or total_duration > upper:
            return (
                False,
                f"{label} total duration must stay near target_duration_sec "
                f"({lower:.1f}s to {upper:.1f}s allowed, got {total_duration:.1f}s).",
            )

    return True, ""


def format_clip_plan_for_prompt(clips: list[dict]) -> str:
    if not clips:
        return "[]"
    compact = []
    for clip in clips:
        compact.append(
            {
                "source_start": clip.get("source_start"),
                "source_end": clip.get("source_end"),
                "duration_sec": round(float(clip.get("source_end", 0.0)) - float(clip.get("source_start", 0.0)), 3),
                "purpose": clip.get("purpose"),
            }
        )
    return json.dumps(compact, ensure_ascii=False, indent=2)


def validate_local_result(result: dict) -> tuple[bool, str]:
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        return False, "Missing candidates list."
    for idx, cand in enumerate(candidates, start=1):
        if not cand.get("candidate_id"):
            return False, f"Candidate {idx} missing candidate_id."
        if not cand.get("core_event"):
            return False, f"Candidate {idx} missing core_event."
        if not cand.get("hook_frame_name"):
            return False, f"Candidate {idx} missing hook_frame_name."
        if not cand.get("hook_frame_reason"):
            return False, f"Candidate {idx} missing hook_frame_reason."
        if not cand.get("viewer_promise"):
            return False, f"Candidate {idx} missing viewer_promise."
        try:
            score = int(cand.get("score"))
            start = float(cand.get("candidate_start"))
            end = float(cand.get("candidate_end"))
        except Exception:
            return False, f"Candidate {idx} has invalid score/start/end."
        if score < 60 or score > 100:
            return False, f"Candidate {idx} score must be 60 to 100."
        if end <= start:
            return False, f"Candidate {idx} end must be greater than start."

        narration_need = cand.get("narration_need")
        if narration_need not in {"none", "low", "medium"}:
            return False, f"Candidate {idx} narration_need must be none, low, or medium."

        hook_moment = cand.get("hook_moment", {})
        payoff_moment = cand.get("payoff_moment", {})
        try:
            hook_time = float(hook_moment.get("time"))
            payoff_time = float(payoff_moment.get("time"))
        except Exception:
            return False, f"Candidate {idx} hook_moment/payoff_moment time invalid."
        if not (start <= hook_time <= end):
            return False, f"Candidate {idx} hook_moment must stay inside candidate range."
        if not (start <= payoff_time <= end):
            return False, f"Candidate {idx} payoff_moment must stay inside candidate range."

        blueprint = cand.get("clip_blueprint")
        ok, message = validate_clip_plan(
            blueprint,
            target_duration=None,
            max_clip_duration=MAX_LOCAL_CLIP_DURATION_SEC,
            label=f"Candidate {idx} clip_blueprint",
        )
        if not ok:
            return False, message
    return True, ""


def validate_global_result(result: dict) -> tuple[bool, str]:
    selected = result.get("selected")
    if not isinstance(selected, list):
        return False, "Missing selected list."
    seen_candidate_ids = set()
    seen_ranks = set()
    for idx, item in enumerate(selected, start=1):
        if not item.get("candidate_id"):
            return False, f"Selection {idx} missing candidate_id."
        try:
            rank = int(item.get("global_rank"))
            score = int(item.get("global_score"))
        except Exception:
            return False, f"Selection {idx} rank/score invalid."
        if rank < 1:
            return False, f"Selection {idx} global_rank must be >= 1."
        if score < 70 or score > 100:
            return False, f"Selection {idx} global_score must be 70 to 100."
        if item["candidate_id"] in seen_candidate_ids:
            return False, f"Selection {idx} candidate_id is duplicated."
        if rank in seen_ranks:
            return False, f"Selection {idx} global_rank is duplicated."
        if not item.get("selection_reason"):
            return False, f"Selection {idx} missing selection_reason."
        seen_candidate_ids.add(item["candidate_id"])
        seen_ranks.add(rank)
    return True, ""


def validate_final_result(result: dict) -> tuple[bool, str]:
    for key in [
        "short_id",
        "core_event",
        "emotion_arc",
        "title_line1",
        "title_line2",
        "selection_pitch",
        "hook_line",
        "protagonist_presence",
        "standalone_clarity",
    ]:
        if not result.get(key):
            return False, f"Missing {key}."
    try:
        score = int(result.get("score"))
        duration = float(result.get("target_duration_sec"))
    except Exception:
        return False, "Invalid score or target_duration_sec."
    if score < 70 or score > 100:
        return False, "score must be 70 to 100."
    if duration < 18 or duration > 55:
        return False, "target_duration_sec must be 18 to 55."

    clips = result.get("source_clips")
    ok, message = validate_clip_plan(
        clips,
        target_duration=duration,
        max_clip_duration=MAX_FINAL_CLIP_DURATION_SEC,
        label="source_clips",
    )
    if not ok:
        return False, message

    narration = result.get("narration", [])
    if not isinstance(narration, list) or len(narration) > 2:
        return False, "narration must contain 0 to 2 items."
    for idx, item in enumerate(narration, start=1):
        if not item.get("text"):
            return False, f"Narration {idx} missing text."
        try:
            target_start = float(item.get("target_start"))
        except Exception:
            return False, f"Narration {idx} target_start invalid."
        if target_start < 0 or target_start > duration:
            return False, f"Narration {idx} target_start must be within the short timeline."

    captions = result.get("point_captions", [])
    if not isinstance(captions, list) or len(captions) > 3:
        return False, "point_captions must contain 0 to 3 items."
    for idx, item in enumerate(captions, start=1):
        if not item.get("text"):
            return False, f"Point caption {idx} missing text."
        try:
            target_start = float(item.get("target_start"))
            target_end = float(item.get("target_end"))
        except Exception:
            return False, f"Point caption {idx} target_start/target_end invalid."
        if not (0 <= target_start < target_end <= duration):
            return False, f"Point caption {idx} must stay within the short timeline."

    fun_tags = result.get("fun_tags", [])
    if not isinstance(fun_tags, list) or len(fun_tags) < 1 or len(fun_tags) > 3:
        return False, "fun_tags must contain 1 to 3 items."

    main_characters = result.get("main_characters", [])
    if not isinstance(main_characters, list) or len(main_characters) < 1 or len(main_characters) > 3:
        return False, "main_characters must contain 1 to 3 items."

    if result.get("protagonist_presence") not in {"high", "medium", "low", "unknown"}:
        return False, "protagonist_presence must be high, medium, low, or unknown."
    if result.get("standalone_clarity") not in {"high", "medium", "low"}:
        return False, "standalone_clarity must be high, medium, or low."

    target_cut_count_min = int(result.get("target_cut_count_min", 0))
    target_cut_count_max = int(result.get("target_cut_count_max", 0))
    recommended = result.get("target_cut_count_recommended", [])
    if target_cut_count_min < MIN_CLIP_COUNT:
        return False, "target_cut_count_min must be at least 5."
    if target_cut_count_max > MAX_CLIP_COUNT:
        return False, "target_cut_count_max must be at most 10."
    if not isinstance(recommended, list) or len(recommended) != 2:
        return False, "target_cut_count_recommended must contain two integers."
    try:
        rec_min = int(recommended[0])
        rec_max = int(recommended[1])
    except Exception:
        return False, "target_cut_count_recommended values must be integers."
    if not (target_cut_count_min <= rec_min <= rec_max <= target_cut_count_max):
        return False, "target_cut_count_recommended must stay within min/max cut counts."
    return True, ""


def model_json_validated(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    validator,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    prompt = user_prompt
    last_error = ""
    for _ in range(MAX_RETRIES):
        result = model_json(client, model, system_prompt, prompt, temperature=temperature)
        ok, message = validator(result)
        if ok:
            return result
        last_error = message
        prompt = (
            user_prompt
            + "\n\nYour previous JSON failed validation.\n"
            + f"Validation error: {message}\n"
            + "Return corrected JSON only. Do not omit required fields."
        )
    raise RuntimeError(f"Model output failed validation after {MAX_RETRIES} tries: {last_error}")


def normalize_text(text: str) -> str:
    value = (text or "").encode("utf-8", "ignore").decode("utf-8", "ignore")
    cleaned_chars = []
    for ch in value:
        category = unicodedata.category(ch)
        if category.startswith("C") and ch not in {" ", "\t"}:
            continue
        cleaned_chars.append(ch)
    return " ".join("".join(cleaned_chars).strip().split())


def clean_segments(segments: list[dict]) -> list[dict]:
    cleaned = []
    for seg in segments:
        text = normalize_text(seg.get("text", ""))
        if not text:
            continue
        if text in NOISE_PATTERNS:
            continue
        cleaned.append(
            {
                "start": round(float(seg["start"]), 3),
                "end": round(float(seg["end"]), 3),
                "text": text,
            }
        )
    return cleaned


def format_segments_for_prompt(segments: list[dict]) -> str:
    return "\n".join(f"[{seg['start']:08.3f}-{seg['end']:08.3f}] {seg['text']}" for seg in segments)


def format_movie_info_for_prompt() -> str:
    if not MOVIE_INFO:
        return "Movie info context: unavailable"

    compact = {
        "query": MOVIE_INFO.get("query", ""),
        "matched_title": MOVIE_INFO.get("matched_title", ""),
        "language": MOVIE_INFO.get("language", ""),
        "description": MOVIE_INFO.get("description", ""),
        "summary": MOVIE_INFO.get("summary", ""),
        "url": MOVIE_INFO.get("url", ""),
    }
    return "Movie info context:\n" + json.dumps(compact, ensure_ascii=False, indent=2)


def load_chunk_transcripts() -> list[dict]:
    chunk_records = []
    for path in sorted(TRANSCRIPTS_DIR.glob("chunk_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)
        offset = float(record.get("chunk_start_sec") or 0.0)
        segments = []
        for seg in record.get("segments", []) or []:
            segments.append(
                {
                    "start": round(float(seg["start"]) + offset, 3),
                    "end": round(float(seg["end"]) + offset, 3),
                    "text": seg.get("text", ""),
                }
            )
        chunk_records.append(
            {
                "chunk_id": path.stem,
                "path": str(path),
                "start_sec": offset,
                "end_sec": float(record.get("chunk_end_sec") or offset),
                "segments": clean_segments(segments),
            }
        )
    return chunk_records


def load_merged_segments() -> list[dict]:
    with open(MERGED_TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return clean_segments(data["segments"])


def extract_window(segments: list[dict], start_sec: float, end_sec: float, pad_sec: float = 20.0) -> list[dict]:
    window_start = max(0.0, start_sec - pad_sec)
    window_end = end_sec + pad_sec
    return [seg for seg in segments if seg["end"] >= window_start and seg["start"] <= window_end]


def build_local_prompt(chunk: dict) -> str:
    return f"""Source title: {SOURCE_TITLE}
{format_movie_info_for_prompt()}
Chunk id: {chunk['chunk_id']}
Chunk range: {chunk['start_sec']:.3f} to {chunk['end_sec']:.3f}

Task:
- Find the top {CHUNK_TOP_K} local shorts candidates from this chunk only.
- Each candidate must be a distinct source scene with a distinct payoff beat.
- Favor hook, power abuse, scam, corruption, money tension, public humiliation, betrayal, reveal, reversal, jealousy, irony, or comeback.
- Each candidate must also have a short, market-facing hook_frame_name in Korean.
- A hook frame should explain why a viewer clicks this short, not just what happens in the plot.
- Good hook frames are short and legible, such as "무명 배우 굴욕", "돈 때문에 버티는 청춘", "권력 상승과 배신", "부패 시스템 폭로", "오해가 부른 파국".
- Prefer candidates where protagonist role, pressure source or opponent, and payoff are easy to understand.
- Prefer candidates that are immediately clickable even if the viewer knows nothing about the movie.
- Every candidate must already be thinkable as a {RECOMMENDED_CLIP_COUNT_MIN} to {RECOMMENDED_CLIP_COUNT_MAX} cut short.
- The hook and the payoff must both exist inside this chunk.
- Reject candidates that need a key payoff from another chunk.
- If two candidates heavily overlap in time or share the same payoff beat, keep only the stronger one.
- If two candidates would use the same hook_frame_name and the same payoff logic, keep only the stronger one.
- If two candidates are basically the same scene with different title angles, keep only the stronger one.
- clip_blueprint must contain {MIN_CLIP_COUNT} to {MAX_CLIP_COUNT} cuts.
- clip_blueprint must open with purpose "hook".
- clip_blueprint should restore context quickly after the opening hook.
- clip_blueprint should usually total about 18 to 45 seconds of source material.
- Each clip_blueprint item should usually be 1 to 6 seconds, should rarely exceed 8 seconds, and must never exceed {MAX_LOCAL_CLIP_DURATION_SEC:.0f} seconds.
- Reject candidates that depend on a single unbroken conversation stretch longer than 10 seconds.
- Split reaction, pause, stare, comeback, and reveal into separate cuts instead of using one long conversation block.
- Use Korean for descriptive text fields.
- If the chunk does not truly contain {CHUNK_TOP_K} strong moments, return fewer.

Return JSON with this exact shape:
{{
  "chunk_id": "{chunk['chunk_id']}",
  "candidates": [
    {{
      "candidate_id": "{chunk['chunk_id']}_cand_01",
      "chunk_id": "{chunk['chunk_id']}",
      "score": 82,
      "hook_frame_name": "시장형 훅 프레임",
      "hook_frame_reason": "이 장면이 어떤 시청자 약속으로 팔리는지 설명",
      "viewer_promise": "시청자가 이 쇼츠에서 기대하는 보상이나 반전",
      "core_event": "짧은 한국어 요약",
      "why_it_works": "왜 쇼츠가 되는지 한국어 설명",
      "emotion_arc": "감정 흐름",
      "candidate_start": 0.0,
      "candidate_end": 0.0,
      "hook_moment": {{
        "time": 0.0,
        "reason": "왜 여기서 훅이 생기는지 한국어 설명"
      }},
      "payoff_moment": {{
        "time": 0.0,
        "reason": "왜 여기가 반전이나 감정 폭발인지 한국어 설명"
      }},
      "narration_need": "none|low|medium",
      "clip_blueprint": [
        {{
          "source_start": 0.0,
          "source_end": 0.0,
          "purpose": "hook|context|reaction|bridge|payoff"
        }},
        {{
          "source_start": 0.0,
          "source_end": 0.0,
          "purpose": "reaction"
        }},
        {{
          "source_start": 0.0,
          "source_end": 0.0,
          "purpose": "context"
        }},
        {{
          "source_start": 0.0,
          "source_end": 0.0,
          "purpose": "bridge"
        }},
        {{
          "source_start": 0.0,
          "source_end": 0.0,
          "purpose": "payoff"
        }}
      ],
      "title_angle": "제목 방향 한국어 메모"
    }}
  ]
}}

Transcript:
{format_segments_for_prompt(chunk['segments'])}
"""


def build_global_prompt(local_candidates: list[dict]) -> str:
    compact = []
    for cand in local_candidates:
        compact.append(
            {
                "candidate_id": cand["candidate_id"],
                "chunk_id": cand["chunk_id"],
                "score": cand["score"],
                "hook_frame_name": cand.get("hook_frame_name", ""),
                "hook_frame_reason": cand.get("hook_frame_reason", ""),
                "viewer_promise": cand.get("viewer_promise", ""),
                "core_event": cand["core_event"],
                "why_it_works": cand["why_it_works"],
                "emotion_arc": cand.get("emotion_arc", ""),
                "candidate_start": cand["candidate_start"],
                "candidate_end": cand["candidate_end"],
                "hook_moment": cand.get("hook_moment", {}),
                "payoff_moment": cand.get("payoff_moment", {}),
                "narration_need": cand.get("narration_need", "low"),
                "clip_blueprint_duration_sec": total_clip_duration_sec(cand.get("clip_blueprint", [])),
                "clip_blueprint": cand.get("clip_blueprint", []),
                "title_angle": cand.get("title_angle", ""),
            }
        )

    return f"""Source title: {SOURCE_TITLE}
{format_movie_info_for_prompt()}
You are selecting the best final shorts from the candidate pool below.

Rules:
- select up to {FINAL_TOP_K}
- prioritize strongest view-driving candidates first
- each selected short must use genuinely distinct source footage
- keep the bar high
- prioritize candidates that can become a satisfying short with at least 5 cuts
- later moments can open the short if that improves hook and clarity
- prefer candidates whose hook_frame_name is easy to package into a short title
- prefer candidates whose viewer_promise is specific and immediately legible
- prefer candidates whose clip_blueprint already forms a clean hook -> context recovery -> payoff sequence
- prefer candidates whose clip_blueprint duration already fits a real short
- treat heavily overlapping candidates from the same scene as duplicates unless the emotional payoff is clearly different
- if the hook_frame_name and payoff logic are basically the same, keep only the stronger one
- if two candidates would mostly reuse the same scene, treat them as the same short even if the title angle is different
- do not reward variety for its own sake
- it is okay to keep multiple shorts from the same frame family if the scenes are different and each one is highly clickable
- penalize candidates that need too much narration or too much setup
- global_score must be an integer from 70 to 100
- selection_reason must be in Korean
- if two candidates are similar, keep only the stronger one

Return JSON with this shape:
{{
  "selected": [
    {{
      "candidate_id": "chunk_001_cand_01",
      "global_rank": 1,
      "global_score": 91,
      "selection_reason": "왜 최종 후보로 살아남았는지 한국어 설명",
      "duplicate_group": "선택적 군집 라벨"
    }}
  ]
}}

Candidate pool:
{json.dumps(compact, ensure_ascii=False, indent=2)}
"""


def build_packaging_prompt(candidate: dict, context_segments: list[dict], rank: int) -> str:
    short_id = f"short_{rank:02d}"
    blueprint = candidate.get("clip_blueprint", []) or []
    blueprint_total = total_clip_duration_sec(blueprint)
    return f"""Source title: {SOURCE_TITLE}
{format_movie_info_for_prompt()}
Final rank: {rank}
Target short id: {short_id}

Candidate summary:
{json.dumps(candidate, ensure_ascii=False, indent=2)}

Candidate clip blueprint (starting skeleton, in playback order):
{format_clip_plan_for_prompt(blueprint)}

Candidate clip blueprint total duration: {blueprint_total:.1f} seconds

Task:
- Create a final shorts package for CapCut rough-cut generation.
- The final short should target about {TARGET_DURATION_MIN:.0f} to {TARGET_DURATION_MAX:.0f} seconds.
- It must be reconstructable into at least {MIN_CLIP_COUNT} cuts.
- source_clips should usually contain {RECOMMENDED_CLIP_COUNT_MIN} to {RECOMMENDED_CLIP_COUNT_MAX} cuts and never fewer than {MIN_CLIP_COUNT}
- if the event feels simple, split reaction, pause, reveal, and aftermath into separate cuts
- source_clips are the playback order of the short
- source order may be rearranged for hook and flow, but only when clarity is preserved
- use candidate.clip_blueprint as your starting skeleton and keep the same core emotional logic
- do not invent a different event from the one described in candidate summary
- the first source_clips item must be the hook
- if the hook is pulled from a later moment, the next 1 to 2 cuts must restore minimum context immediately
- the last source_clips item must land on payoff, reveal, reaction, or aftermath
- keep the sum of source_clips durations close to target_duration_sec and within about +/- {TARGET_DURATION_TOLERANCE_SEC:.0f} seconds
- each individual clip should usually be 1 to 8 seconds and should almost never exceed {MAX_FINAL_CLIP_DURATION_SEC:.0f} seconds
- do not pad with long unbroken context if a tighter reaction or bridge cut would work
- title must be exactly 2 lines in Korean
- title should feel like a high-performing short headline, not a neutral recap
- title should surface role, target, conflict, or payoff fast
- selection_pitch must explain in one Korean sentence why a human would want to click this short
- fun_tags must be 1 to 3 items such as 웃김, 통쾌, 긴장, 반전, 황당, 설렘, 캐릭터, 관계, 돈, 직장, 가족
- hook_line must be the strongest short dialogue or moment summary
- main_characters must be 1 to 3 role labels or names
- protagonist_presence must be one of high, medium, low, unknown
- standalone_clarity must be one of high, medium, low
- narration should be omitted unless needed
- point caption times must be relative to the short timeline
- narration target_start must also be relative to the short timeline
- use natural Korean suitable for movie/drama shorts
- avoid bland generic phrasing
- do not repeat the exact meaning of the 2-line title inside point captions
- help the user choose this short before making a CapCut draft
- prefer ending on reaction rather than explanation when possible

Return JSON with this exact shape:
{{
  "short_id": "{short_id}",
  "score": 88,
  "core_event": "한 문장 요약",
  "emotion_arc": "감정 흐름",
  "title_line1": "제목 1줄",
  "title_line2": "제목 2줄",
  "selection_pitch": "이 쇼츠가 왜 볼만한지 한 줄 설명",
  "fun_tags": ["긴장", "돈"],
  "hook_line": "혹시... 사채?",
  "main_characters": ["남자 주인공", "당첨금을 의심하는 사람들"],
  "protagonist_presence": "high|medium|low|unknown",
  "standalone_clarity": "high|medium|low",
  "target_duration_sec": 30,
  "target_cut_count_min": 5,
  "target_cut_count_recommended": [5, 8],
  "target_cut_count_max": 10,
  "source_clips": [
    {{
      "source_start": 0.0,
      "source_end": 0.0,
      "purpose": "hook|context|reaction|bridge|payoff"
    }},
    {{
      "source_start": 0.0,
      "source_end": 0.0,
      "purpose": "reaction"
    }},
    {{
      "source_start": 0.0,
      "source_end": 0.0,
      "purpose": "context"
    }},
    {{
      "source_start": 0.0,
      "source_end": 0.0,
      "purpose": "bridge"
    }},
    {{
      "source_start": 0.0,
      "source_end": 0.0,
      "purpose": "payoff"
    }}
  ],
  "narration": [
    {{
      "target_start": 0.0,
      "text": "짧은 나레이션"
    }}
  ],
  "point_captions": [
    {{
      "target_start": 3.0,
      "target_end": 5.0,
      "text": "짧은 포인트 자막"
    }}
  ],
  "edit_notes": [
    "훅 컷을 먼저 열고 필요한 문맥만 보강",
    "마지막은 설명보다 반응으로 끊기"
  ]
}}

Local transcript context:
{format_segments_for_prompt(context_segments)}
"""


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def cleanup_legacy_final_jsons() -> None:
    final_dir = OUTPUT_DIR / "final"
    for path in final_dir.glob("short_*.json"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def run_local_extraction(client: OpenAI, model: str, chunk_records: list[dict], force: bool) -> list[dict]:
    all_candidates = []
    total = len(chunk_records)
    for index, chunk in enumerate(chunk_records, start=1):
        print(f"[package] local extraction {index}/{total} -> {chunk['chunk_id']}", flush=True)
        out_path = OUTPUT_DIR / "local" / f"{chunk['chunk_id']}.json"
        if out_path.exists() and not force:
            print(f"[package] using cached local result -> {out_path.name}", flush=True)
            with open(out_path, "r", encoding="utf-8") as f:
                result = json.load(f)
        else:
            print(f"[package] requesting local candidates -> {out_path.name}", flush=True)
            result = model_json_validated(
                client,
                model,
                LOCAL_EXTRACTION_SYSTEM,
                build_local_prompt(chunk),
                validate_local_result,
            )
            save_json(out_path, result)
            print(f"[package] saved local candidates -> {out_path.name}", flush=True)
        for candidate in result.get("candidates", []) or []:
            candidate["chunk_id"] = candidate.get("chunk_id") or chunk["chunk_id"]
            all_candidates.append(candidate)
    return all_candidates


def run_global_selection(client: OpenAI, model: str, local_candidates: list[dict], force: bool) -> dict:
    out_path = OUTPUT_DIR / "global" / "selected_candidates.json"
    if out_path.exists() and not force:
        print(f"[package] using cached global selection -> {out_path.name}", flush=True)
        with open(out_path, "r", encoding="utf-8") as f:
            result = json.load(f)
    else:
        print(f"[package] requesting global selection from {len(local_candidates)} candidates", flush=True)
        result = model_json_validated(
            client,
            model,
            GLOBAL_RERANK_SYSTEM,
            build_global_prompt(local_candidates),
            validate_global_result,
        )
        save_json(out_path, result)
        print(f"[package] saved global selection -> {out_path.name}", flush=True)
    return result


def run_final_packaging(
    client: OpenAI,
    model: str,
    merged_segments: list[dict],
    local_candidates_by_id: dict,
    selected: list[dict],
    force: bool,
) -> list[dict]:
    final_packages = []
    ordered = sorted(selected, key=lambda x: x["global_rank"])
    total = len(ordered)
    for index, item in enumerate(ordered, start=1):
        print(f"[package] final packaging {index}/{total} -> rank {item['global_rank']}", flush=True)
        candidate = dict(local_candidates_by_id[item["candidate_id"]])
        candidate["global_rank"] = item["global_rank"]
        candidate["global_score"] = item["global_score"]
        candidate["selection_reason"] = item.get("selection_reason", "")
        context_segments = extract_window(
            merged_segments,
            float(candidate["candidate_start"]),
            float(candidate["candidate_end"]),
            pad_sec=25.0,
        )
        package_name = f"short_{int(item['global_rank']):02d}.json"
        print(f"[package] requesting final package -> {package_name}", flush=True)
        result = model_json_validated(
            client,
            model,
            PACKAGING_SYSTEM,
            build_packaging_prompt(candidate, context_segments, int(item["global_rank"])),
            validate_final_result,
        )
        result["candidate_id"] = item["candidate_id"]
        result["global_rank"] = item["global_rank"]
        result["global_score"] = item["global_score"]
        result["hook_frame_name"] = candidate.get("hook_frame_name", "")
        result["viewer_promise"] = candidate.get("viewer_promise", "")
        final_packages.append(result)
    return final_packages


def write_summary(final_packages: list[dict]) -> None:
    lines = []
    for pkg in sorted(final_packages, key=lambda x: x["global_rank"]):
        clip_total = total_clip_duration_sec(pkg.get("source_clips", []) or [])
        lines.append(f"## {pkg['global_rank']}. {pkg['title_line1']} / {pkg['title_line2']}")
        lines.append(f"- short_id: {pkg.get('short_id', '')}")
        lines.append(f"- score: {pkg.get('global_score', pkg.get('score', ''))}")
        lines.append(f"- selection_pitch: {pkg.get('selection_pitch', '')}")
        lines.append(f"- hook_frame_name: {pkg.get('hook_frame_name', '')}")
        lines.append(f"- fun_tags: {', '.join(pkg.get('fun_tags', []) or [])}")
        lines.append(f"- hook_line: {pkg.get('hook_line', '')}")
        lines.append(f"- protagonist_presence: {pkg.get('protagonist_presence', '')}")
        lines.append(f"- standalone_clarity: {pkg.get('standalone_clarity', '')}")
        lines.append(f"- core_event: {pkg.get('core_event', '')}")
        lines.append(f"- target_duration_sec: {pkg.get('target_duration_sec', '')}")
        lines.append(f"- source_clips_total_sec: {clip_total}")
        lines.append(f"- clip_count: {len(pkg.get('source_clips', []) or [])}")
        lines.append(f"- narration_count: {len(pkg.get('narration', []) or [])}")
        lines.append(f"- point_caption_count: {len(pkg.get('point_captions', []) or [])}")
        lines.append("")
    with open(OUTPUT_DIR / "final" / "final_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = build_parser().parse_args()
    configure_runtime(args.analysis_dir.resolve(), args.source_title, args.movie_info.resolve() if args.movie_info else None)

    ensure_output_dirs()
    print(f"[package] analysis_dir={ANALYSIS_DIR}", flush=True)
    print(f"[package] source_title={SOURCE_TITLE}", flush=True)
    print(f"[package] model={args.model}", flush=True)
    client = load_client()
    chunk_records = load_chunk_transcripts()
    merged_segments = load_merged_segments()
    print(f"[package] loaded chunk_records={len(chunk_records)}", flush=True)
    print(f"[package] loaded merged_segments={len(merged_segments)}", flush=True)

    local_candidates = run_local_extraction(client, args.model, chunk_records, args.force)
    save_json(OUTPUT_DIR / "local" / "all_candidates.json", {"candidates": local_candidates})
    print(f"[package] local_candidates={len(local_candidates)}", flush=True)

    selected_data = run_global_selection(client, args.model, local_candidates, args.force)
    selected_items = selected_data.get("selected", []) or []
    print(f"[package] selected_items={len(selected_items)}", flush=True)

    local_candidates_by_id = {cand["candidate_id"]: cand for cand in local_candidates}
    aggregate_path = OUTPUT_DIR / "final" / "shorts_packages.json"
    if aggregate_path.exists() and not args.force:
        print(f"[package] using cached aggregate packages -> {aggregate_path.name}", flush=True)
        with open(aggregate_path, "r", encoding="utf-8") as f:
            final_payload = json.load(f)
        final_packages = final_payload.get("shorts", [])
    else:
        final_packages = run_final_packaging(
            client=client,
            model=args.model,
            merged_segments=merged_segments,
            local_candidates_by_id=local_candidates_by_id,
            selected=selected_items,
            force=args.force,
        )

        final_payload = {
            "source_title": SOURCE_TITLE,
            "model": args.model,
            "movie_info": MOVIE_INFO,
            "shorts": sorted(final_packages, key=lambda x: x["global_rank"]),
        }
        cleanup_legacy_final_jsons()
        save_json(aggregate_path, final_payload)
        write_summary(final_packages)

    print(f"[package] output_dir={OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
