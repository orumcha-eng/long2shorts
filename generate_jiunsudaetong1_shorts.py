import argparse
import json
import os
from pathlib import Path
import unicodedata

from openai import OpenAI

from env_loader import format_checked_env_paths, load_project_env


BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = BASE_DIR / "analysis" / "jiunsudaetong1"
TRANSCRIPTS_DIR = ANALYSIS_DIR / "transcripts"
MERGED_TRANSCRIPT_PATH = ANALYSIS_DIR / "merged" / "merged_transcript.json"
OUTPUT_DIR = ANALYSIS_DIR / "shorts_candidates"

SOURCE_TITLE = "지은수대통 1회"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_TEMPERATURE = 0.2
CHUNK_TOP_K = 4
FINAL_TOP_K = 10
MAX_RETRIES = 3

NOISE_PATTERNS = {
    "마이콜과 마이콜의 대화",
}


LOCAL_EXTRACTION_SYSTEM = """You are a senior Korean movie/drama shorts editor.
You find reliable shorts candidates from transcripts.

You must optimize for consistency, not novelty.
Always use the same standards:
- hook within first 1 to 3 seconds
- clear conflict, misunderstanding, humiliation, money tension, relationship tension, reversal, or payoff
- understandable even if character names are incomplete
- editable into at least 5 cuts

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

Ignore obvious ASR junk, repeated hallucinations, and non-dialogue artifacts.
Return JSON only.
"""


GLOBAL_RERANK_SYSTEM = """You are the lead Korean movie/drama shorts editor.
You choose the strongest final shorts from a pool of local candidates.

Priorities:
1. strong opening hook
2. emotional clarity
3. clear conflict and payoff
4. diversity between selected shorts
5. editability into a satisfying 5+ cut short

Do not keep weak duplicates.
Prefer a mix of money tension, workplace tension, family tension, romantic tension, humiliation, reversal, and reveal when available.
Return JSON only.
"""


PACKAGING_SYSTEM = """You package one Korean movie/drama short for CapCut rough-cut generation.

Rules:
- title must be exactly 2 lines
- title should feel clickable and specific, not bland
- narration should be omitted unless needed
- maximum 2 narration lines
- maximum 3 point captions
- the short must be reconstructable into at least 5 cuts
- source order may be rearranged for hook and flow
- point caption times must be relative to the short timeline
- narration target_start must also be relative to the short timeline
- use natural Korean
- if names are uncertain, use roles or relationship labels

Return JSON only.
"""


def load_client() -> OpenAI:
    load_project_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY not found. Checked: {format_checked_env_paths()}")
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


def validate_local_result(result: dict) -> tuple[bool, str]:
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        return False, "Missing candidates list."
    for idx, cand in enumerate(candidates, start=1):
        if not cand.get("candidate_id"):
            return False, f"Candidate {idx} missing candidate_id."
        if not cand.get("core_event"):
            return False, f"Candidate {idx} missing core_event."
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
        blueprint = cand.get("clip_blueprint")
        if not isinstance(blueprint, list) or len(blueprint) == 0:
            return False, f"Candidate {idx} clip_blueprint must not be empty."
    return True, ""


def validate_global_result(result: dict) -> tuple[bool, str]:
    selected = result.get("selected")
    if not isinstance(selected, list):
        return False, "Missing selected list."
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
    return True, ""


def validate_final_result(result: dict) -> tuple[bool, str]:
    for key in ["short_id", "core_event", "emotion_arc", "title_line1", "title_line2"]:
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
    if not isinstance(clips, list) or len(clips) < 5 or len(clips) > 10:
        return False, "source_clips must contain 5 to 10 items."
    for idx, clip in enumerate(clips, start=1):
        try:
            start = float(clip.get("source_start"))
            end = float(clip.get("source_end"))
        except Exception:
            return False, f"Clip {idx} has invalid source_start/source_end."
        if end <= start:
            return False, f"Clip {idx} end must be greater than start."

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
Chunk id: {chunk['chunk_id']}
Chunk range: {chunk['start_sec']:.3f} to {chunk['end_sec']:.3f}

Task:
- Find the top {CHUNK_TOP_K} local shorts candidates from this chunk only.
- Each candidate must be a distinct emotional event.
- Favor hook, misunderstanding, money tension, status tension, family pressure, workplace humiliation, reveal, reversal, jealousy, or irony.
- Every candidate must already be thinkable as a 5 to 8 cut short.
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
                "core_event": cand["core_event"],
                "why_it_works": cand["why_it_works"],
                "emotion_arc": cand.get("emotion_arc", ""),
                "candidate_start": cand["candidate_start"],
                "candidate_end": cand["candidate_end"],
                "hook_moment": cand.get("hook_moment", {}),
                "payoff_moment": cand.get("payoff_moment", {}),
                "narration_need": cand.get("narration_need", "low"),
                "title_angle": cand.get("title_angle", ""),
            }
        )

    return f"""Source title: {SOURCE_TITLE}
You are selecting the best final shorts from the candidate pool below.

Rules:
- select up to {FINAL_TOP_K}
- prefer variety, not repetition
- each selected short must feel meaningfully different
- keep the bar high
- prioritize candidates that can become a satisfying short with at least 5 cuts
- later moments can open the short if that improves hook and clarity
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
    return f"""Source title: {SOURCE_TITLE}
Final rank: {rank}
Target short id: {short_id}

Candidate summary:
{json.dumps(candidate, ensure_ascii=False, indent=2)}

Task:
- Create a final shorts package for CapCut rough-cut generation.
- The final short should target about 22 to 40 seconds.
- It must be reconstructable into at least 5 cuts.
- source_clips should usually contain 5 to 8 cuts and never fewer than 5
- if the event feels simple, split reaction, pause, reveal, and aftermath into separate cuts
- source order may be rearranged for hook and flow
- title must be exactly 2 lines in Korean
- narration should be omitted unless needed
- point caption times must be relative to the short timeline
- narration target_start must also be relative to the short timeline
- use natural Korean suitable for movie/drama shorts
- avoid bland generic phrasing

Return JSON with this exact shape:
{{
  "short_id": "{short_id}",
  "score": 88,
  "core_event": "한 문장 요약",
  "emotion_arc": "감정 흐름",
  "title_line1": "제목 1줄",
  "title_line2": "제목 2줄",
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


def run_local_extraction(client: OpenAI, model: str, chunk_records: list[dict], force: bool) -> list[dict]:
    all_candidates = []
    for chunk in chunk_records:
        out_path = OUTPUT_DIR / "local" / f"{chunk['chunk_id']}.json"
        if out_path.exists() and not force:
            with open(out_path, "r", encoding="utf-8") as f:
                result = json.load(f)
        else:
            result = model_json_validated(
                client,
                model,
                LOCAL_EXTRACTION_SYSTEM,
                build_local_prompt(chunk),
                validate_local_result,
            )
            save_json(out_path, result)
        for candidate in result.get("candidates", []) or []:
            candidate["chunk_id"] = candidate.get("chunk_id") or chunk["chunk_id"]
            all_candidates.append(candidate)
    return all_candidates


def run_global_selection(client: OpenAI, model: str, local_candidates: list[dict], force: bool) -> dict:
    out_path = OUTPUT_DIR / "global" / "selected_candidates.json"
    if out_path.exists() and not force:
        with open(out_path, "r", encoding="utf-8") as f:
            result = json.load(f)
    else:
        result = model_json_validated(
            client,
            model,
            GLOBAL_RERANK_SYSTEM,
            build_global_prompt(local_candidates),
            validate_global_result,
        )
        save_json(out_path, result)
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
    for item in sorted(selected, key=lambda x: x["global_rank"]):
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
        out_path = OUTPUT_DIR / "final" / f"short_{int(item['global_rank']):02d}.json"
        if out_path.exists() and not force:
            with open(out_path, "r", encoding="utf-8") as f:
                result = json.load(f)
        else:
            result = model_json_validated(
                client,
                model,
                PACKAGING_SYSTEM,
                build_packaging_prompt(candidate, context_segments, int(item["global_rank"])),
                validate_final_result,
            )
            save_json(out_path, result)
        result["candidate_id"] = item["candidate_id"]
        result["global_rank"] = item["global_rank"]
        result["global_score"] = item["global_score"]
        final_packages.append(result)
    return final_packages


def write_summary(final_packages: list[dict]) -> None:
    lines = []
    for pkg in sorted(final_packages, key=lambda x: x["global_rank"]):
        lines.append(f"## {pkg['global_rank']}. {pkg['title_line1']} / {pkg['title_line2']}")
        lines.append(f"- short_id: {pkg.get('short_id', '')}")
        lines.append(f"- score: {pkg.get('global_score', pkg.get('score', ''))}")
        lines.append(f"- core_event: {pkg.get('core_event', '')}")
        lines.append(f"- target_duration_sec: {pkg.get('target_duration_sec', '')}")
        lines.append(f"- clip_count: {len(pkg.get('source_clips', []) or [])}")
        lines.append(f"- narration_count: {len(pkg.get('narration', []) or [])}")
        lines.append(f"- point_caption_count: {len(pkg.get('point_captions', []) or [])}")
        lines.append("")
    with open(OUTPUT_DIR / "final" / "final_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_output_dirs()
    client = load_client()
    chunk_records = load_chunk_transcripts()
    merged_segments = load_merged_segments()

    local_candidates = run_local_extraction(client, args.model, chunk_records, args.force)
    save_json(OUTPUT_DIR / "local" / "all_candidates.json", {"candidates": local_candidates})

    selected_data = run_global_selection(client, args.model, local_candidates, args.force)
    selected_items = selected_data.get("selected", []) or []

    local_candidates_by_id = {cand["candidate_id"]: cand for cand in local_candidates}
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
        "shorts": sorted(final_packages, key=lambda x: x["global_rank"]),
    }
    save_json(OUTPUT_DIR / "final" / "shorts_packages.json", final_payload)
    write_summary(final_packages)

    print(f"model={args.model}")
    print(f"local_candidates={len(local_candidates)}")
    print(f"selected={len(selected_items)}")
    print(f"output_dir={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
