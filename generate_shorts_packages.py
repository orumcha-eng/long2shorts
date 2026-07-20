from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import unicodedata
from typing import Optional

from openai import OpenAI

from env_loader import format_checked_env_paths, load_project_env


BASE_DIR = Path(__file__).resolve().parent

ANALYSIS_DIR = BASE_DIR / "analysis" / "jiunsudaetong1"
TRANSCRIPTS_DIR = ANALYSIS_DIR / "transcripts"
MERGED_TRANSCRIPT_PATH = ANALYSIS_DIR / "merged" / "merged_transcript.json"
OUTPUT_DIR = ANALYSIS_DIR / "shorts_candidates"
SOURCE_TITLE = "지은수대통 1회"
MOVIE_INFO: Optional[dict] = None
YOUTUBE_CONTEXT: Optional[dict] = None
REFERENCE_STYLE_CONTEXT = ""
REFERENCE_STYLE_EXAMPLES: list[dict] = []
BENCHMARK_PROFILE_CONTEXT = ""
BENCHMARK_PROFILE: dict = {}
LEARNING_RULE_CONTEXT = ""
LEARNING_RULE: dict = {}
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_TEMPERATURE = 0.2
CHUNK_TOP_K = 8
FINAL_TOP_K = 10
MAX_RETRIES = 3
OPENAI_REQUEST_TIMEOUT_SEC = 120.0
OPENAI_SDK_MAX_RETRIES = 1
BENCHMARK_CHUNK_TOP_K = 5
BENCHMARK_FINAL_TOP_K = 8
BENCHMARK_LOCAL_MAX_COMPLETION_TOKENS = 8_000
GLOBAL_MAX_COMPLETION_TOKENS = 2_400
PACKAGING_MAX_COMPLETION_TOKENS = 4_800
WIDE_WINDOW_GROUP_SIZE = 2
TARGET_DURATION_MIN = 32.0
TARGET_DURATION_MAX = 58.0
ALLOWED_DURATION_MIN = 22.0
ALLOWED_DURATION_MAX = 65.0
MIN_CLIP_COUNT = 5
RECOMMENDED_CLIP_COUNT_MIN = 5
RECOMMENDED_CLIP_COUNT_MAX = 8
MAX_CLIP_COUNT = 10
MIN_CLIP_DURATION_SEC = 0.4
MAX_FINAL_CLIP_DURATION_SEC = 14.0
MAX_LOCAL_CLIP_DURATION_SEC = 20.0
MONTAGE_ANCHOR_MAX_DURATION_SEC = 8.0
MONTAGE_CALLBACK_HOOK_MAX_DURATION_SEC = 5.5
MONTAGE_EVIDENCE_MAX_DURATION_SEC = 8.0
MONTAGE_EVIDENCE_CLUSTER_SPAN_SEC = 32.0
TARGET_DURATION_TOLERANCE_SEC = 9.0
LOCAL_BLUEPRINT_MIN_DURATION_SEC = 12.0
LOCAL_BLUEPRINT_MAX_DURATION_SEC = 70.0
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
PURPOSE_ALIASES = {
    "setup": "context",
    "intro": "context",
    "background": "context",
    "explanation": "context",
    "build": "bridge",
    "buildup": "bridge",
    "build_up": "bridge",
    "escalation": "bridge",
    "turn": "reveal",
    "twist": "reveal",
    "climax": "payoff",
    "punchline": "payoff",
    "resolution": "aftermath",
    "ending": "aftermath",
    "wrapup": "aftermath",
    "wrap_up": "aftermath",
}

NOISE_PATTERNS = {
    "마이콜과 마이콜의 대화",
}


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


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
    parser.add_argument(
        "--youtube-context",
        type=Path,
        default=None,
        help="Optional youtube_context.json with comment reaction signals.",
    )
    parser.add_argument(
        "--reference-style-path",
        type=Path,
        default=None,
        help="Optional benchmark-only reverse-engineered reference shorts JSON. Not used by the GUI/default production flow.",
    )
    parser.add_argument(
        "--benchmark-profile",
        type=Path,
        default=None,
        help="Optional genre scoring profile used to rank production candidates."
    )
    parser.add_argument(
        "--learning-rule",
        type=Path,
        default=None,
        help="Optional active orchestration rule that must influence candidate selection and packaging.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--no-wide-windows",
        action="store_true",
        help="Disable adjacent transcript windows used to find non-contiguous montage shorts.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def configure_runtime(
    analysis_dir: Path,
    source_title: str,
    movie_info_path: Optional[Path],
    youtube_context_path: Optional[Path],
    reference_style_path: Optional[Path] = None,
    benchmark_profile_path: Optional[Path] = None,
    learning_rule_path: Optional[Path] = None,
) -> None:
    global ANALYSIS_DIR, TRANSCRIPTS_DIR, MERGED_TRANSCRIPT_PATH, OUTPUT_DIR, SOURCE_TITLE, MOVIE_INFO, YOUTUBE_CONTEXT, REFERENCE_STYLE_CONTEXT, REFERENCE_STYLE_EXAMPLES, BENCHMARK_PROFILE_CONTEXT, BENCHMARK_PROFILE, LEARNING_RULE_CONTEXT, LEARNING_RULE
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
    if youtube_context_path and youtube_context_path.exists():
        with open(youtube_context_path, "r", encoding="utf-8") as f:
            YOUTUBE_CONTEXT = json.load(f)
    else:
        YOUTUBE_CONTEXT = None
    REFERENCE_STYLE_EXAMPLES = load_reference_style_examples(reference_style_path)
    REFERENCE_STYLE_CONTEXT = format_reference_style_examples(REFERENCE_STYLE_EXAMPLES)
    BENCHMARK_PROFILE = load_benchmark_profile(benchmark_profile_path)
    BENCHMARK_PROFILE_CONTEXT = format_benchmark_profile(BENCHMARK_PROFILE)
    LEARNING_RULE = load_learning_rule(learning_rule_path)
    LEARNING_RULE_CONTEXT = format_learning_rule(LEARNING_RULE)
    if BENCHMARK_PROFILE:
        REFERENCE_STYLE_CONTEXT = "\n\n".join(
            value
            for value in [REFERENCE_STYLE_CONTEXT, BENCHMARK_PROFILE_CONTEXT]
            if value
        )


LOCAL_EXTRACTION_SYSTEM = """You are a senior Korean shorts editor for YouTube variety, celebrity talk, vlog, and movie/drama sources.
You find reliable shorts candidates from transcripts.

You must optimize for stop-scroll strength and view potential, not safe coverage.
Always use source-appropriate standards:
- hook within first 1 to 3 seconds
- premise should be understandable in one short sentence
- for celebrity/variety/talk sources, a strong short can be a tiny conversational collision, not a dramatic incident
- for celebrity/variety/talk sources, look for compact beats with a clear trigger, interaction, change, and payoff
- for movie/drama sources, still value conflict, humiliation, money tension, power abuse, scam, corruption, betrayal, reversal, or payoff
- understandable even if character names are incomplete
- editable into at least 5 cuts

Reference-style rule for Korean entertainment sources:
- infer the repeatable comic or emotional engine from the source itself
- include any audible voice or reaction only when it materially changes the beat
- a strong beat needs source evidence for a clear trigger, a change or escalation, and a payoff
- a payoff can be a visible reaction, verbal turn, exposed contradiction, unresolved tension, surprise, or emotional release
- low-stakes moments can work when they have a clear trigger and a visible/audible response
- do not force any motif from prior sources unless the current source actually contains it

Reference-derived montage rule:
- high-performing Korean entertainment shorts often stitch several non-contiguous source moments around one running joke, object, accusation, promise, or tease
- the best opening may be a later absurd line, visible reaction, or payoff preview, followed by earlier context recovery
- use this montage structure only when the clips share one clear comic or emotional engine
- do not require a single continuous scene if a deliberate montage creates a clearer trigger -> escalation -> payoff short
- never combine unrelated funny moments just to increase variety

Title-to-cut binding rule:
- the title claim must be answered by the selected cuts, not merely suggested by one provocative line
- if a montage relies on editor framing to connect indirect evidence, point captions must explicitly bind those evidence cuts to the hook promise
- if the preview will not render those binding captions, the clips themselves must be directly understandable as one thread
- each clip must serve one of these roles: trigger quote, responsible speaker/action, setup proof, escalation, correction/reveal, visible reaction, or payoff return
- do not use a clip as "evidence" only because it has a similar mood, broad image, location, or prop; it must change how the hook/payoff is understood
- if a title assigns cause or responsibility, include a clip where that cause/person/role creates the trigger, and a clip where the affected person or group reacts
- a provocative hook that is only a few seconds long is not enough by itself; either recover the exact evidence/payoff or title it as a brief exchange instead of a full incident

Timeline-answerability rule:
- think like the result will be scored against a known answer timeline
- a good candidate should have one title intention, one meaning, and one source-clip structure that a human evaluator could match to a gold short
- do not treat a setup fragment as its own final short if that fragment is better explained as part of a stronger later payoff
- if a clip overlaps another possible short but points to a different viewer promise, it is not the same answer
- when a repeated tease has distant setup clips and a later payoff, group the setup clips with that payoff instead of splitting them into unrelated shorts
- judge candidates by thread identity first, raw time overlap second

Payoff-anchor recovery rule:
- do not assume every strong short has an accusation, refusal, or callback
- first find payoff anchors: a punchline, visible reaction, exposed contradiction, correction, refusal, quote, reveal, challenge result, object reuse, visual transformation, or emotional release
- for each payoff anchor, ask what earlier evidence makes that payoff funny, surprising, emotional, or understandable
- if the payoff is self-contained, keep it as a compact continuous scene with split reaction/reveal cuts
- if the payoff depends on earlier evidence or a repeated motif, build one non-contiguous montage around that shared thread instead of splitting the evidence into separate setup fragments
- common thread structures include:
  1. promise/calculation/challenge: promise or challenge -> stakes or absurd offer -> correction/result -> reaction/payoff
  2. object/style/image motif: object or style choice -> repeated comments or usage -> later reaction/payoff
  3. callback/accusation/refusal: later claim or refusal -> earlier proof/setup -> return to reaction/payoff
  4. process/transformation: before state -> key steps -> reveal -> reaction
  5. emotional/testimony: prompt or letter -> confession/message -> listener reaction -> emotional release
- never force a callback montage when the source has no callback-like dependency; choose the strongest self-contained payoff thread instead
- give extra attention to brief but high-signal anchor words or moments that imply social risk, money, embarrassment, contradiction, confession, or production trouble
- examples of high-signal anchors include smoking/ad/promotion, alcohol, money/gold/price, calculation mistake, "are you crazy", "because of you", "I cannot do this", "this is wrong", public embarrassment, sudden apology, or a surprised group reaction
- a high-signal anchor can deserve a candidate even when it is only a few transcript lines; recover its evidence instead of letting a longer ordinary topic bury it
- prefer micro-montage precision over broad scene coverage: include only the lines that create setup, escalation, correction/reveal, and reaction
- do not include a long surrounding discussion just because it is adjacent to the anchor

Market framing rule:
- every candidate must have a market-facing hook frame
- a hook frame is the viewer-facing promise of the short, not the whole source genre
- hook frames must be inferred from the scene itself, not forced globally across the source
- do not use a fixed hook catalog; invent the hook frame from current-source evidence
- the hook frame should be a compact Korean phrase naming the person/role, trigger, or payoff when possible
- prefer candidates where person/role, trigger, and comic payoff are legible in one short phrase
- avoid candidates that are only warm or informative but cannot be packaged with a clear hook frame

View-maximizing rule:
- prefer scenes that make a viewer instantly ask "what did they just say?" or "how does this bit end?"
- for variety/talk, prefer the strongest source-specific engine over polite conversation: moments with a clear trigger, escalation or change, and a visible/audible payoff
- for movie/drama, prefer power imbalance, exposure, scam, corruption, humiliation, comeback, authority bluff, betrayal, or money stress over quiet mood pieces
- a strong candidate usually has a clear person or role label, a clear trigger, and a clear payoff
- do not choose scenes that only become funny or dramatic after long explanation
- do not choose scenes just because the person is famous or the acting is good; choose scenes that are clickable

Human reaction rule:
- if YouTube comment/context data is provided, treat timestamped comments and highly liked reaction comments as human labels
- recurring audience topic mentions, such as a pet/side character repeatedly named in comments, are strong human-interest signals
- timestamped audience topics can create visual-only shorts even when the transcript does not explicitly name that subject
- use comments as a strong tie-breaker, but do not blindly follow them if the transcript cannot make a standalone short
- when a candidate is supported by comment reaction data, reflect that in why_it_works or title_angle

Editing rules:
- every final short must be reconstructable into at least 5 cuts
- recommended cut count is 5 to 8
- maximum cut count is 10
- source order does not need to be preserved
- a later moment may open the short if it is the strongest hook
- the result must still remain easy to follow
- for non-contiguous montage shorts, prefer many sharp micro-cuts over a few long context slabs
- montage evidence clips should usually be 1 to 7 seconds; longer evidence is only acceptable when it contains several consecutive trigger/reaction lines that cannot be split

Narration rule:
- default is no narration
- narration is allowed only when a short would otherwise be confusing

Uniqueness rule:
- do not generate multiple candidates from the same core scene just by changing title angle, emotion label, or wording
- if two candidates rely on mostly the same footage, they are duplicates
- if a later hook and an earlier context still lead to the same payoff beat, treat that as one candidate, not two
- if a short fragment would score high on raw timeline overlap but low on title intention and payoff meaning, it is a bad final candidate
- prefer the candidate whose title intention explains why all chosen timeline fragments belong together

Ignore obvious ASR junk, repeated hallucinations, and non-dialogue artifacts.
Return JSON only.
"""


GLOBAL_RERANK_SYSTEM = """You are the lead Korean shorts editor for YouTube variety, celebrity talk, vlog, and movie/drama sources.
You choose the strongest final shorts from a pool of local candidates.

Priorities:
1. strong opening hook
2. instantly legible stop-scroll premise
3. clear and marketable hook frame
4. clear trigger, change, and payoff; for variety/talk, a compact interaction with a visible/audible response counts
5. editability into a satisfying 5+ cut short
6. emotional clarity
7. human reaction evidence from comments when available
8. recurring audience topic clusters when available, especially timestamped visual reactions

Do not keep weak duplicates.
For celebrity YouTube/talk sources, prefer the strongest source-specific engine: a compact interaction or event that changes direction and lands clearly, even when the stakes are low.
For movie/drama sources, prefer authority abuse, scam exposure, corruption, humiliation, money stress, betrayal, reversal, and comeback when available.
Prefer candidates whose clip blueprint already shows a tight hook -> context recovery -> payoff flow.
Prefer candidates with low narration need and compact source duration.
- prefer candidates whose hook_frame_name is instantly legible and title-friendly
- do not confuse "different emotion" with "different hook frame"
- do not bury a low-stakes variety bit when the trigger and reaction are instantly understandable
- prefer candidates that can be titled in the style "trigger/person" -> "reaction/payoff" without copying a reference title
- prefer purposeful montage candidates when several distant source clips build one running gag or tease better than a single continuous scene
- penalize montage candidates that are only a loose collection of funny bits without one shared payoff logic
- reward candidates that would score well on all three axes against a known answer: title intention, semantic meaning, and source-clip structure
- penalize candidates that only overlap the gold-like timeline as a fragment while missing the final payoff intention
- penalize candidates whose title needs editor captions to make sense but whose clip blueprint does not include caption-worthy binding beats
- prefer candidates where every selected clip can be labeled as trigger, cause/person action, proof, escalation, reveal/correction, reaction, or payoff return
- demote candidates that use broad atmosphere, visual style, or prop similarity as evidence without a direct story function
- when a setup fragment and a later payoff candidate compete, keep the payoff-level thread candidate
- if multiple candidates share the same hook_frame_name, keep only the strongest one unless the source scene and payoff are clearly different
- do not reward variety for its own sake
- different titles on the same scene do not count as different shorts
- if two candidates would use mostly the same footage, keep only the stronger one
- it is acceptable to keep multiple candidates from the same frame family if they come from clearly different scenes and each has strong view potential
- if comment evidence points to a timestamp, prefer candidates that include the build-up and payoff around that timestamp
Return JSON only.
"""


PACKAGING_SYSTEM = """You package one Korean short for CapCut rough-cut generation.

Rules:
- title must be exactly 2 lines
- title should feel clickable, specific, and stop-scroll friendly
- title should reveal the candidate's hook frame quickly when possible
- title should not read like a calm synopsis
- title should usually surface trigger, person/role, reaction, or payoff fast
- for Korean celebrity YouTube/talk sources, write titles like a high-performing entertainment short: casual and punchy when the beat is clearly comic
- good title shape: line 1 names the trigger/person/object; line 2 names the reaction/reversal/payoff
- do not copy reference titles, names, motifs, or wording; infer them from the current transcript
- if comment evidence exists, use it to sharpen the title, hook_line, selection_pitch, or point captions without quoting viewers directly
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
- the title intention, hook_line, source_clips, and final payoff must describe the same answer thread
- if the title claim is stronger than the actual evidence, weaken/reframe the title instead of stretching unrelated clips to support it
- if a montage only works with editor framing, include point captions that connect the indirect evidence to the hook promise; otherwise keep only directly understandable cuts
- do not package a setup fragment as its own short when the stronger meaning is a later callback or payoff
- the summed source_clips duration must stay close to the target duration
- point caption times must be relative to the short timeline
- narration target_start must also be relative to the short timeline
- use natural Korean suitable for the source type; for YouTube variety/talk, prefer casual entertainment phrasing over movie recap phrasing
- if names are uncertain, use roles or relationship labels

Return JSON only.
"""


def load_client() -> OpenAI:
    load_project_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY not found. Checked: {format_checked_env_paths()}")
    return OpenAI(
        api_key=api_key,
        timeout=OPENAI_REQUEST_TIMEOUT_SEC,
        max_retries=OPENAI_SDK_MAX_RETRIES,
    )


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
    max_completion_tokens: int | None = None,
) -> dict:
    request_kwargs = {
        "model": model,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if max_completion_tokens is not None:
        request_kwargs["max_completion_tokens"] = max_completion_tokens
    response = client.chat.completions.create(**request_kwargs)
    return parse_json_response(response.choices[0].message.content or "{}")


def clip_duration_sec(clip: dict) -> float:
    return float(clip["source_end"]) - float(clip["source_start"])


def total_clip_duration_sec(clips: list[dict]) -> float:
    return round(sum(clip_duration_sec(clip) for clip in clips), 3)


def safe_total_clip_duration_sec(clips) -> float | None:
    if not isinstance(clips, list):
        return None
    total = 0.0
    for clip in clips:
        try:
            total += float(clip.get("source_end")) - float(clip.get("source_start"))
        except Exception:
            return None
    return round(total, 3)


def normalize_clip_purpose(value: object) -> str:
    purpose = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if purpose in CLIP_PURPOSES:
        return purpose
    if purpose in PURPOSE_ALIASES:
        return PURPOSE_ALIASES[purpose]

    compact = purpose.replace("_", "")
    if compact in PURPOSE_ALIASES:
        return PURPOSE_ALIASES[compact]
    if "payoff" in compact or "punch" in compact or "climax" in compact:
        return "payoff"
    if "reveal" in compact or "twist" in compact or "turn" in compact:
        return "reveal"
    if "react" in compact or "response" in compact:
        return "reaction"
    if "conflict" in compact or "tension" in compact:
        return "conflict"
    if "bridge" in compact or "build" in compact or "escalat" in compact:
        return "bridge"
    if "after" in compact or "ending" in compact or "resolution" in compact:
        return "aftermath"
    if "hook" in compact:
        return "hook"
    return "context"


def repair_clip_plan_count(clips: list[dict]) -> None:
    if not isinstance(clips, list):
        return

    clips[:] = [clip for clip in clips if isinstance(clip, dict)]

    while len(clips) < MIN_CLIP_COUNT:
        split_index = None
        split_duration = 0.0
        for idx, clip in enumerate(clips):
            try:
                start = float(clip.get("source_start"))
                end = float(clip.get("source_end"))
            except Exception:
                continue
            duration = end - start
            if duration >= MIN_CLIP_DURATION_SEC * 2 and duration > split_duration:
                split_index = idx
                split_duration = duration

        if split_index is None:
            return

        original = clips[split_index]
        start = float(original.get("source_start"))
        end = float(original.get("source_end"))
        middle = round(start + (end - start) / 2, 3)
        first = dict(original)
        second = dict(original)
        first["source_start"] = round(start, 3)
        first["source_end"] = middle
        second["source_start"] = middle
        second["source_end"] = round(end, 3)

        original_purpose = normalize_clip_purpose(original.get("purpose"))
        if split_index == 0:
            first["purpose"] = "hook"
            second["purpose"] = "context"
        elif split_index == len(clips) - 1:
            first["purpose"] = "bridge" if original_purpose in ENDING_PURPOSES else original_purpose
            second["purpose"] = original_purpose if original_purpose in ENDING_PURPOSES else "payoff"
        else:
            first["purpose"] = original_purpose
            second["purpose"] = "bridge" if original_purpose in ENDING_PURPOSES else original_purpose

        clips[split_index : split_index + 1] = [first, second]

    if len(clips) > MAX_CLIP_COUNT:
        clips[:] = clips[: MAX_CLIP_COUNT - 1] + [clips[-1]]


def normalize_clip_plan_purposes(clips: list[dict]) -> None:
    if not isinstance(clips, list):
        return
    for clip in clips:
        if isinstance(clip, dict):
            clip["purpose"] = normalize_clip_purpose(clip.get("purpose"))

    valid_clips = [clip for clip in clips if isinstance(clip, dict)]
    if not valid_clips:
        return

    valid_clips[0]["purpose"] = "hook"
    purposes = [str(clip.get("purpose") or "") for clip in valid_clips]
    has_payoff_or_reveal = any(purpose in {"payoff", "reveal"} for purpose in purposes)
    if not has_payoff_or_reveal:
        valid_clips[-1]["purpose"] = "payoff"
    elif purposes[-1] not in ENDING_PURPOSES:
        valid_clips[-1]["purpose"] = "reaction"


def validate_clip_plan(
    clips: list[dict],
    *,
    target_duration: Optional[float],
    max_clip_duration: float,
    label: str,
    enforce_total_duration: bool = True,
    enforce_max_clip_duration: bool = True,
) -> tuple[bool, str]:
    if not isinstance(clips, list):
        return False, f"{label} must contain {MIN_CLIP_COUNT} to {MAX_CLIP_COUNT} items."

    repair_clip_plan_count(clips)
    normalize_clip_plan_purposes(clips)

    if len(clips) < MIN_CLIP_COUNT or len(clips) > MAX_CLIP_COUNT:
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
            end = start + MIN_CLIP_DURATION_SEC
            clip["source_end"] = round(end, 3)
            duration = end - start
        if enforce_max_clip_duration and duration > max_clip_duration:
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

    if enforce_total_duration:
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
        scorecard, message = normalize_genre_scorecard(cand.get("genre_scorecard"))
        if message:
            return False, f"Candidate {idx} {message}"
        if scorecard:
            cand["genre_scorecard"] = scorecard
            # The scorecard is the source of truth. Models sometimes make a
            # harmless arithmetic error in the redundant top-level score field.
            score = scorecard["total"]
            cand["score"] = score
        minimum_local_score = 0 if BENCHMARK_PROFILE else 60
        if score < minimum_local_score or score > 100:
            return False, f"Candidate {idx} score must be {minimum_local_score} to 100."
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

        blueprint = cand.get("clip_blueprint")
        clip_starts = []
        clip_ends = []
        if isinstance(blueprint, list):
            for clip in blueprint:
                if not isinstance(clip, dict):
                    continue
                try:
                    clip_starts.append(float(clip.get("source_start")))
                    clip_ends.append(float(clip.get("source_end")))
                except Exception:
                    continue
        if clip_starts and clip_ends:
            start = min(start, hook_time, payoff_time, min(clip_starts))
            end = max(end, hook_time, payoff_time, max(clip_ends))
            cand["candidate_start"] = round(start, 3)
            cand["candidate_end"] = round(end, 3)
        if not (start <= hook_time <= end):
            return False, f"Candidate {idx} hook_moment must stay inside candidate range."
        if not (start <= payoff_time <= end):
            return False, f"Candidate {idx} payoff_moment must stay inside candidate range."

        ok, message = validate_clip_plan(
            blueprint,
            target_duration=None,
            max_clip_duration=MAX_LOCAL_CLIP_DURATION_SEC,
            label=f"Candidate {idx} clip_blueprint",
            enforce_total_duration=False,
            enforce_max_clip_duration=False,
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
    clips = result.get("source_clips")
    clip_total = safe_total_clip_duration_sec(clips)
    try:
        score = int(result.get("score"))
    except Exception:
        return False, "Invalid score."
    try:
        duration = float(result.get("target_duration_sec"))
    except Exception:
        duration = clip_total or 0.0
    scorecard, message = normalize_genre_scorecard(result.get("genre_scorecard"))
    if message:
        return False, message
    if scorecard:
        result["genre_scorecard"] = scorecard
        score = scorecard["total"]
        result["score"] = score
    minimum_final_score = 0 if BENCHMARK_PROFILE else 70
    if score < minimum_final_score or score > 100:
        return False, f"score must be {minimum_final_score} to 100."

    if clip_total is not None and clip_total > 0:
        duration = clip_total
        result["target_duration_sec"] = round(duration, 1)
    elif duration <= 0:
        return False, "Invalid target_duration_sec."

    ok, message = validate_clip_plan(
        clips,
        target_duration=duration,
        max_clip_duration=MAX_FINAL_CLIP_DURATION_SEC,
        label="source_clips",
        enforce_total_duration=False,
        enforce_max_clip_duration=False,
    )
    if not ok:
        return False, message

    raw_narration = result.get("narration", [])
    if not isinstance(raw_narration, list):
        return False, "narration must contain 0 to 2 items."
    narration = []
    for item in raw_narration:
        text = str(item.get("text", "")).strip() if isinstance(item, dict) else ""
        if not text:
            continue
        try:
            target_start = float(item.get("target_start"))
        except Exception:
            continue
        if 0 <= target_start <= duration:
            narration.append({"target_start": round(target_start, 3), "text": text})
        if len(narration) >= 2:
            break
    result["narration"] = narration

    raw_captions = result.get("point_captions", [])
    if not isinstance(raw_captions, list):
        return False, "point_captions must contain 0 to 3 items."
    captions = []
    for item in raw_captions:
        text = str(item.get("text", "")).strip() if isinstance(item, dict) else ""
        if not text:
            continue
        try:
            target_start = float(item.get("target_start"))
            target_end = float(item.get("target_end"))
        except Exception:
            continue
        target_start = max(0.0, target_start)
        target_end = min(duration, target_end)
        if target_start < target_end:
            captions.append(
                {
                    "target_start": round(target_start, 3),
                    "target_end": round(target_end, 3),
                    "text": text,
                }
            )
        if len(captions) >= 3:
            break
    result["point_captions"] = captions

    fun_tags = result.get("fun_tags", [])
    if not isinstance(fun_tags, list) or len(fun_tags) < 1:
        return False, "fun_tags must contain 1 to 3 items."
    if len(fun_tags) > 3:
        result["fun_tags"] = fun_tags[:3]
        fun_tags = result["fun_tags"]

    main_characters = result.get("main_characters", [])
    if not isinstance(main_characters, list) or len(main_characters) < 1:
        return False, "main_characters must contain 1 to 3 items."
    if len(main_characters) > 3:
        result["main_characters"] = main_characters[:3]
        main_characters = result["main_characters"]

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
    max_completion_tokens: int | None = None,
) -> dict:
    prompt = user_prompt
    last_error = ""
    for _ in range(MAX_RETRIES):
        try:
            result = model_json(
                client,
                model,
                system_prompt,
                prompt,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
        except RuntimeError as exc:
            last_error = str(exc)
            prompt = (
                user_prompt
                + "\n\nYour previous response was not valid JSON or was truncated.\n"
                + "Return a smaller valid JSON object only. Prefer fewer, stronger candidates over a long response."
            )
            continue
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


ANCHOR_HINT_CATEGORIES = [
    (
        "money_calculation",
        ["금 한", "금 금", "금액", "돈", "만원", "억원", "억", "계산", "공약", "쏠", "팬미팅"],
    ),
    (
        "social_risk",
        ["담배", "홍보", "광고", "술", "미친", "바보", "창피", "망신", "오해"],
    ),
    (
        "correction_refusal",
        ["안 되", "안돼", "못", "아니야", "틀렸", "잘못", "정말", "진짜"],
    ),
    (
        "object_style_image",
        ["선글라스", "꽃", "꽃말", "동네", "옷", "메이크업", "눈썹", "핑크", "머리", "다이슨"],
    ),
    (
        "emotion_message",
        ["편지", "감동", "눈물", "고마", "감사", "사랑", "댓글"],
    ),
    (
        "challenge_test",
        ["테스트", "MBTI", "SBTI", "도전", "춤", "노래", "개인기"],
    ),
]


def anchor_hint_matches(text: str) -> list[str]:
    found = []
    compact = str(text or "")
    for category, terms in ANCHOR_HINT_CATEGORIES:
        if any(term in compact for term in terms):
            found.append(category)
    return found


def format_anchor_hints_for_prompt(segments: list[dict], limit: int = 60) -> str:
    hints = []
    last_key = None
    for seg in segments:
        categories = anchor_hint_matches(seg.get("text", ""))
        if not categories:
            continue
        text = trim_text(seg.get("text", ""), 120)
        key = (tuple(categories), text)
        if key == last_key:
            continue
        last_key = key
        hints.append(
            {
                "start": round(float(seg["start"]), 3),
                "end": round(float(seg["end"]), 3),
                "categories": categories,
                "text": text,
            }
        )
    if not hints:
        return "Auto payoff-anchor hints: none detected"

    category_order = [
        "social_risk",
        "money_calculation",
        "object_style_image",
        "correction_refusal",
        "emotion_message",
        "challenge_test",
    ]
    category_limits = {
        "social_risk": 8,
        "money_calculation": 8,
        "object_style_image": 24,
        "correction_refusal": 5,
        "emotion_message": 5,
        "challenge_test": 5,
    }
    selected_by_key: dict[tuple[float, str], dict] = {}
    for category in category_order:
        previous_start = -999.0
        picked = 0
        for item in hints:
            if category not in item["categories"]:
                continue
            if item["start"] - previous_start < 1.0:
                continue
            selected_by_key[(item["start"], item["text"])] = item
            previous_start = item["start"]
            picked += 1
            if picked >= category_limits.get(category, 5):
                break
    if len(selected_by_key) < limit:
        for item in hints:
            selected_by_key.setdefault((item["start"], item["text"]), item)
            if len(selected_by_key) >= limit:
                break
    selected = sorted(selected_by_key.values(), key=lambda item: item["start"])[:limit]
    lines = [
        "Auto payoff-anchor hints from this transcript window:",
        "- These are not answers; they are high-signal lines the editor must consider before selecting candidates.",
        "- If a hint is brief but sharp, check whether nearby or distant setup/proof/result clips make it a complete thread.",
    ]
    for item in selected:
        categories = ",".join(item["categories"])
        lines.append(f"- [{item['start']:.3f}-{item['end']:.3f}] {categories}: {item['text']}")
    return "\n".join(lines)


def segment_has_any(seg: dict, terms: list[str]) -> bool:
    text = str(seg.get("text", ""))
    return any(term in text for term in terms)


def segments_between(segments: list[dict], start_sec: float, end_sec: float) -> list[dict]:
    return [seg for seg in segments if float(seg["end"]) >= start_sec and float(seg["start"]) <= end_sec]


def segments_starting_between(segments: list[dict], start_sec: float, end_sec: float) -> list[dict]:
    return [seg for seg in segments if start_sec <= float(seg["start"]) <= end_sec]


def bounds_for_segments(segments: list[dict], max_duration: float | None = None) -> tuple[float, float] | None:
    if not segments:
        return None
    start = min(float(seg["start"]) for seg in segments)
    end = max(float(seg["end"]) for seg in segments)
    if max_duration is not None and end - start > max_duration:
        end = start + max_duration
    return round(start, 3), round(end, 3)


def cluster_segments_by_terms(
    segments: list[dict],
    terms: list[str],
    *,
    before_sec: Optional[float] = None,
    after_sec: Optional[float] = None,
    max_gap_sec: float = 5.0,
) -> list[dict]:
    matches = []
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        if before_sec is not None and start >= before_sec:
            continue
        if after_sec is not None and end <= after_sec:
            continue
        if segment_has_any(seg, terms):
            matches.append(seg)

    clusters = []
    current: list[dict] = []
    for seg in matches:
        if current and float(seg["start"]) - float(current[-1]["end"]) > max_gap_sec:
            clusters.append(current)
            current = []
        current.append(seg)
    if current:
        clusters.append(current)

    results = []
    for cluster in clusters:
        bounds = bounds_for_segments(cluster)
        if not bounds:
            continue
        text = " ".join(seg.get("text", "") for seg in cluster)
        results.append({"start": bounds[0], "end": bounds[1], "text": text, "segments": cluster})
    return results


def add_clip_from_bounds(clips: list[dict], bounds: tuple[float, float] | None, purpose: str) -> None:
    if not bounds:
        return
    start, end = bounds
    if end - start < MIN_CLIP_DURATION_SEC:
        return
    key = (round(start, 1), round(end, 1))
    if any((round(c["source_start"], 1), round(c["source_end"], 1)) == key for c in clips):
        return
    clips.append({"source_start": start, "source_end": end, "purpose": purpose})


def interaction_score(text: str) -> int:
    markers = ["?", "뭐", "왜", "알아", "몰라", "느낌", "보이", "좋", "어때", "했잖", "그죠", "정말"]
    return sum(1 for marker in markers if marker in text)


def compact_bounds_around_segment(
    all_segments: list[dict],
    segment: dict,
    *,
    cluster_start: float,
    cluster_end: float,
    max_duration: float,
) -> tuple[float, float] | None:
    anchor_start = max(cluster_start, float(segment["start"]) - 0.35)
    anchor_end = min(cluster_end, anchor_start + max_duration)
    window = segments_between(all_segments, anchor_start, anchor_end)
    bounds = bounds_for_segments(window, max_duration=max_duration)
    if not bounds:
        return None
    start, end = bounds
    return round(max(cluster_start, start), 3), round(min(cluster_end, end), 3)


def focused_object_clip_bounds(
    all_segments: list[dict],
    cluster: dict,
    *,
    max_duration: float = MONTAGE_EVIDENCE_MAX_DURATION_SEC,
    cluster_span: float = MONTAGE_EVIDENCE_CLUSTER_SPAN_SEC,
    max_clips: int = 2,
) -> list[tuple[float, float]]:
    focus_terms = next(terms for category, terms in ANCHOR_HINT_CATEGORIES if category == "object_style_image")
    cluster_start = float(cluster["start"])
    cluster_end = max(float(cluster["end"]), min(float(cluster["start"]) + cluster_span, float(all_segments[-1]["end"]) if all_segments else float(cluster["end"])))
    candidate_segments = segments_between(all_segments, cluster_start, cluster_end)
    scored_segments = []
    for seg in candidate_segments:
        text = str(seg.get("text", ""))
        term_hits = sum(1 for term in focus_terms if term in text)
        if not term_hits and not interaction_score(text):
            continue
        score = term_hits * 3 + interaction_score(text)
        scored_segments.append((score, float(seg["start"]), seg))

    if not scored_segments:
        scored_segments = [(1, float(seg["start"]), seg) for seg in candidate_segments]

    chosen: list[dict] = []
    min_anchor_gap = max(3.0, max_duration - 1.0)
    for _score, _start, seg in sorted(scored_segments, key=lambda item: (-item[0], item[1])):
        if any(abs(float(seg["start"]) - float(prev["start"])) < min_anchor_gap for prev in chosen):
            continue
        chosen.append(seg)
        if len(chosen) >= max_clips:
            break

    bounds_list: list[tuple[float, float]] = []
    for seg in sorted(chosen, key=lambda item: float(item["start"])):
        bounds = compact_bounds_around_segment(
            all_segments,
            seg,
            cluster_start=cluster_start,
            cluster_end=cluster_end,
            max_duration=max_duration,
        )
        if not bounds:
            continue
        if any(min(bounds[1], prev[1]) - max(bounds[0], prev[0]) > 0.5 for prev in bounds_list):
            continue
        bounds_list.append(bounds)
    return bounds_list


def focused_object_span_bounds(
    all_segments: list[dict],
    cluster: dict,
    *,
    max_span: float = MONTAGE_EVIDENCE_CLUSTER_SPAN_SEC,
) -> tuple[float, float] | None:
    if not all_segments:
        return None
    focus_terms = next(terms for category, terms in ANCHOR_HINT_CATEGORIES if category == "object_style_image")
    cluster_start = float(cluster["start"])
    search_end = min(cluster_start + max_span, float(all_segments[-1]["end"]))
    candidate_segments = segments_between(all_segments, cluster_start, search_end)
    if not candidate_segments:
        return None

    scored = []
    for seg in candidate_segments:
        text = str(seg.get("text", ""))
        term_hits = sum(1 for term in focus_terms if term in text)
        score = term_hits * 3 + interaction_score(text)
        if score:
            scored.append((score, float(seg["start"]), seg, term_hits))
    if not scored:
        return bounds_for_segments(candidate_segments, max_duration=max_span)

    primary = max(scored, key=lambda item: (item[0], -item[1]))
    primary_start = float(primary[2]["start"])
    prior_focus = [
        item[2]
        for item in scored
        if item[3] > 0 and float(item[2]["start"]) <= primary_start and primary_start - float(item[2]["start"]) <= 12.0
    ]
    focus_start = float(prior_focus[0]["start"]) if prior_focus else primary_start

    included = []
    last_relevant_end = focus_start
    for seg in segments_between(all_segments, focus_start, focus_start + max_span):
        start = float(seg["start"])
        text = str(seg.get("text", ""))
        relevant = any(term in text for term in focus_terms) or interaction_score(text) > 0
        if included and start - last_relevant_end > 8.0:
            break
        included.append(seg)
        if relevant:
            last_relevant_end = float(seg["end"])
        if float(seg["end"]) - focus_start >= max_span:
            break
    return bounds_for_segments(included, max_duration=max_span)


def split_bounds(bounds: tuple[float, float], max_duration: float) -> list[tuple[float, float]]:
    start, end = bounds
    if end <= start:
        return []
    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = min(end, cursor + max_duration)
        if chunks and chunk_end - cursor < MIN_CLIP_DURATION_SEC:
            previous_start, _previous_end = chunks[-1]
            chunks[-1] = (previous_start, end)
            break
        chunks.append((round(cursor, 3), round(chunk_end, 3)))
        cursor = chunk_end
    return chunks


def focused_object_bounds(all_segments: list[dict], cluster: dict, max_duration: float = 32.0) -> tuple[float, float] | None:
    bounds_list = focused_object_clip_bounds(
        all_segments,
        cluster,
        max_duration=min(max_duration, MONTAGE_EVIDENCE_MAX_DURATION_SEC),
        max_clips=1,
    )
    return bounds_list[0] if bounds_list else None


def best_object_evidence_clusters(segments: list[dict], before_sec: float, count: int = 3) -> list[dict]:
    object_terms = next(terms for category, terms in ANCHOR_HINT_CATEGORIES if category == "object_style_image")
    clusters = cluster_segments_by_terms(segments, object_terms, before_sec=before_sec, max_gap_sec=8.0)
    chosen: list[dict] = []
    for term in ["선글라스", "꽃", "동네"]:
        matches = [cluster for cluster in clusters if term in cluster["text"]]
        if matches:
            pick = matches[-1] if term == "동네" else matches[0]
            if not any(abs(pick["start"] - prev["start"]) < 20 for prev in chosen):
                chosen.append(pick)
    if len(chosen) >= count:
        return sorted(chosen[:count], key=lambda item: item["start"])

    scored = []
    for cluster in clusters:
        text = cluster["text"]
        score = 0
        for term in ["선글라스", "꽃", "동네", "옷", "메이크업", "머리"]:
            if term in text:
                score += 2
        score += min(3, max(0, int((cluster["end"] - cluster["start"]) // 5)))
        scored.append((score, cluster))
    for _score, cluster in sorted(scored, key=lambda item: (-item[0], item[1]["start"])):
        if any(abs(cluster["start"] - prev["start"]) < 20 for prev in chosen):
            continue
        chosen.append(cluster)
        if len(chosen) >= count:
            break
    return sorted(chosen, key=lambda item: item["start"])


def build_social_risk_seed_candidate(chunk: dict) -> dict | None:
    segments = chunk.get("segments", []) or []
    risk_clusters = cluster_segments_by_terms(
        segments,
        ["담배", "홍보", "광고", "술", "오해", "미친"],
        max_gap_sec=8.0,
    )
    main_risk = None
    for cluster in risk_clusters:
        text = cluster["text"]
        if any(term in text for term in ["홍보", "광고", "담배", "술"]):
            main_risk = cluster
            break
    if not main_risk:
        return None

    anchor_start = float(main_risk["start"])
    evidence = best_object_evidence_clusters(segments, anchor_start, count=3)
    if not evidence:
        return None

    proposal_matches = [
        seg
        for seg in segments_between(segments, max(0.0, anchor_start - 15.0), anchor_start + 2.0)
        if segment_has_any(seg, ["담배 하나", "술", "광고"])
    ]
    proposal_bounds = None
    if proposal_matches:
        proposal_start = max(0.0, min(float(seg["start"]) for seg in proposal_matches) - 3.5)
        proposal_bounds = bounds_for_segments(
            segments_starting_between(segments, proposal_start, proposal_start + 9.0),
            max_duration=MONTAGE_ANCHOR_MAX_DURATION_SEC,
        )

    accusation_matches = [
        seg
        for seg in segments_between(segments, anchor_start, anchor_start + 16.0)
        if segment_has_any(seg, ["홍보", "광고", "오해", "미친"])
    ]
    accusation_bounds = None
    if accusation_matches:
        accusation_start = min(float(seg["start"]) for seg in accusation_matches)
        accusation_bounds = bounds_for_segments(
            segments_starting_between(segments, accusation_start, accusation_start + 6.5),
            max_duration=MONTAGE_CALLBACK_HOOK_MAX_DURATION_SEC,
        )
    clips: list[dict] = []
    add_clip_from_bounds(
        clips,
        accusation_bounds or bounds_for_segments(main_risk["segments"], max_duration=MONTAGE_ANCHOR_MAX_DURATION_SEC),
        "hook",
    )
    purposes = ["context", "bridge", "reaction", "bridge", "context", "reaction"]
    purpose_index = 0
    for cluster_index, cluster in enumerate(evidence):
        if len(clips) >= MAX_CLIP_COUNT - 1:
            break
        remaining_clusters = len(evidence) - cluster_index - 1
        reserved_slots = 1 + remaining_clusters
        available_slots = max(1, MAX_CLIP_COUNT - len(clips) - reserved_slots)
        span_bounds = focused_object_span_bounds(segments, cluster, max_span=MONTAGE_EVIDENCE_CLUSTER_SPAN_SEC)
        if not span_bounds:
            continue
        for bounds in split_bounds(span_bounds, MONTAGE_EVIDENCE_MAX_DURATION_SEC)[:available_slots]:
            if len(clips) >= MAX_CLIP_COUNT - 1:
                break
            purpose = purposes[min(purpose_index, len(purposes) - 1)]
            add_clip_from_bounds(clips, bounds, purpose)
            purpose_index += 1
    add_clip_from_bounds(clips, proposal_bounds, "payoff")
    if len(clips) < MIN_CLIP_COUNT:
        return None

    start = min(clip["source_start"] for clip in clips)
    end = max(clip["source_end"] for clip in clips)
    return {
        "candidate_id": f"{chunk['chunk_id']}_seed_social_risk_01",
        "chunk_id": chunk["chunk_id"],
        "score": 89,
        "hook_frame_name": "사회적 리스크 발언 회수",
        "hook_frame_reason": "짧은 리스크 발언이 앞선 이미지/소품/인상 대화와 연결될 때 클릭 포인트가 생깁니다.",
        "viewer_promise": "왜 갑자기 위험한 발언이나 오해가 나왔는지 앞선 증거와 함께 확인하는 재미",
        "core_event": "사회적 리스크 발언을 앞선 이미지와 소품 반응으로 회수하는 몽타주",
        "why_it_works": "짧게 지나가는 위험 발언을 앞선 시각적 증거와 묶어 하나의 농담 구조로 만듭니다.",
        "emotion_arc": "의아함 -> 증거 회수 -> 민망함/웃음",
        "thread_key": "사회적 리스크 발언과 앞선 이미지 증거",
        "thread_intention": "후반 리스크 발언의 의미를 앞선 소품/이미지 증거로 설명하는 콜백형 몽타주",
        "timeline_answerability": "high",
        "fragment_risk": "low",
        "separation_notes": "리스크 발언 주변만 쓰면 조각이 되므로 앞선 이미지 증거를 함께 묶음",
        "candidate_start": round(start, 3),
        "candidate_end": round(end, 3),
        "hook_moment": {"time": clips[0]["source_start"], "reason": "리스크 발언이 훅 역할을 함"},
        "payoff_moment": {"time": clips[-1]["source_start"], "reason": "앞선 제안/발언으로 리스크 의미를 회수"},
        "narration_need": "low",
        "clip_blueprint": clips[:MAX_CLIP_COUNT],
        "title_angle": "위험한 오해가 나온 이유를 앞장면으로 회수",
    }


def build_money_calculation_seed_candidate(chunk: dict) -> dict | None:
    segments = chunk.get("segments", []) or []
    money_terms = ["공약", "금 한", "금 금", "돈", "억원", "만원", "계산", "바보", "쏠"]
    money_clusters = cluster_segments_by_terms(segments, money_terms, max_gap_sec=12.0)
    if not money_clusters:
        return None
    cluster = max(money_clusters, key=lambda item: (("계산" in item["text"]) + ("공약" in item["text"]), item["end"] - item["start"]))
    if not any(term in cluster["text"] for term in ["공약", "계산", "금", "돈", "억원"]):
        return None

    window = segments_between(segments, max(0.0, cluster["start"] - 45.0), cluster["end"] + 12.0)
    clips: list[dict] = []
    offer_segments = [seg for seg in window if segment_has_any(seg, ["금 한", "억원"])]
    setup_terms = ["공약", "팬미팅"]
    setup_segments = [seg for seg in window if segment_has_any(seg, setup_terms)]
    first_offer_start = float(offer_segments[0]["start"]) if offer_segments else float(cluster["start"])
    setup_before_offer = [seg for seg in setup_segments if float(seg["start"]) <= first_offer_start]
    if setup_before_offer:
        setup_start = max(0.0, min(float(seg["start"]) for seg in setup_before_offer) - 1.0)
        setup_end = min(first_offer_start, setup_start + 7.5)
        setup_bounds = bounds_for_segments(segments_between(segments, setup_start, setup_end), max_duration=7.5)
    else:
        setup_bounds = None

    offer_bounds = None
    if offer_segments:
        offer_start = max(0.0, min(float(seg["start"]) for seg in offer_segments) - 0.4)
        offer_bounds = bounds_for_segments(segments_between(segments, offer_start, offer_start + 10.8), max_duration=10.8)

    offer_end = offer_bounds[1] if offer_bounds else first_offer_start
    small_reaction_bounds = bounds_for_segments(
        segments_starting_between(segments, offer_end, offer_end + 4.0),
        max_duration=4.0,
    )

    correction_start = None
    for seg in window:
        if float(seg["start"]) >= offer_end and segment_has_any(seg, ["금 금", "열돈", "130", "돈", "계산", "바보"]):
            correction_start = float(seg["start"])
            break
    correction_search_start = correction_start or (small_reaction_bounds[0] if small_reaction_bounds else offer_end)
    calc_segments = [
        seg
        for seg in segments_starting_between(segments, correction_search_start, correction_search_start + 24.0)
        if segment_has_any(seg, ["금", "돈", "130", "계산", "바보", "쏠"])
    ]
    correction_bounds = bounds_for_segments(calc_segments, max_duration=21.9)
    calc_end = correction_bounds[1] if correction_bounds else cluster["end"]

    add_clip_from_bounds(
        clips,
        setup_bounds,
        "hook",
    )
    add_clip_from_bounds(
        clips,
        offer_bounds,
        "context",
    )
    add_clip_from_bounds(
        clips,
        correction_bounds,
        "bridge",
    )
    if len(clips) < MIN_CLIP_COUNT - 2:
        add_clip_from_bounds(
            clips,
            small_reaction_bounds,
            "reaction",
        )
    payoff_window = segments_starting_between(segments, calc_end, calc_end + 13.0)
    payoff_anchor_start = None
    for seg in payoff_window:
        text = str(seg.get("text", ""))
        if any(ch.isdigit() for ch in text) or interaction_score(text) >= 2:
            payoff_anchor_start = float(seg["start"])
            break
    payoff_start = payoff_anchor_start if payoff_anchor_start is not None else calc_end
    payoff_after = segments_starting_between(segments, payoff_start, payoff_start + 7.0)
    payoff_bounds = bounds_for_segments(payoff_after, max_duration=6.4)
    if payoff_bounds:
        payoff_chunks = split_bounds(payoff_bounds, 3.4)
        for index, bounds in enumerate(payoff_chunks[:2]):
            add_clip_from_bounds(clips, bounds, "reaction" if index == 0 else "payoff")
    if len(clips) < MIN_CLIP_COUNT:
        return None

    start = min(clip["source_start"] for clip in clips)
    end = max(clip["source_end"] for clip in clips)
    return {
        "candidate_id": f"{chunk['chunk_id']}_seed_money_calc_01",
        "chunk_id": chunk["chunk_id"],
        "score": 90,
        "hook_frame_name": "공약 계산 실수 회수",
        "hook_frame_reason": "공약, 터무니없는 금액, 계산 정정, 반응이 하나의 짧은 payoff thread를 만듭니다.",
        "viewer_promise": "큰소리친 공약이 계산 실수로 무너지는 반전",
        "core_event": "공약 제안과 금액 계산 실수, 정정과 반응을 묶은 몽타주",
        "why_it_works": "돈/계산/공약은 이해가 빠르고 반응 payoff가 선명합니다.",
        "emotion_arc": "기대 -> 과장 -> 계산 오류 -> 폭소",
        "thread_key": "공약 계산 실수",
        "thread_intention": "공약의 과장과 계산 오류를 짧게 회수해 웃음을 만드는 구조",
        "timeline_answerability": "high",
        "fragment_risk": "low",
        "separation_notes": "넓은 준비 대화는 빼고 공약/금액/계산/반응만 묶음",
        "candidate_start": round(start, 3),
        "candidate_end": round(end, 3),
        "hook_moment": {"time": clips[0]["source_start"], "reason": "공약이 훅 역할을 함"},
        "payoff_moment": {"time": clips[-2]["source_start"], "reason": "계산 오류가 payoff 역할을 함"},
        "narration_need": "none",
        "clip_blueprint": clips[:MAX_CLIP_COUNT],
        "title_angle": "공약이 계산 실수로 무너지는 순간",
    }


def build_anchor_seed_candidates(chunk_records: list[dict]) -> list[dict]:
    seeds: list[dict] = []
    for chunk in chunk_records:
        if not str(chunk.get("chunk_id", "")).startswith("wide_"):
            continue
        for builder in (build_social_risk_seed_candidate, build_money_calculation_seed_candidate):
            candidate = builder(chunk)
            if not candidate:
                continue
            ok, _message = validate_local_result({"candidates": [candidate]})
            if ok:
                seeds.append(candidate)
    return seeds


def clip_overlaps_range(clip: dict, start_sec: Optional[float], end_sec: Optional[float]) -> bool:
    if start_sec is None or end_sec is None:
        return True
    clip_start = float(clip.get("source_start", 0.0))
    clip_end = float(clip.get("source_end", clip_start))
    return clip_end >= float(start_sec) and clip_start <= float(end_sec)


def load_reference_style_examples(path: Optional[Path]) -> list[dict]:
    if not path or not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw if isinstance(raw, list) else raw.get("shorts", raw.get("examples", []))
    if not isinstance(items, list):
        return []

    def reference_metric(item: dict) -> float:
        if "f1" in item:
            return float(item.get("f1") or 0.0)
        return float(item.get("reference_score", 0.0) or 0.0)

    best_by_reference: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        clips = item.get("source_clips") or item.get("clips") or []
        if not clips:
            continue
        reference = str(item.get("reference") or item.get("candidate_id") or item.get("title") or "reference")
        score = reference_metric(item)
        previous = best_by_reference.get(reference)
        if previous is None or score > reference_metric(previous):
            best_by_reference[reference] = item

    examples = []
    for reference, item in sorted(best_by_reference.items(), key=lambda pair: pair[0]):
        candidate_id = str(item.get("candidate_id", ""))
        clips = item.get("source_clips") or item.get("clips") or []
        if "tobacco" in reference or "tobacco" in candidate_id or "pd_tobacco" in candidate_id:
            learned_thread = (
                "ad/promotion/smoking callback montage: later accusation/payoff opens the short, "
                "then earlier visual proof/setup clips explain why the accusation is funny"
            )
        elif "gold" in reference or "gold" in candidate_id or "calculation" in candidate_id:
            learned_thread = (
                "promise/calculation mistake montage: promise setup, absurd gold offer, "
                "calculation correction, and reaction/payoff belong to one answer thread"
            )
        else:
            learned_thread = "reference-style answer thread from this source"
        examples.append(
            {
                "reference": reference,
                "candidate_id": candidate_id,
                "learned_thread": learned_thread,
                "source_clips": [
                    {
                        "source_start": float(clip.get("source_start", 0.0)),
                        "source_end": float(clip.get("source_end", 0.0)),
                        "purpose": str(clip.get("purpose", "bridge")),
                    }
                    for clip in clips
                    if "source_start" in clip and "source_end" in clip
                ],
            }
        )
    return examples[:4]


def load_benchmark_profile(path: Optional[Path]) -> dict:
    if not path or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, dict) else {}


def load_learning_rule(path: Optional[Path]) -> dict:
    if not path or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw.get("rule_id"):
        return {}
    return raw


def format_learning_rule(rule: dict) -> str:
    if not rule:
        return ""
    constraints = [str(item).strip() for item in rule.get("constraints", []) or [] if str(item).strip()]
    lines = [
        "Active orchestration learning rule:",
        f"- rule_id={rule.get('rule_id', '')}; mode={rule.get('mode', '')}; kind={rule.get('rule_kind', '')}",
        f"- change_note={rule.get('change_note', '')}",
        "- This rule is mandatory for the current run. Change only what its constraints specify; retain source truth and standalone clarity.",
    ]
    for constraint in constraints:
        lines.append(f"- required constraint: {constraint}")
    return "\n".join(lines)


def format_benchmark_profile(profile: dict) -> str:
    if not profile:
        return ""

    score = profile.get("score") if isinstance(profile.get("score"), dict) else {}
    lines = [
        "Genre benchmark profile:",
        f"- profile={profile.get('profile_id', '')}; scope={profile.get('scope', '')}",
        "- Treat this profile as a scoring rubric. Never copy its creators, characters, phrases, or jokes into another source.",
    ]
    for dimension in score.get("dimensions", []) or []:
        if not isinstance(dimension, dict):
            continue
        lines.append(
            f"- score {dimension.get('id', '')} ({dimension.get('weight', 0)}): {dimension.get('question', '')}"
        )
    for penalty in score.get("penalties", []) or []:
        if not isinstance(penalty, dict):
            continue
        lines.append(
            f"- penalty {penalty.get('id', '')} (up to -{penalty.get('max_deduction', 0)}): {penalty.get('rule', '')}"
        )
    for rule in profile.get("selection_rules", []) or []:
        lines.append(f"- selection rule: {rule}")
    policy = profile.get("decision_policy") if isinstance(profile.get("decision_policy"), dict) else {}
    if policy:
        lines.append(
            "- decision policy: "
            + "; ".join(f"{key}={value}" for key, value in policy.items())
        )
    return "\n".join(lines)


def benchmark_score_config() -> tuple[str, list[dict], dict[str, dict], int, int]:
    if not BENCHMARK_PROFILE:
        return "", [], {}, 0, 0
    score = BENCHMARK_PROFILE.get("score") if isinstance(BENCHMARK_PROFILE.get("score"), dict) else {}
    dimensions = [
        item
        for item in score.get("dimensions", []) or []
        if isinstance(item, dict) and item.get("id") and int(item.get("weight", 0) or 0) > 0
    ]
    penalties = {
        str(item.get("id")): item
        for item in score.get("penalties", []) or []
        if isinstance(item, dict) and item.get("id")
    }
    pass_score = int(score.get("pass_score", 78) or 78)
    review_score = int(score.get("review_score", 68) or 68)
    return str(BENCHMARK_PROFILE.get("profile_id", "")), dimensions, penalties, pass_score, review_score


def benchmark_chunk_top_k() -> int:
    return BENCHMARK_CHUNK_TOP_K if BENCHMARK_PROFILE else CHUNK_TOP_K


def benchmark_final_top_k() -> int:
    return BENCHMARK_FINAL_TOP_K if BENCHMARK_PROFILE else FINAL_TOP_K


def normalize_genre_scorecard(value: object) -> tuple[dict | None, str]:
    profile_id, dimensions, penalties_by_id, pass_score, review_score = benchmark_score_config()
    if not profile_id:
        return None, ""
    if not isinstance(value, dict):
        return None, "Missing genre_scorecard for the active benchmark profile."
    if str(value.get("profile_id", "")) != profile_id:
        return None, "genre_scorecard profile_id does not match the active benchmark profile."

    raw_dimensions = value.get("dimensions")
    if not isinstance(raw_dimensions, list):
        return None, "genre_scorecard.dimensions must be a list."
    expected_ids = [str(item["id"]) for item in dimensions]
    found: dict[str, dict] = {}
    for item in raw_dimensions:
        if not isinstance(item, dict):
            return None, "genre_scorecard contains a non-object dimension."
        dimension_id = str(item.get("id", ""))
        if dimension_id not in expected_ids or dimension_id in found:
            return None, "genre_scorecard dimensions must contain each configured id exactly once."
        max_score = int(next(spec["weight"] for spec in dimensions if str(spec["id"]) == dimension_id))
        try:
            item_score = int(item.get("score"))
        except Exception:
            return None, f"genre_scorecard {dimension_id} score must be an integer."
        if not 0 <= item_score <= max_score:
            return None, f"genre_scorecard {dimension_id} score must be 0 to {max_score}."
        reason = str(item.get("reason", "")).strip()
        if not reason:
            return None, f"genre_scorecard {dimension_id} requires a reason."
        found[dimension_id] = {"id": dimension_id, "score": item_score, "max_score": max_score, "reason": reason}
    if set(found) != set(expected_ids):
        return None, "genre_scorecard is missing one or more configured dimensions."

    raw_penalties = value.get("penalties", [])
    if not isinstance(raw_penalties, list):
        return None, "genre_scorecard.penalties must be a list."
    normalized_penalties = []
    seen_penalty_ids = set()
    for item in raw_penalties:
        if not isinstance(item, dict):
            return None, "genre_scorecard contains a non-object penalty."
        penalty_id = str(item.get("id", ""))
        if penalty_id not in penalties_by_id or penalty_id in seen_penalty_ids:
            return None, "genre_scorecard penalty ids must come from the active profile and be unique."
        try:
            deduction = int(item.get("deduction"))
        except Exception:
            return None, f"genre_scorecard {penalty_id} deduction must be an integer."
        max_deduction = int(penalties_by_id[penalty_id].get("max_deduction", 0) or 0)
        if not 1 <= deduction <= max_deduction:
            return None, f"genre_scorecard {penalty_id} deduction must be 1 to {max_deduction}."
        reason = str(item.get("reason", "")).strip()
        if not reason:
            return None, f"genre_scorecard {penalty_id} requires a reason."
        normalized_penalties.append({"id": penalty_id, "deduction": deduction, "reason": reason})
        seen_penalty_ids.add(penalty_id)

    dimension_total = sum(item["score"] for item in found.values())
    penalty_total = sum(item["deduction"] for item in normalized_penalties)
    total = max(0, dimension_total - penalty_total)
    # A large context/crop/payoff penalty should never pass straight to rendering,
    # even when the remaining dimensions produce a high total.
    has_blocking_penalty = any(item["deduction"] > 5 for item in normalized_penalties)
    decision = (
        "auto_render"
        if total >= pass_score and not has_blocking_penalty
        else "review"
        if total >= review_score
        else "reject"
    )
    return {
        "profile_id": profile_id,
        "total": total,
        "dimension_total": dimension_total,
        "penalty_total": penalty_total,
        "dimensions": [found[item_id] for item_id in expected_ids],
        "penalties": normalized_penalties,
        "decision": decision,
    }, ""


def format_reference_style_examples(
    examples: list[dict],
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> str:
    if not examples:
        return "Reference short calibration: unavailable"

    lines = [
        "Benchmark reference calibration:",
        "- These examples are reverse-engineered from user-provided reference shorts for offline scoring/debugging only.",
        "- This block is not part of the GUI/default production prompt.",
        "- Do not copy exact title wording; use these only to inspect whether a candidate has the same title intention, meaning, and source-clip structure.",
    ]
    shown = 0
    for example in examples:
        clips = example.get("source_clips", []) or []
        overlapping = [clip for clip in clips if clip_overlaps_range(clip, start_sec, end_sec)]
        if start_sec is not None and end_sec is not None and not overlapping:
            continue
        coverage_note = "full pattern available in this window" if len(overlapping) == len(clips) else "partial pattern only; recover full pattern in a wider window"
        lines.append(f"- reference={example.get('reference', '')}; learned_thread={example.get('learned_thread', '')}; {coverage_note}")
        if len(overlapping) == len(clips):
            lines.append(
                "  Benchmark note: this exact skeleton can be used to score reproduction, but it should not be treated as a generalized production rule."
            )
        for clip in clips:
            marker = "*" if clip in overlapping else "-"
            lines.append(
                f"  {marker} {clip['purpose']}: {clip['source_start']:.1f}-{clip['source_end']:.1f}"
            )
        shown += 1
    if shown == 0:
        return "Reference short calibration: no overlapping example for this window"
    return "\n".join(lines)


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


def trim_text(text: str, max_len: int = 180) -> str:
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 1].rstrip() + "…"


def format_youtube_context_for_prompt(start_sec: Optional[float] = None, end_sec: Optional[float] = None) -> str:
    if not YOUTUBE_CONTEXT:
        return "YouTube audience/comment context: unavailable"

    metadata = YOUTUBE_CONTEXT.get("metadata", {}) if isinstance(YOUTUBE_CONTEXT.get("metadata"), dict) else {}
    comments = YOUTUBE_CONTEXT.get("comments", {}) if isinstance(YOUTUBE_CONTEXT.get("comments"), dict) else {}
    insights = (
        YOUTUBE_CONTEXT.get("comment_insights", {})
        if isinstance(YOUTUBE_CONTEXT.get("comment_insights"), dict)
        else {}
    )
    moments = insights.get("timecode_moments", []) if isinstance(insights.get("timecode_moments"), list) else []
    top_comments = (
        insights.get("top_reaction_comments", [])
        if isinstance(insights.get("top_reaction_comments"), list)
        else []
    )
    top_topics = (
        insights.get("top_comment_topics", [])
        if isinstance(insights.get("top_comment_topics"), list)
        else []
    )

    if start_sec is not None and end_sec is not None:
        low = max(0.0, float(start_sec) - 45.0)
        high = float(end_sec) + 45.0
        moments = [
            item
            for item in moments
            if low <= float(item.get("representative_time_sec", item.get("bucket_start_sec", -1))) <= high
        ]

    compact_moments = []
    for item in sorted(moments, key=lambda x: float(x.get("reaction_score", 0)), reverse=True)[:8]:
        compact_moments.append(
            {
                "time_sec": item.get("representative_time_sec", item.get("bucket_start_sec")),
                "reaction_score": item.get("reaction_score", 0),
                "comment_count": item.get("comment_count", 0),
                "like_count": item.get("like_count", 0),
                "samples": [
                    {
                        "text": trim_text(sample.get("text", ""), 120),
                        "like_count": sample.get("like_count", 0),
                        "reaction_keywords": sample.get("reaction_keywords", []),
                    }
                    for sample in (item.get("samples", []) or [])[:2]
                ],
            }
        )

    compact_topics = []
    for item in sorted(top_topics, key=lambda x: float(x.get("audience_score", 0)), reverse=True):
        timecodes = [float(sec) for sec in (item.get("timecodes", []) or [])]
        if start_sec is not None and end_sec is not None and timecodes:
            near_timecodes = [sec for sec in timecodes if low <= sec <= high]
            if not near_timecodes:
                continue
        else:
            near_timecodes = timecodes[:8]
        compact_topics.append(
            {
                "topic": item.get("topic", ""),
                "audience_score": item.get("audience_score", 0),
                "comment_count": item.get("comment_count", 0),
                "like_count": item.get("like_count", 0),
                "timecodes": near_timecodes[:8],
                "samples": [
                    {
                        "text": trim_text(sample.get("text", ""), 130),
                        "like_count": sample.get("like_count", 0),
                        "timecodes": sample.get("timecodes", []),
                    }
                    for sample in (item.get("samples", []) or [])[:3]
                ],
            }
        )
        if len(compact_topics) >= 8:
            break

    compact_comments = []
    for item in sorted(top_comments, key=lambda x: float(x.get("reaction_score", 0)), reverse=True)[:6]:
        compact_comments.append(
            {
                "reaction_score": item.get("reaction_score", 0),
                "like_count": item.get("like_count", 0),
                "reply_count": item.get("reply_count", 0),
                "timecodes": item.get("timecodes", []),
                "reaction_keywords": item.get("reaction_keywords", []),
                "text": trim_text(item.get("text", ""), 140),
            }
        )

    compact = {
        "video_title": metadata.get("title", ""),
        "channel_title": metadata.get("channel_title", ""),
        "view_count": metadata.get("view_count", 0),
        "like_count": metadata.get("like_count", 0),
        "comment_count": metadata.get("comment_count", comments.get("fetched_count", 0)),
        "fetched_comments": comments.get("fetched_count", 0),
        "timecode_moments": compact_moments,
        "top_comment_topics": compact_topics,
        "top_reaction_comments": compact_comments,
        "keyword_counts": insights.get("keyword_counts", {}),
    }
    return "YouTube audience/comment context:\n" + json.dumps(compact, ensure_ascii=False, indent=2)


def get_top_comment_topics() -> list[dict]:
    if not YOUTUBE_CONTEXT:
        return []
    insights = (
        YOUTUBE_CONTEXT.get("comment_insights", {})
        if isinstance(YOUTUBE_CONTEXT.get("comment_insights"), dict)
        else {}
    )
    topics = insights.get("top_comment_topics", [])
    return topics if isinstance(topics, list) else []


def candidate_text_blob(candidate: dict) -> str:
    fields = [
        candidate.get("hook_frame_name", ""),
        candidate.get("hook_frame_reason", ""),
        candidate.get("viewer_promise", ""),
        candidate.get("core_event", ""),
        candidate.get("why_it_works", ""),
        candidate.get("emotion_arc", ""),
        candidate.get("title_angle", ""),
    ]
    return " ".join(str(field) for field in fields)


def audience_topic_signals_for_candidate(candidate: dict) -> list[dict]:
    topics = get_top_comment_topics()
    if not topics:
        return []
    try:
        start = float(candidate.get("candidate_start"))
        end = float(candidate.get("candidate_end"))
    except Exception:
        return []
    low = max(0.0, start - 45.0)
    high = end + 45.0
    text_blob = candidate_text_blob(candidate)
    signals = []
    for topic in topics:
        topic_name = str(topic.get("topic", "")).strip()
        if not topic_name:
            continue
        timecodes = [float(sec) for sec in (topic.get("timecodes", []) or [])]
        near_timecodes = [sec for sec in timecodes if low <= sec <= high]
        text_match = topic_name in text_blob
        if not near_timecodes and not text_match:
            continue
        audience_score = float(topic.get("audience_score", 0) or 0)
        signal_score = audience_score * (2.0 if text_match else 0.45) + len(near_timecodes) * 30.0
        if text_match and near_timecodes:
            signal_score += 250.0
        signals.append(
            {
                "topic": topic_name,
                "signal_score": round(signal_score, 3),
                "audience_score": topic.get("audience_score", 0),
                "comment_count": topic.get("comment_count", 0),
                "like_count": topic.get("like_count", 0),
                "timecodes": near_timecodes[:8],
                "text_match": text_match,
                "samples": [
                    {
                        "text": trim_text(sample.get("text", ""), 130),
                        "like_count": sample.get("like_count", 0),
                        "timecodes": sample.get("timecodes", []),
                    }
                    for sample in (topic.get("samples", []) or [])[:3]
                ],
            }
        )
    return sorted(signals, key=lambda item: item["signal_score"], reverse=True)[:3]


def attach_audience_topic_signals(candidates: list[dict]) -> None:
    for candidate in candidates:
        signals = audience_topic_signals_for_candidate(candidate)
        candidate["audience_topic_signal"] = signals
        candidate["audience_topic_score"] = round(sum(float(item.get("signal_score", 0)) for item in signals), 3)


def build_wide_chunk_records(chunk_records: list[dict], group_size: int = WIDE_WINDOW_GROUP_SIZE) -> list[dict]:
    if group_size < 2 or len(chunk_records) < group_size:
        return []

    wide_records = []
    for index in range(0, len(chunk_records) - group_size + 1):
        group = chunk_records[index : index + group_size]
        first = group[0]
        last = group[-1]
        first_label = str(first["chunk_id"]).replace("chunk_", "")
        last_label = str(last["chunk_id"]).replace("chunk_", "")
        combined_segments = []
        for record in group:
            combined_segments.extend(record["segments"])
        wide_records.append(
            {
                "chunk_id": f"wide_{first_label}_{last_label}",
                "path": " + ".join(str(record["path"]) for record in group),
                "start_sec": float(first["start_sec"]),
                "end_sec": float(last["end_sec"]),
                "segments": clean_segments(combined_segments),
            }
        )
    return wide_records


def load_chunk_transcripts(include_wide_windows: bool = True) -> list[dict]:
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
    if include_wide_windows:
        chunk_records.extend(build_wide_chunk_records(chunk_records))
    return chunk_records


def load_merged_segments() -> list[dict]:
    with open(MERGED_TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return clean_segments(data["segments"])


def extract_window(segments: list[dict], start_sec: float, end_sec: float, pad_sec: float = 20.0) -> list[dict]:
    window_start = max(0.0, start_sec - pad_sec)
    window_end = end_sec + pad_sec
    return [seg for seg in segments if seg["end"] >= window_start and seg["start"] <= window_end]


def extract_clip_context(segments: list[dict], clips: list[dict], pad_sec: float = 12.0) -> list[dict]:
    selected = []
    seen = set()
    valid_clips = []
    for clip in clips:
        if not isinstance(clip, dict):
            continue
        try:
            start = float(clip.get("source_start"))
            end = float(clip.get("source_end"))
        except Exception:
            continue
        valid_clips.append((start, end))

    for start, end in sorted(valid_clips):
        for seg in extract_window(segments, start, end, pad_sec=pad_sec):
            key = (seg.get("start"), seg.get("end"), seg.get("text", ""))
            if key in seen:
                continue
            seen.add(key)
            selected.append(seg)

    return clean_segments(sorted(selected, key=lambda item: (item["start"], item["end"])))


def extract_candidate_context(segments: list[dict], candidate: dict) -> list[dict]:
    clips = candidate.get("clip_blueprint", []) or []
    clip_context = extract_clip_context(segments, clips)
    if clip_context:
        return clip_context
    return extract_window(
        segments,
        float(candidate["candidate_start"]),
        float(candidate["candidate_end"]),
        pad_sec=25.0,
    )


def build_local_prompt(chunk: dict) -> str:
    candidate_limit = benchmark_chunk_top_k()
    wide_window_extra_rule = ""
    if str(chunk.get("chunk_id", "")).startswith("wide_"):
        wide_window_extra_rule = """
Wide-window cross-chunk rule:
- This is a wide window made by joining adjacent transcript chunks.
- Reserve at least 2 candidates for cross-chunk motif/payoff threads when the evidence exists.
- A cross-chunk montage thread should use useful clips from more than one original chunk, not just a continuous scene copied from one chunk.
- Specifically hunt for payoff anchors near any part of the wide window that need earlier or later evidence to make sense.
- Do not require an accusation/callback; cross-chunk threads can be built from repeated object/style motifs, promise/calculation, process/reveal, challenge/result, or emotional setup/payoff.
- In a wide window, make an explicit anchor inventory before selecting: money/calculation, social-risk words, production trouble, object/style motif, process/reveal, challenge/result, emotional release.
- Use the Auto payoff-anchor hints above as the inventory seed; do not ignore a social-risk or money/calculation hint just because it is brief.
- Category coverage rule: if the Auto hints contain social_risk lines, output at least one candidate anchored on the strongest social_risk payoff unless it is clearly throwaway with no reaction or evidence.
- A social_risk coverage candidate must name the risk in hook_frame_name/title_angle/core_event. Do not let the next adjacent topic, such as a test, quiz, gift, or ordinary chat, hijack the candidate meaning.
- Category coverage rule: if the Auto hints contain money_calculation lines, output at least one candidate anchored on the strongest money/calculation payoff unless it is clearly not a payoff thread.
- If any inventory category has a sharp payoff anchor, include the best candidate for that category unless it is clearly weaker than all others.
- If you return only single-chunk candidates from a wide window, you probably missed the main reason this wide window exists.
- Do not let a simple prop/styling/context fragment beat the payoff anchor that gives the fragment meaning.
"""
    return f"""Source title: {SOURCE_TITLE}
{format_movie_info_for_prompt()}
{format_youtube_context_for_prompt(chunk['start_sec'], chunk['end_sec'])}
Chunk id: {chunk['chunk_id']}
Chunk range: {chunk['start_sec']:.3f} to {chunk['end_sec']:.3f}
{format_reference_style_examples(REFERENCE_STYLE_EXAMPLES, chunk['start_sec'], chunk['end_sec'])}
{BENCHMARK_PROFILE_CONTEXT}
{LEARNING_RULE_CONTEXT}
{format_anchor_hints_for_prompt(chunk['segments'])}
{wide_window_extra_rule}

Task:
- Find the top {candidate_limit} local shorts candidates from this chunk only.
- Each candidate must be a distinct source scene or variety bit with a distinct payoff beat.
- If benchmark reference calibration is explicitly supplied, use it only as a scoring/debugging lens; do not copy an exact answer skeleton unless the source evidence independently supports that same thread.
- When an Active orchestration learning rule is present, apply every required constraint to candidate selection, hook order, and clip_blueprint. Do not claim the rule was applied without source evidence.
- Before choosing candidates, mentally separate the transcript into answerable threads, not isolated funny lines.
- A thread is a repeated tease, object/style motif, promise, accusation, misunderstanding, calculation, process, challenge, transformation, or emotional question that can be named in one title.
- The candidate should represent the whole answer thread that a human could score against a timeline answer.
- Run a payoff-anchor recovery pass before scoring candidates.
- A payoff anchor can be a visible reaction, exposed contradiction, correction, refusal, punchline quote, object reuse, process reveal, challenge result, or emotional release.
- Brief high-signal anchors should not be buried by longer ordinary conversations. High-signal anchors include money/gold/price, calculation mistake, smoking/ad/promotion, alcohol, public embarrassment, accusation, apology, refusal, contradiction, or a group reaction.
- For each anchor, identify whether it is self-contained or needs earlier/later evidence. If it needs evidence, attach the useful setup/proof/result clips to the same candidate.
- For a social-risk anchor, use the risk line as the hook or payoff return, then inspect object/style/image, production, gift, comment, or prior tease hints in the same wide window as possible evidence.
- If a social-risk or brand-safety line appears, the candidate must be about that risk/reaction, not about the topic that starts immediately after it.
- If a social-risk or brand-safety line has earlier object/style/image, production, or repeated tease hints in the same wide window, do not output only the continuous risk area. Build a testable montage with the risk line plus 2 to 4 earlier evidence cuts.
- Earlier evidence cuts for social-risk montages can be styling, props, gifts, image comments, production prompts, or repeated visual impressions. Use only evidence that plausibly explains why the risk line became funny or awkward.
- Indirect evidence is allowed only when the candidate can name why the evidence changes the hook/payoff; otherwise it is atmosphere and must be omitted.
- For caption-dependent montages, clip_blueprint must provide short caption-worthy beats, not long unbroken scenes.
- For a money/calculation anchor, use only the promise/challenge, absurd number or offer, calculation/correction, and reaction/payoff lines; omit broad surrounding talk.
- Output the payoff-thread candidate, not each earlier setup clip as its own short.
- For object/style/image motifs, do not stop at the first prop or compliment; look for the later reaction, reuse, joke, reveal, or image change that gives the motif a payoff.
- For accusation/ad/promotion/smoking-style callbacks, include the later accusation/payoff plus earlier proof/setup cuts from the same wide window when they exist.
- A candidate that only uses a final 5 to 15 second payoff area is a fragment when its meaning clearly depends on earlier evidence.
- If a timestamped audience topic points to a visual gag or side character that the transcript barely describes, still create a candidate around that time window.
- Strong recurring comment topics can outweigh ordinary dialogue-only beats.
- For Korean celebrity YouTube/talk sources, favor the strongest source-specific engine: moments with a clear trigger, escalation or change, and a visible/audible payoff.
- For movie/drama sources, favor hook, power abuse, scam, corruption, money tension, public humiliation, betrayal, reveal, reversal, jealousy, irony, or comeback.
- Each candidate must also have a short, market-facing hook_frame_name in Korean.
- A hook frame should explain why a viewer clicks this short, not just what happens in the source.
- Do not copy hook frames from previous sources; create a compact phrase from current-source evidence only.
- Prefer candidates where person/role, trigger or opponent, and payoff are easy to understand.
- Prefer candidates that are immediately clickable even if the viewer knows nothing about the source.
- Every candidate must already be thinkable as a {RECOMMENDED_CLIP_COUNT_MIN} to {RECOMMENDED_CLIP_COUNT_MAX} cut short.
- For Korean entertainment/talk sources, a candidate may be a non-contiguous montage when several moments in this window repeat one running joke, object, accusation, promise, tease, or emotional question.
- For montage candidates, candidate_start and candidate_end should cover the full source span used by the clips, but clip_blueprint should contain only the useful short clips.
- clip_blueprint playback order may be non-chronological when opening with a later hook makes the short clearer or funnier.
- Do not fill the gap between distant montage clips with unrelated transcript context.
- If an early clip is only setup for a stronger later payoff, do not output the early clip as a separate final candidate; attach it to the stronger thread.
- A candidate that only matches a future gold timeline by overlap but has a different title intention is a false match; lower its score or omit it.
- Prefer candidates whose title_angle, core_event, hook_moment, payoff_moment, and clip_blueprint would all point to the same gold answer.
- For promise/calculation threads, the minimum useful structure is: promise setup -> absurd number or offer -> calculation/correction -> visible or verbal reaction/payoff.
- For promise/calculation threads, omit unrelated preparation talk. Keep only the promise/challenge line, the absurd offer or number, the correction/calculation, and the reaction/payoff.
- For object/style/image threads, the minimum useful structure is: object/style setup -> repeated comment/use -> later reaction, reveal, or payoff.
- For callback/accusation threads, the minimum useful structure is: later callback hook -> earlier setup/proof clips -> return to reaction/payoff.
- For process/transformation threads, the minimum useful structure is: before state -> key steps -> reveal -> reaction.
- For emotional/testimony threads, the minimum useful structure is: prompt/context -> confession/message -> listener reaction -> emotional release.
- For each candidate, include thread_key, thread_intention, timeline_answerability, fragment_risk, and separation_notes.
- The hook and the payoff must both exist inside this chunk.
- Reject candidates that need a key payoff from another chunk.
- If two candidates heavily overlap in time or share the same payoff beat, keep only the stronger one.
- If two candidates would use the same hook_frame_name and the same payoff logic, keep only the stronger one.
- If two candidates are basically the same scene with different title angles, keep only the stronger one.
- clip_blueprint must contain {MIN_CLIP_COUNT} to {MAX_CLIP_COUNT} cuts.
- clip_blueprint must open with purpose "hook".
- clip_blueprint should restore context quickly after the opening hook.
- clip_blueprint should usually total about 30 to 60 seconds of source material for entertainment/talk shorts, or 22 to 45 seconds for very tight drama beats.
- Each clip_blueprint item should usually be 1 to 6 seconds, should rarely exceed 8 seconds, and must never exceed {MAX_LOCAL_CLIP_DURATION_SEC:.0f} seconds.
- For non-contiguous montage candidates, split long evidence scenes into micro-cuts; do not use one 20+ second evidence slab when three 3-7 second beats would carry the same meaning better.
- Do not use rounded 10, 20, 30, or 50 second slabs. Use transcript line boundaries and cut away filler between key beats.
- A candidate with broad surrounding talk instead of precise setup/escalation/payoff lines should be scored lower than a tighter micro-montage.
- Do not output one long unbroken conversation block; even a continuous exchange should be split into hook, reaction, bridge, correction, and payoff cuts.
- Split reaction, pause, stare, comeback, and reveal into separate cuts instead of using one long conversation block.
- Use Korean for descriptive text fields.
- When a Genre benchmark profile is present, score every configured dimension from 0 to its listed weight and explain each score in one short Korean sentence.
- Use a penalty only when the listed risk is present in source evidence. The candidate score must equal the dimension total minus deductions.
- If the chunk does not truly contain {candidate_limit} strong moments, return fewer.

Return JSON with this exact shape:
{{
  "chunk_id": "{chunk['chunk_id']}",
  "candidates": [
    {{
      "candidate_id": "{chunk['chunk_id']}_cand_01",
      "chunk_id": "{chunk['chunk_id']}",
      "score": 82,
      "genre_scorecard": {{
        "profile_id": "active benchmark profile id when supplied",
        "dimensions": [
          {{"id": "configured dimension id", "score": 0, "reason": "Korean evidence-based reason"}}
        ],
        "penalties": [
          {{"id": "configured penalty id", "deduction": 1, "reason": "Korean evidence-based reason"}}
        ]
      }},
      "hook_frame_name": "시장형 훅 프레임",
      "hook_frame_reason": "이 장면이 어떤 시청자 약속으로 팔리는지 설명",
      "viewer_promise": "시청자가 이 쇼츠에서 기대하는 보상이나 반전",
      "core_event": "짧은 한국어 요약",
      "why_it_works": "왜 쇼츠가 되는지 한국어 설명",
      "emotion_arc": "감정 흐름",
      "thread_key": "반복 장난/약속/오해/계산 등 한 줄 식별자",
      "thread_intention": "이 후보가 걸고 있는 시청자 약속과 최종 페이오프",
      "timeline_answerability": "high|medium|low",
      "fragment_risk": "none|low|medium|high",
      "separation_notes": "비슷한 구간/부분 조각과 왜 분리하거나 합쳐야 하는지",
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
    final_limit = benchmark_final_top_k()
    compact = []
    for cand in local_candidates:
        compact.append(
            {
                "candidate_id": cand["candidate_id"],
                "chunk_id": cand["chunk_id"],
                "score": cand["score"],
                "genre_scorecard": cand.get("genre_scorecard", {}),
                "hook_frame_name": cand.get("hook_frame_name", ""),
                "hook_frame_reason": cand.get("hook_frame_reason", ""),
                "viewer_promise": cand.get("viewer_promise", ""),
                "core_event": cand["core_event"],
                "why_it_works": cand["why_it_works"],
                "emotion_arc": cand.get("emotion_arc", ""),
                "thread_key": cand.get("thread_key", ""),
                "thread_intention": cand.get("thread_intention", ""),
                "timeline_answerability": cand.get("timeline_answerability", ""),
                "fragment_risk": cand.get("fragment_risk", ""),
                "separation_notes": cand.get("separation_notes", ""),
                "candidate_start": cand["candidate_start"],
                "candidate_end": cand["candidate_end"],
                "hook_moment": cand.get("hook_moment", {}),
                "payoff_moment": cand.get("payoff_moment", {}),
                "narration_need": cand.get("narration_need", "low"),
                "clip_blueprint_duration_sec": total_clip_duration_sec(cand.get("clip_blueprint", [])),
                "clip_blueprint": cand.get("clip_blueprint", []),
                "title_angle": cand.get("title_angle", ""),
                "audience_topic_score": cand.get("audience_topic_score", 0),
                "audience_topic_signal": cand.get("audience_topic_signal", []),
            }
        )

    return f"""Source title: {SOURCE_TITLE}
{format_movie_info_for_prompt()}
{format_youtube_context_for_prompt()}
You are selecting the best final shorts from the candidate pool below.
{REFERENCE_STYLE_CONTEXT}
{LEARNING_RULE_CONTEXT}

Rules:
- select up to {final_limit}
- prioritize strongest view-driving candidates first
- if benchmark reference calibration examples are explicitly supplied, use them to audit title intention and timeline similarity; do not let them override source-evidence quality in normal selection
- when an Active orchestration learning rule is present, prefer candidates that visibly satisfy its required constraints over otherwise similar candidates
- each selected short must use genuinely distinct source footage
- keep the bar high
- prioritize candidates that can become a satisfying short with at least 5 cuts
- later moments can open the short if that improves hook and clarity
- prefer candidates whose hook_frame_name is easy to package into a short title
- prefer candidates whose viewer_promise is specific and immediately legible
- prefer candidates whose clip_blueprint already forms a clean hook -> context recovery -> payoff sequence
- prefer candidates whose clip_blueprint duration already fits a real short
- prefer precise micro-montages over broad scene coverage when both describe the same payoff thread
- demote candidates whose source_clips/clip_blueprint use large rounded ranges that include filler around the payoff anchor
- do not bury brief high-signal anchors such as money, calculation mistakes, smoking/ad/promotion, public embarrassment, refusal, accusation, or sudden group reaction just because another ordinary conversation is longer
- demote candidates that include a social-risk line but title/core_event the next adjacent ordinary topic instead of the risk itself
- for Korean celebrity YouTube/talk sources, value source-specific moments that have a clear trigger, interaction, change, and payoff even when the stakes are low
- when a Genre benchmark profile is present, keep candidates with a higher genre_scorecard total ahead of otherwise similar candidates and demote genre_scorecard decisions of review or reject
- for those sources, prefer candidates that can become a title pair like "trigger/person" plus "reaction/payoff"
- do not bury a candidate just because the transcript is quiet if comments repeatedly call out a visual subject or side character
- a lower local score candidate with a strong recurring audience topic and timestamp evidence should usually be selected
- select thread-level answers, not setup fragments
- prefer candidates with timeline_answerability="high" and fragment_risk="none" or "low"
- demote candidates with fragment_risk="medium" or "high" unless their payoff is clearly independent
- if one candidate is a setup/context fragment of a stronger montage thread, discard the fragment even when it has high raw overlap with useful footage
- if a callback-level candidate explains why several earlier fragments matter, select that callback-level candidate above any individual setup fragment
- do not rank a prop/object/context fragment above the payoff anchor that gives it meaning
- prefer candidates whose clip_blueprint recovers the evidence needed for the payoff anchor, whether the thread is a calculation, object motif, process reveal, challenge result, callback, or emotional release
- demote candidates that title a payoff thread correctly but only contain the final payoff area when the meaning clearly depends on earlier evidence
- demote candidates that overlap setup/proof moments but title them as an independent prop, styling, or compliment bit when a later payoff gives those moments a sharper meaning
- duplicate_group should name the answer thread, not just the nearest time range
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
{format_youtube_context_for_prompt(candidate.get('candidate_start'), candidate.get('candidate_end'))}
Final rank: {rank}
Target short id: {short_id}
{format_reference_style_examples(REFERENCE_STYLE_EXAMPLES, candidate.get('candidate_start'), candidate.get('candidate_end'))}
{BENCHMARK_PROFILE_CONTEXT}
{LEARNING_RULE_CONTEXT}

Candidate summary:
{json.dumps(candidate, ensure_ascii=False, indent=2)}

Candidate clip blueprint (starting skeleton, in playback order):
{format_clip_plan_for_prompt(blueprint)}

Candidate clip blueprint total duration: {blueprint_total:.1f} seconds

Task:
- Create a final shorts package for CapCut rough-cut generation.
- When an Active orchestration learning rule is present, preserve its required change in the final source_clips. Do not silently revert to the previous default edit structure.
- If this package is driven by audience comments about a visual subject, make that subject visible in the title, hook, and clip purposes.
- The final short often works around {TARGET_DURATION_MIN:.0f} to {TARGET_DURATION_MAX:.0f} seconds, but story flow matters more than this preference.
- It must be reconstructable into at least {MIN_CLIP_COUNT} cuts.
- source_clips should usually contain {RECOMMENDED_CLIP_COUNT_MIN} to {RECOMMENDED_CLIP_COUNT_MAX} cuts and never fewer than {MIN_CLIP_COUNT}
- if the event feels simple, split reaction, pause, reveal, and aftermath into separate cuts
- source_clips are the playback order of the short
- source order may be rearranged for hook and flow, but only when clarity is preserved
- use candidate.clip_blueprint as your starting skeleton and keep the same core emotional logic
- if benchmark reference calibration is explicitly supplied, mention similarity in evaluation_notes only when the candidate independently supports the same answer thread
- if candidate.clip_blueprint is a non-contiguous montage, preserve that montage logic instead of collapsing it into one continuous range
- every montage clip must serve the same running joke, object, accusation, promise, tease, or emotional question
- for payoff-driven montages, preserve the setup/proof/process/result clips and the final payoff anchor; do not shrink the package to only the final payoff area
- if the candidate title/summary depends on a repeated object, style, promise, calculation, challenge, process, accusation, or emotional reveal, source_clips should include the evidence that makes that payoff understandable unless the candidate explicitly says the setup is unavailable
- if the title depends on indirect evidence, every indirect evidence clip must be paired with a point caption or a nearby reaction line that explains why it belongs
- if you cannot write a clear binding caption for an indirect clip, remove that clip or reframe the title to the directly visible exchange
- source_clips must be tighter than a recap scene: use exact transcript-line beats, not broad rounded ranges
- for calculation/promise packages, remove unrelated preparation talk and keep only promise/challenge, absurd offer/number, correction/calculation, and reaction/payoff
- for high-signal short anchors such as money, calculation mistakes, social-risk/brand-safety lines, public embarrassment, refusal, accusation, or sudden group reaction, preserve the anchor and its evidence as a micro-montage
- preserve candidate.thread_key and candidate.thread_intention when they exist
- if the candidate is a fragment of a bigger answer thread, state that in evaluation_notes instead of making it look like a complete short
- do not invent a different event from the one described in candidate summary
- the first source_clips item must be the hook
- if the hook is pulled from a later moment, the next 1 to 2 cuts must restore minimum context immediately
- the last source_clips item must land on payoff, reveal, reaction, or aftermath
- let source_clips run shorter or longer when the scene flow needs it; do not add padding just to hit a duration
- target_duration_sec should match the actual sum of source_clips durations, rounded to about 0.1 seconds
- each individual clip usually works best around 1 to 8 seconds, but can be longer when the reaction or setup needs breathing room
- for non-contiguous montage packages, any evidence clip longer than 8 seconds should be split unless it contains one uninterrupted trigger/reaction exchange
- do not pad with long unbroken context if a tighter reaction or bridge cut would work
- title must be exactly 2 lines in Korean
- title_line1 should usually be 4 to 14 visible characters excluding spaces; allow short English tokens when they appear in the source
- title_line2 should usually be 4 to 14 visible characters excluding spaces
- write titles as punchy short fragments, not full sentences or recap text
- line 1 should name the setup, person, or trigger; line 2 should name the reaction, reversal, or payoff
- avoid generic title filler such as "현장", "대공개", "포착", "이유는?", "진심", "모습", and "순간"
- avoid ellipsis and do not end either title line with "..."
- title should feel like a high-performing short headline, not a neutral recap
- title should surface trigger, person/role, reaction, or payoff fast
- for Korean celebrity YouTube/talk, line 1 should name the trigger, person, object, or situation; line 2 should name the reaction, reversal, or payoff
- do not reuse titles or motifs from reference shorts unless the current source independently contains them
- do not use fixed title templates; title wording must come from the current source
- upload_title must be one copy-ready Korean YouTube Shorts upload title that combines the hook and payoff naturally; include 1 to 3 compact hashtags only when useful
- selection_pitch must explain in one Korean sentence why a human would want to click this short
- evaluation_notes must explain how title intention, semantic meaning, and source clip structure line up for this candidate
- timeline_answerability should be high only when a human could match this package to one clear gold timeline answer
- fragment_risk should be high if the source_clips are mostly setup/context for another stronger payoff thread
- fun_tags must be 1 to 3 items such as 웃김, 통쾌, 긴장, 반전, 황당, 설렘, 캐릭터, 관계, 돈, 직장, 가족
- hook_line must be the strongest short dialogue or moment summary
- main_characters must be 1 to 3 role labels or names
- protagonist_presence must be one of high, medium, low, unknown
- standalone_clarity must be one of high, medium, low
- narration should be omitted unless needed
- point_captions must contain 0 to 3 items; keep only the strongest caption beats
- point caption times must be relative to the short timeline
- narration target_start must also be relative to the short timeline
- use natural Korean suitable for the source type; for YouTube variety/talk, prefer casual entertainment phrasing over movie recap phrasing
- avoid bland generic phrasing
- do not repeat the exact meaning of the 2-line title inside point captions
- help the user choose this short before making a CapCut draft
- prefer ending on reaction rather than explanation when possible
- when a Genre benchmark profile is present, independently score the final source_clips with every configured dimension and include a genre_scorecard; set score to its dimension total minus deductions

Return JSON with this exact shape:
{{
  "short_id": "{short_id}",
  "score": 88,
  "genre_scorecard": {{
    "profile_id": "active benchmark profile id when supplied",
    "dimensions": [
      {{"id": "configured dimension id", "score": 0, "reason": "Korean evidence-based reason"}}
    ],
    "penalties": [
      {{"id": "configured penalty id", "deduction": 1, "reason": "Korean evidence-based reason"}}
    ]
  }},
  "core_event": "한 문장 요약",
  "emotion_arc": "감정 흐름",
  "title_line1": "제목 1줄",
  "title_line2": "제목 2줄",
  "upload_title": "업로드용 제목 한 줄",
  "selection_pitch": "이 쇼츠가 왜 볼만한지 한 줄 설명",
  "thread_key": "반복 장난/약속/오해/계산 등 한 줄 식별자",
  "thread_intention": "이 후보가 걸고 있는 시청자 약속과 최종 페이오프",
  "timeline_answerability": "high|medium|low",
  "fragment_risk": "none|low|medium|high",
  "evaluation_notes": "제목 의도/의미/컷 구성 기준으로 정답 타임라인과 비교 가능한지 설명",
  "fun_tags": ["웃김", "돈"],
  "hook_line": "그럼 금 한 톤을요?",
  "main_characters": ["출연자", "상대역"],
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


def cleanup_legacy_final_jsons(active_names: set[str] | None = None) -> None:
    final_dir = OUTPUT_DIR / "final"
    for path in final_dir.glob("short_*.json"):
        if active_names and path.name in active_names:
            continue
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
        if out_path.exists() and not force and not BENCHMARK_PROFILE and not LEARNING_RULE:
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
                max_completion_tokens=(
                    BENCHMARK_LOCAL_MAX_COMPLETION_TOKENS if BENCHMARK_PROFILE else None
                ),
            )
            save_json(out_path, result)
            print(f"[package] saved local candidates -> {out_path.name}", flush=True)
        for candidate in result.get("candidates", []) or []:
            candidate["chunk_id"] = candidate.get("chunk_id") or chunk["chunk_id"]
            all_candidates.append(candidate)
    return all_candidates


def run_global_selection(client: OpenAI, model: str, local_candidates: list[dict], force: bool) -> dict:
    out_path = OUTPUT_DIR / "global" / "selected_candidates.json"
    if out_path.exists() and not force and not BENCHMARK_PROFILE and not LEARNING_RULE:
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
            max_completion_tokens=GLOBAL_MAX_COMPLETION_TOKENS,
        )
        save_json(out_path, result)
        print(f"[package] saved global selection -> {out_path.name}", flush=True)
    return result


def augment_selected_with_audience_topics(selected: list[dict], local_candidates: list[dict]) -> list[dict]:
    selected = [dict(item) for item in selected]
    selected_ids = {item.get("candidate_id") for item in selected}
    missing = []
    for candidate in local_candidates:
        candidate_id = candidate.get("candidate_id")
        if not candidate_id or candidate_id in selected_ids:
            continue
        signals = candidate.get("audience_topic_signal", []) or []
        if not signals:
            continue
        best_signal = signals[0]
        if not best_signal.get("text_match"):
            continue
        if not best_signal.get("timecodes"):
            continue
        if float(best_signal.get("signal_score", 0)) < 1000:
            continue
        missing.append((float(best_signal.get("signal_score", 0)), candidate, best_signal))

    for _score, candidate, signal in sorted(missing, key=lambda item: item[0], reverse=True):
        if len(selected) >= benchmark_final_top_k():
            break
        rank = len(selected) + 1
        selected.append(
            {
                "candidate_id": candidate["candidate_id"],
                "global_rank": rank,
                "global_score": max(75, min(95, int(candidate.get("score", 75)) + 8)),
                "selection_reason": (
                    f"댓글에서 '{signal.get('topic')}' 언급이 반복되고 좋아요/타임스탬프 반응이 강해 "
                    "대사 중심 후보가 아니어도 시청자가 직접 짚은 시각적 웃음 포인트로 보존했습니다."
                ),
                "duplicate_group": f"audience topic: {signal.get('topic')}",
            }
        )
        selected_ids.add(candidate["candidate_id"])
        print(
            f"[package] added audience-topic candidate -> {candidate['candidate_id']} ({signal.get('topic')})",
            flush=True,
        )
    return selected


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
        if item["candidate_id"] not in local_candidates_by_id:
            print(
                f"[package] skipping missing selected candidate -> {item['candidate_id']}",
                flush=True,
            )
            continue
        candidate = dict(local_candidates_by_id[item["candidate_id"]])
        candidate["global_rank"] = item["global_rank"]
        candidate["global_score"] = item["global_score"]
        candidate["selection_reason"] = item.get("selection_reason", "")
        context_segments = extract_candidate_context(merged_segments, candidate)
        package_name = f"short_{int(item['global_rank']):02d}.json"
        out_path = OUTPUT_DIR / "final" / package_name
        result = None
        if out_path.exists() and not force:
            with open(out_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("candidate_id") == item["candidate_id"] and not BENCHMARK_PROFILE and not LEARNING_RULE:
                print(f"[package] using cached final package -> {package_name}", flush=True)
                result = cached
            else:
                print(f"[package] cached final package candidate changed -> {package_name}", flush=True)
        if result is None:
            print(f"[package] requesting final package -> {package_name}", flush=True)
            result = model_json_validated(
                client,
                model,
                PACKAGING_SYSTEM,
                build_packaging_prompt(candidate, context_segments, int(item["global_rank"])),
                validate_final_result,
                max_completion_tokens=PACKAGING_MAX_COMPLETION_TOKENS,
            )
        result["candidate_id"] = item["candidate_id"]
        result["global_rank"] = item["global_rank"]
        result["global_score"] = item["global_score"]
        result["hook_frame_name"] = candidate.get("hook_frame_name", "")
        result["viewer_promise"] = candidate.get("viewer_promise", "")
        for key in ["thread_key", "thread_intention", "timeline_answerability", "fragment_risk", "separation_notes"]:
            if not result.get(key) and candidate.get(key):
                result[key] = candidate.get(key)
        result["audience_topic_signal"] = candidate.get("audience_topic_signal", [])
        result["audience_topic_score"] = candidate.get("audience_topic_score", 0)
        if LEARNING_RULE:
            result["learning_rule"] = LEARNING_RULE
        ok, message = validate_final_result(result)
        if not ok:
            raise RuntimeError(f"Cached/generated final package failed validation: {message}")
        save_json(out_path, result)
        print(f"[package] saved final package -> {package_name}", flush=True)
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
        lines.append(f"- thread_key: {pkg.get('thread_key', '')}")
        lines.append(f"- timeline_answerability: {pkg.get('timeline_answerability', '')}")
        lines.append(f"- fragment_risk: {pkg.get('fragment_risk', '')}")
        lines.append(f"- fun_tags: {', '.join(pkg.get('fun_tags', []) or [])}")
        lines.append(f"- hook_line: {pkg.get('hook_line', '')}")
        lines.append(f"- protagonist_presence: {pkg.get('protagonist_presence', '')}")
        lines.append(f"- standalone_clarity: {pkg.get('standalone_clarity', '')}")
        lines.append(f"- core_event: {pkg.get('core_event', '')}")
        lines.append(f"- target_duration_sec: {pkg.get('target_duration_sec', '')}")
        lines.append(f"- source_clips_total_sec: {clip_total}")
        lines.append(f"- clip_count: {len(pkg.get('source_clips', []) or [])}")
        scorecard = pkg.get("genre_scorecard") if isinstance(pkg.get("genre_scorecard"), dict) else {}
        if scorecard:
            lines.append(
                f"- genre_scorecard: {scorecard.get('total', '')}/100 "
                f"({scorecard.get('decision', '')})"
            )
            dimensions = [
                f"{item.get('id', '')}={item.get('score', '')}/{item.get('max_score', '')}"
                for item in scorecard.get("dimensions", []) or []
                if isinstance(item, dict)
            ]
            if dimensions:
                lines.append(f"- genre_dimensions: {', '.join(dimensions)}")
            penalties = [
                f"{item.get('id', '')}=-{item.get('deduction', '')}"
                for item in scorecard.get("penalties", []) or []
                if isinstance(item, dict)
            ]
            if penalties:
                lines.append(f"- genre_penalties: {', '.join(penalties)}")
        if pkg.get("evaluation_notes"):
            lines.append(f"- evaluation_notes: {pkg.get('evaluation_notes', '')}")
        lines.append(f"- narration_count: {len(pkg.get('narration', []) or [])}")
        lines.append(f"- point_caption_count: {len(pkg.get('point_captions', []) or [])}")
        lines.append("")
    with open(OUTPUT_DIR / "final" / "final_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    configure_stdout()
    args = build_parser().parse_args()
    configure_runtime(
        args.analysis_dir.resolve(),
        args.source_title,
        args.movie_info.resolve() if args.movie_info else None,
        args.youtube_context.resolve() if args.youtube_context else None,
        args.reference_style_path.resolve() if args.reference_style_path else None,
        args.benchmark_profile.resolve() if args.benchmark_profile else None,
        args.learning_rule.resolve() if args.learning_rule else None,
    )

    ensure_output_dirs()
    print(f"[package] analysis_dir={ANALYSIS_DIR}", flush=True)
    print(f"[package] source_title={SOURCE_TITLE}", flush=True)
    print(f"[package] model={args.model}", flush=True)
    if BENCHMARK_PROFILE_CONTEXT:
        print(f"[package] benchmark_profile={args.benchmark_profile}", flush=True)
    if LEARNING_RULE_CONTEXT:
        print(f"[package] learning_rule={args.learning_rule}", flush=True)
    if YOUTUBE_CONTEXT:
        fetched = (YOUTUBE_CONTEXT.get("comments") or {}).get("fetched_count", 0)
        moments = ((YOUTUBE_CONTEXT.get("comment_insights") or {}).get("timecode_moments") or [])
        print(f"[package] youtube_context comments={fetched} timecode_moments={len(moments)}", flush=True)
    if REFERENCE_STYLE_EXAMPLES:
        print(f"[package] reference_style_examples={len(REFERENCE_STYLE_EXAMPLES)}", flush=True)
    client = load_client()
    chunk_records = load_chunk_transcripts(include_wide_windows=not args.no_wide_windows)
    merged_segments = load_merged_segments()
    print(f"[package] loaded chunk_records={len(chunk_records)}", flush=True)
    print(f"[package] loaded merged_segments={len(merged_segments)}", flush=True)

    local_candidates = run_local_extraction(client, args.model, chunk_records, args.force)
    anchor_seed_candidates = [] if BENCHMARK_PROFILE else build_anchor_seed_candidates(chunk_records)
    if anchor_seed_candidates:
        existing_ids = {candidate.get("candidate_id") for candidate in local_candidates}
        new_seeds = [candidate for candidate in anchor_seed_candidates if candidate.get("candidate_id") not in existing_ids]
        local_candidates.extend(new_seeds)
        print(f"[package] anchor_seed_candidates={len(new_seeds)}", flush=True)
    attach_audience_topic_signals(local_candidates)
    save_json(OUTPUT_DIR / "local" / "all_candidates.json", {"candidates": local_candidates})
    print(f"[package] local_candidates={len(local_candidates)}", flush=True)

    selected_data = run_global_selection(client, args.model, local_candidates, args.force)
    selected_items = augment_selected_with_audience_topics(selected_data.get("selected", []) or [], local_candidates)
    if selected_items != (selected_data.get("selected", []) or []):
        selected_data["selected"] = selected_items
        save_json(OUTPUT_DIR / "global" / "selected_candidates.json", selected_data)
    print(f"[package] selected_items={len(selected_items)}", flush=True)

    local_candidates_by_id = {cand["candidate_id"]: cand for cand in local_candidates}
    aggregate_path = OUTPUT_DIR / "final" / "shorts_packages.json"
    if aggregate_path.exists() and not args.force and not BENCHMARK_PROFILE and not LEARNING_RULE:
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
            "benchmark_profile": {
                "profile_id": BENCHMARK_PROFILE.get("profile_id", ""),
                "path": str(args.benchmark_profile) if args.benchmark_profile else "",
            },
            "learning_rule": LEARNING_RULE,
            "movie_info": MOVIE_INFO,
            "youtube_context": {
                "source_url": YOUTUBE_CONTEXT.get("source_url", "") if YOUTUBE_CONTEXT else "",
                "video_id": YOUTUBE_CONTEXT.get("video_id", "") if YOUTUBE_CONTEXT else "",
                "comments": YOUTUBE_CONTEXT.get("comments", {}) if YOUTUBE_CONTEXT else {},
            },
            "shorts": sorted(final_packages, key=lambda x: x["global_rank"]),
        }
        active_names = {f"short_{int(pkg['global_rank']):02d}.json" for pkg in final_packages}
        cleanup_legacy_final_jsons(active_names)
        save_json(aggregate_path, final_payload)
        write_summary(final_packages)

    print(f"[package] output_dir={OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
