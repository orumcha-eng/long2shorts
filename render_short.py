from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
import cv2
import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from pymediainfo import MediaInfo

from env_loader import load_project_env
from usage_ledger import append_event


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FONT_PATH = Path(r"C:\Windows\Fonts\Pretendard-ExtraBold.ttf")
FALLBACK_FONT_PATH = Path(r"C:\Windows\Fonts\malgunbd.ttf")
CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920
# Brand template reconstructed from the published "이것도봐바" Shorts.  The
# footage stays in a compact player card between a white editorial header and
# the familiar player controls instead of being surrounded by a generic matte.
VIDEO_VISIBLE_TOP = 412
VIDEO_VISIBLE_BOTTOM = 1488
VIDEO_VISIBLE_HEIGHT = VIDEO_VISIBLE_BOTTOM - VIDEO_VISIBLE_TOP
FPS = 30
VIDEO_FILE_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi"}
DEFAULT_TEMPLATE_IMAGE_CANDIDATES = [
    BASE_DIR / "templates" / "shorts_template.png",
    BASE_DIR / "short_templet.png",
    BASE_DIR / "short_template.png",
]
BRAND_CHANNEL_NAME = os.environ.get("SHORTS_BRAND_NAME", "이것도봐바").strip() or "이것도봐바"
BRAND_HANDLE = os.environ.get("SHORTS_BRAND_HANDLE", "@이것도재밌다").strip() or "@이것도재밌다"
# A compact one-line headline sits inside the rounded white title chip.
TITLE_SINGLE_LINE_BOX = (112, 260, 968, 370)
SOURCE_CREDIT_BOX = (92, 1518, 740, 1594)
TEMPLATE_BACKGROUND = (250, 250, 247, 255)
TEMPLATE_INK = (18, 18, 18, 255)
TEMPLATE_GREEN = (20, 228, 108, 255)
SFX_DIR = BASE_DIR / "필수 효과음 100종"
SFX_FILE_HINTS = {
    "entrance": ("01-9 띠링", "01-8 뽕", "01-7 띵"),
    "transition": ("06-1 훅", "06-2 훅", "06-3 훅"),
    "surprise": ("29 띠요옹", "05-6 띠요옹"),
    "impact": ("18 뚜둥탁", "12-2 충격적인 등장", "12-1 극적인 순간"),
    "correct": ("02-6 QnA 띠리링", "02-7 QnA 정답"),
    "wrong": ("02-9 틀렸을 때", "02-7 QnA 삐익"),
    "awkward": ("30 어이없는 상황", "05-3 오리 꽥"),
    "applause": ("03-11 박수+환호", "25-1 응원 소리"),
}
COLOR_PALETTES = (
    {"title1": (18, 18, 18), "title2": (20, 228, 108), "caption": (255, 238, 92), "source": (204, 204, 204)},
)
# Dialogue subtitles are deliberately independent from any text baked into the
# source video.  A side-by-side reframe can crop that baked text, whereas this
# track remains centred, legible, and consistently branded.
DIALOGUE_CAPTION_MAX_CHARS = 19

# The visible footage area is deliberately shorter than the whole 9:16 canvas
# because the channel header and source credit sit above and below it.
VIDEO_ASPECT_RATIO = CANVAS_WIDTH / VIDEO_VISIBLE_HEIGHT
SPLIT_PANEL_WIDTH = CANVAS_WIDTH // 2
SPLIT_PANEL_ASPECT_RATIO = SPLIT_PANEL_WIDTH / VIDEO_VISIBLE_HEIGHT
FACE_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
# Keep a single feminine-presenting Korean editor voice across every Short.
# Changing voices between packages makes the channel feel automated instead of
# edited by one consistent narrator.
NARRATION_VOICE = "shimmer"
NARRATION_SPEED = 1.04
NARRATION_INSTRUCTIONS = (
    "Speak natural Korean like one restrained, warm female variety-show editor. "
    "Use a conversational creator tone with clear diction and a light reaction, never an announcer, "
    "synthetic assistant, or theatrical character voice."
)
# A narration is a brief editor comment, not the programme dialogue.  Keep its
# label safely below any source caption which commonly sits at the top of a
# broadcast frame.
NARRATION_LABEL_X = 70
NARRATION_LABEL_Y = 448
NARRATION_LABEL_WIDTH = CANVAS_WIDTH - (NARRATION_LABEL_X * 2)
NARRATION_LABEL_HEIGHT = 92
# Reframing should feel like one camera operator, not a new zoom every beat.
# A cut may change framing, but within a continuous cut the crop only moves
# when the subject has genuinely moved out of the safe area.
TRACK_SEGMENT_SEC = 2.4
TRACK_DEAD_ZONE = 0.055
TRACK_MAX_STEP = 0.055


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def has_audio_track(path: Path) -> bool:
    media = MediaInfo.parse(str(path))
    return any(track.track_type == "Audio" for track in media.tracks)


def positive_float(value: object, fallback: float) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return fallback


def approved_narration_items(package: dict) -> list[dict]:
    """Keep narration as a high-bar editorial device, never a generic filler line."""
    generic_phrases = (
        "독특하네", "재밌네", "웃기네", "신기하네", "대박", "난리", "이 집", "감성",
        "그냥 봐도", "역시", "레전드",
    )
    approved: list[dict] = []
    for item in package.get("narration", []) or []:
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get("text") or "").split()).rstrip(".!?")
        visible_length = len("".join(text.split()))
        if not 8 <= visible_length <= 24 or any(phrase in text for phrase in generic_phrases):
            continue
        start = positive_float(item.get("target_start"), 0.0)
        approved.append({**item, "text": text, "target_start": start})
        # One very good line is the maximum.  More narration turns a Short
        # into a summary instead of an edited reaction story.
        break
    return approved


def palette_for_package(package: dict) -> dict[str, tuple[int, int, int]]:
    key = str(package.get("short_id") or package.get("candidate_id") or "default")
    return COLOR_PALETTES[sum(ord(char) for char in key) % len(COLOR_PALETTES)]


def ass_bgr_color(rgb: tuple[int, int, int]) -> str:
    red, green, blue = rgb
    return f"&H00{blue:02X}{green:02X}{red:02X}"


def format_ass_time(seconds: float) -> str:
    total_centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(total_centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole_seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def escape_ass_text(value: str) -> str:
    return value.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")


def animated_ass_text(value: str) -> str:
    text = escape_ass_text(value)
    return r"{\fad(80,120)\fscx86\fscy86\t(0,160,\fscx114\fscy114)\t(160,320,\fscx100\fscy100)}" + text


def escape_filter_path(path: Path) -> str:
    return path.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")


def resolve_font_path() -> Path:
    if DEFAULT_FONT_PATH.exists():
        return DEFAULT_FONT_PATH
    if FALLBACK_FONT_PATH.exists():
        return FALLBACK_FONT_PATH
    raise RuntimeError("No supported Korean subtitle font was found.")


def split_display_lines(value: str, max_chars: int) -> list[str]:
    text = " ".join(value.split())
    if not text:
        return []
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    if len(lines) == 1 and len(lines[0]) > max_chars:
        return [lines[0][index : index + max_chars] for index in range(0, len(lines[0]), max_chars)]
    return lines[:2]


def validated_title_line(value: str, *, field: str, max_visible_chars: int = 18) -> str:
    """Return a render-safe headline or stop before it is silently damaged.

    A title is editorial content, not disposable UI text. The prior renderer
    shortened long lines with ``...``; that created broken headlines such as
    ``노...`` even when the package itself looked valid.
    """
    text = " ".join(str(value or "").split())
    visible_length = len("".join(text.split()))
    if not text:
        raise RuntimeError(f"{field} is empty.")
    if "..." in text or "…" in text:
        raise RuntimeError(f"{field} contains an ellipsis and must be rewritten, not truncated.")
    if visible_length > max_visible_chars:
        raise RuntimeError(
            f"{field} has {visible_length} visible characters; maximum is {max_visible_chars}. Rewrite the title line."
        )
    return text


def infer_analysis_dir(package_path: Path) -> Path | None:
    for parent in package_path.parents:
        if (parent / "youtube_context.json").exists():
            return parent
    return None


def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def first_text(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def infer_channel_name(package_path: Path, package: dict, explicit_name: str = "") -> str:
    if explicit_name.strip():
        return explicit_name.strip()
    youtube_context = package.get("youtube_context", {}) if isinstance(package.get("youtube_context"), dict) else {}
    metadata = youtube_context.get("metadata", {}) if isinstance(youtube_context.get("metadata"), dict) else {}
    channel_name = first_text(
        package.get("source_label"),
        package.get("source_credit"),
        package.get("channel_title"),
        metadata.get("channel_title"),
    )
    if channel_name:
        return channel_name
    analysis_dir = infer_analysis_dir(package_path)
    if analysis_dir:
        context = load_json_if_exists(analysis_dir / "youtube_context.json")
        context_metadata = context.get("metadata", {}) if isinstance(context.get("metadata"), dict) else {}
        channel_name = first_text(context_metadata.get("channel_title"))
        if channel_name:
            return channel_name
    return first_text(package.get("source_title"), package.get("movie_title"))


def resolve_template_image() -> Path | None:
    for candidate in DEFAULT_TEMPLATE_IMAGE_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()
    return None


def prepare_template_layer(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, CANVAS_WIDTH, VIDEO_VISIBLE_TOP), fill=TEMPLATE_BACKGROUND)

    # Reuse the established player-control artwork from the old template;
    # only its former channel header is deliberately discarded.
    template = resolve_template_image()
    if template:
        with Image.open(template) as source:
            footer = source.convert("RGBA").resize((CANVAS_WIDTH, CANVAS_HEIGHT), Image.Resampling.LANCZOS)
            image.alpha_composite(footer.crop((0, VIDEO_VISIBLE_BOTTOM, CANVAS_WIDTH, CANVAS_HEIGHT)), (0, VIDEO_VISIBLE_BOTTOM))
    else:
        draw.rectangle((0, VIDEO_VISIBLE_BOTTOM, CANVAS_WIDTH, CANVAS_HEIGHT), fill=TEMPLATE_BACKGROUND)

    font_path = resolve_font_path()
    regular_path = Path(r"C:\Windows\Fonts\malgun.ttf")
    regular = regular_path if regular_path.exists() else font_path
    # Top-left channel mark.
    draw.rounded_rectangle((46, 42, 250, 164), radius=25, fill=(255, 255, 255, 255), outline=TEMPLATE_INK, width=5)
    draw.ellipse((69, 70, 137, 138), fill=TEMPLATE_GREEN, outline=TEMPLATE_INK, width=4)
    draw.polygon(((95, 84), (95, 124), (123, 104)), fill=(255, 255, 255, 255))
    mark_font = ImageFont.truetype(str(font_path), 37)
    draw.multiline_text((151, 57), "이것도\n봐바", font=mark_font, fill=TEMPLATE_INK, spacing=-7)
    title_font = ImageFont.truetype(str(font_path), 52)
    handle_font = ImageFont.truetype(str(regular), 28)
    draw.text((294, 69), BRAND_CHANNEL_NAME, font=title_font, fill=TEMPLATE_INK)
    draw.text((296, 130), f"{BRAND_HANDLE}  |  SHORTS", font=handle_font, fill=(70, 70, 70, 255))
    draw.line((294, 184, 1008, 184), fill=(222, 222, 216, 255), width=3)
    draw.line((294, 184, 470, 184), fill=TEMPLATE_GREEN, width=5)
    draw.rounded_rectangle((80, 246, 1000, 383), radius=22, fill=TEMPLATE_BACKGROUND, outline=(215, 215, 210, 255), width=2)
    image.save(output_path)
    return output_path


def fit_font(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font_path: Path,
    max_width: int,
    start_size: int,
    min_size: int = 36,
    stroke_width: int = 10,
) -> ImageFont.FreeTypeFont:
    for size in range(start_size, min_size - 1, -2):
        font = ImageFont.truetype(str(font_path), size)
        if all(draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)[2] <= max_width for line in lines):
            return font
    return ImageFont.truetype(str(font_path), min_size)


def draw_boxed_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    *,
    font_path: Path,
    font_size: int,
    min_font_size: int,
    fill: tuple[int, int, int, int],
    align: str = "center",
    stroke_width: int = 0,
    stroke_fill: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> None:
    if not text:
        return
    width = box[2] - box[0]
    height = box[3] - box[1]
    font = fit_font(draw, [text], font_path, width, font_size, min_font_size, stroke_width=stroke_width)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    if align == "left":
        x = box[0]
    elif align == "right":
        x = box[2] - text_width
    else:
        x = box[0] + max(0, (width - text_width) // 2)
    y = box[1] + max(0, (height - text_height) // 2)
    draw.text(
        (x - bbox[0], y - bbox[1]),
        text,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )


def draw_highlighted_headline(
    draw: ImageDraw.ImageDraw,
    text: str,
    highlight: str,
    box: tuple[int, int, int, int],
    *,
    font_path: Path,
    font_size: int,
    min_font_size: int,
    fill: tuple[int, int, int, int],
    highlight_fill: tuple[int, int, int, int],
    align: str = "left",
    stroke_width: int = 0,
    stroke_fill: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> None:
    """Draw one headline with exactly one editorially chosen coloured phrase."""
    highlight = " ".join(str(highlight or "").split())
    if not highlight or highlight not in text:
        draw_boxed_text(
            draw, text, box, font_path=font_path, font_size=font_size,
            min_font_size=min_font_size, fill=fill, align=align,
            stroke_width=stroke_width, stroke_fill=stroke_fill,
        )
        return
    font = fit_font(
        draw, [text], font_path, box[2] - box[0], font_size,
        min_font_size, stroke_width=stroke_width,
    )
    before, after = text.split(highlight, 1)
    full_bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_height = full_bbox[3] - full_bbox[1]
    y = box[1] + max(0, ((box[3] - box[1]) - text_height) // 2)
    text_width = full_bbox[2] - full_bbox[0]
    x = box[0] if align == "left" else box[0] + max(0, ((box[2] - box[0]) - text_width) // 2)
    for segment, color in ((before, fill), (highlight, highlight_fill), (after, fill)):
        if not segment:
            continue
        segment_bbox = draw.textbbox((0, 0), segment, font=font, stroke_width=stroke_width)
        draw.text(
            (x - segment_bbox[0], y - segment_bbox[1]),
            segment,
            font=font,
            fill=color,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        x += draw.textlength(segment, font=font)


def build_text_overlay(package: dict, package_path: Path, output_path: Path, channel_name: str = "") -> Path:
    font_path = resolve_font_path()
    image = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headline_raw = first_text(
        package.get("on_video_title"),
        " ".join(
            value for value in (
                str(package.get("title_line1") or "").strip(),
                str(package.get("title_line2") or "").strip(),
            )
            if value
        ),
    )
    headline = validated_title_line(headline_raw, field="on_video_title", max_visible_chars=30)
    channel_name = channel_name.strip() or infer_channel_name(package_path, package)
    palette = palette_for_package(package)

    draw_highlighted_headline(
        draw,
        headline,
        package.get("title_highlight") or "",
        TITLE_SINGLE_LINE_BOX,
        font_path=font_path,
        font_size=58,
        min_font_size=38,
        fill=(*palette["title1"], 255),
        highlight_fill=(*palette["title2"], 255),
        align="left",
        stroke_width=0,
    )
    # The visible source is editorially useful and makes the re-cut's origin
    # unambiguous.  A package may deliberately omit it only for owned footage.
    source_credit = " ".join(str(package.get("source_attribution") or "").split())
    if source_credit:
        draw_boxed_text(
            draw,
            source_credit,
            SOURCE_CREDIT_BOX,
            font_path=font_path,
            font_size=36,
            min_font_size=26,
            fill=(*palette["title1"], 255),
            align="left",
        )
    image.save(output_path)
    return output_path


def build_caption_ass(package: dict, output_path: Path) -> Path:
    font_name = "Pretendard ExtraBold" if DEFAULT_FONT_PATH.exists() else "Malgun Gothic"
    palette = palette_for_package(package)
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {CANVAS_WIDTH}",
        f"PlayResY: {CANVAS_HEIGHT}",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        f"Style: Point,{font_name},82,{ass_bgr_color(palette['caption'])},&H000000FF,&H00101010,&H9A000000,-1,0,0,0,100,100,0,0,1,9,3,2,70,70,920,1",
        # BorderStyle=3 makes a stable opaque box in libass.  This masks
        # clipped programme subtitles underneath and keeps the two speakers
        # readable even over a busy split screen.
        f"Style: SpeakerWhite,{font_name},60,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,3,14,0,2,86,86,650,1",
        f"Style: SpeakerGreen,{font_name},60,&H006CE414,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,3,14,0,2,86,86,650,1",
        "Style: Narration,Pretendard ExtraBold,48,&H00FFFFFF,&H000000FF,&H00251A38,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,8,70,70,0,1",
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    dialogue_captions = [
        item for item in (package.get("dialogue_captions", []) or []) if isinstance(item, dict)
    ]
    # When dialogue captions are supplied, they replace sparse editor labels.
    # Running both stacks made the visual language noisy and can obscure the
    # speaker's own line.
    for caption in dialogue_captions:
        start = positive_float(caption.get("target_start"), 0.0)
        end = positive_float(caption.get("target_end"), start + 2.0)
        raw_text = str(caption.get("text") or "").strip()
        display_lines = split_display_lines(raw_text, DIALOGUE_CAPTION_MAX_CHARS)
        # Pass real line breaks to the ASS escaper.  Passing a literal ``\N``
        # first caused it to be escaped once more and could show a stray
        # backslash instead of a normal two-line subtitle.
        text = animated_ass_text("\n".join(display_lines)) if display_lines else ""
        speaker = str(caption.get("speaker") or caption.get("speaker_side") or "").strip().casefold()
        style = "SpeakerGreen" if speaker in {"b", "right", "speaker_b", "speaker_b_right", "green", "연두"} else "SpeakerWhite"
        if text and end > start:
            lines.append(f"Dialogue: 0,{format_ass_time(start)},{format_ass_time(end)},{style},,0,0,0,,{text}")

    for caption in ([] if dialogue_captions else (package.get("point_captions", []) or [])):
        if not isinstance(caption, dict):
            continue
        start = positive_float(caption.get("target_start"), 0.0)
        end = positive_float(caption.get("target_end"), start + 2.0)
        text = animated_ass_text(str(caption.get("text") or "").strip())
        if text and end > start:
            lines.append(f"Dialogue: 0,{format_ass_time(start)},{format_ass_time(end)},Point,,0,0,0,,{text}")
    for narration in approved_narration_items(package):
        start = positive_float(narration.get("target_start"), 0.0)
        text = animated_ass_text(str(narration.get("text") or "").strip())
        # The voice clip usually runs 1 to 2 seconds.  Keep this caption long
        # enough to read without competing with the larger reaction caption.
        end = positive_float(narration.get("target_end"), start + 2.1)
        if text and end > start:
            lines.append(
                f"Dialogue: 1,{format_ass_time(start)},{format_ass_time(end)},Narration,,0,0,{NARRATION_LABEL_Y + 15},,{text}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def default_output_path(package_path: Path) -> Path:
    analysis_dir = infer_analysis_dir(package_path) or package_path.parent.parent.parent
    batch_id = ""
    for parent in package_path.parents:
        if parent.parent.name == "batches":
            batch_id = parent.name
            break
    output_dir = analysis_dir / "productions"
    if batch_id:
        output_dir = output_dir / "batches" / batch_id
    return output_dir / f"{package_path.stem}.mp4"


def resolve_sound_effect(cue: str) -> Path | None:
    if not SFX_DIR.exists():
        return None
    files = sorted(path for path in SFX_DIR.iterdir() if path.is_file() and path.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac", ".ogg"})
    for hint in SFX_FILE_HINTS.get(cue, ()):
        match = next((path for path in files if hint.casefold() in path.name.casefold()), None)
        if match:
            return match
    return None


def resolved_sound_effects(package: dict) -> list[dict]:
    resolved: list[dict] = []
    for item in package.get("sound_effects", []) or []:
        if not isinstance(item, dict):
            continue
        cue = str(item.get("cue") or "").strip().lower()
        path = resolve_sound_effect(cue)
        if not path:
            continue
        try:
            target_start = max(0.0, float(item.get("target_start") or 0.0))
        except (TypeError, ValueError):
            target_start = 0.0
        try:
            volume = min(0.45, max(0.10, float(item.get("volume", 0.24))))
        except (TypeError, ValueError):
            volume = 0.24
        resolved.append({"cue": cue, "path": path, "target_start": target_start, "volume": volume})
        if len(resolved) >= 3:
            break
    return resolved


def media_duration_sec(path: Path) -> float:
    info = MediaInfo.parse(str(path))
    for track in info.tracks:
        if track.track_type == "Audio" and track.duration:
            return max(0.0, float(track.duration) / 1000.0)
    return 0.0


def build_narration_tracks(package: dict, output_path: Path) -> list[dict]:
    """Create concise Korean editor narration clips and cache them per text."""
    items = approved_narration_items(package)
    if not items:
        return []
    load_project_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Narration requested but OPENAI_API_KEY is missing from .env.")
    client = OpenAI(api_key=api_key)
    tracks: list[dict] = []
    for index, item in enumerate(items, start=1):
        text = " ".join(str(item.get("text") or "").split())[:80]
        if not text:
            continue
        start = positive_float(item.get("target_start"), 0.0)
        cache_key = f"{NARRATION_VOICE}|{NARRATION_SPEED}|{text}"
        digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:10]
        audio_path = output_path.with_name(f"{output_path.stem}.narration_{index}_{digest}.mp3")
        if not audio_path.exists() or audio_path.stat().st_size < 1024:
            with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice=NARRATION_VOICE,
                input=text,
                instructions=NARRATION_INSTRUCTIONS,
                response_format="mp3",
                speed=NARRATION_SPEED,
            ) as response:
                response.stream_to_file(audio_path)
            append_event(
                stage="narration_tts",
                model="gpt-4o-mini-tts",
                extra={"character_count": len(text), "voice": NARRATION_VOICE},
            )
        duration = media_duration_sec(audio_path)
        if duration > 0.05:
            caption_end = positive_float(item.get("target_end"), start + duration)
            tracks.append({
                "path": audio_path,
                "target_start": start,
                "target_end": max(start + duration, caption_end),
                "duration": duration,
                "text": text,
            })
    return tracks


def even_number(value: float, minimum: int = 2) -> int:
    return max(minimum, int(round(value)) // 2 * 2)


def face_cascade_runtime_path() -> Path | None:
    """Use an ASCII path because OpenCV on Windows cannot open some Hangul paths."""
    if not FACE_CASCADE_PATH.exists():
        return None
    destination = Path(tempfile.gettempdir()) / "long2shorts_haarcascade_frontalface_default.xml"
    try:
        if not destination.exists() or destination.stat().st_size != FACE_CASCADE_PATH.stat().st_size:
            shutil.copyfile(FACE_CASCADE_PATH, destination)
    except OSError:
        return None
    return destination


def clamp_crop_x(center_x: float, crop_width: int, source_width: int) -> int:
    maximum = max(0, source_width - crop_width)
    return max(0, min(maximum, int(round(center_x - crop_width / 2))))


def detect_faces_at(cap: cv2.VideoCapture, cascade: cv2.CascadeClassifier, time_sec: float) -> list[tuple[float, float, float]]:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_sec) * 1000)
    ok, frame = cap.read()
    if not ok or frame is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(32, 32))
    return sorted(
        ((x + width / 2, y + height / 2, width * height) for x, y, width, height in faces),
        key=lambda face: face[2],
        reverse=True,
    )


def smooth_centers(samples: list[tuple[float, float]], fallback: float) -> list[dict]:
    """Turn intermittent face detections into a calm, bounded camera path."""
    if not samples:
        return [{"time": 0.0, "center_x": fallback}]
    smoothed: list[dict] = []
    prior = fallback
    for index, (time_sec, center) in enumerate(samples):
        nearby = [value for _, value in samples[max(0, index - 1) : index + 2]]
        target = float(np.median(nearby))
        # Ignore small detector wobble and never let one frame whip the crop
        # across the screen.  The crop must feel held, not constantly zoomed.
        if abs(target - prior) < TRACK_DEAD_ZONE:
            target = prior
        target = max(prior - TRACK_MAX_STEP, min(prior + TRACK_MAX_STEP, target))
        prior = max(0.0, min(1.0, target))
        smoothed.append({"time": round(time_sec, 3), "center_x": round(prior, 4)})
    return smoothed


def interpolate_center(keyframes: list[dict], time_sec: float, key: str, fallback: float) -> float:
    """Linearly interpolate a normalized tracked centre at a source time."""
    usable = [frame for frame in keyframes if isinstance(frame, dict) and key in frame]
    if not usable:
        return fallback
    if time_sec <= float(usable[0].get("time", 0.0)):
        return float(usable[0].get(key, fallback))
    if time_sec >= float(usable[-1].get("time", 0.0)):
        return float(usable[-1].get(key, fallback))
    for left, right in zip(usable, usable[1:]):
        left_time, right_time = float(left.get("time", 0.0)), float(right.get("time", 0.0))
        if left_time <= time_sec <= right_time:
            amount = 0.0 if right_time <= left_time else (time_sec - left_time) / (right_time - left_time)
            return float(left.get(key, fallback)) + (float(right.get(key, fallback)) - float(left.get(key, fallback))) * amount
    return fallback


def plan_clip_reframes(source_video: Path, clips: list[dict]) -> list[dict]:
    """Plan a real time-varying crop path for each source cut.

    Face detection is sampled roughly twice a second, then smoothed.  The
    renderer turns those points into short contiguous crop segments.  This is
    intentionally more honest than the previous 'tracked' label: that version
    chose one crop per cut and did not follow a moving person at all.
    """
    cap = cv2.VideoCapture(str(source_video))
    source_width = even_number(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 2)
    source_height = even_number(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 2)
    cascade_path = face_cascade_runtime_path()
    if not cap.isOpened() or source_width <= 2 or source_height <= 2 or not cascade_path:
        cap.release()
        return [{"mode": "focus", "center_x": 0.5, "source_width": source_width, "source_height": source_height} for _ in clips]
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        cap.release()
        return [{"mode": "focus", "center_x": 0.5, "source_width": source_width, "source_height": source_height} for _ in clips]
    plans: list[dict] = []
    full_crop_width = min(source_width, even_number(source_height * VIDEO_ASPECT_RATIO))
    panel_crop_width = min(source_width, even_number(source_height * SPLIT_PANEL_ASPECT_RATIO))
    try:
        for clip in clips:
            start = positive_float(clip.get("source_start"), 0.0)
            end = positive_float(clip.get("source_end"), start + 0.5)
            duration = max(0.25, end - start)
            # A reviewed package may name the speaker/subject that must stay
            # in frame.  Use this explicit composition before face detection:
            # a detector can notice only the listener in a two-person shot,
            # which otherwise leaves the actual speaker half out of frame.
            manual_focus = clip.get("reframe_focus_x")
            try:
                manual_focus = float(manual_focus)
            except (TypeError, ValueError):
                manual_focus = None
            if manual_focus is not None:
                manual_focus = max(0.0, min(1.0, manual_focus))
                plans.append({
                    "mode": "focus",
                    "center_x": manual_focus,
                    "keyframes": [{"time": start, "center_x": manual_focus}],
                    "source_width": source_width,
                    "source_height": source_height,
                })
                continue
            sample_times = list(np.arange(start + 0.12, max(start + 0.13, end - 0.10), 0.55))
            if not sample_times or sample_times[-1] < end - 0.12:
                sample_times.append(max(start + 0.12, end - 0.12))
            pairs: list[tuple[float, float, float]] = []
            singles: list[tuple[float, float]] = []
            for sample_time in sample_times:
                faces = detect_faces_at(cap, cascade, sample_time)
                if len(faces) >= 2:
                    pair = sorted(faces[:2], key=lambda face: face[0])
                    pairs.append((sample_time, pair[0][0] / source_width, pair[1][0] / source_width))
                elif faces:
                    singles.append((sample_time, faces[0][0] / source_width))
            if pairs and len(pairs) >= max(2, len(sample_times) // 3):
                spans = [(right - left) * source_width for _, left, right in pairs]
                span = float(np.median(spans))
                # If both faces safely fit in one vertical crop, preserve their
                # shared reaction.  Otherwise use two panels; never centre-crop
                # a dialogue so that only an arm or half a face survives.
                if span <= full_crop_width * 0.62:
                    centers = [(time_sec, (left + right) / 2) for time_sec, left, right in pairs]
                    plans.append({
                        "mode": "focus",
                        "center_x": float(np.median([center for _, center in centers])),
                        "keyframes": smooth_centers(centers, 0.5),
                        "source_width": source_width,
                        "source_height": source_height,
                    })
                else:
                    left_path = smooth_centers([(time_sec, left) for time_sec, left, _ in pairs], 0.30)
                    right_path = smooth_centers([(time_sec, right) for time_sec, _, right in pairs], 0.70)
                    keyframes = [
                        {
                            "time": point["time"],
                            "left_x": point["center_x"],
                            "right_x": interpolate_center(right_path, float(point["time"]), "center_x", 0.70),
                        }
                        for point in left_path
                    ]
                    plans.append({
                        "mode": "split",
                        "left_x": float(np.median([left for _, left, _ in pairs])),
                        "right_x": float(np.median([right for _, _, right in pairs])),
                        "keyframes": keyframes,
                        "source_width": source_width,
                        "source_height": source_height,
                        "panel_crop_width": panel_crop_width,
                    })
            elif singles:
                keyframes = smooth_centers(singles, 0.5)
                plans.append({
                    "mode": "focus",
                    "center_x": float(np.median([center for _, center in singles])),
                    "keyframes": keyframes,
                    "source_width": source_width,
                    "source_height": source_height,
                })
            else:
                plans.append({"mode": "focus", "center_x": 0.5, "keyframes": [], "source_width": source_width, "source_height": source_height})
    finally:
        cap.release()
    return plans


def build_filter(
    clips: list[dict],
    include_audio: bool,
    captions_path: Path | None,
    sound_effects: list[dict],
    reframes: list[dict],
    narration_tracks: list[dict],
) -> str:
    parts: list[str] = []
    concat_inputs: list[str] = []
    concat_count = 0
    for index, clip in enumerate(clips):
        start = positive_float(clip.get("source_start"), -1.0)
        end = positive_float(clip.get("source_end"), -1.0)
        if end <= start or start < 0:
            raise RuntimeError(f"Invalid source clip timing at index {index}.")
        reframe = reframes[index] if index < len(reframes) else {"mode": "focus", "center_x": 0.5}
        source_width = max(2, int(reframe.get("source_width") or CANVAS_WIDTH))
        source_height = max(2, int(reframe.get("source_height") or VIDEO_VISIBLE_HEIGHT))
        full_crop_width = min(source_width, even_number(source_height * VIDEO_ASPECT_RATIO))
        # Hold the composition for at least 2.4 seconds. This still follows a
        # real subject change, but avoids the nauseating micro-pan caused by
        # updating the crop at every short detector sample.
        segment_edges = list(np.arange(start, end, TRACK_SEGMENT_SEC)) + [end]
        if len(segment_edges) < 2:
            segment_edges = [start, end]
        held_focus_center: float | None = None
        held_left_center: float | None = None
        held_right_center: float | None = None
        for segment_index, (segment_start, segment_end) in enumerate(zip(segment_edges, segment_edges[1:])):
            if segment_end - segment_start < 0.04:
                continue
            label = f"{index}_{segment_index}"
            midpoint = (segment_start + segment_end) / 2
            if reframe.get("mode") == "split" and source_width / source_height > VIDEO_ASPECT_RATIO:
                panel_crop_width = min(source_width, int(reframe.get("panel_crop_width") or even_number(source_height * SPLIT_PANEL_ASPECT_RATIO)))
                left_center = interpolate_center(reframe.get("keyframes", []), midpoint, "left_x", float(reframe.get("left_x") or 0.30))
                right_center = interpolate_center(reframe.get("keyframes", []), midpoint, "right_x", float(reframe.get("right_x") or 0.70))
                if held_left_center is not None and abs(left_center - held_left_center) < TRACK_DEAD_ZONE:
                    left_center = held_left_center
                if held_right_center is not None and abs(right_center - held_right_center) < TRACK_DEAD_ZONE:
                    right_center = held_right_center
                held_left_center, held_right_center = left_center, right_center
                left_x = clamp_crop_x(left_center * source_width, panel_crop_width, source_width)
                right_x = clamp_crop_x(right_center * source_width, panel_crop_width, source_width)
                parts.append(f"[0:v]trim=start={segment_start:.3f}:end={segment_end:.3f},setpts=PTS-STARTPTS,split=2[vl{label}][vr{label}]")
                parts.append(f"[vl{label}]crop={panel_crop_width}:{source_height}:{left_x}:0,scale={SPLIT_PANEL_WIDTH}:{VIDEO_VISIBLE_HEIGHT}:flags=lanczos[left{label}]")
                parts.append(f"[vr{label}]crop={panel_crop_width}:{source_height}:{right_x}:0,scale={SPLIT_PANEL_WIDTH}:{VIDEO_VISIBLE_HEIGHT}:flags=lanczos[right{label}]")
                parts.append(f"[left{label}][right{label}]hstack=inputs=2,pad={CANVAS_WIDTH}:{CANVAS_HEIGHT}:0:{VIDEO_VISIBLE_TOP}:color=black,setsar=1,fps={FPS}[v{label}]")
            elif source_width / source_height > VIDEO_ASPECT_RATIO:
                center = interpolate_center(reframe.get("keyframes", []), midpoint, "center_x", float(reframe.get("center_x") or 0.5))
                if held_focus_center is not None and abs(center - held_focus_center) < TRACK_DEAD_ZONE:
                    center = held_focus_center
                held_focus_center = center
                crop_x = clamp_crop_x(center * source_width, full_crop_width, source_width)
                parts.append(f"[0:v]trim=start={segment_start:.3f}:end={segment_end:.3f},setpts=PTS-STARTPTS,crop={full_crop_width}:{source_height}:{crop_x}:0,scale={CANVAS_WIDTH}:{VIDEO_VISIBLE_HEIGHT}:flags=lanczos,pad={CANVAS_WIDTH}:{CANVAS_HEIGHT}:0:{VIDEO_VISIBLE_TOP}:color=black,setsar=1,fps={FPS}[v{label}]")
            else:
                parts.append(f"[0:v]trim=start={segment_start:.3f}:end={segment_end:.3f},setpts=PTS-STARTPTS,scale={CANVAS_WIDTH}:{VIDEO_VISIBLE_HEIGHT}:force_original_aspect_ratio=increase,crop={CANVAS_WIDTH}:{VIDEO_VISIBLE_HEIGHT},pad={CANVAS_WIDTH}:{CANVAS_HEIGHT}:0:{VIDEO_VISIBLE_TOP}:color=black,setsar=1,fps={FPS}[v{label}]")
            concat_inputs.append(f"[v{label}]")
            if include_audio:
                parts.append(f"[0:a]atrim=start={segment_start:.3f}:end={segment_end:.3f},asetpts=PTS-STARTPTS[a{label}]")
                concat_inputs.append(f"[a{label}]")
            concat_count += 1
    if include_audio:
        parts.append(f"{''.join(concat_inputs)}concat=n={concat_count}:v=1:a=1[vconcat][aconcat]")
        audio_label = "aconcat"
        for index, narration in enumerate(narration_tracks):
            start = max(0.0, float(narration["target_start"]))
            end = start + max(0.1, float(narration["duration"]))
            duck_label = f"duck{index}"
            parts.append(f"[{audio_label}]volume=enable='between(t,{start:.3f},{end:.3f})':volume=0.28[{duck_label}]")
            audio_label = duck_label
        mix_labels = [f"[{audio_label}]"]
        if sound_effects:
            for index, effect in enumerate(sound_effects):
                delay_ms = max(0, round(float(effect["target_start"]) * 1000))
                input_index = 3 + index
                label = f"sfx{index}"
                parts.append(f"[{input_index}:a]adelay={delay_ms}:all=1,volume={float(effect['volume']):.3f}[{label}]")
                mix_labels.append(f"[{label}]")
        for index, narration in enumerate(narration_tracks):
            delay_ms = max(0, round(float(narration["target_start"]) * 1000))
            label = f"narr{index}"
            input_index = 3 + len(sound_effects) + index
            parts.append(f"[{input_index}:a]adelay={delay_ms}:all=1,volume=1.00[{label}]")
            mix_labels.append(f"[{label}]")
        if len(mix_labels) > 1:
            parts.append(f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=first:dropout_transition=0[aout]")
        else:
            parts.append(f"[{audio_label}]anull[aout]")
    else:
        parts.append(f"{''.join(concat_inputs)}concat=n={concat_count}:v=1:a=0[vconcat]")

    video_label = "vconcat"
    parts.append(f"[{video_label}][1:v]overlay=0:0:shortest=1:format=auto[vtemplate]")
    video_label = "vtemplate"
    parts.append(f"[{video_label}][2:v]overlay=0:0:shortest=1:format=auto[vtext]")
    video_label = "vtext"
    # Draw the narration label separately.  libass opaque boxes vary by build
    # on Windows, whereas drawbox produces the same readable label everywhere.
    for index, narration in enumerate(narration_tracks):
        start = max(0.0, float(narration["target_start"]))
        end = max(start + 0.1, float(narration.get("target_end") or (start + narration.get("duration", 2.1))))
        next_label = f"vnarrationbox{index}"
        parts.append(
            f"[{video_label}]drawbox=x={NARRATION_LABEL_X}:y={NARRATION_LABEL_Y}:"
            f"w={NARRATION_LABEL_WIDTH}:h={NARRATION_LABEL_HEIGHT}:"
            f"color=0x251A38@0.92:t=fill:enable='between(t,{start:.3f},{end:.3f})'[{next_label}]"
        )
        video_label = next_label
    if captions_path and captions_path.exists():
        parts.append(f"[{video_label}]ass=filename='{escape_filter_path(captions_path)}'[vout]")
    else:
        parts.append(f"[{video_label}]null[vout]")
    return ";".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a vertical Shorts MP4 directly from an approved package.")
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--channel-name", default="")
    parser.add_argument("--no-point-captions", action="store_true")
    return parser


def main() -> None:
    configure_stdout()
    args = build_parser().parse_args()
    source_video = args.source_video.resolve()
    package_path = args.package.resolve()
    if not source_video.exists():
        raise FileNotFoundError(f"Source video not found: {source_video}")
    if source_video.suffix.lower() not in VIDEO_FILE_SUFFIXES:
        raise RuntimeError(f"Source must be a video file, not {source_video.suffix or 'an extensionless file'}: {source_video}")
    if not package_path.exists():
        raise FileNotFoundError(f"Package JSON not found: {package_path}")

    package = json.loads(package_path.read_text(encoding="utf-8"))
    clips = [clip for clip in package.get("source_clips", []) or [] if isinstance(clip, dict)]
    if not clips:
        raise RuntimeError("Package has no source clips.")

    output_path = args.output.resolve() if args.output else default_output_path(package_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template_path = output_path.with_suffix(".template.png")
    overlay_path = output_path.with_suffix(".text_overlay.png")
    captions_path = output_path.with_suffix(".captions.ass")
    prepare_template_layer(template_path)
    build_text_overlay(package, package_path, overlay_path, args.channel_name)
    if not args.no_point_captions:
        build_caption_ass(package, captions_path)
    elif captions_path.exists():
        captions_path.unlink()

    include_audio = has_audio_track(source_video)
    sound_effects = resolved_sound_effects(package) if include_audio else []
    narration_tracks = build_narration_tracks(package, output_path) if include_audio else []
    reframes = plan_clip_reframes(source_video, clips)
    filter_complex = build_filter(
        clips,
        include_audio,
        None if args.no_point_captions else captions_path,
        sound_effects,
        reframes,
        narration_tracks,
    )
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_video),
        "-loop",
        "1",
        "-framerate",
        str(FPS),
        "-i",
        str(template_path),
        "-loop",
        "1",
        "-framerate",
        str(FPS),
        "-i",
        str(overlay_path),
    ]
    for effect in sound_effects:
        command.extend(["-i", str(effect["path"])])
    for narration in narration_tracks:
        command.extend(["-i", str(narration["path"])])
    command.extend(["-filter_complex", filter_complex, "-map", "[vout]"])
    if include_audio:
        command.extend(["-map", "[aout]"])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    )
    if include_audio:
        command.extend(["-c:a", "aac", "-b:a", "160k"])
    command.extend(["-shortest", str(output_path)])
    subprocess.run(command, check=True)
    print(f"[render] output={output_path}", flush=True)
    print(f"[render] template_overlay={template_path}", flush=True)
    print(f"[render] text_overlay={overlay_path}", flush=True)
    if not args.no_point_captions:
        print(f"[render] captions={captions_path}", flush=True)
    for effect in sound_effects:
        print(f"[render] sound_effect={effect['cue']} at={effect['target_start']:.2f}s file={effect['path'].name}", flush=True)
    for narration in narration_tracks:
        print(f"[render] narration at={narration['target_start']:.2f}s text={narration['text']}", flush=True)


if __name__ == "__main__":
    main()
