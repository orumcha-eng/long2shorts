from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont
from pymediainfo import MediaInfo


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FONT_PATH = Path(r"C:\Windows\Fonts\Pretendard-ExtraBold.ttf")
FALLBACK_FONT_PATH = Path(r"C:\Windows\Fonts\malgunbd.ttf")
CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920
# The existing branded template has a transparent footage window between the
# header/logo and the lower controls.  Keep that visual language instead of
# silently replacing it with a bare full-screen render.
VIDEO_VISIBLE_TOP = 260
VIDEO_VISIBLE_BOTTOM = 1488
VIDEO_VISIBLE_HEIGHT = VIDEO_VISIBLE_BOTTOM - VIDEO_VISIBLE_TOP
FPS = 30
VIDEO_FILE_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi"}
DEFAULT_TEMPLATE_IMAGE_CANDIDATES = [
    BASE_DIR / "templates" / "shorts_template.png",
    BASE_DIR / "short_templet.png",
    BASE_DIR / "short_template.png",
]
# The channel mark occupies the left side of the template header.  The title
# belongs beside it, not over the first shot where it hides the visual hook.
TITLE_LINE1_BOX = (270, 42, 1040, 138)
TITLE_LINE2_BOX = (270, 135, 1040, 248)
SOURCE_CREDIT_BOX = (90, 1514, 990, 1605)
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
    {"title1": (255, 80, 118), "title2": (255, 202, 40), "caption": (255, 229, 62), "source": (230, 70, 105)},
    {"title1": (48, 125, 255), "title2": (132, 72, 255), "caption": (87, 222, 255), "source": (57, 100, 210)},
    {"title1": (255, 116, 35), "title2": (38, 180, 120), "caption": (255, 148, 65), "source": (220, 90, 26)},
    {"title1": (214, 55, 179), "title2": (30, 181, 211), "caption": (255, 104, 188), "source": (180, 40, 148)},
)


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
    template = resolve_template_image()
    if template:
        with Image.open(template) as source:
            source = source.convert("RGBA").resize((CANVAS_WIDTH, CANVAS_HEIGHT), Image.Resampling.LANCZOS)
            # Keep the existing channel mark and player controls, but remove
            # the empty white block that used to sit between them.  The new
            # header is intentionally compact so the footage starts high.
            image = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (0, 0, 0, 0))
            # Header and footer are template chrome; the footage window between
            # them must remain transparent so it is never painted white over
            # the rendered video.
            ImageDraw.Draw(image).rectangle((0, 0, CANVAS_WIDTH, VIDEO_VISIBLE_TOP), fill=(255, 255, 255, 255))
            image.alpha_composite(source.crop((0, 0, 270, VIDEO_VISIBLE_TOP)), (0, 0))
            image.alpha_composite(source.crop((0, VIDEO_VISIBLE_BOTTOM, CANVAS_WIDTH, CANVAS_HEIGHT)), (0, VIDEO_VISIBLE_BOTTOM))
    else:
        image = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (0, 0, 0, 0))
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


def build_text_overlay(package: dict, package_path: Path, output_path: Path, channel_name: str = "") -> Path:
    font_path = resolve_font_path()
    image = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    title_line1 = validated_title_line(package.get("title_line1"), field="title_line1")
    title_line2 = validated_title_line(package.get("title_line2"), field="title_line2")
    channel_name = channel_name.strip() or infer_channel_name(package_path, package)
    palette = palette_for_package(package)

    draw_boxed_text(
        draw,
        title_line1,
        TITLE_LINE1_BOX,
        font_path=font_path,
        font_size=70,
        min_font_size=46,
        fill=(*palette["title1"], 255),
        align="left",
        stroke_width=5,
        stroke_fill=(17, 17, 17, 255),
    )
    draw_boxed_text(
        draw,
        title_line2,
        TITLE_LINE2_BOX,
        font_path=font_path,
        font_size=82,
        min_font_size=52,
        fill=(*palette["title2"], 255),
        align="center",
        stroke_width=6,
        stroke_fill=(17, 17, 17, 255),
    )
    if channel_name:
        draw_boxed_text(
            draw,
            f"출처 | {channel_name}",
            SOURCE_CREDIT_BOX,
            font_path=font_path,
            font_size=48,
            min_font_size=34,
            fill=(*palette["source"], 255),
            align="center",
            stroke_width=0,
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
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    for caption in package.get("point_captions", []) or []:
        if not isinstance(caption, dict):
            continue
        start = positive_float(caption.get("target_start"), 0.0)
        end = positive_float(caption.get("target_end"), start + 2.0)
        text = animated_ass_text(str(caption.get("text") or "").strip())
        if text and end > start:
            lines.append(f"Dialogue: 0,{format_ass_time(start)},{format_ass_time(end)},Point,,0,0,0,,{text}")
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


def build_filter(clips: list[dict], include_audio: bool, captions_path: Path | None, sound_effects: list[dict]) -> str:
    parts: list[str] = []
    concat_inputs: list[str] = []
    for index, clip in enumerate(clips):
        start = positive_float(clip.get("source_start"), -1.0)
        end = positive_float(clip.get("source_end"), -1.0)
        if end <= start or start < 0:
            raise RuntimeError(f"Invalid source clip timing at index {index}.")
        # A landscape talk-show shot cannot safely be centre-cropped into a
        # vertical frame: a two-person shot turns into a face edge or an arm.
        # Preserve the full foreground frame and fill the remaining window
        # with a softened copy of that same shot. This keeps both speakers
        # visible; a later explicit tracking mode can still choose a tighter
        # crop only when the edit plan names one person to follow.
        parts.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS,split=2[vbg{index}][vfg{index}]")
        parts.append(
            f"[vbg{index}]scale={CANVAS_WIDTH}:{VIDEO_VISIBLE_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={CANVAS_WIDTH}:{VIDEO_VISIBLE_HEIGHT},gblur=sigma=22,eq=brightness=-0.18:saturation=0.72[bg{index}]"
        )
        parts.append(
            f"[vfg{index}]scale={CANVAS_WIDTH}:{VIDEO_VISIBLE_HEIGHT}:force_original_aspect_ratio=decrease[fg{index}]"
        )
        parts.append(
            f"[bg{index}][fg{index}]overlay=(W-w)/2:(H-h)/2,"
            f"pad={CANVAS_WIDTH}:{CANVAS_HEIGHT}:0:{VIDEO_VISIBLE_TOP}:color=black,setsar=1,fps={FPS}[v{index}]"
        )
        concat_inputs.append(f"[v{index}]")
        if include_audio:
            parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{index}]")
            concat_inputs.append(f"[a{index}]")
    if include_audio:
        parts.append(f"{''.join(concat_inputs)}concat=n={len(clips)}:v=1:a=1[vconcat][aconcat]")
        if sound_effects:
            effect_labels = []
            for index, effect in enumerate(sound_effects):
                delay_ms = max(0, round(float(effect["target_start"]) * 1000))
                input_index = 3 + index
                label = f"sfx{index}"
                parts.append(f"[{input_index}:a]adelay={delay_ms}:all=1,volume={float(effect['volume']):.3f}[{label}]")
                effect_labels.append(f"[{label}]")
            parts.append(f"[aconcat]{''.join(effect_labels)}amix=inputs={len(effect_labels) + 1}:duration=first:dropout_transition=0[aout]")
        else:
            parts.append("[aconcat]anull[aout]")
    else:
        parts.append(f"{''.join(concat_inputs)}concat=n={len(clips)}:v=1:a=0[vconcat]")

    video_label = "vconcat"
    parts.append(f"[{video_label}][1:v]overlay=0:0:shortest=1:format=auto[vtemplate]")
    video_label = "vtemplate"
    parts.append(f"[{video_label}][2:v]overlay=0:0:shortest=1:format=auto[vtext]")
    video_label = "vtext"
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
    filter_complex = build_filter(clips, include_audio, None if args.no_point_captions else captions_path, sound_effects)
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


if __name__ == "__main__":
    main()
