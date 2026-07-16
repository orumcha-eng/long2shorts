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
VIDEO_VISIBLE_TOP = 432
VIDEO_VISIBLE_BOTTOM = 1488
VIDEO_VISIBLE_HEIGHT = VIDEO_VISIBLE_BOTTOM - VIDEO_VISIBLE_TOP
FPS = 30
DEFAULT_TEMPLATE_IMAGE_CANDIDATES = [
    BASE_DIR / "templates" / "shorts_template.png",
    BASE_DIR / "short_templet.png",
    BASE_DIR / "short_template.png",
]
TITLE_LINE1_BOX = (285, 108, 1050, 205)
TITLE_LINE2_BOX = (130, 205, 1040, 325)
CHANNEL_BOX = (0, 1515, 1080, 1610)


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


def compact_line(value: str, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(1, max_chars - 3)].rstrip() + "..."


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
    template = resolve_template_image()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if template:
        image = Image.open(template).convert("RGBA").resize((CANVAS_WIDTH, CANVAS_HEIGHT))
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
    title_line1 = compact_line(package.get("title_line1"), 12)
    title_line2 = compact_line(package.get("title_line2"), 14)
    channel_text = infer_channel_name(package_path, package, channel_name)

    draw_boxed_text(
        draw,
        title_line1,
        TITLE_LINE1_BOX,
        font_path=font_path,
        font_size=80,
        min_font_size=58,
        fill=(255, 255, 255, 255),
        align="left",
        stroke_width=7,
        stroke_fill=(17, 17, 17, 255),
    )
    draw_boxed_text(
        draw,
        title_line2,
        TITLE_LINE2_BOX,
        font_path=font_path,
        font_size=82,
        min_font_size=56,
        fill=(190, 255, 0, 255),
        align="center",
        stroke_width=10,
        stroke_fill=(17, 17, 17, 255),
    )
    draw_boxed_text(
        draw,
        channel_text,
        CHANNEL_BOX,
        font_path=font_path,
        font_size=72,
        min_font_size=52,
        fill=(17, 17, 17, 255),
        align="center",
        stroke_width=2,
        stroke_fill=(255, 255, 255, 255),
    )

    image.save(output_path)
    return output_path


def build_caption_ass(package: dict, output_path: Path) -> Path:
    font_name = "Pretendard ExtraBold" if DEFAULT_FONT_PATH.exists() else "Malgun Gothic"
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {CANVAS_WIDTH}",
        f"PlayResY: {CANVAS_HEIGHT}",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        f"Style: Point,{font_name},74,&H00FFFFFF,&H000000FF,&H00101010,&H9A000000,-1,0,0,0,100,100,0,0,1,9,3,2,70,70,500,1",
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
    return package_path.parent.parent.parent / "productions" / f"{package_path.stem}.mp4"


def build_filter(clips: list[dict], include_audio: bool, captions_path: Path | None) -> str:
    parts: list[str] = []
    concat_inputs: list[str] = []
    for index, clip in enumerate(clips):
        start = positive_float(clip.get("source_start"), -1.0)
        end = positive_float(clip.get("source_end"), -1.0)
        if end <= start or start < 0:
            raise RuntimeError(f"Invalid source clip timing at index {index}.")
        parts.append(
            f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS,"
            f"scale={CANVAS_WIDTH}:{VIDEO_VISIBLE_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={CANVAS_WIDTH}:{VIDEO_VISIBLE_HEIGHT},"
            f"pad={CANVAS_WIDTH}:{CANVAS_HEIGHT}:0:{VIDEO_VISIBLE_TOP}:color=black,"
            f"setsar=1,fps={FPS}[v{index}]"
        )
        concat_inputs.append(f"[v{index}]")
        if include_audio:
            parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{index}]")
            concat_inputs.append(f"[a{index}]")
    if include_audio:
        parts.append(f"{''.join(concat_inputs)}concat=n={len(clips)}:v=1:a=1[vconcat][aconcat]")
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
    filter_complex = build_filter(clips, include_audio, None if args.no_point_captions else captions_path)
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
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
    ]
    if include_audio:
        command.extend(["-map", "[aconcat]"])
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


if __name__ == "__main__":
    main()
