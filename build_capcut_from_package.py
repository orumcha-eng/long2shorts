import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pycapcut as cc
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageColor, ImageDraw, ImageFont
from pymediainfo import MediaInfo


DRAFT_ROOT = Path(r"C:\Users\user\AppData\Local\CapCut\User Data\Projects\com.lveditor.draft")
ROOT_META_PATH = DRAFT_ROOT / "root_meta_info.json"
BASE_DIR = Path(__file__).resolve().parent
SHORTS_ENV_PATH = BASE_DIR.parent / "auto_Youtube" / "shorts" / ".env"
AUDIO_CACHE_DIR = BASE_DIR / "generated_audio"
TEXT_OVERLAY_CACHE_DIR = BASE_DIR / "generated_text_overlays"
FONT_DIR = BASE_DIR / "assets" / "fonts"
DEFAULT_JUA_FONT = FONT_DIR / "Jua-Regular.ttf"
TEXT_OVERLAY_RENDER_VERSION = 2
REFERENCE_CAPCUT_FONT_ID = "7577600442964725008"
REFERENCE_CAPCUT_FONT_PATH = Path(
    r"C:\Users\user\AppData\Local\CapCut\User Data\Cache\effect"
    r"\7577600442964725008\c1431e2eae4fe955caa6f71a7d60e08f\font.ttf"
)
DEFAULT_TEMPLATE_IMAGE = BASE_DIR / "templates" / "shorts_template.png"
DEFAULT_TEMPLATE_IMAGE_CANDIDATES = [
    DEFAULT_TEMPLATE_IMAGE,
    BASE_DIR / "short_templet.png",
    BASE_DIR / "short_template.png",
]
DRAFT_WIDTH = 1080
DRAFT_HEIGHT = 1920
DRAFT_FPS = 30
TTS_VOICE = "onyx"
TTS_SPEED = 1.15
TITLE_DURATION_S = 4.5
TITLE_FULL_DURATION = True
TITLE_SIZE = 10.5
TITLE_Y = -0.82
TITLE_FONT = "Poppins_Bold"
TITLE_IMAGE_FONT_SIZE = 72
TITLE_IMAGE_MIN_FONT_SIZE = 48
TITLE_IMAGE_COLOR = "#111111"
TITLE_IMAGE_BOX = (300, 58, 1030, 290)
TITLE_IMAGE_MAX_CHARS_PER_LINE = 13
TITLE_IMAGE_MAX_LINES = 2
TITLE_LINE1_IMAGE_FONT_SIZE = 78
TITLE_LINE1_IMAGE_MIN_FONT_SIZE = 58
TITLE_LINE1_IMAGE_COLOR = "#111111"
TITLE_LINE1_IMAGE_BOX = (290, 70, 1040, 165)
TITLE_LINE1_IMAGE_MAX_CHARS = 12
TITLE_LINE2_IMAGE_FONT_SIZE = 76
TITLE_LINE2_IMAGE_MIN_FONT_SIZE = 56
TITLE_LINE2_IMAGE_COLOR = "#00E846"
TITLE_LINE2_IMAGE_STROKE_COLOR = "#111111"
TITLE_LINE2_IMAGE_STROKE_WIDTH = 9
TITLE_LINE2_IMAGE_BOX = (80, 245, 1000, 380)
TITLE_LINE2_IMAGE_MAX_CHARS = 11
TITLE_LINE1_CAPCUT_X = 0.23809523809523817
TITLE_LINE1_CAPCUT_Y = 0.8806584362139918
TITLE_LINE2_CAPCUT_X = 0.04395604395604402
TITLE_LINE2_CAPCUT_Y = 0.6707818930041152
CHANNEL_CAPCUT_X = 0.0
CHANNEL_CAPCUT_Y = -0.6337448559670782
REFERENCE_CAPCUT_TEXT_SIZE = 15.0
REFERENCE_CAPCUT_LINE_MAX_WIDTH = 0.82
REFERENCE_CAPCUT_LINE_SPACING = 0.02
REFERENCE_CAPCUT_STROKE_WIDTH = 0.06
REFERENCE_CAPCUT_TITLE_LINE1_COLOR = "#000000"
REFERENCE_CAPCUT_TITLE_LINE1_STROKE = "#FFFFFF"
REFERENCE_CAPCUT_TITLE_LINE2_COLOR = "#00FF18"
REFERENCE_CAPCUT_TITLE_LINE2_STROKE = "#000000"
REFERENCE_CAPCUT_CHANNEL_COLOR = "#000000"
REFERENCE_CAPCUT_CHANNEL_STROKE = "#FFFFFF"
TITLE_GENERIC_REMOVE_PHRASES = (
    "현장 대공개",
    "대공개",
    "이유는",
    "이유",
    "현장",
    "포착",
    "순간",
    "장면",
    "모습",
)
TITLE_PHRASE_REPLACEMENTS = (
    ("폭력적으로 웃는", "폭력 웃음"),
    ("폭력적으로", "폭력"),
    ("웃음 유발", "웃음"),
    ("미안해하는", "미안한"),
    ("열심히 하는", "열심"),
    ("갑자기 터진", "터진"),
    ("유쾌한 굴욕", "굴욕"),
    ("케미가 폭발하는", "케미 폭발"),
    ("케미 폭발", "케미"),
)
POINT_SIZE = 10.0
POINT_Y = 0.1
POINT_FONT = "Montserrat"
CHANNEL_SIZE = 5.6
CHANNEL_Y = 0.88
CHANNEL_FONT = "Montserrat"
CHANNEL_IMAGE_FONT_SIZE = 72
CHANNEL_IMAGE_MIN_FONT_SIZE = 52
CHANNEL_IMAGE_COLOR = "#111111"
CHANNEL_IMAGE_BOX = (0, 1515, 1080, 1610)
VIDEO_VISIBLE_TOP = 432
VIDEO_VISIBLE_BOTTOM = 1488
VIDEO_VOLUME = 1.0
NARRATION_VOLUME = 1.0


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def us(seconds: float) -> int:
    return int(seconds * 1_000_000)


def load_media_info(path: Path) -> tuple[int, int, int]:
    media = MediaInfo.parse(str(path))
    general = next(track for track in media.tracks if track.track_type == "General")
    video = next(track for track in media.tracks if track.track_type == "Video")
    duration_us = int(float(general.duration) * 1000)
    return duration_us, int(video.width), int(video.height)


def load_duration_us(path: Path) -> int:
    media = MediaInfo.parse(str(path))
    general = next(track for track in media.tracks if track.track_type == "General")
    return int(float(general.duration) * 1000)


def get_openai_client() -> OpenAI:
    load_dotenv(SHORTS_ENV_PATH)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY not found in {SHORTS_ENV_PATH}")
    return OpenAI(api_key=api_key)


def safe_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False
    return default


def as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def positive_float(value: Any, default: float) -> float:
    parsed = as_float(value, default)
    return parsed if parsed > 0 else default


def as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def positive_int(value: Any, default: int) -> int:
    parsed = as_int(value, default)
    return parsed if parsed > 0 else default


def normalize_token(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def resolve_font(font_name: Any) -> cc.FontType | None:
    if not isinstance(font_name, str) or not font_name.strip():
        return None

    target = normalize_token(font_name)
    for font in cc.FontType:
        if target in {normalize_token(font.name), normalize_token(font.value.name)}:
            return font
    return None


def resolve_color(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, str):
        raw = value.strip().lstrip("#")
        if len(raw) == 6:
            try:
                return tuple(int(raw[idx : idx + 2], 16) / 255.0 for idx in range(0, 6, 2))
            except ValueError:
                return default

    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            channels = [float(channel) for channel in value]
        except (TypeError, ValueError):
            return default

        if any(abs(channel) > 1.0 for channel in channels):
            channels = [min(max(channel, 0.0), 255.0) / 255.0 for channel in channels]
        else:
            channels = [min(max(channel, 0.0), 1.0) for channel in channels]
        return tuple(channels)

    return default


def color_list(value: Any, default: str) -> list[float]:
    fallback = resolve_color(default, (0.0, 0.0, 0.0))
    return [round(channel, 8) for channel in resolve_color(value, fallback)]


def resolve_hex_color(value: Any, default: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return default
    raw = value.strip()
    if not raw.startswith("#"):
        raw = "#" + raw
    return raw


def resolve_path(value: Any, base_dir: Path = BASE_DIR) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None

    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def first_nonempty_string(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def resolve_box(value: Any, default: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if isinstance(value, dict):
        try:
            if {"x", "y", "width", "height"}.issubset(value):
                x = int(value["x"])
                y = int(value["y"])
                return (x, y, x + int(value["width"]), y + int(value["height"]))
            if {"left", "top", "right", "bottom"}.issubset(value):
                return (int(value["left"]), int(value["top"]), int(value["right"]), int(value["bottom"]))
        except (TypeError, ValueError):
            return default

    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return tuple(int(part) for part in value)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return default

    return default


def resolve_rgba(value: Any, default: str) -> tuple[int, int, int, int]:
    try:
        return ImageColor.getcolor(resolve_hex_color(value, default), "RGBA")
    except ValueError:
        return ImageColor.getcolor(default, "RGBA")


def resolve_align(value: Any, default: int) -> int:
    if isinstance(value, int):
        return min(max(value, 0), 2)

    if isinstance(value, str):
        align_map = {
            "left": 0,
            "start": 0,
            "center": 1,
            "centre": 1,
            "middle": 1,
            "right": 2,
            "end": 2,
        }
        return align_map.get(value.strip().lower(), default)

    return default


def build_text_style(config: dict[str, Any], fallback: cc.TextStyle) -> cc.TextStyle:
    style_config = safe_dict(config.get("style"))
    return cc.TextStyle(
        size=as_float(style_config.get("size"), fallback.size),
        bold=as_bool(style_config.get("bold"), fallback.bold),
        italic=as_bool(style_config.get("italic"), fallback.italic),
        underline=as_bool(style_config.get("underline"), fallback.underline),
        color=resolve_color(style_config.get("color"), fallback.color),
        alpha=as_float(style_config.get("alpha"), fallback.alpha),
        align=resolve_align(style_config.get("align"), fallback.align),
        vertical=as_bool(style_config.get("vertical"), fallback.vertical),
        letter_spacing=as_int(style_config.get("letter_spacing"), fallback.letter_spacing),
        line_spacing=as_int(style_config.get("line_spacing"), fallback.line_spacing),
        auto_wrapping=as_bool(style_config.get("auto_wrapping"), fallback.auto_wrapping),
        max_line_width=as_float(style_config.get("max_line_width"), fallback.max_line_width),
    )


def build_text_border(config: dict[str, Any]) -> cc.TextBorder | None:
    border_config = safe_dict(config.get("border"))
    if not border_config or border_config.get("enabled") is False:
        return None

    return cc.TextBorder(
        alpha=as_float(border_config.get("alpha"), 1.0),
        color=resolve_color(border_config.get("color"), (0.0, 0.0, 0.0)),
        width=positive_float(border_config.get("width"), 40.0),
    )


def build_text_background(config: dict[str, Any]) -> cc.TextBackground | None:
    background_config = safe_dict(config.get("background"))
    if not background_config or background_config.get("enabled") is False:
        return None

    color = resolve_hex_color(background_config.get("color"), "#000000")
    return cc.TextBackground(
        color=color,
        style=1 if as_int(background_config.get("style"), 1) != 2 else 2,
        alpha=as_float(background_config.get("alpha"), 1.0),
        round_radius=as_float(background_config.get("round_radius"), 0.0),
        height=positive_float(background_config.get("height"), 0.14),
        width=positive_float(background_config.get("width"), 0.14),
        horizontal_offset=as_float(background_config.get("horizontal_offset"), 0.5),
        vertical_offset=as_float(background_config.get("vertical_offset"), 0.5),
    )


def build_clip_settings(config: dict[str, Any], fallback: cc.ClipSettings) -> cc.ClipSettings:
    clip_config = safe_dict(config.get("clip"))
    return cc.ClipSettings(
        alpha=as_float(clip_config.get("alpha"), fallback.alpha),
        flip_horizontal=as_bool(clip_config.get("flip_horizontal"), fallback.flip_horizontal),
        flip_vertical=as_bool(clip_config.get("flip_vertical"), fallback.flip_vertical),
        rotation=as_float(clip_config.get("rotation"), fallback.rotation),
        scale_x=as_float(clip_config.get("scale_x"), fallback.scale_x),
        scale_y=as_float(clip_config.get("scale_y"), fallback.scale_y),
        transform_x=as_float(clip_config.get("transform_x"), fallback.transform_x),
        transform_y=as_float(clip_config.get("transform_y"), fallback.transform_y),
    )


def extract_effect_id(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        effect_id = value.get("effect_id") or value.get("id")
        if effect_id:
            return str(effect_id).strip()
    return None


def build_text_segment(
    text: str,
    timerange: cc.Timerange,
    config: dict[str, Any],
    fallback_style: cc.TextStyle,
    fallback_clip: cc.ClipSettings,
) -> cc.TextSegment:
    font = resolve_font(config.get("font"))
    segment = cc.TextSegment(
        text,
        timerange,
        font=font,
        style=build_text_style(config, fallback_style),
        clip_settings=build_clip_settings(config, fallback_clip),
        border=build_text_border(config),
        background=build_text_background(config),
    )

    effect_id = extract_effect_id(config.get("effect"))
    if effect_id:
        segment.add_effect(effect_id)

    bubble_config = safe_dict(config.get("bubble"))
    bubble_effect_id = bubble_config.get("effect_id")
    bubble_resource_id = bubble_config.get("resource_id")
    if bubble_effect_id and bubble_resource_id:
        segment.add_bubble(str(bubble_effect_id), str(bubble_resource_id))

    return segment


def build_reference_capcut_text_config(
    *,
    transform_x: float,
    transform_y: float,
    fill_color: str,
    stroke_color: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return deep_merge(
        {
            "style": {
                "size": REFERENCE_CAPCUT_TEXT_SIZE,
                "color": fill_color,
                "align": 1,
                "auto_wrapping": False,
                "max_line_width": REFERENCE_CAPCUT_LINE_MAX_WIDTH,
                "line_spacing": 0,
                "letter_spacing": 0,
            },
            "clip": {
                "transform_x": transform_x,
                "transform_y": transform_y,
                "scale_x": 1.0,
                "scale_y": 1.0,
                "rotation": 0.0,
                "alpha": 1.0,
            },
            "border": {
                "enabled": True,
                "color": stroke_color,
                "alpha": 1.0,
                "width": 24.0,
            },
        },
        safe_dict(overrides),
    )


def register_reference_capcut_text_patch(
    material_patches: dict[str, dict[str, Any]],
    content_patches: dict[str, dict[str, Any]],
    segment: cc.TextSegment,
    text: str,
    *,
    fill_color: str,
    stroke_color: str,
    overrides: dict[str, Any] | None = None,
) -> None:
    config = safe_dict(overrides)
    font_id = str(config.get("font_resource_id") or REFERENCE_CAPCUT_FONT_ID)
    font_path = resolve_path(config.get("font_path") or config.get("font_file")) or REFERENCE_CAPCUT_FONT_PATH
    font_path_value = font_path.as_posix()
    font_size = positive_float(config.get("font_size"), REFERENCE_CAPCUT_TEXT_SIZE)
    stroke_width = positive_float(config.get("stroke_width"), REFERENCE_CAPCUT_STROKE_WIDTH)
    line_max_width = positive_float(config.get("line_max_width"), REFERENCE_CAPCUT_LINE_MAX_WIDTH)
    line_spacing = as_float(config.get("line_spacing"), REFERENCE_CAPCUT_LINE_SPACING)
    fill_hex = resolve_hex_color(config.get("fill_color") or config.get("color") or fill_color, fill_color)
    stroke_hex = resolve_hex_color(config.get("stroke_color") or stroke_color, stroke_color)

    material_patches[segment.material_id] = {
        "alignment": 1,
        "border_alpha": 1.0,
        "border_color": stroke_hex,
        "border_mode": 0,
        "border_width": 0.08,
        "font_path": font_path_value,
        "font_resource_id": font_id,
        "font_size": font_size,
        "font_source_platform": 1,
        "global_alpha": 1.0,
        "letter_spacing": 0.0,
        "line_feed": 1,
        "line_max_width": line_max_width,
        "line_spacing": line_spacing,
        "oneline_cutoff": False,
        "text_alpha": 1.0,
        "text_color": fill_hex,
        "text_size": 30,
        "use_effect_default_color": True,
    }
    content_patches[segment.material_id] = {
        "styles": [
            {
                "fill": {
                    "alpha": 1.0,
                    "content": {
                        "render_type": "solid",
                        "solid": {
                            "alpha": 1.0,
                            "color": color_list(fill_hex, fill_color),
                        },
                    },
                },
                "font": {
                    "id": font_id,
                    "path": font_path_value,
                },
                "range": [0, len(text)],
                "size": font_size,
                "strokes": [
                    {
                        "alpha": 1.0,
                        "content": {
                            "render_type": "solid",
                            "solid": {
                                "alpha": 1.0,
                                "color": color_list(stroke_hex, stroke_color),
                            },
                        },
                        "mode": 0,
                        "width": stroke_width,
                    }
                ],
                "useLetterColor": True,
            }
        ],
        "text": text,
    }


def collect_text_patches(
    material_patches: dict[str, dict[str, Any]],
    content_patches: dict[str, dict[str, Any]],
    segment: cc.TextSegment,
    config: dict[str, Any],
) -> None:
    material_patch = safe_dict(config.get("material_patch"))
    content_patch = safe_dict(config.get("content_patch"))

    if material_patch:
        material_patches[segment.material_id] = material_patch
    if content_patch:
        content_patches[segment.material_id] = content_patch


def apply_text_material_patches(
    draft_path: Path,
    material_patches: dict[str, dict[str, Any]],
    content_patches: dict[str, dict[str, Any]],
) -> int:
    if not material_patches and not content_patches:
        return 0

    draft_content_path = draft_path / "draft_content.json"
    with open(draft_content_path, "r", encoding="utf-8") as f:
        draft_content = json.load(f)

    patched_count = 0
    for text_material in draft_content.get("materials", {}).get("texts", []):
        material_id = text_material.get("id")
        touched = False

        material_patch = material_patches.get(material_id)
        if material_patch:
            merged = deep_merge(text_material, material_patch)
            text_material.clear()
            text_material.update(merged)
            touched = True

        content_patch = content_patches.get(material_id)
        if content_patch:
            raw_content = text_material.get("content")
            if isinstance(raw_content, str) and raw_content.strip():
                try:
                    decoded_content = json.loads(raw_content)
                except json.JSONDecodeError:
                    decoded_content = {}
            else:
                decoded_content = {}

            text_material["content"] = json.dumps(deep_merge(decoded_content, content_patch), ensure_ascii=False)
            touched = True

        if touched:
            patched_count += 1

    if patched_count:
        with open(draft_content_path, "w", encoding="utf-8") as f:
            json.dump(draft_content, f, ensure_ascii=False, indent=2)

    return patched_count


def generate_narration_audio(
    text: str,
    index: int,
    client: OpenAI,
    voice: str = TTS_VOICE,
    speed: float = TTS_SPEED,
) -> Path:
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest_source = json.dumps({"text": text, "voice": voice, "speed": speed}, ensure_ascii=False, sort_keys=True)
    digest = hashlib.md5(digest_source.encode("utf-8")).hexdigest()[:10]
    output_path = AUDIO_CACHE_DIR / f"pkg_narration_{index:02d}_{digest}.mp3"
    if output_path.exists():
        return output_path

    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        speed=speed,
    )
    output_path.write_bytes(response.content)
    return output_path


def build_draft_name(source_video: Path, package: dict, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    return f"{source_video.stem}_{package['short_id']}"


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def infer_analysis_dir(package_path: Path) -> Path | None:
    for parent in package_path.parents:
        if (parent / "youtube_context.json").exists():
            return parent
    return None


def infer_channel_name(package_path: Path, package: dict[str, Any], explicit_name: str | None) -> str:
    if explicit_name and explicit_name.strip():
        return explicit_name.strip()

    package_context = safe_dict(package.get("youtube_context"))
    package_metadata = safe_dict(package_context.get("metadata"))
    channel_name = first_nonempty_string(
        package.get("source_label"),
        package.get("source_credit"),
        package.get("channel_title"),
        package_metadata.get("channel_title"),
    )
    if channel_name:
        return channel_name

    movie_info = safe_dict(package.get("movie_info"))
    movie_title = first_nonempty_string(
        package.get("source_title"),
        package.get("movie_title"),
        movie_info.get("matched_title"),
        movie_info.get("query"),
        movie_info.get("title"),
    )
    if movie_title:
        return movie_title

    analysis_dir = infer_analysis_dir(package_path)
    if not analysis_dir:
        return ""

    youtube_context = load_json_if_exists(analysis_dir / "youtube_context.json")
    metadata = safe_dict(youtube_context.get("metadata"))
    channel_name = first_nonempty_string(metadata.get("channel_title"))
    if channel_name:
        return channel_name

    movie_info = load_json_if_exists(analysis_dir / "movie_info.json")
    movie_title = first_nonempty_string(
        movie_info.get("matched_title"),
        movie_info.get("query"),
        movie_info.get("title"),
    )
    if movie_title:
        return movie_title

    package_index = load_json_if_exists(analysis_dir / "shorts_candidates" / "final" / "shorts_packages.json")
    index_movie_info = safe_dict(package_index.get("movie_info"))
    return first_nonempty_string(
        package_index.get("source_label"),
        package_index.get("source_title"),
        index_movie_info.get("matched_title"),
        index_movie_info.get("query"),
        index_movie_info.get("title"),
    )


def format_channel_text(channel_name: str, config: dict[str, Any]) -> str:
    if not channel_name:
        return ""

    explicit_text = first_nonempty_string(config.get("text"))
    if explicit_text:
        return explicit_text.format(channel=channel_name)

    prefix = config.get("prefix")
    if not isinstance(prefix, str):
        suffix = config.get("suffix")
        if not isinstance(suffix, str):
            suffix = ""
        return f"{channel_name}{suffix}"
    return f"{prefix}{channel_name}"


def resolve_template_image_path(config: dict[str, Any], explicit_path: Path | None) -> Path | None:
    if explicit_path:
        return explicit_path.resolve()

    configured_path = resolve_path(config.get("image_path") or config.get("path"))
    if configured_path:
        return configured_path

    for candidate in DEFAULT_TEMPLATE_IMAGE_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_overlay_font_path(config: dict[str, Any], fallback: Path = DEFAULT_JUA_FONT) -> Path:
    for value in (
        config.get("font_path"),
        config.get("font_file"),
        config.get("font"),
    ):
        path = resolve_path(value)
        if path and path.exists():
            return path

    for candidate in (
        fallback,
        FONT_DIR / "BMJUA.ttf",
        FONT_DIR / "BMJUA_ttf.ttf",
        FONT_DIR / "Jua-Regular.ttf",
        Path(r"C:\Windows\Fonts\BMJUA_ttf.ttf"),
        Path(r"C:\Windows\Fonts\Jua-Regular.ttf"),
    ):
        if candidate.exists():
            return candidate.resolve()

    return Path(r"C:\Windows\Fonts\malgunbd.ttf")


def load_overlay_font(font_path: Path, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default(size=size)


def split_text_by_char_limit(text: str, max_chars: int) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []

    parts: list[str] = []
    current = ""
    for word in text.split(" "):
        if len(word) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.extend(word[idx : idx + max_chars] for idx in range(0, len(word), max_chars))
            continue

        candidate = word if not current else f"{current} {word}"
        if len(candidate.replace(" ", "")) <= max_chars:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = word

    if current:
        parts.append(current)
    return parts


def display_char_count(text: str) -> int:
    return len(text.replace(" ", ""))


def clip_line_without_ellipsis(text: str, max_chars: int) -> str:
    text = " ".join(text.split()).strip()
    if display_char_count(text) <= max_chars:
        return text

    clipped: list[str] = []
    count = 0
    for char in text:
        if char.isspace():
            if clipped and clipped[-1] != " ":
                clipped.append(" ")
            continue
        if count >= max_chars:
            break
        clipped.append(char)
        count += 1
    return "".join(clipped).strip()


def compact_title_display_line(text: str, max_chars: int) -> str:
    text = " ".join(text.replace("…", "").replace("...", "").split()).strip()
    if not text:
        return ""

    text = re.sub(r"([가-힣A-Za-z0-9]+)(?:과|와)\s+([가-힣A-Za-z0-9]+)(?:의)?", r"\1x\2", text)
    text = re.sub(r"([가-힣A-Za-z0-9]{2,})(?:이|가|은|는|의)\s+", r"\1 ", text)
    for old, new in TITLE_PHRASE_REPLACEMENTS:
        text = text.replace(old, new)
    for phrase in TITLE_GENERIC_REMOVE_PHRASES:
        text = text.replace(phrase, "")
    text = re.sub(r"[?!！？，,.:;~]+", "", text)
    text = " ".join(text.split()).strip()

    if display_char_count(text) <= max_chars:
        return text

    words = text.split()
    selected: list[str] = []
    for word in words:
        candidate = " ".join(selected + [word])
        if display_char_count(candidate) <= max_chars:
            selected.append(word)
    if selected:
        return " ".join(selected)

    return clip_line_without_ellipsis(text, max_chars)


def truncate_line(text: str, max_chars: int) -> str:
    return clip_line_without_ellipsis(text, max_chars)


def limit_title_lines(text: str, max_chars: int, max_lines: int) -> list[str]:
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    wrapped: list[str] = []
    for line in raw_lines:
        wrapped.extend(split_text_by_char_limit(line, max_chars))

    if len(wrapped) <= max_lines:
        return [truncate_line(line, max_chars) for line in wrapped]

    lines = wrapped[:max_lines]
    remainder = "".join(part.replace(" ", "") for part in wrapped[max_lines - 1 :])
    lines[-1] = truncate_line(remainder, max_chars)
    return lines


def measure_multiline(draw: ImageDraw.ImageDraw, lines: list[str], font: ImageFont.ImageFont, line_gap: int) -> tuple[int, int]:
    if not lines:
        return (0, 0)

    widths: list[int] = []
    heights: list[int] = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    return max(widths), sum(heights) + line_gap * max(0, len(lines) - 1)


def fit_overlay_font(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font_path: Path,
    box: tuple[int, int, int, int],
    max_size: int,
    min_size: int,
    line_gap_ratio: float,
) -> tuple[ImageFont.ImageFont, int]:
    box_width = max(1, box[2] - box[0])
    box_height = max(1, box[3] - box[1])
    for size in range(max_size, min_size - 1, -2):
        font = load_overlay_font(font_path, size)
        line_gap = max(0, int(size * line_gap_ratio))
        text_width, text_height = measure_multiline(draw, lines, font, line_gap)
        if text_width <= box_width and text_height <= box_height:
            return font, line_gap

    font = load_overlay_font(font_path, min_size)
    return font, max(0, int(min_size * line_gap_ratio))


def draw_text_block(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    box: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    *,
    align: str,
    valign: str,
    line_gap: int,
    stroke_width: int = 0,
    stroke_fill: tuple[int, int, int, int] | None = None,
) -> None:
    if not lines:
        return

    line_boxes = [draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width) for line in lines]
    line_widths = [bbox[2] - bbox[0] for bbox in line_boxes]
    line_heights = [bbox[3] - bbox[1] for bbox in line_boxes]
    total_height = sum(line_heights) + line_gap * max(0, len(lines) - 1)

    if valign == "center":
        y = box[1] + max(0, (box[3] - box[1] - total_height) // 2)
    elif valign == "bottom":
        y = box[3] - total_height
    else:
        y = box[1]

    for line, bbox, line_width, line_height in zip(lines, line_boxes, line_widths, line_heights):
        if align == "center":
            x = box[0] + max(0, (box[2] - box[0] - line_width) // 2)
        elif align == "right":
            x = box[2] - line_width
        else:
            x = box[0]
        draw.text(
            (x - bbox[0], y - bbox[1]),
            line,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill or fill,
        )
        y += line_height + line_gap


def overlay_style_value(config: dict[str, Any], key: str, default: Any) -> Any:
    style = safe_dict(config.get("style"))
    return config.get(key, style.get(key, default))


def merge_text_render_config(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    return deep_merge(defaults, overrides)


def common_title_render_config(title_config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: title_config[key]
        for key in ("font_path", "font_file")
        if key in title_config
    }


def render_single_line_overlay(
    draw: ImageDraw.ImageDraw,
    text: str,
    config: dict[str, Any],
    *,
    default_box: tuple[int, int, int, int],
    default_font_size: int,
    default_min_font_size: int,
    default_color: str,
    default_max_chars: int,
    default_align: str,
    default_valign: str = "center",
    default_stroke_width: int = 0,
    default_stroke_color: str = "#FFFFFF",
) -> None:
    max_chars = positive_int(overlay_style_value(config, "max_chars", default_max_chars), default_max_chars)
    text = compact_title_display_line(text, max_chars)
    if not text:
        return

    box = resolve_box(overlay_style_value(config, "box", default_box), default_box)
    font_path = resolve_overlay_font_path(config)
    font_size = positive_int(overlay_style_value(config, "image_font_size", default_font_size), default_font_size)
    min_font_size = positive_int(
        overlay_style_value(config, "min_image_font_size", default_min_font_size),
        default_min_font_size,
    )
    font, line_gap = fit_overlay_font(
        draw,
        [text],
        font_path,
        box,
        font_size,
        min_font_size,
        line_gap_ratio=0.0,
    )
    draw_text_block(
        draw,
        [text],
        box,
        font,
        resolve_rgba(overlay_style_value(config, "image_color", default_color), default_color),
        align=str(overlay_style_value(config, "image_align", default_align)),
        valign=str(overlay_style_value(config, "image_valign", default_valign)),
        line_gap=line_gap,
        stroke_width=max(0, as_int(overlay_style_value(config, "image_stroke_width", default_stroke_width), default_stroke_width)),
        stroke_fill=resolve_rgba(
            overlay_style_value(config, "image_stroke_color", default_stroke_color),
            default_stroke_color,
        ),
    )


def create_text_overlay_image(
    draft_name: str,
    draft_width: int,
    draft_height: int,
    title_text: str,
    channel_text: str,
    title_config: dict[str, Any],
    channel_config: dict[str, Any],
    overlay_config: dict[str, Any],
) -> Path:
    raw_title_lines = [line.strip() for line in title_text.splitlines() if line.strip()]
    title_line1 = raw_title_lines[0] if raw_title_lines else ""
    title_line2 = " ".join(raw_title_lines[1:]) if len(raw_title_lines) > 1 else ""
    title_common_config = common_title_render_config(title_config)
    title_line1_config = merge_text_render_config(
        {
            "font_path": title_config.get("font_path"),
            "box": TITLE_LINE1_IMAGE_BOX,
            "max_chars": TITLE_LINE1_IMAGE_MAX_CHARS,
            "image_font_size": TITLE_LINE1_IMAGE_FONT_SIZE,
            "min_image_font_size": TITLE_LINE1_IMAGE_MIN_FONT_SIZE,
            "image_color": TITLE_LINE1_IMAGE_COLOR,
            "image_align": "left",
            "image_valign": "center",
        },
        deep_merge(title_common_config, safe_dict(title_config.get("line1"))),
    )
    title_line2_config = merge_text_render_config(
        {
            "font_path": title_config.get("font_path"),
            "box": TITLE_LINE2_IMAGE_BOX,
            "max_chars": TITLE_LINE2_IMAGE_MAX_CHARS,
            "image_font_size": TITLE_LINE2_IMAGE_FONT_SIZE,
            "min_image_font_size": TITLE_LINE2_IMAGE_MIN_FONT_SIZE,
            "image_color": TITLE_LINE2_IMAGE_COLOR,
            "image_align": "center",
            "image_valign": "center",
            "image_stroke_width": TITLE_LINE2_IMAGE_STROKE_WIDTH,
            "image_stroke_color": TITLE_LINE2_IMAGE_STROKE_COLOR,
        },
        deep_merge(title_common_config, safe_dict(title_config.get("line2"))),
    )
    title_line1_display = compact_title_display_line(
        title_line1,
        positive_int(overlay_style_value(title_line1_config, "max_chars", TITLE_LINE1_IMAGE_MAX_CHARS), TITLE_LINE1_IMAGE_MAX_CHARS),
    )
    title_line2_display = compact_title_display_line(
        title_line2,
        positive_int(overlay_style_value(title_line2_config, "max_chars", TITLE_LINE2_IMAGE_MAX_CHARS), TITLE_LINE2_IMAGE_MAX_CHARS),
    )
    channel_box = resolve_box(overlay_style_value(channel_config, "box", CHANNEL_IMAGE_BOX), CHANNEL_IMAGE_BOX)
    channel_font_size = positive_int(
        overlay_style_value(channel_config, "image_font_size", CHANNEL_IMAGE_FONT_SIZE),
        CHANNEL_IMAGE_FONT_SIZE,
    )
    channel_min_font_size = positive_int(
        overlay_style_value(channel_config, "min_image_font_size", CHANNEL_IMAGE_MIN_FONT_SIZE),
        CHANNEL_IMAGE_MIN_FONT_SIZE,
    )
    channel_font_path = resolve_overlay_font_path(channel_config)

    cache_payload = {
        "text_overlay_render_version": TEXT_OVERLAY_RENDER_VERSION,
        "draft_width": draft_width,
        "draft_height": draft_height,
        "title_line1": title_line1,
        "title_line2": title_line2,
        "title_line1_display": title_line1_display,
        "title_line2_display": title_line2_display,
        "channel_text": channel_text,
        "title_line1_config": title_line1_config,
        "title_line2_config": title_line2_config,
        "channel_config": channel_config,
        "channel_box": channel_box,
        "channel_font_size": channel_font_size,
        "channel_min_font_size": channel_min_font_size,
        "overlay_config": overlay_config,
        "channel_font_path": channel_font_path.as_posix(),
    }
    digest = hashlib.md5(json.dumps(cache_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in draft_name)
    output_path = TEXT_OVERLAY_CACHE_DIR / f"{safe_name}_{digest}.png"
    if output_path.exists():
        return output_path

    TEXT_OVERLAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (draft_width, draft_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    channel_color = resolve_rgba(overlay_style_value(channel_config, "image_color", CHANNEL_IMAGE_COLOR), CHANNEL_IMAGE_COLOR)
    channel_stroke_width = max(0, as_int(overlay_style_value(channel_config, "image_stroke_width", 0), 0))
    channel_stroke_fill = resolve_rgba(overlay_style_value(channel_config, "image_stroke_color", "#FFFFFF"), "#FFFFFF")

    render_single_line_overlay(
        draw,
        title_line1,
        title_line1_config,
        default_box=TITLE_LINE1_IMAGE_BOX,
        default_font_size=TITLE_LINE1_IMAGE_FONT_SIZE,
        default_min_font_size=TITLE_LINE1_IMAGE_MIN_FONT_SIZE,
        default_color=TITLE_LINE1_IMAGE_COLOR,
        default_max_chars=TITLE_LINE1_IMAGE_MAX_CHARS,
        default_align="left",
    )
    render_single_line_overlay(
        draw,
        title_line2,
        title_line2_config,
        default_box=TITLE_LINE2_IMAGE_BOX,
        default_font_size=TITLE_LINE2_IMAGE_FONT_SIZE,
        default_min_font_size=TITLE_LINE2_IMAGE_MIN_FONT_SIZE,
        default_color=TITLE_LINE2_IMAGE_COLOR,
        default_max_chars=TITLE_LINE2_IMAGE_MAX_CHARS,
        default_align="center",
        default_stroke_width=TITLE_LINE2_IMAGE_STROKE_WIDTH,
        default_stroke_color=TITLE_LINE2_IMAGE_STROKE_COLOR,
    )

    if channel_text:
        channel_font, channel_line_gap = fit_overlay_font(
            draw,
            [channel_text],
            channel_font_path,
            channel_box,
            channel_font_size,
            channel_min_font_size,
            line_gap_ratio=0.0,
        )
        draw_text_block(
            draw,
            [channel_text],
            channel_box,
            channel_font,
            channel_color,
            align=str(overlay_style_value(channel_config, "image_align", "center")),
            valign=str(overlay_style_value(channel_config, "image_valign", "center")),
            line_gap=channel_line_gap,
            stroke_width=channel_stroke_width,
            stroke_fill=channel_stroke_fill,
        )

    image.save(output_path)
    return output_path


def build_video_meta_value(video_material: dict[str, Any], now_sec: int, now_us: int, fallback_path: Path) -> dict[str, Any]:
    material_path = Path(video_material.get("path") or fallback_path)
    local_material_id = (
        video_material.get("local_material_id")
        or video_material.get("material_id")
        or video_material.get("id")
        or ""
    )
    material_type = str(video_material.get("type") or "video")

    return {
        "ai_group_type": "",
        "create_time": now_sec,
        "duration": int(video_material.get("duration") or 0),
        "enter_from": 0,
        "extra_info": video_material.get("material_name") or material_path.name,
        "file_Path": material_path.as_posix(),
        "height": int(video_material.get("height") or 0),
        "id": local_material_id,
        "import_time": now_sec,
        "import_time_ms": now_us,
        "item_source": 1,
        "md5": "",
        "metetype": material_type,
        "roughcut_time_range": {"duration": int(video_material.get("duration") or 0), "start": 0},
        "sub_time_range": {"duration": -1, "start": -1},
        "type": 0,
        "width": int(video_material.get("width") or 0),
    }


def update_folder_meta(
    draft_path: Path,
    draft_id: str,
    draft_name: str,
    duration_us: int,
    video_materials: list[dict],
    audio_materials: list[dict],
    material_paths: list[Path],
    source_video: Path,
) -> None:
    now_sec = int(time.time())
    now_us = int(time.time() * 1_000_000)
    file_size = sum(path.stat().st_size for path in material_paths if path.exists())

    with open(draft_path / "draft_meta_info.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta["draft_id"] = draft_id
    meta["draft_name"] = draft_name
    meta["draft_fold_path"] = draft_path.as_posix()
    meta["draft_root_path"] = DRAFT_ROOT.as_posix()
    meta["draft_cover"] = ""
    meta["draft_new_version"] = ""
    meta["draft_is_invisible"] = False
    meta["draft_timeline_materials_size_"] = file_size
    meta["tm_draft_create"] = now_us
    meta["tm_draft_modified"] = now_us
    meta["tm_duration"] = duration_us

    audio_meta_values = []
    for audio in audio_materials:
        audio_meta_values.append(
            {
                "ai_group_type": "",
                "create_time": now_sec,
                "duration": int(audio.get("duration") or 0),
                "enter_from": 0,
                "extra_info": audio.get("name") or Path(audio.get("path", "")).name,
                "file_Path": Path(audio.get("path", "")).as_posix(),
                "height": 0,
                "id": audio.get("id", ""),
                "import_time": now_sec,
                "import_time_ms": now_us,
                "item_source": 1,
                "md5": "",
                "metetype": "audio",
                "roughcut_time_range": {"duration": int(audio.get("duration") or 0), "start": 0},
                "sub_time_range": {"duration": -1, "start": -1},
                "type": 1,
                "width": 0,
            }
        )

    meta["draft_materials"] = [
        {
            "type": 0,
            "value": [
                build_video_meta_value(video_material, now_sec, now_us, source_video)
                for video_material in video_materials
            ],
        },
        {"type": 1, "value": audio_meta_values},
        {"type": 2, "value": []},
        {"type": 3, "value": []},
        {"type": 6, "value": []},
        {"type": 7, "value": []},
        {"type": 8, "value": []},
    ]

    with open(draft_path / "draft_meta_info.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)


def update_root_meta(draft_path: Path, draft_id: str, draft_name: str, duration_us: int, material_paths: list[Path]) -> None:
    now_us = int(time.time() * 1_000_000)
    file_size = sum(path.stat().st_size for path in material_paths if path.exists())

    with open(ROOT_META_PATH, "r", encoding="utf-8") as f:
        root_meta = json.load(f)

    entries = root_meta.get("all_draft_store", [])
    entries = [entry for entry in entries if entry.get("draft_fold_path") != draft_path.as_posix()]
    entries.append(
        {
            "cloud_draft_cover": False,
            "cloud_draft_sync": False,
            "draft_cloud_last_action_download": False,
            "draft_cloud_purchase_info": "",
            "draft_cloud_template_id": "",
            "draft_cloud_tutorial_info": "",
            "draft_cloud_videocut_purchase_info": "",
            "draft_cover": "",
            "draft_fold_path": draft_path.as_posix(),
            "draft_id": draft_id,
            "draft_is_ai_shorts": False,
            "draft_is_cloud_temp_draft": False,
            "draft_is_invisible": False,
            "draft_is_web_article_video": False,
            "draft_json_file": (draft_path / "draft_content.json").as_posix(),
            "draft_name": draft_name,
            "draft_new_version": "",
            "draft_root_path": DRAFT_ROOT.as_posix(),
            "draft_timeline_materials_size": file_size,
            "draft_type": "",
            "draft_web_article_video_enter_from": "",
            "streaming_edit_draft_ready": True,
            "tm_draft_cloud_completed": "",
            "tm_draft_cloud_entry_id": -1,
            "tm_draft_cloud_modified": 0,
            "tm_draft_cloud_parent_entry_id": -1,
            "tm_draft_cloud_space_id": -1,
            "tm_draft_cloud_user_id": -1,
            "tm_draft_create": now_us,
            "tm_draft_modified": now_us,
            "tm_draft_removed": 0,
            "tm_duration": duration_us,
        }
    )
    root_meta["all_draft_store"] = entries
    root_meta["draft_ids"] = len(entries)

    with open(ROOT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(root_meta, f, ensure_ascii=False, indent=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a CapCut draft from a shorts package JSON.")
    parser.add_argument(
        "--source-video",
        type=Path,
        default=Path("지은수대통1회.mp4"),
        help="Source long-form video path.",
    )
    parser.add_argument(
        "--package",
        type=Path,
        default=Path("analysis/jiunsudaetong1/shorts_candidates/final/short_05.json"),
        help="Shorts package JSON path.",
    )
    parser.add_argument(
        "--draft-name",
        type=str,
        default=None,
        help="CapCut draft name override.",
    )
    parser.add_argument(
        "--template-image",
        type=Path,
        default=None,
        help="Overlay image path. Defaults to templates/shorts_template.png or short_templet.png when present.",
    )
    parser.add_argument(
        "--channel-name",
        type=str,
        default=None,
        help="Bottom source label. Auto-detected as YouTube channel name or movie title when possible.",
    )
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
    if not DRAFT_ROOT.exists():
        raise FileNotFoundError(f"CapCut draft root not found: {DRAFT_ROOT}")

    source_duration_us, source_width, source_height = load_media_info(source_video)
    if source_duration_us <= 0 or source_width <= 0 or source_height <= 0:
        raise RuntimeError("Source video media info could not be parsed correctly.")

    package = json.loads(package_path.read_text(encoding="utf-8"))
    capcut_config = safe_dict(package.get("capcut"))
    draft_config = safe_dict(capcut_config.get("draft"))
    title_config = deep_merge(
        {"font": TITLE_FONT, "border": {"color": "#111111", "width": 32, "alpha": 1.0}},
        safe_dict(capcut_config.get("title")),
    )
    point_config = deep_merge(
        {"font": POINT_FONT},
        safe_dict(capcut_config.get("point")),
    )
    narration_config = safe_dict(capcut_config.get("narration"))
    template_config = deep_merge(
        safe_dict(capcut_config.get("template")),
        safe_dict(capcut_config.get("template_image")),
    )
    text_overlay_config = safe_dict(capcut_config.get("text_overlay"))
    channel_config = deep_merge(
        {"font": CHANNEL_FONT, "border": {"color": "#111111", "width": 24, "alpha": 1.0}},
        safe_dict(capcut_config.get("channel")),
    )
    title_text = f"{package['title_line1']}\n{package['title_line2']}"
    title_line1_display = compact_title_display_line(str(package["title_line1"]), TITLE_LINE1_IMAGE_MAX_CHARS)
    title_line2_display = compact_title_display_line(str(package["title_line2"]), TITLE_LINE2_IMAGE_MAX_CHARS)
    draft_name = build_draft_name(source_video, package, args.draft_name)
    draft_width = positive_int(draft_config.get("width"), DRAFT_WIDTH)
    draft_height = positive_int(draft_config.get("height"), DRAFT_HEIGHT)
    draft_fps = positive_int(draft_config.get("fps"), DRAFT_FPS)
    video_volume = as_float(draft_config.get("video_volume"), VIDEO_VOLUME)
    title_duration_s = positive_float(draft_config.get("title_duration_sec"), TITLE_DURATION_S)
    title_full_duration = as_bool(draft_config.get("title_full_duration"), TITLE_FULL_DURATION)
    narration_voice = str(narration_config.get("voice") or TTS_VOICE)
    narration_speed = positive_float(narration_config.get("speed"), TTS_SPEED)
    narration_volume = as_float(narration_config.get("volume"), NARRATION_VOLUME)
    template_enabled = as_bool(template_config.get("enabled"), True)
    template_image_path = (
        resolve_template_image_path(template_config, args.template_image)
        if template_enabled
        else None
    )
    if template_image_path and not template_image_path.exists():
        raise FileNotFoundError(f"Template image not found: {template_image_path}")

    channel_name = infer_channel_name(package_path, package, args.channel_name)
    channel_text = format_channel_text(channel_name, channel_config)
    channel_enabled = as_bool(channel_config.get("enabled"), True)
    text_overlay_enabled = as_bool(text_overlay_config.get("enabled"), False)

    draft_folder = cc.DraftFolder(str(DRAFT_ROOT))
    if draft_folder.has_draft(draft_name):
        draft_folder.remove(draft_name)

    script = draft_folder.create_draft(draft_name, width=draft_width, height=draft_height, fps=draft_fps)
    script.add_track(cc.TrackType.video, "main")
    if template_image_path:
        script.add_track(cc.TrackType.video, "template", relative_index=20)
    if text_overlay_enabled:
        script.add_track(cc.TrackType.video, "text_overlay", relative_index=30)
    else:
        script.add_track(cc.TrackType.text, "title_line1", relative_index=30)
        script.add_track(cc.TrackType.text, "title_line2", relative_index=31)
        if channel_enabled and channel_text:
            script.add_track(cc.TrackType.text, "channel", relative_index=32)
    script.add_track(cc.TrackType.text, "point", relative_index=5)

    narration_items = package.get("narration", [])
    if narration_items:
        script.add_track(cc.TrackType.audio, "narration")

    video = cc.VideoMaterial(str(source_video))
    script.add_material(video)
    title_default_style = cc.TextStyle(size=TITLE_SIZE, bold=True, align=1, auto_wrapping=True)
    point_default_style = cc.TextStyle(size=POINT_SIZE, bold=True, align=1, auto_wrapping=True)
    channel_default_style = cc.TextStyle(size=CHANNEL_SIZE, bold=True, align=1, auto_wrapping=True)
    title_default_clip = cc.ClipSettings(transform_y=TITLE_Y)
    point_default_clip = cc.ClipSettings(transform_y=POINT_Y)
    channel_default_clip = cc.ClipSettings(transform_y=CHANNEL_Y)
    material_patches: dict[str, dict[str, Any]] = {}
    content_patches: dict[str, dict[str, Any]] = {}

    current_start_us = 0
    for clip in package.get("source_clips", []):
        start_s = float(clip["source_start"])
        end_s = float(clip["source_end"])
        duration_s = max(0.0, end_s - start_s)
        duration_us = us(duration_s)
        script.add_segment(
            cc.VideoSegment(
                video,
                cc.Timerange(current_start_us, duration_us),
                source_timerange=cc.Timerange(us(start_s), duration_us),
                volume=video_volume,
            ),
            track_name="main",
        )
        current_start_us += duration_us

    template_material_path: Path | None = None
    if template_image_path and current_start_us:
        template_material = cc.VideoMaterial(str(template_image_path))
        template_clip = build_clip_settings(template_config, cc.ClipSettings())
        script.add_segment(
            cc.VideoSegment(
                template_material,
                cc.Timerange(0, current_start_us),
                source_timerange=cc.Timerange(0, current_start_us),
                volume=0.0,
                clip_settings=template_clip,
            ),
            track_name="template",
        )
        template_material_path = template_image_path

    text_overlay_path: Path | None = None
    if text_overlay_enabled and current_start_us:
        text_overlay_path = create_text_overlay_image(
            draft_name,
            draft_width,
            draft_height,
            title_text,
            channel_text if channel_enabled else "",
            title_config,
            channel_config,
            text_overlay_config,
        )
        text_overlay_material = cc.VideoMaterial(str(text_overlay_path))
        script.add_segment(
            cc.VideoSegment(
                text_overlay_material,
                cc.Timerange(0, current_start_us),
                source_timerange=cc.Timerange(0, current_start_us),
                volume=0.0,
            ),
            track_name="text_overlay",
        )

    generated_audio_paths: list[Path] = []
    if narration_items:
        client = get_openai_client()
        for index, narration in enumerate(narration_items, start=1):
            item_config = safe_dict(narration.get("capcut"))
            item_voice = str(item_config.get("voice") or narration.get("voice") or narration_voice)
            item_speed = positive_float(item_config.get("speed") or narration.get("speed"), narration_speed)
            item_volume = as_float(item_config.get("volume") or narration.get("volume"), narration_volume)
            audio_path = generate_narration_audio(
                narration["text"],
                index,
                client,
                voice=item_voice,
                speed=item_speed,
            )
            audio_duration_us = load_duration_us(audio_path)
            audio_material = cc.AudioMaterial(str(audio_path))
            script.add_material(audio_material)
            script.add_segment(
                cc.AudioSegment(
                    audio_material,
                    cc.Timerange(us(float(narration["target_start"])), audio_duration_us),
                    source_timerange=cc.Timerange(0, audio_duration_us),
                    volume=item_volume,
                ),
                track_name="narration",
            )
            generated_audio_paths.append(audio_path)

    if title_full_duration and current_start_us:
        title_duration_us = current_start_us
    else:
        title_duration_us = min(us(title_duration_s), current_start_us) if current_start_us else us(title_duration_s)
    if not text_overlay_enabled:
        title_line1_config = build_reference_capcut_text_config(
            transform_x=TITLE_LINE1_CAPCUT_X,
            transform_y=TITLE_LINE1_CAPCUT_Y,
            fill_color=REFERENCE_CAPCUT_TITLE_LINE1_COLOR,
            stroke_color=REFERENCE_CAPCUT_TITLE_LINE1_STROKE,
            overrides=safe_dict(title_config.get("line1_text")),
        )
        title_line1_segment = build_text_segment(
            title_line1_display,
            cc.Timerange(0, title_duration_us),
            title_line1_config,
            fallback_style=cc.TextStyle(size=REFERENCE_CAPCUT_TEXT_SIZE, align=1, auto_wrapping=False),
            fallback_clip=cc.ClipSettings(transform_x=TITLE_LINE1_CAPCUT_X, transform_y=TITLE_LINE1_CAPCUT_Y),
        )
        script.add_segment(title_line1_segment, track_name="title_line1")
        register_reference_capcut_text_patch(
            material_patches,
            content_patches,
            title_line1_segment,
            title_line1_display,
            fill_color=REFERENCE_CAPCUT_TITLE_LINE1_COLOR,
            stroke_color=REFERENCE_CAPCUT_TITLE_LINE1_STROKE,
            overrides=safe_dict(title_config.get("line1_text")),
        )

        title_line2_config = build_reference_capcut_text_config(
            transform_x=TITLE_LINE2_CAPCUT_X,
            transform_y=TITLE_LINE2_CAPCUT_Y,
            fill_color=REFERENCE_CAPCUT_TITLE_LINE2_COLOR,
            stroke_color=REFERENCE_CAPCUT_TITLE_LINE2_STROKE,
            overrides=safe_dict(title_config.get("line2_text")),
        )
        title_line2_segment = build_text_segment(
            title_line2_display,
            cc.Timerange(0, title_duration_us),
            title_line2_config,
            fallback_style=cc.TextStyle(size=REFERENCE_CAPCUT_TEXT_SIZE, align=1, auto_wrapping=False),
            fallback_clip=cc.ClipSettings(transform_x=TITLE_LINE2_CAPCUT_X, transform_y=TITLE_LINE2_CAPCUT_Y),
        )
        script.add_segment(title_line2_segment, track_name="title_line2")
        register_reference_capcut_text_patch(
            material_patches,
            content_patches,
            title_line2_segment,
            title_line2_display,
            fill_color=REFERENCE_CAPCUT_TITLE_LINE2_COLOR,
            stroke_color=REFERENCE_CAPCUT_TITLE_LINE2_STROKE,
            overrides=safe_dict(title_config.get("line2_text")),
        )

    if not text_overlay_enabled and channel_enabled and channel_text and current_start_us:
        reference_channel_config = build_reference_capcut_text_config(
            transform_x=CHANNEL_CAPCUT_X,
            transform_y=CHANNEL_CAPCUT_Y,
            fill_color=REFERENCE_CAPCUT_CHANNEL_COLOR,
            stroke_color=REFERENCE_CAPCUT_CHANNEL_STROKE,
            overrides=safe_dict(channel_config.get("text")),
        )
        channel_segment = build_text_segment(
            channel_text,
            cc.Timerange(0, current_start_us),
            reference_channel_config,
            fallback_style=cc.TextStyle(size=REFERENCE_CAPCUT_TEXT_SIZE, align=1, auto_wrapping=False),
            fallback_clip=cc.ClipSettings(transform_x=CHANNEL_CAPCUT_X, transform_y=CHANNEL_CAPCUT_Y),
        )
        script.add_segment(channel_segment, track_name="channel")
        register_reference_capcut_text_patch(
            material_patches,
            content_patches,
            channel_segment,
            channel_text,
            fill_color=REFERENCE_CAPCUT_CHANNEL_COLOR,
            stroke_color=REFERENCE_CAPCUT_CHANNEL_STROKE,
            overrides=safe_dict(channel_config.get("text")),
        )

    for caption in package.get("point_captions", []):
        start_s = float(caption["target_start"])
        end_s = float(caption["target_end"])
        duration_s = max(0.0, end_s - start_s)
        caption_config = deep_merge(point_config, safe_dict(caption.get("capcut")))
        caption_segment = build_text_segment(
            caption["text"],
            cc.Timerange(us(start_s), us(duration_s)),
            caption_config,
            fallback_style=point_default_style,
            fallback_clip=point_default_clip,
        )
        script.add_segment(caption_segment, track_name="point")
        collect_text_patches(material_patches, content_patches, caption_segment, caption_config)

    script.save()

    draft_path = DRAFT_ROOT / draft_name
    patched_text_materials = apply_text_material_patches(draft_path, material_patches, content_patches)
    with open(draft_path / "draft_content.json", "r", encoding="utf-8") as f:
        content = json.load(f)

    video_materials = content["materials"]["videos"]
    audio_materials = content["materials"].get("audios", [])
    duration_us = int(content["duration"])
    draft_id = str(uuid.uuid4()).upper()
    material_paths = [
        path
        for path in [source_video, template_material_path, text_overlay_path, *generated_audio_paths]
        if path is not None
    ]

    update_folder_meta(
        draft_path=draft_path,
        draft_id=draft_id,
        draft_name=draft_name,
        duration_us=duration_us,
        video_materials=video_materials,
        audio_materials=audio_materials,
        material_paths=material_paths,
        source_video=source_video,
    )
    update_root_meta(
        draft_path=draft_path,
        draft_id=draft_id,
        draft_name=draft_name,
        duration_us=duration_us,
        material_paths=material_paths,
    )

    print(f"[capcut] created_draft={draft_path}", flush=True)
    print(f"[capcut] source_video={source_video}", flush=True)
    print(f"[capcut] package={package_path}", flush=True)
    print(f"[capcut] draft_name={draft_name}", flush=True)
    print(f"[capcut] title={title_text}", flush=True)
    print(f"[capcut] title_display={title_line1_display} / {title_line2_display}", flush=True)
    print(f"[capcut] title_full_duration={title_full_duration}", flush=True)
    print(f"[capcut] template_image={template_image_path or ''}", flush=True)
    print(f"[capcut] text_overlay={text_overlay_path or ''}", flush=True)
    print(f"[capcut] text_mode={'png_overlay' if text_overlay_enabled else 'capcut_text_tracks'}", flush=True)
    print(f"[capcut] source_label={channel_name}", flush=True)
    print(f"[capcut] clips={len(package.get('source_clips', []))}", flush=True)
    print(f"[capcut] point_captions={len(package.get('point_captions', []))}", flush=True)
    print(f"[capcut] narration_items={len(narration_items)}", flush=True)
    print(f"[capcut] patched_text_materials={patched_text_materials}", flush=True)


if __name__ == "__main__":
    main()
