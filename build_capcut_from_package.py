import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import pycapcut as cc
from dotenv import load_dotenv
from openai import OpenAI
from pymediainfo import MediaInfo


DRAFT_ROOT = Path(r"C:\Users\user\AppData\Local\CapCut\User Data\Projects\com.lveditor.draft")
ROOT_META_PATH = DRAFT_ROOT / "root_meta_info.json"
BASE_DIR = Path(__file__).resolve().parent
SHORTS_ENV_PATH = BASE_DIR.parent / "auto_Youtube" / "shorts" / ".env"
AUDIO_CACHE_DIR = BASE_DIR / "generated_audio"
DRAFT_WIDTH = 1080
DRAFT_HEIGHT = 1920
DRAFT_FPS = 30
TTS_VOICE = "onyx"
TTS_SPEED = 1.15
TITLE_DURATION_S = 4.5
TITLE_SIZE = 12.0
POINT_SIZE = 10.0
POINT_Y = 0.1
VIDEO_VOLUME = 1.0
NARRATION_VOLUME = 1.0


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


def resolve_hex_color(value: Any, default: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return default
    raw = value.strip()
    if not raw.startswith("#"):
        raw = "#" + raw
    return raw


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


def update_folder_meta(
    draft_path: Path,
    draft_id: str,
    draft_name: str,
    duration_us: int,
    video_material: dict,
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

    video_local_material_id = (
        video_material.get("local_material_id")
        or video_material.get("material_id")
        or video_material.get("id")
        or ""
    )

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
                {
                    "ai_group_type": "",
                    "create_time": now_sec,
                    "duration": int(video_material.get("duration") or 0),
                    "enter_from": 0,
                    "extra_info": video_material.get("material_name") or source_video.name,
                    "file_Path": Path(video_material.get("path", str(source_video))).as_posix(),
                    "height": int(video_material.get("height") or 0),
                    "id": video_local_material_id,
                    "import_time": now_sec,
                    "import_time_ms": now_us,
                    "item_source": 1,
                    "md5": "",
                    "metetype": "video",
                    "roughcut_time_range": {"duration": int(video_material.get("duration") or 0), "start": 0},
                    "sub_time_range": {"duration": -1, "start": -1},
                    "type": 0,
                    "width": int(video_material.get("width") or 0),
                }
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
    return parser


def main() -> None:
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
    title_config = safe_dict(capcut_config.get("title"))
    point_config = safe_dict(capcut_config.get("point"))
    narration_config = safe_dict(capcut_config.get("narration"))
    draft_name = build_draft_name(source_video, package, args.draft_name)
    draft_width = positive_int(draft_config.get("width"), DRAFT_WIDTH)
    draft_height = positive_int(draft_config.get("height"), DRAFT_HEIGHT)
    draft_fps = positive_int(draft_config.get("fps"), DRAFT_FPS)
    video_volume = as_float(draft_config.get("video_volume"), VIDEO_VOLUME)
    title_duration_s = positive_float(draft_config.get("title_duration_sec"), TITLE_DURATION_S)
    narration_voice = str(narration_config.get("voice") or TTS_VOICE)
    narration_speed = positive_float(narration_config.get("speed"), TTS_SPEED)
    narration_volume = as_float(narration_config.get("volume"), NARRATION_VOLUME)

    draft_folder = cc.DraftFolder(str(DRAFT_ROOT))
    if draft_folder.has_draft(draft_name):
        draft_folder.remove(draft_name)

    script = draft_folder.create_draft(draft_name, width=draft_width, height=draft_height, fps=draft_fps)
    script.add_track(cc.TrackType.video, "main")
    script.add_track(cc.TrackType.text, "title")
    script.add_track(cc.TrackType.text, "point")

    narration_items = package.get("narration", [])
    if narration_items:
        script.add_track(cc.TrackType.audio, "narration")

    video = cc.VideoMaterial(str(source_video))
    script.add_material(video)
    title_default_style = cc.TextStyle(size=TITLE_SIZE, bold=True)
    point_default_style = cc.TextStyle(size=POINT_SIZE, bold=True)
    title_default_clip = cc.ClipSettings()
    point_default_clip = cc.ClipSettings(transform_y=POINT_Y)
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

    title_text = f"{package['title_line1']}\n{package['title_line2']}"
    title_duration_us = min(us(title_duration_s), current_start_us) if current_start_us else us(title_duration_s)
    title_segment = build_text_segment(
        title_text,
        cc.Timerange(0, title_duration_us),
        title_config,
        fallback_style=title_default_style,
        fallback_clip=title_default_clip,
    )
    script.add_segment(title_segment, track_name="title")
    collect_text_patches(material_patches, content_patches, title_segment, title_config)

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

    video_material = content["materials"]["videos"][0]
    audio_materials = content["materials"].get("audios", [])
    duration_us = int(content["duration"])
    draft_id = str(uuid.uuid4()).upper()
    material_paths = [source_video, *generated_audio_paths]

    update_folder_meta(
        draft_path=draft_path,
        draft_id=draft_id,
        draft_name=draft_name,
        duration_us=duration_us,
        video_material=video_material,
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
    print(f"[capcut] clips={len(package.get('source_clips', []))}", flush=True)
    print(f"[capcut] point_captions={len(package.get('point_captions', []))}", flush=True)
    print(f"[capcut] narration_items={len(narration_items)}", flush=True)
    print(f"[capcut] patched_text_materials={patched_text_materials}", flush=True)


if __name__ == "__main__":
    main()
