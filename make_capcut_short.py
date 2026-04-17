import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import pycapcut as cc
from dotenv import load_dotenv
from openai import OpenAI
from pymediainfo import MediaInfo


DRAFT_ROOT = Path(r"C:\Users\user\AppData\Local\CapCut\User Data\Projects\com.lveditor.draft")
ROOT_META_PATH = DRAFT_ROOT / "root_meta_info.json"
SOURCE_VIDEO = Path(r"C:\Users\user\Downloads\final_longform_1771901754.mp4")
DRAFT_NAME = "codex_short_poc_02"
BASE_DIR = Path(__file__).resolve().parent
SHORTS_ENV_PATH = BASE_DIR.parent / "auto_Youtube" / "shorts" / ".env"
AUDIO_CACHE_DIR = BASE_DIR / "generated_audio"
TTS_VOICE = "onyx"
TTS_SPEED = 1.2
VIDEO_VOLUME = 0.35

# Arbitrary proof-of-concept cuts from the source video.
SEGMENTS = [
    {"source_start_s": 20.0, "duration_s": 8.0},
    {"source_start_s": 95.0, "duration_s": 7.0},
    {"source_start_s": 205.0, "duration_s": 10.0},
]

POINT_CAPTION = {
    "text": "자막TEST",
    "start_s": 11.0,
    "duration_s": 2.5,
}

NARRATION_SEGMENTS = [
    {
        "text": "회장딸이 젊은 남자와 데이트하는 걸 목격했는데.",
        "start_s": 0.0,
    },
    {
        "text": "다음날 회장과 있는 걸 봤다고 고백하는 임창정.",
        "start_s": 15.5,
    },
]


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


def generate_narration_audio(text: str, index: int, client: OpenAI) -> Path:
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    output_path = AUDIO_CACHE_DIR / f"narration_{index:02d}_{digest}.mp3"
    if output_path.exists():
        return output_path

    response = client.audio.speech.create(
        model="tts-1",
        voice=TTS_VOICE,
        input=text,
        speed=TTS_SPEED,
    )
    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path


def update_folder_meta(
    draft_path: Path,
    draft_id: str,
    duration_us: int,
    video_material: dict,
    audio_materials: list[dict],
    material_paths: list[Path],
) -> None:
    now_sec = int(time.time())
    now_us = int(time.time() * 1_000_000)
    file_size = sum(path.stat().st_size for path in material_paths if path.exists())

    with open(draft_path / "draft_meta_info.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta["draft_id"] = draft_id
    meta["draft_name"] = DRAFT_NAME
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
                    "extra_info": video_material.get("material_name") or SOURCE_VIDEO.name,
                    "file_Path": Path(video_material.get("path", str(SOURCE_VIDEO))).as_posix(),
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


def update_root_meta(draft_path: Path, draft_id: str, duration_us: int, material_paths: list[Path]) -> None:
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
            "draft_name": DRAFT_NAME,
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


def main() -> None:
    if not SOURCE_VIDEO.exists():
        raise FileNotFoundError(f"Source video not found: {SOURCE_VIDEO}")
    if not DRAFT_ROOT.exists():
        raise FileNotFoundError(f"CapCut draft root not found: {DRAFT_ROOT}")

    source_duration_us, source_width, source_height = load_media_info(SOURCE_VIDEO)
    if source_duration_us <= 0 or source_width <= 0 or source_height <= 0:
        raise RuntimeError("Source video media info could not be parsed correctly.")

    draft_folder = cc.DraftFolder(str(DRAFT_ROOT))
    openai_client = get_openai_client()

    if draft_folder.has_draft(DRAFT_NAME):
        draft_folder.remove(DRAFT_NAME)

    script = draft_folder.create_draft(DRAFT_NAME, width=1080, height=1920, fps=30)
    script.add_track(cc.TrackType.video, "main")
    script.add_track(cc.TrackType.audio, "narration")
    script.add_track(cc.TrackType.text, "title")
    script.add_track(cc.TrackType.text, "point")

    video = cc.VideoMaterial(str(SOURCE_VIDEO))
    script.add_material(video)

    current_start_us = 0
    for segment in SEGMENTS:
        duration_us = us(segment["duration_s"])
        script.add_segment(
            cc.VideoSegment(
                video,
                cc.Timerange(current_start_us, duration_us),
                source_timerange=cc.Timerange(us(segment["source_start_s"]), duration_us),
                volume=VIDEO_VOLUME,
            ),
            track_name="main",
        )
        current_start_us += duration_us

    generated_audio_paths: list[Path] = []
    for index, narration in enumerate(NARRATION_SEGMENTS, start=1):
        audio_path = generate_narration_audio(narration["text"], index, openai_client)
        audio_duration_us = load_duration_us(audio_path)
        audio_material = cc.AudioMaterial(str(audio_path))
        script.add_material(audio_material)
        script.add_segment(
            cc.AudioSegment(
                audio_material,
                cc.Timerange(us(narration["start_s"]), audio_duration_us),
                source_timerange=cc.Timerange(0, audio_duration_us),
                volume=1.0,
            ),
            track_name="narration",
        )
        generated_audio_paths.append(audio_path)

    script.add_segment(
        cc.TextSegment("POC Short\nArbitrary Cut Test", cc.Timerange(0, us(4.0))),
        track_name="title",
    )
    script.add_segment(
        cc.TextSegment(
            POINT_CAPTION["text"],
            cc.Timerange(us(POINT_CAPTION["start_s"]), us(POINT_CAPTION["duration_s"])),
            style=cc.TextStyle(size=10.0, bold=True),
            clip_settings=cc.ClipSettings(transform_y=0.1),
        ),
        track_name="point",
    )
    script.save()

    draft_path = DRAFT_ROOT / DRAFT_NAME
    with open(draft_path / "draft_content.json", "r", encoding="utf-8") as f:
        content = json.load(f)

    video_material = content["materials"]["videos"][0]
    audio_materials = content["materials"].get("audios", [])
    duration_us = int(content["duration"])
    draft_id = str(uuid.uuid4()).upper()
    material_paths = [SOURCE_VIDEO, *generated_audio_paths]

    update_folder_meta(
        draft_path=draft_path,
        draft_id=draft_id,
        duration_us=duration_us,
        video_material=video_material,
        audio_materials=audio_materials,
        material_paths=material_paths,
    )
    update_root_meta(
        draft_path=draft_path,
        draft_id=draft_id,
        duration_us=duration_us,
        material_paths=material_paths,
    )

    print(f"Created draft: {draft_path}")
    print(f"Source video: {SOURCE_VIDEO}")
    print(f"Segments: {SEGMENTS}")
    print(f"Narration files: {[str(path) for path in generated_audio_paths]}")


if __name__ == "__main__":
    main()
