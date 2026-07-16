from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

from env_loader import load_project_env


BASE_DIR = Path(__file__).resolve().parent
YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
USER_AGENT = "Long2Shorts/0.1"
DEFAULT_CHUNK_SECONDS = 600
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm"}
TIME_RE = re.compile(r"(?<!\d)(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{2})(?!\d)")
TOPIC_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]{2,}")
VTT_TIME_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}[.,]\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2}[.,]\d{3})"
)
REACTION_KEYWORDS = [
    "ㅋㅋ",
    "ㅎㅎ",
    "웃",
    "레전드",
    "미쳤",
    "소름",
    "대박",
    "개웃",
    "터짐",
    "공감",
    "명장면",
    "핵심",
    "팩폭",
    "반전",
    "통쾌",
    "슬프",
    "눈물",
    "crazy",
    "legend",
    "lol",
    "lmao",
]
COMMENT_TOPIC_STOPWORDS = {
    "ㅋㅋ",
    "ㅎㅎ",
    "ㅠㅠ",
    "영상",
    "오늘",
    "너무",
    "진짜",
    "정말",
    "완전",
    "항상",
    "보고",
    "보는",
    "있는",
    "없는",
    "해서",
    "하고",
    "하면",
    "같이",
    "이번",
    "다음",
    "우리",
    "제가",
    "저는",
    "근데",
    "그리고",
    "미숙",
    "언니",
    "배우",
    "배우님",
    "피디",
    "PD",
}
COMMENT_TOPIC_ALIASES = {
    "나쵸": "나초",
    "강쥐": "강아지",
    "댕댕": "강아지",
    "댕댕이": "강아지",
    "멍멍": "강아지",
    "멍뭉": "강아지",
    "반려견": "강아지",
}
COMMENT_TOPIC_SUFFIXES = ("으로", "에서", "에게", "한테", "처럼", "까지", "부터", "은", "는", "이", "가", "을", "를", "도", "만", "에", "와", "과", "야", "아")


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect YouTube metadata and comment reaction signals.")
    parser.add_argument("--url", required=True, help="YouTube watch/shorts URL.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for youtube_context.json.")
    parser.add_argument("--download-dir", type=Path, default=BASE_DIR / "downloads")
    parser.add_argument("--download-video", action="store_true", help="Download the video with yt-dlp.")
    parser.add_argument(
        "--prepare-transcript",
        action="store_true",
        help="Download captions first, or audio+Whisper if captions are unavailable.",
    )
    parser.add_argument(
        "--captions-only",
        action="store_true",
        help="Only use downloadable public captions. Do not download audio or start transcription when captions are unavailable.",
    )
    parser.add_argument(
        "--generate-packages",
        action="store_true",
        help="Generate shorts packages as soon as transcript preparation is ready.",
    )
    parser.add_argument(
        "--benchmark-profile",
        type=Path,
        default=None,
        help="Optional genre scoring profile passed to package generation.",
    )
    parser.add_argument("--max-comments", type=int, default=300)
    parser.add_argument("--api-key", default="", help="YouTube Data API key. Defaults to env.")
    return parser


def extract_video_id(url: str) -> str:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().replace("www.", "")
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
        if video_id:
            return video_id
    if host.endswith("youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
            if video_id:
                return video_id
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
            return parts[1]
    raise ValueError(f"Cannot find a YouTube video id from URL: {url}")


def default_output_dir(video_id: str) -> Path:
    return BASE_DIR / "analysis" / f"youtube_{video_id}"


def get_api_key(explicit: str) -> str:
    if explicit:
        return explicit
    load_project_env()
    return os.environ.get("YOUTUBE_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""


def fetch_json(path: str, params: dict[str, str | int]) -> dict:
    url = f"{YOUTUBE_API_BASE}/{path}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def collect_metadata(video_id: str, api_key: str) -> dict:
    if not api_key:
        return {"available": False, "reason": "YOUTUBE_API_KEY or GOOGLE_API_KEY is not set."}

    data = fetch_json(
        "videos",
        {
            "part": "snippet,statistics,contentDetails",
            "id": video_id,
            "key": api_key,
        },
    )
    items = data.get("items", [])
    if not items:
        return {"available": False, "reason": "Video was not found or is not public."}

    item = items[0]
    snippet = item.get("snippet", {})
    statistics = item.get("statistics", {})
    content_details = item.get("contentDetails", {})
    return {
        "available": True,
        "video_id": video_id,
        "title": snippet.get("title", ""),
        "channel_title": snippet.get("channelTitle", ""),
        "published_at": snippet.get("publishedAt", ""),
        "description": snippet.get("description", ""),
        "duration": content_details.get("duration", ""),
        "view_count": int(statistics.get("viewCount", 0) or 0),
        "like_count": int(statistics.get("likeCount", 0) or 0),
        "comment_count": int(statistics.get("commentCount", 0) or 0),
    }


def collect_comments(video_id: str, api_key: str, max_comments: int) -> tuple[list[dict], str]:
    if not api_key:
        return [], "YOUTUBE_API_KEY or GOOGLE_API_KEY is not set."

    comments: list[dict] = []
    page_token = ""
    reason = ""
    while len(comments) < max_comments:
        params: dict[str, str | int] = {
            "part": "snippet,replies",
            "videoId": video_id,
            "maxResults": min(100, max_comments - len(comments)),
            "order": "relevance",
            "textFormat": "plainText",
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            data = fetch_json("commentThreads", params)
        except Exception as exc:
            reason = str(exc)
            break

        for item in data.get("items", []) or []:
            snippet = item.get("snippet", {})
            top = snippet.get("topLevelComment", {}).get("snippet", {})
            text = str(top.get("textDisplay", "")).strip()
            if not text:
                continue
            comments.append(
                {
                    "id": item.get("id", ""),
                    "text": text,
                    "like_count": int(top.get("likeCount", 0) or 0),
                    "reply_count": int(snippet.get("totalReplyCount", 0) or 0),
                    "published_at": top.get("publishedAt", ""),
                    "updated_at": top.get("updatedAt", ""),
                }
            )
        page_token = data.get("nextPageToken", "")
        if not page_token:
            break
    return comments, reason


def parse_timecodes(text: str) -> list[int]:
    seconds: list[int] = []
    for match in TIME_RE.finditer(text):
        h = int(match.group("h") or 0)
        m = int(match.group("m"))
        s = int(match.group("s"))
        total = h * 3600 + m * 60 + s
        seconds.append(total)
    return seconds


def reaction_keyword_hits(text: str) -> list[str]:
    lowered = text.lower()
    hits = []
    for keyword in REACTION_KEYWORDS:
        if keyword.lower() in lowered:
            hits.append(keyword)
    return hits


def short_text(text: str, max_len: int = 180) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 1].rstrip() + "…"


def comment_reaction_score(comment: dict) -> float:
    hits = reaction_keyword_hits(comment.get("text", ""))
    timecodes = parse_timecodes(comment.get("text", ""))
    return round(
        comment.get("like_count", 0) * 2.0
        + comment.get("reply_count", 0) * 3.0
        + len(timecodes) * 12.0
        + len(hits) * 5.0
        + min(len(comment.get("text", "")) / 90.0, 3.0),
        3,
    )


def normalize_comment_topic(token: str) -> str:
    value = COMMENT_TOPIC_ALIASES.get(token, token)
    for suffix in COMMENT_TOPIC_SUFFIXES:
        if value.endswith(suffix) and len(value) > len(suffix) + 1:
            value = value[: -len(suffix)]
            value = COMMENT_TOPIC_ALIASES.get(value, value)
            break
    if value in COMMENT_TOPIC_STOPWORDS:
        return ""
    if len(value) < 2:
        return ""
    if value.isdigit():
        return ""
    if value in REACTION_KEYWORDS:
        return ""
    return value


def comment_topic_hits(text: str) -> list[str]:
    topics = set()
    for alias, canonical in COMMENT_TOPIC_ALIASES.items():
        if alias in text:
            topics.add(canonical)
    for token in TOPIC_TOKEN_RE.findall(text):
        topic = normalize_comment_topic(token)
        if topic:
            topics.add(topic)
    return sorted(topics)


def analyze_comments(comments: list[dict]) -> dict:
    keyword_counts: dict[str, int] = {}
    topic_buckets: dict[str, dict] = {}
    moment_buckets: dict[int, dict] = {}
    enriched = []

    for comment in comments:
        text = comment.get("text", "")
        timecodes = parse_timecodes(text)
        hits = reaction_keyword_hits(text)
        topics = comment_topic_hits(text)
        score = comment_reaction_score(comment)
        enriched_comment = {
            **comment,
            "timecodes": timecodes,
            "reaction_keywords": hits,
            "comment_topics": topics,
            "reaction_score": score,
        }
        enriched.append(enriched_comment)

        for hit in hits:
            keyword_counts[hit] = keyword_counts.get(hit, 0) + 1

        for topic in topics:
            bucket = topic_buckets.setdefault(
                topic,
                {
                    "topic": topic,
                    "comment_count": 0,
                    "like_count": 0,
                    "reply_count": 0,
                    "reaction_score": 0.0,
                    "timecodes": [],
                    "samples": [],
                },
            )
            bucket["comment_count"] += 1
            bucket["like_count"] += int(comment.get("like_count", 0) or 0)
            bucket["reply_count"] += int(comment.get("reply_count", 0) or 0)
            bucket["reaction_score"] += score
            bucket["timecodes"].extend(timecodes)
            bucket["samples"].append(
                {
                    "text": short_text(text),
                    "like_count": int(comment.get("like_count", 0) or 0),
                    "reply_count": int(comment.get("reply_count", 0) or 0),
                    "timecodes": timecodes,
                    "reaction_keywords": hits,
                }
            )

        for sec in timecodes:
            bucket = int(sec // 10) * 10
            moment = moment_buckets.setdefault(
                bucket,
                {
                    "bucket_start_sec": bucket,
                    "bucket_end_sec": bucket + 10,
                    "comment_count": 0,
                    "like_count": 0,
                    "reply_count": 0,
                    "reaction_score": 0.0,
                    "timecode_seconds": [],
                    "samples": [],
                },
            )
            moment["comment_count"] += 1
            moment["like_count"] += int(comment.get("like_count", 0) or 0)
            moment["reply_count"] += int(comment.get("reply_count", 0) or 0)
            moment["reaction_score"] += score
            moment["timecode_seconds"].append(sec)
            moment["samples"].append(
                {
                    "time_sec": sec,
                    "text": short_text(text),
                    "like_count": int(comment.get("like_count", 0) or 0),
                    "reply_count": int(comment.get("reply_count", 0) or 0),
                    "reaction_keywords": hits,
                }
            )

    moments = []
    for moment in moment_buckets.values():
        times = moment.pop("timecode_seconds")
        moment["representative_time_sec"] = round(sum(times) / len(times), 1) if times else moment["bucket_start_sec"]
        moment["reaction_score"] = round(moment["reaction_score"], 3)
        moment["samples"] = sorted(
            moment["samples"],
            key=lambda item: (item["like_count"], item["reply_count"]),
            reverse=True,
        )[:3]
        moments.append(moment)

    topic_mentions = []
    for topic in topic_buckets.values():
        topic["reaction_score"] = round(topic["reaction_score"], 3)
        topic["audience_score"] = round(
            topic["comment_count"] * 6.0
            + topic["like_count"] * 1.5
            + topic["reply_count"] * 3.0
            + len(topic["timecodes"]) * 10.0,
            3,
        )
        topic["timecodes"] = sorted(set(topic["timecodes"]))
        topic["samples"] = sorted(
            topic["samples"],
            key=lambda item: (item["like_count"], item["reply_count"], len(item.get("timecodes", []))),
            reverse=True,
        )[:5]
        topic_mentions.append(topic)

    return {
        "timecode_moments": sorted(moments, key=lambda item: item["reaction_score"], reverse=True)[:30],
        "top_reaction_comments": sorted(enriched, key=lambda item: item["reaction_score"], reverse=True)[:30],
        "keyword_counts": dict(sorted(keyword_counts.items(), key=lambda item: item[1], reverse=True)),
        "top_comment_topics": sorted(topic_mentions, key=lambda item: item["audience_score"], reverse=True)[:30],
    }


def yt_dlp_command() -> list[str] | None:
    try:
        import yt_dlp  # noqa: F401
    except Exception:
        exe = shutil.which("yt-dlp")
        if exe:
            return [exe]
        return None
    return [sys.executable, "-m", "yt_dlp"]


def yt_dlp_ffmpeg_args() -> list[str]:
    try:
        import imageio_ffmpeg
    except Exception:
        return []
    try:
        ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return []
    if ffmpeg_path.exists():
        return ["--ffmpeg-location", str(ffmpeg_path)]
    return []


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def select_downloaded_video_file(download_dir: Path, video_id: str) -> Path | None:
    expected_paths = [
        download_dir / f"youtube_{video_id}_video.mp4",
        download_dir / f"youtube_{video_id}.mp4",
    ]
    for expected in expected_paths:
        if expected.exists():
            return expected

    matches: list[Path] = []
    for pattern in [f"youtube_{video_id}_video.*", f"youtube_{video_id}.*"]:
        for path in download_dir.glob(pattern):
            name = path.name.lower()
            if "_audio." in name:
                continue
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                matches.append(path)
    if not matches:
        return None
    return sorted(set(matches), key=lambda item: (-file_size(item), item.name))[0]


def download_video(url: str, video_id: str, download_dir: Path) -> Path | None:
    command_prefix = yt_dlp_command()
    if not command_prefix:
        print("[youtube] download_skipped=yt-dlp not installed", flush=True)
        return None

    download_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(download_dir / f"youtube_{video_id}_video.%(ext)s")
    command = [
        *command_prefix,
        "--no-playlist",
        *yt_dlp_ffmpeg_args(),
        "-f",
        "bv*[vcodec^=avc1][ext=mp4]+ba[ext=m4a]/bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        output_template,
        url,
    ]
    print("[youtube] downloading_video", flush=True)
    subprocess.run(command, check=True)

    return select_downloaded_video_file(download_dir, video_id)


def safe_download_video(url: str, video_id: str, download_dir: Path) -> tuple[Path | None, str]:
    try:
        return download_video(url, video_id, download_dir), ""
    except Exception as exc:
        return None, str(exc)


def parse_vtt_time(value: str) -> float:
    normalized = value.replace(",", ".")
    hours, minutes, seconds = normalized.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def clean_vtt_text(text: str) -> str:
    text = re.sub(r"<\d{2}:\d{2}:\d{2}[.,]\d{3}>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return " ".join(text.split()).strip()


def parse_vtt_segments(path: Path) -> list[dict]:
    segments: list[dict] = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        match = VTT_TIME_RE.search(line)
        if not match:
            idx += 1
            continue

        start = parse_vtt_time(match.group("start"))
        end = parse_vtt_time(match.group("end"))
        idx += 1
        text_lines = []
        while idx < len(lines) and lines[idx].strip():
            text_lines.append(lines[idx].strip())
            idx += 1
        text = clean_vtt_text(" ".join(text_lines))
        if text and end > start:
            if segments and segments[-1]["text"] == text and start <= float(segments[-1]["end"]) + 0.25:
                segments[-1]["end"] = round(max(float(segments[-1]["end"]), end), 3)
            else:
                segments.append({"start": round(start, 3), "end": round(end, 3), "text": text})
        idx += 1
    return segments


def write_transcript_files_from_segments(
    output_dir: Path,
    source_label: str,
    segments: list[dict],
    chunk_seconds: int = DEFAULT_CHUNK_SECONDS,
) -> dict:
    transcripts_dir = output_dir / "transcripts"
    merged_dir = output_dir / "merged"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    duration = max((float(seg["end"]) for seg in segments), default=0.0)
    total_chunks = max(1, int((duration + chunk_seconds - 1) // chunk_seconds))
    for chunk_index in range(1, total_chunks + 1):
        chunk_start = (chunk_index - 1) * chunk_seconds
        chunk_end = min(chunk_start + chunk_seconds, duration)
        chunk_segments = [
            {
                "start": round(float(seg["start"]) - chunk_start, 3),
                "end": round(float(seg["end"]) - chunk_start, 3),
                "text": seg["text"],
            }
            for seg in segments
            if float(seg["end"]) >= chunk_start and float(seg["start"]) <= chunk_start + chunk_seconds
        ]
        chunk_payload = {
            "text": " ".join(seg["text"] for seg in chunk_segments),
            "duration": max(0.0, chunk_end - chunk_start),
            "segments": chunk_segments,
            "chunk_file": source_label,
            "chunk_start_sec": float(chunk_start),
            "chunk_end_sec": float(chunk_end),
            "transcript_source": source_label,
        }
        (transcripts_dir / f"chunk_{chunk_index:03d}.json").write_text(
            json.dumps(chunk_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    merged_payload = {
        "source_video": source_label,
        "chunk_seconds": chunk_seconds,
        "segment_count": len(segments),
        "segments": segments,
    }
    (merged_dir / "merged_transcript.json").write_text(
        json.dumps(merged_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    merged_lines = [f"[{seg['start']:08.3f} - {seg['end']:08.3f}] {seg['text']}" for seg in segments]
    (merged_dir / "merged_transcript.txt").write_text("\n".join(merged_lines), encoding="utf-8")

    metadata = {
        "source_video": source_label,
        "duration_sec": round(duration, 3),
        "chunk_seconds": chunk_seconds,
        "processed_chunks": total_chunks,
        "max_chunks": total_chunks,
        "transcribe_model": "youtube_captions",
        "language": "auto",
    }
    (output_dir / "job_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "source": source_label,
        "duration_sec": round(duration, 3),
        "segment_count": len(segments),
        "chunk_count": total_chunks,
        "merged_transcript": str(merged_dir / "merged_transcript.json"),
    }


def download_subtitles(url: str, video_id: str, output_dir: Path) -> Path | None:
    command_prefix = yt_dlp_command()
    if not command_prefix:
        print("[youtube] subtitle_skipped=yt-dlp not installed", flush=True)
        return None

    subtitles_dir = output_dir / "subtitles"
    subtitles_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(subtitles_dir / f"youtube_{video_id}.%(ext)s")
    command = [
        *command_prefix,
        "--no-playlist",
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        "ko.*,ko",
        "--sub-format",
        "vtt",
        "-o",
        output_template,
        url,
    ]
    print("[youtube] subtitle_download_started=true", flush=True)
    result = subprocess.run(command, check=False)
    candidates = sorted(subtitles_dir.glob(f"youtube_{video_id}*.vtt"))
    if not candidates:
        if result.returncode != 0:
            print(f"[youtube] subtitle_download_failed=exit {result.returncode}", flush=True)
        return None
    preferred = sorted(candidates, key=lambda item: (".ko" not in item.name.lower(), len(item.name)))
    return preferred[0]


def download_audio(url: str, video_id: str, download_dir: Path) -> Path | None:
    command_prefix = yt_dlp_command()
    if not command_prefix:
        print("[youtube] audio_skipped=yt-dlp not installed", flush=True)
        return None

    download_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(download_dir / f"youtube_{video_id}_audio.%(ext)s")
    command = [
        *command_prefix,
        "--no-playlist",
        "-f",
        "ba[ext=m4a]/ba/bestaudio",
        "-o",
        output_template,
        url,
    ]
    print("[youtube] audio_download_started=true", flush=True)
    subprocess.run(command, check=True)
    matches = sorted(download_dir.glob(f"youtube_{video_id}_audio.*"))
    return matches[0] if matches else None


def transcribe_audio(audio_path: Path, output_dir: Path) -> None:
    command = [
        sys.executable,
        "-u",
        str(BASE_DIR / "analyze_longform.py"),
        "--source-video",
        str(audio_path),
        "--output-dir",
        str(output_dir),
    ]
    print("[youtube] audio_transcription_started=true", flush=True)
    process = subprocess.Popen(
        command,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        clean = line.rstrip()
        if clean:
            print(clean, flush=True)
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"Audio transcription failed with exit code {code}")


def prepare_transcript(
    url: str,
    video_id: str,
    output_dir: Path,
    download_dir: Path,
    allow_audio_fallback: bool = True,
) -> dict:
    result = {
        "status": "not_started",
        "source": "",
        "subtitle_file": "",
        "audio_file": "",
        "merged_transcript": "",
        "error": "",
    }
    try:
        subtitle_path = download_subtitles(url, video_id, output_dir)
        if subtitle_path:
            print(f"[youtube] subtitle_file={subtitle_path}", flush=True)
            segments = parse_vtt_segments(subtitle_path)
            if segments:
                transcript_info = write_transcript_files_from_segments(
                    output_dir,
                    str(subtitle_path),
                    segments,
                )
                result.update(
                    {
                        "status": "ready",
                        "source": "youtube_captions",
                        "subtitle_file": str(subtitle_path),
                        "merged_transcript": transcript_info["merged_transcript"],
                        "segment_count": transcript_info["segment_count"],
                        "chunk_count": transcript_info["chunk_count"],
                    }
                )
                print(f"[youtube] transcript_from_subtitles={transcript_info['merged_transcript']}", flush=True)
                return result
            print("[youtube] subtitle_empty=true", flush=True)
        else:
            print("[youtube] subtitle_file=none", flush=True)

        if not allow_audio_fallback:
            result.update(
                {
                    "status": "unavailable",
                    "error": "No downloadable captions were available and audio fallback was disabled.",
                }
            )
            return result

        audio_path = download_audio(url, video_id, download_dir)
        if not audio_path:
            result.update({"status": "failed", "error": "Could not download subtitles or audio."})
            return result
        print(f"[youtube] audio_file={audio_path}", flush=True)
        transcribe_audio(audio_path, output_dir)
        merged_path = output_dir / "merged" / "merged_transcript.json"
        result.update(
            {
                "status": "ready",
                "source": "audio_whisper",
                "audio_file": str(audio_path),
                "merged_transcript": str(merged_path) if merged_path.exists() else "",
            }
        )
        print(f"[youtube] transcript_from_audio={merged_path}", flush=True)
        return result
    except Exception as exc:
        result.update({"status": "failed", "error": str(exc)})
        print(f"[youtube] transcript_error={exc}", flush=True)
        return result


def safe_prepare_transcript(
    url: str,
    video_id: str,
    output_dir: Path,
    download_dir: Path,
    allow_audio_fallback: bool = True,
) -> dict:
    return prepare_transcript(url, video_id, output_dir, download_dir, allow_audio_fallback=allow_audio_fallback)


def generate_packages_from_prepared_transcript(
    output_dir: Path,
    source_title: str,
    youtube_context_path: Path,
    benchmark_profile_path: Path | None = None,
) -> dict:
    result = {"status": "not_started", "output": "", "error": ""}
    command = [
        sys.executable,
        "-u",
        str(BASE_DIR / "generate_shorts_packages.py"),
        "--analysis-dir",
        str(output_dir),
        "--source-title",
        source_title,
        "--youtube-context",
        str(youtube_context_path),
    ]
    if benchmark_profile_path and benchmark_profile_path.exists():
        command.extend(["--benchmark-profile", str(benchmark_profile_path)])
    print("[youtube] package_generation_started=true", flush=True)
    process = subprocess.Popen(
        command,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        clean = line.rstrip()
        if clean:
            print(clean, flush=True)
    code = process.wait()
    if code != 0:
        result.update({"status": "failed", "error": f"Package generation failed with exit code {code}"})
        print(f"[youtube] package_generation_failed={result['error']}", flush=True)
        return result
    package_path = output_dir / "shorts_candidates" / "final" / "shorts_packages.json"
    result.update({"status": "ready", "output": str(package_path)})
    print(f"[youtube] package_generation_done={package_path}", flush=True)
    return result


def main() -> None:
    configure_stdout()
    args = build_parser().parse_args()
    video_id = extract_video_id(args.url)
    api_key = get_api_key(args.api_key)
    output_dir = (args.output_dir or default_output_dir(video_id)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[youtube] video_id={video_id}", flush=True)
    downloaded_video = None
    download_error = ""
    download_future: Future[tuple[Path | None, str]] | None = None
    transcript_future: Future[dict] | None = None
    output_path = output_dir / "youtube_context.json"
    existing_context: dict = {}
    if output_path.exists():
        try:
            existing_context = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_context = {}
    existing_transcript = existing_context.get("transcript_preparation")
    transcript_preparation = {
        "status": "not_requested",
        "source": "",
        "subtitle_file": "",
        "audio_file": "",
        "merged_transcript": "",
        "error": "",
    }
    if not args.prepare_transcript and isinstance(existing_transcript, dict):
        transcript_preparation = existing_transcript
    package_generation = {"status": "not_requested", "output": "", "error": ""}

    with ThreadPoolExecutor(max_workers=3) as executor:
        if args.download_video:
            print("[youtube] parallel_download_started=true", flush=True)
            download_future = executor.submit(
                safe_download_video,
                args.url,
                video_id,
                args.download_dir.resolve(),
            )
        if args.prepare_transcript:
            print("[youtube] transcript_preparation_started=true", flush=True)
            transcript_future = executor.submit(
                safe_prepare_transcript,
                args.url,
                video_id,
                output_dir,
                args.download_dir.resolve(),
                not args.captions_only,
            )

        print("[youtube] collecting_context_while_download=true", flush=True)
        metadata = collect_metadata(video_id, api_key)
        if metadata.get("available"):
            print(f"[youtube] metadata title={metadata.get('title', '')}", flush=True)
        else:
            print(f"[youtube] metadata_unavailable={metadata.get('reason', '')}", flush=True)

        comments, comment_error = collect_comments(video_id, api_key, max(0, args.max_comments))
        print(f"[youtube] comments_fetched={len(comments)}", flush=True)
        if comment_error:
            print(f"[youtube] comments_note={comment_error}", flush=True)

        comment_insights = analyze_comments(comments)
        print(f"[youtube] timecode_moments={len(comment_insights['timecode_moments'])}", flush=True)

        payload = {
            "source_url": args.url,
            "video_id": video_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata,
            "comments": {
                "available": bool(comments),
                "fetched_count": len(comments),
                "error": comment_error,
            },
            "comment_insights": comment_insights,
            "transcript_preparation": transcript_preparation,
            "package_generation": package_generation,
            "downloaded_video": "",
            "download_error": "",
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[youtube] context_preliminary={output_path}", flush=True)

        if transcript_future:
            print("[youtube] waiting_for_transcript=true", flush=True)
            transcript_preparation = transcript_future.result()
            payload["transcript_preparation"] = transcript_preparation
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if transcript_preparation.get("status") == "ready":
                print(f"[youtube] prepared_transcript={transcript_preparation.get('merged_transcript', '')}", flush=True)
            else:
                print(f"[youtube] prepared_transcript_failed={transcript_preparation.get('error', '')}", flush=True)
            print(f"[youtube] context_transcript={output_path}", flush=True)

            if args.generate_packages and transcript_preparation.get("status") == "ready":
                source_title = metadata.get("title") if metadata.get("available") else video_id
                package_generation = generate_packages_from_prepared_transcript(
                    output_dir,
                    source_title or video_id,
                    output_path,
                    args.benchmark_profile.resolve() if args.benchmark_profile else None,
                )
                payload["package_generation"] = package_generation
                output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[youtube] context_packages={output_path}", flush=True)

        if download_future:
            print("[youtube] waiting_for_download=true", flush=True)
            downloaded_video, download_error = download_future.result()

    if args.download_video:
        if downloaded_video:
            print(f"[youtube] downloaded_video={downloaded_video}", flush=True)
        elif download_error:
            print(f"[youtube] download_error={download_error}", flush=True)

    payload["downloaded_video"] = str(downloaded_video) if downloaded_video else ""
    payload["download_error"] = download_error
    payload["transcript_preparation"] = transcript_preparation
    payload["package_generation"] = package_generation
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[youtube] context={output_path}", flush=True)


if __name__ == "__main__":
    main()
