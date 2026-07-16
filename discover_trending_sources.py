from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from env_loader import load_project_env


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "analysis" / "trends"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "latest_trend_candidates.json"
HISTORY_PATH = OUTPUT_DIR / "trend_history.json"
YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
USER_AGENT = "Long2Shorts/0.1"

DEFAULT_CONFIG: dict[str, Any] = {
    "window_hours": 72,
    "candidate_limit": 20,
    "search_results_per_query": 20,
    "min_duration_sec": 8 * 60,
    "max_duration_sec": 2 * 60 * 60,
    "region_code": "KR",
    "language": "ko",
    "queries": [
        "한국 예능",
        "연예인 토크쇼",
        "유재석 예능",
        "예능 게스트",
        "아이돌 예능",
        "핑계고",
        "놀면 뭐하니",
        "런닝맨",
        "빠더너스",
        "연예 예능 화제",
    ],
    "green_channel_keywords": [
        "뜬뜬",
        "ddeunddeun",
        "놀면 뭐하니",
        "hangout with yoo",
        "런닝맨",
        "running man",
        "sbs",
        "mbc",
        "kbs",
        "jtbc",
        "tvn",
        "diggle",
        "스튜디오 와플",
        "studio waffle",
        "채널십오야",
        "15ya",
        "빠더너스",
        "bdns",
        "teo",
        "요정재형",
        "숙스러운 미숙씨",
    ],
    "block_title_keywords": [
        "불륜",
        "사망",
        "도박",
        "마약",
        "폭로",
        "이혼",
        "고소",
        "충격",
        "해외도피",
        "전말",
    ],
    "skip_title_keywords": [
        "live",
        "실시간",
        "streaming",
        "스트리밍",
        "예고",
        "preview",
        "teaser",
        "티저",
    ],
}


ISO_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path and path.exists():
        config = deep_merge(config, json.loads(path.read_text(encoding="utf-8")))
    return config


def parse_iso_duration(value: str) -> int:
    match = ISO_DURATION_RE.match(value or "")
    if not match:
        return 0
    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        pass
    try:
        return datetime.strptime(value, "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def load_history(path: Path = HISTORY_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"processed_sources": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"processed_sources": {}}
    if not isinstance(data, dict):
        return {"processed_sources": {}}
    data.setdefault("processed_sources", {})
    return data


def save_history(history: dict[str, Any], path: Path = HISTORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def mark_processed_source(
    video_id: str,
    source_url: str,
    title: str,
    channel_title: str = "",
    shorts_created: int = 0,
    capcut_drafts_created: int = 0,
    path: Path = HISTORY_PATH,
) -> None:
    history = load_history(path)
    history["processed_sources"][video_id] = {
        "video_id": video_id,
        "source_url": source_url,
        "title": title,
        "channel_title": channel_title,
        "shorts_created": shorts_created,
        "capcut_drafts_created": capcut_drafts_created,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "status": "processed",
    }
    save_history(history, path)


def get_api_key(explicit: str = "") -> str:
    if explicit:
        return explicit
    load_project_env()
    return os.environ.get("YOUTUBE_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""


def youtube_api_get(path: str, params: dict[str, Any], api_key: str) -> dict[str, Any]:
    query = dict(params)
    query["key"] = api_key
    url = f"{YOUTUBE_API_BASE}/{path}?{urlencode(query)}"
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=25) as response:
        return json.loads(response.read().decode("utf-8"))


def batched(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def collect_ids_from_api(config: dict[str, Any], api_key: str, published_after: datetime) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    region_code = str(config.get("region_code") or "KR")
    language = str(config.get("language") or "ko")
    max_results = max(1, min(50, int(config.get("search_results_per_query") or 20)))

    for category_id in ["24", "23"]:
        try:
            data = youtube_api_get(
                "videos",
                {
                    "part": "snippet,statistics,contentDetails",
                    "chart": "mostPopular",
                    "regionCode": region_code,
                    "videoCategoryId": category_id,
                    "maxResults": 50,
                },
                api_key,
            )
        except Exception as exc:
            print(f"[trend] chart_error={category_id}:{exc}", flush=True)
            continue
        for item in data.get("items", []) or []:
            video_id = str(item.get("id") or "")
            if video_id and video_id not in seen:
                seen.add(video_id)
                ids.append(video_id)

    for query in config.get("queries", []) or []:
        try:
            data = youtube_api_get(
                "search",
                {
                    "part": "snippet",
                    "type": "video",
                    "q": str(query),
                    "regionCode": region_code,
                    "relevanceLanguage": language,
                    "publishedAfter": published_after.isoformat().replace("+00:00", "Z"),
                    "order": "viewCount",
                    "safeSearch": "moderate",
                    "maxResults": max_results,
                },
                api_key,
            )
        except Exception as exc:
            print(f"[trend] search_error={query}:{exc}", flush=True)
            continue
        for item in data.get("items", []) or []:
            video_id = ((item.get("id") or {}).get("videoId") or "").strip()
            if video_id and video_id not in seen:
                seen.add(video_id)
                ids.append(video_id)
    return ids


def fetch_video_details_api(ids: list[str], api_key: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for group in batched(ids, 50):
        data = youtube_api_get(
            "videos",
            {
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(group),
                "maxResults": 50,
            },
            api_key,
        )
        items.extend(data.get("items", []) or [])
    return items


def yt_dlp_json(args: list[str]) -> dict[str, Any] | None:
    command = [sys.executable, "-m", "yt_dlp", "--no-warnings", "--dump-single-json", *args]
    process = subprocess.run(
        command,
        cwd=str(BASE_DIR),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if process.returncode != 0 or not process.stdout.strip():
        return None
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError:
        return None


def collect_items_from_ytdlp(config: dict[str, Any], published_after: datetime) -> list[dict[str, Any]]:
    ids: list[str] = []
    seen: set[str] = set()
    max_results = max(1, min(20, int(config.get("search_results_per_query") or 10)))

    for query in (config.get("queries", []) or [])[:8]:
        data = yt_dlp_json(["--flat-playlist", f"ytsearch{max_results}:{query}"])
        for item in (data or {}).get("entries", []) or []:
            video_id = str(item.get("id") or "")
            if video_id and video_id not in seen:
                seen.add(video_id)
                ids.append(video_id)

    items: list[dict[str, Any]] = []
    for video_id in ids[:80]:
        data = yt_dlp_json([f"https://www.youtube.com/watch?v={video_id}", "--skip-download"])
        if not data:
            continue
        upload_dt = parse_datetime(str(data.get("upload_date") or ""))
        if upload_dt and upload_dt < published_after:
            continue
        items.append(
            {
                "id": video_id,
                "snippet": {
                    "title": data.get("title") or "",
                    "channelTitle": data.get("channel") or data.get("uploader") or "",
                    "channelId": data.get("channel_id") or "",
                    "publishedAt": upload_dt.isoformat().replace("+00:00", "Z") if upload_dt else "",
                    "description": data.get("description") or "",
                },
                "statistics": {
                    "viewCount": data.get("view_count") or 0,
                    "likeCount": data.get("like_count") or 0,
                    "commentCount": data.get("comment_count") or 0,
                },
                "contentDetails": {
                    "duration": seconds_to_iso_duration(safe_int(data.get("duration") or 0)),
                },
            }
        )
    return items


def seconds_to_iso_duration(seconds: int) -> str:
    seconds = max(0, seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    parts = "PT"
    if hours:
        parts += f"{hours}H"
    if minutes:
        parts += f"{minutes}M"
    if seconds or parts == "PT":
        parts += f"{seconds}S"
    return parts


def contains_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(str(keyword).lower() in lowered for keyword in keywords)


def classify_candidate(candidate: dict[str, Any], config: dict[str, Any], history: dict[str, Any], now: datetime) -> tuple[str, list[str]]:
    reasons: list[str] = []
    title = str(candidate.get("title") or "")
    channel = str(candidate.get("channel_title") or "")
    combined = f"{title} {channel}"
    video_id = str(candidate.get("video_id") or "")
    duration_sec = safe_int(candidate.get("duration_sec"))
    age_hours = float(candidate.get("age_hours") or 9999)

    if video_id in (history.get("processed_sources") or {}):
        return "processed", ["이미 처리한 원본"]
    if duration_sec < int(config.get("min_duration_sec") or 0):
        return "red", ["길이가 너무 짧음"]
    if duration_sec > int(config.get("max_duration_sec") or 999999):
        return "red", ["길이가 너무 김"]
    if age_hours > float(config.get("window_hours") or 72):
        return "red", ["최근성 기준 초과"]
    if contains_any(combined, list(config.get("skip_title_keywords") or [])):
        return "red", ["라이브/예고/티저성 제목"]
    if contains_any(title, list(config.get("block_title_keywords") or [])):
        return "red", ["루머/자극 키워드"]

    is_green_channel = contains_any(channel, list(config.get("green_channel_keywords") or []))
    if is_green_channel:
        reasons.append("공식/준공식 채널 키워드")
        return "green", reasons

    if safe_int(candidate.get("view_count")) >= 100000 and safe_int(candidate.get("comment_count")) >= 100:
        reasons.append("반응은 좋지만 미확인 채널")
        return "yellow", reasons

    reasons.append("자동 처리하기엔 신뢰 신호 부족")
    return "yellow", reasons


def score_candidate(candidate: dict[str, Any], status: str) -> float:
    age_hours = max(1.0, float(candidate.get("age_hours") or 1.0))
    views = safe_int(candidate.get("view_count"))
    likes = safe_int(candidate.get("like_count"))
    comments = safe_int(candidate.get("comment_count"))
    duration_sec = safe_int(candidate.get("duration_sec"))

    views_per_hour = views / age_hours
    comments_per_hour = comments / age_hours
    like_rate = likes / max(1, views)
    comment_rate = comments / max(1, views)
    duration_bonus = 8.0 if 12 * 60 <= duration_sec <= 75 * 60 else 3.0
    trust_bonus = 15.0 if status == "green" else 0.0
    recency_bonus = max(0.0, 24.0 - age_hours * 0.25)

    return round(
        math.log1p(views_per_hour) * 9.0
        + math.log1p(comments_per_hour) * 14.0
        + min(12.0, like_rate * 600.0)
        + min(12.0, comment_rate * 2500.0)
        + duration_bonus
        + trust_bonus
        + recency_bonus,
        3,
    )


def normalize_video_item(item: dict[str, Any], now: datetime) -> dict[str, Any] | None:
    video_id = str(item.get("id") or "")
    if not video_id:
        return None
    snippet = item.get("snippet") or {}
    statistics = item.get("statistics") or {}
    content_details = item.get("contentDetails") or {}
    published_at = parse_datetime(str(snippet.get("publishedAt") or ""))
    if not published_at:
        return None
    duration_sec = parse_iso_duration(str(content_details.get("duration") or ""))
    age_hours = max(0.0, (now - published_at).total_seconds() / 3600.0)
    return {
        "video_id": video_id,
        "source_url": f"https://www.youtube.com/watch?v={video_id}",
        "title": snippet.get("title") or "",
        "channel_title": snippet.get("channelTitle") or "",
        "channel_id": snippet.get("channelId") or "",
        "published_at": published_at.isoformat(),
        "age_hours": round(age_hours, 2),
        "duration_sec": duration_sec,
        "duration_min": round(duration_sec / 60.0, 2),
        "view_count": safe_int(statistics.get("viewCount")),
        "like_count": safe_int(statistics.get("likeCount")),
        "comment_count": safe_int(statistics.get("commentCount")),
    }


def discover_candidates(
    config: dict[str, Any],
    api_key: str = "",
    include_processed: bool = False,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    window_hours = int(config.get("window_hours") or 72)
    published_after = now - timedelta(hours=window_hours)
    history = {"processed_sources": {}} if include_processed else load_history()
    source = "youtube_api" if api_key else "yt_dlp"

    if api_key:
        ids = collect_ids_from_api(config, api_key, published_after)
        raw_items = fetch_video_details_api(ids, api_key)
    else:
        raw_items = collect_items_from_ytdlp(config, published_after)

    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_items:
        candidate = normalize_video_item(item, now)
        if not candidate or candidate["video_id"] in seen:
            continue
        seen.add(candidate["video_id"])
        status, reasons = classify_candidate(candidate, config, history, now)
        candidate["status"] = status
        candidate["reasons"] = reasons
        candidate["score"] = score_candidate(candidate, status)
        if status in {"green", "yellow"}:
            candidates.append(candidate)
        else:
            rejected.append(candidate)

    candidates.sort(key=lambda value: (value["status"] == "green", value["score"]), reverse=True)
    limit = max(1, int(config.get("candidate_limit") or 20))
    selected = next((item for item in candidates if item.get("status") == "green"), None)
    return {
        "created_at": now.isoformat(),
        "source": source,
        "window_hours": window_hours,
        "candidate_limit": limit,
        "candidates": candidates[:limit],
        "rejected_count": len(rejected),
        "selected": selected,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover recent Korean entertainment longform candidates.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--window-hours", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-processed", action="store_true")
    parser.add_argument("--mark-processed", action="store_true")
    parser.add_argument("--video-id", default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--title", default="")
    parser.add_argument("--channel-title", default="")
    parser.add_argument("--shorts-created", type=int, default=0)
    parser.add_argument("--capcut-drafts-created", type=int, default=0)
    return parser


def print_candidate_summary(result: dict[str, Any]) -> None:
    print(f"[trend] source={result.get('source', '')}", flush=True)
    print(f"[trend] window_hours={result.get('window_hours', '')}", flush=True)
    print(f"[trend] candidates={len(result.get('candidates', []))}", flush=True)
    print(f"[trend] rejected_count={result.get('rejected_count', 0)}", flush=True)
    for index, item in enumerate(result.get("candidates", [])[:10], start=1):
        print(
            "[trend] "
            f"rank={index} "
            f"status={item.get('status')} "
            f"score={item.get('score')} "
            f"age_h={item.get('age_hours')} "
            f"views={item.get('view_count')} "
            f"comments={item.get('comment_count')} "
            f"title={item.get('title')} "
            f"url={item.get('source_url')}",
            flush=True,
        )
    selected = result.get("selected") or {}
    if selected:
        print(f"[trend] selected_video_id={selected.get('video_id', '')}", flush=True)
        print(f"[trend] selected_url={selected.get('source_url', '')}", flush=True)
        print(f"[trend] selected_title={selected.get('title', '')}", flush=True)
        print(f"[trend] selected_channel={selected.get('channel_title', '')}", flush=True)
    else:
        print("[trend] selected_url=", flush=True)


def main() -> None:
    configure_stdout()
    args = build_parser().parse_args()

    if args.mark_processed:
        if not args.video_id:
            raise SystemExit("--video-id is required with --mark-processed")
        mark_processed_source(
            video_id=args.video_id,
            source_url=args.source_url or f"https://www.youtube.com/watch?v={args.video_id}",
            title=args.title,
            channel_title=args.channel_title,
            shorts_created=args.shorts_created,
            capcut_drafts_created=args.capcut_drafts_created,
        )
        print(f"[trend] marked_processed={args.video_id}", flush=True)
        return

    config = load_config(args.config)
    if args.window_hours is not None:
        config["window_hours"] = args.window_hours
    if args.limit is not None:
        config["candidate_limit"] = args.limit

    api_key = get_api_key(args.api_key)
    result = discover_candidates(config, api_key=api_key, include_processed=args.include_processed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print_candidate_summary(result)
    print(f"[trend] output={args.output}", flush=True)


if __name__ == "__main__":
    main()
