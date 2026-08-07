from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
import re
import sqlite3
import statistics
import subprocess
import sys
import traceback
from typing import Any

import cv2
import numpy as np

from env_loader import load_project_env
from usage_ledger import summarize_usage


BASE_DIR = Path(__file__).resolve().parent
AUTOMATION_DIR = BASE_DIR / "analysis" / "automation"
DEFAULT_DB_PATH = AUTOMATION_DIR / "orchestrator.sqlite3"
DEFAULT_CONFIG_PATH = BASE_DIR / "automation_config.json"
DEFAULT_BENCHMARK_PROFILE = "templates/benchmark_profiles/rescene_gyaru_variety.json"
VIDEO_FILE_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi"}
# YouTube's Entertainment category is not a content-format guarantee: it
# frequently contains stock-market, political-commentary and breaking-news
# channels. These are not a suitable input for reaction-led Korean variety
# Shorts, independently of whether they are technically in category 24.
ALWAYS_BLOCKED_SOURCE_TERMS = (
    "런닝맨", "running man", "정치", "대통령", "국회", "선거", "국회의원", "민주당", "국민의힘",
    "주식", "증시", "대폭락", "코인", "비트코인", "시황", "부동산", "경제 브리핑",
    "뉴스", "속보", "브리핑", "사건사고", "긴급", "시사",
)
# Shorts usually reveal their initial response quickly.  Keep the feedback
# loop inside the first day instead of waiting several days for every test.
CHECKPOINT_HOURS = (1, 3, 6, 12, 24)
ANALYTICS_SCOPES = [
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]


def project_python_executable() -> str:
    """Use the project runtime even when the CLI was launched by a bare IDE Python."""
    venv_python = BASE_DIR / ".venv" / "Scripts" / "python.exe"
    return str(venv_python if venv_python.exists() else Path(sys.executable))


def subprocess_environment(config: dict[str, Any]) -> dict[str, str]:
    """Pass the per-run usage ledger to every OpenAI-using child process."""
    environment = dict(os.environ)
    usage_path = str(config.get("_usage_log_path") or "").strip()
    if usage_path:
        environment["SHORTS_USAGE_LOG_PATH"] = usage_path
    return environment

DEFAULT_CONFIG: dict[str, Any] = {
    "library": {
        "sources": [],
        "max_sources_per_run": 1,
        "max_source_transcript_chars": 12000,
    },
    "trend": {
        "enabled": True,
        "config_path": "templates/trend_config.example.json",
        "window_hours": 72,
        "candidate_limit": 20,
    },
    "source_acquisition": {
        "enabled": True,
        "strategy": "selected_trend",
        "download_video": True,
        "prepare_transcript": True,
        "captions_only": False,
        "download_dir": "downloads",
        "max_comments": 300,
        "max_attempts": 5,
        "mark_processed_after_render": True,
    },
    "benchmark_profile": DEFAULT_BENCHMARK_PROFILE,
    "generation": {
        "candidate_model": "gpt-5.4",
        "final_model": "gpt-5.6-sol",
        "include_wide_windows": True,
        "timeline_first": True,
        "final_candidate_limit": 5,
        "run_budget_usd": 0.0,
        "final_package_reserve_usd": 0.0,
        "budget_safety_usd": 0.0,
        "candidate_visual_refinement": {
            "enabled": True,
            "model": "gpt-5.4",
            "frames_per_clip": 2,
            "candidate_limit": 10,
        },
        "force": False,
    },
    "visual_events": {
        "enabled": True,
        "model": "gpt-5.4",
        "sample_interval_sec": 12,
        "frames_per_batch": 6,
        "max_frames": 180,
    },
    "source_visual_gate": {
        "enabled": True,
        "sample_count": 12,
        "max_bright_document_ratio": 0.35,
    },
    "production": {
        "enabled": True,
        "minimum_packages_per_source": "auto",
        "minimum_score": 85,
        "minimum_duration_sec_exclusive": 20.0,
        "minimum_visual_shortability_score": 6,
        "rendered_visual_qa": {
            "enabled": True,
            "max_bright_document_ratio": 0.25,
            "sample_count": 5,
        },
        "allowed_decisions": ["auto_render"],
    },
    "source_safety": {
        "enabled": True,
        "require_verified_owned_source": False,
        "blocked_terms": ["\ub7f0\ub2dd\ub9e8", "running man", "\uc815\uce58", "\ub300\ud1b5\ub839", "\ubbfc\uc8fc\ub2f9", "\uad6d\ubbfc\uc758\ud798", "\uc120\uac70", "\uad6d\ud68c", "\uc774\uc7ac\uba85", "\ud55c\ub3d9\ud6c8"],
        "blocked_channels": [
            "\ub7f0\ub2dd\ub9e8 - \uc2a4\ube0c\uc2a4 \uacf5\uc2dd \ucc44\ub110", "\uac1c\uadf8\ucf58\uc11c\ud2b8",
            "KBS Entertain", "KBS Drama", "KBS Joy", "SBS Entertainment", "SBS Plus", "SBS Catch",
            "MBCentertainment", "MBC every1", "TV-People", "TVPP", "JTBC Entertainment", "tvN D ENT", "TVCHOSUN - TV\uc870\uc120", "\ub180\uba74 \ubb50\ud558\ub2c8?", "\uc5b4\uc11c\uc640 \ud55c\uad6d\uc740 \ucc98\uc74c\uc774\uc9c0",
        ],
        "blocked_channel_keywords": [
            "kbs", "sbs", "mbc", "jtbc", "tvn", "ebs", "mbn", "채널a", "channel a",
            "tv chosun", "tv조선", "ytn", "연합뉴스tv", "kbs world", "sbs now",
        ],
        "blocked_source_title_keywords": [
            "| kbs", "| sbs", "| mbc", "| jtbc", "| tvn", "| ebs", "kbs 방송", "sbs 방송",
            "mbc 방송", "jtbc 방송", "tvn 방송", "ebs 방송",
        ],
    },
    "review": {
        "enabled": True,
    },
    "learning": {
        "min_evaluated_videos": 3,
        "low_average_view_percentage": 60.0,
    },
    "youtube_analytics": {
        "enabled": False,
        "oauth_client_secret_path": "secrets/youtube_oauth_client.json",
        "token_path": "secrets/youtube_analytics_token.json",
        "interactive_on_first_run": False,
    },
    "youtube_upload": {
        "enabled": False,
        "require_review_approval": True,
        "privacy_status": "private",
        "publish_schedule": {
            "enabled": False,
            "mode": "interval",
            "interval_hours": 4,
            "daily_slots": ["09:00", "12:00", "15:00", "18:00", "21:00"],
        },
        "category_id": "24",
        "made_for_kids": False,
    },
}


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_time(value: datetime | None = None) -> str:
    return (value or utc_now()).replace(microsecond=0).isoformat()


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def resolve_path(value: str | Path | None, *, base: Path = BASE_DIR) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def read_json(path: Path, fallback: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict[str, Any]:
    payload = read_json(path, {}) if path.exists() else {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Orchestrator config must be a JSON object: {path}")
    return deep_merge(DEFAULT_CONFIG, payload)


def source_key_for(path: Path) -> str:
    digest = hashlib.sha256(str(path).casefold().encode("utf-8")).hexdigest()[:16]
    return f"source_{digest}"


def run_id_for(now: datetime | None = None) -> str:
    stamp = (now or utc_now()).strftime("%Y%m%dT%H%M%SZ")
    return f"daily_{stamp}_{hashlib.sha1(os.urandom(12)).hexdigest()[:6]}"


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_database(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            requested_at TEXT NOT NULL,
            status TEXT NOT NULL,
            dry_run INTEGER NOT NULL,
            summary_path TEXT NOT NULL,
            learning_rule_id TEXT,
            error_text TEXT
        );

        CREATE TABLE IF NOT EXISTS library_sources (
            source_key TEXT PRIMARY KEY,
            analysis_dir TEXT NOT NULL UNIQUE,
            source_video TEXT,
            source_title TEXT NOT NULL,
            owned INTEGER NOT NULL,
            active INTEGER NOT NULL,
            priority INTEGER NOT NULL DEFAULT 0,
            registered_at TEXT NOT NULL,
            last_selected_at TEXT
        );

        CREATE TABLE IF NOT EXISTS generated_packages (
            package_path TEXT PRIMARY KEY,
            source_key TEXT NOT NULL,
            run_id TEXT NOT NULL,
            short_id TEXT,
            score INTEGER,
            decision TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(source_key) REFERENCES library_sources(source_key),
            FOREIGN KEY(run_id) REFERENCES runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS published_shorts (
            youtube_video_id TEXT PRIMARY KEY,
            package_path TEXT NOT NULL,
            source_key TEXT,
            published_at TEXT NOT NULL,
            status TEXT NOT NULL,
            next_check_at TEXT,
            last_checked_at TEXT,
            FOREIGN KEY(source_key) REFERENCES library_sources(source_key)
        );

        CREATE TABLE IF NOT EXISTS metrics_snapshots (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            youtube_video_id TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            checkpoint TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            FOREIGN KEY(youtube_video_id) REFERENCES published_shorts(youtube_video_id)
        );

        CREATE TABLE IF NOT EXISTS learning_rules (
            rule_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            mode TEXT NOT NULL,
            rule_kind TEXT NOT NULL,
            rule_json TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS rendered_uploads (
            content_sha256 TEXT PRIMARY KEY,
            output_path TEXT NOT NULL,
            package_path TEXT NOT NULL,
            youtube_video_id TEXT NOT NULL,
            privacy_status TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            response_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS review_items (
            content_sha256 TEXT PRIMARY KEY,
            output_path TEXT NOT NULL,
            package_path TEXT NOT NULL,
            source_key TEXT NOT NULL,
            run_id TEXT NOT NULL,
            short_id TEXT,
            score INTEGER,
            decision TEXT,
            auto_qa_status TEXT NOT NULL,
            auto_qa_json TEXT NOT NULL,
            review_status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            decided_at TEXT,
            reviewer_note TEXT,
            FOREIGN KEY(source_key) REFERENCES library_sources(source_key),
            FOREIGN KEY(run_id) REFERENCES runs(run_id)
        );
        """
    )
    conn.commit()


def write_default_config(path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing config: {path}")
    write_json(path, DEFAULT_CONFIG)


def register_owned_source(
    config_path: Path,
    *,
    source_video: Path,
    analysis_dir: Path,
    source_title: str,
    priority: int,
) -> dict[str, Any]:
    if not source_video.exists():
        raise RuntimeError(f"Owned source video was not found: {source_video}")

    raw_config = read_json(config_path, {}) if config_path.exists() else {}
    if not isinstance(raw_config, dict):
        raise RuntimeError(f"Orchestrator config must be a JSON object: {config_path}")
    if not raw_config:
        raw_config = json.loads(json.dumps(DEFAULT_CONFIG))

    library = raw_config.setdefault("library", {})
    if not isinstance(library, dict):
        raise RuntimeError("Config library must be a JSON object.")
    entries = library.setdefault("sources", [])
    if not isinstance(entries, list):
        raise RuntimeError("Config library.sources must be a JSON array.")

    resolved_analysis_dir = analysis_dir.resolve()
    entry = {
        "analysis_dir": str(resolved_analysis_dir),
        "source_video": str(source_video.resolve()),
        "source_title": source_title.strip() or source_video.stem,
        "owned": True,
        "active": True,
        "priority": int(priority),
    }
    replaced = False
    for index, existing in enumerate(entries):
        if not isinstance(existing, dict):
            continue
        existing_dir = resolve_path(existing.get("analysis_dir"))
        existing_video = resolve_path(existing.get("source_video"))
        if existing_dir == resolved_analysis_dir or existing_video == source_video.resolve():
            entries[index] = entry
            replaced = True
            break
    if not replaced:
        entries.append(entry)
    write_json(config_path, raw_config)
    return entry


def first_string(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def read_source_title(entry: dict[str, Any], analysis_dir: Path) -> str:
    context = read_json(analysis_dir / "youtube_context.json", {})
    metadata = context.get("metadata", {}) if isinstance(context, dict) else {}
    movie_info = read_json(analysis_dir / "movie_info.json", {})
    return first_string(
        entry.get("source_title"),
        entry.get("title"),
        metadata.get("title") if isinstance(metadata, dict) else "",
        movie_info.get("matched_title") if isinstance(movie_info, dict) else "",
        analysis_dir.name,
    )


def read_source_video(entry: dict[str, Any], analysis_dir: Path) -> Path | None:
    def usable_video(value: str | Path | None) -> Path | None:
        path = resolve_path(value)
        if path and path.exists() and path.is_file() and path.suffix.lower() in VIDEO_FILE_SUFFIXES:
            return path
        return None

    configured = usable_video(entry.get("source_video"))
    if configured:
        return configured
    context = read_json(analysis_dir / "youtube_context.json", {})
    if isinstance(context, dict):
        downloaded = usable_video(context.get("downloaded_video"))
        if downloaded:
            return downloaded
    metadata = read_json(analysis_dir / "job_metadata.json", {})
    if isinstance(metadata, dict):
        source_video = usable_video(metadata.get("source_video"))
        if source_video:
            return source_video
    return None


def read_source_duration_sec(source: dict[str, Any]) -> float:
    for value in [source.get("duration_sec"), source.get("source_duration_sec")]:
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            return duration

    analysis_dir = source.get("analysis_dir")
    if not isinstance(analysis_dir, Path):
        analysis_dir = resolve_path(analysis_dir)
    if not analysis_dir:
        return 0.0

    context = read_json(analysis_dir / "youtube_context.json", {})
    metadata = context.get("metadata", {}) if isinstance(context, dict) else {}
    for value in [metadata.get("duration_sec") if isinstance(metadata, dict) else None]:
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            return duration

    job_metadata = read_json(analysis_dir / "job_metadata.json", {})
    if isinstance(job_metadata, dict):
        try:
            duration = float(job_metadata.get("duration_sec") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        if duration > 0:
            return duration
    return 0.0


def minimum_render_count_for_duration(duration_sec: float) -> int:
    minutes = duration_sec / 60.0 if duration_sec > 0 else 0.0
    if minutes >= 60:
        return 5
    if minutes >= 40:
        return 4
    if minutes >= 20:
        return 3
    if minutes >= 10:
        return 2
    return 1


def production_minimum_render_count(production: dict[str, Any], source: dict[str, Any]) -> int:
    """Return the minimum number of approved shorts to produce for one source.

    ``max_packages_per_source`` is retained as a fallback for older local
    configuration files, but new configurations use the minimum-oriented key.
    """
    configured = production.get("minimum_packages_per_source")
    if configured is None:
        configured = production.get("max_packages_per_source", "auto")
    if isinstance(configured, str) and configured.strip().lower() == "auto":
        return minimum_render_count_for_duration(read_source_duration_sec(source))
    try:
        return max(1, int(configured or 1))
    except (TypeError, ValueError):
        return minimum_render_count_for_duration(read_source_duration_sec(source))


def normalized_tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[^\W\d_]{2,}", value or "", flags=re.UNICODE)
        if len(token) >= 2
    }


def source_text_for_match(source: dict[str, Any], max_chars: int) -> str:
    transcript = source["analysis_dir"] / "merged" / "merged_transcript.txt"
    transcript_text = ""
    try:
        transcript_text = transcript.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except OSError:
        pass
    return f"{source['source_title']}\n{transcript_text}"


def trend_text_from_snapshot(payload: dict[str, Any]) -> str:
    values: list[str] = []
    for item in payload.get("candidates", []) or []:
        if not isinstance(item, dict):
            continue
        values.extend(
            [
                str(item.get("title") or ""),
                str(item.get("description") or ""),
                str((item.get("snippet") or {}).get("title") or ""),
                str((item.get("snippet") or {}).get("description") or ""),
            ]
        )
    selected = payload.get("selected") if isinstance(payload.get("selected"), dict) else {}
    values.extend([str(selected.get("title") or ""), str(selected.get("description") or "")])
    return "\n".join(values)


def upsert_library_source(conn: sqlite3.Connection, source: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO library_sources (
            source_key, analysis_dir, source_video, source_title, owned, active, priority, registered_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_key) DO UPDATE SET
            analysis_dir=excluded.analysis_dir,
            source_video=excluded.source_video,
            source_title=excluded.source_title,
            owned=excluded.owned,
            active=excluded.active,
            priority=excluded.priority
        """,
        (
            source["source_key"],
            str(source["analysis_dir"]),
            str(source["source_video"] or ""),
            source["source_title"],
            int(source["owned"]),
            int(source["active"]),
            int(source["priority"]),
            iso_time(),
        ),
    )
    conn.commit()


def configured_sources(config: dict[str, Any], library_dirs: list[Path]) -> list[dict[str, Any]]:
    library = config.get("library", {}) if isinstance(config.get("library"), dict) else {}
    entries = list(library.get("sources", []) or [])
    entries.extend({"analysis_dir": str(path), "owned": True} for path in library_dirs)
    sources = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            continue
        analysis_dir = resolve_path(raw_entry.get("analysis_dir"))
        if not analysis_dir:
            continue
        owned = bool(raw_entry.get("owned", False))
        active = bool(raw_entry.get("active", True))
        if not owned or not active:
            continue
        source_video = read_source_video(raw_entry, analysis_dir)
        sources.append(
            {
                "source_key": source_key_for(analysis_dir),
                "analysis_dir": analysis_dir,
                "source_video": source_video,
                "source_title": read_source_title(raw_entry, analysis_dir),
                "duration_sec": raw_entry.get("duration_sec") or raw_entry.get("source_duration_sec") or 0,
                "owned": owned,
                "active": active,
                "priority": int(raw_entry.get("priority", 0) or 0),
                "ready": (analysis_dir / "merged" / "merged_transcript.json").exists(),
            }
        )
    return sources


def run_trend_research(
    config: dict[str, Any],
    run_dir: Path,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    trend = config.get("trend", {}) if isinstance(config.get("trend"), dict) else {}
    if not trend.get("enabled", True):
        return {"status": "disabled", "path": "", "tokens": []}

    trends_dir = BASE_DIR / "analysis" / "trends"
    latest_path = trends_dir / "latest_trend_candidates.json"
    live_path = trends_dir / "live_hot_candidates.json"
    max_live_age_minutes = max(10, int(trend.get("live_snapshot_max_age_minutes", 25) or 25))

    def live_snapshot() -> dict[str, Any]:
        snapshot = read_json(live_path, {}) if live_path.exists() else {}
        created_at = parse_time(str(snapshot.get("created_at") or "")) if isinstance(snapshot, dict) else None
        if not created_at or utc_now() - created_at > timedelta(minutes=max_live_age_minutes):
            return {}
        return snapshot

    if dry_run:
        snapshot = live_snapshot() or (read_json(latest_path, {}) if latest_path.exists() else {})
        return {
            "status": "dry_run_live" if snapshot and snapshot.get("created_at") else ("dry_run_cached" if snapshot else "dry_run_unavailable"),
            "path": str(live_path if snapshot and snapshot.get("created_at") else latest_path) if snapshot else "",
            "snapshot": snapshot,
            "tokens": sorted(normalized_tokens(trend_text_from_snapshot(snapshot)))[:300],
        }

    snapshot = live_snapshot()
    if snapshot.get("candidates"):
        output_path = run_dir / "trend_snapshot.json"
        write_json(output_path, snapshot)
        return {
            "status": "live_monitor",
            "path": str(output_path),
            "snapshot": snapshot,
            "tokens": sorted(normalized_tokens(trend_text_from_snapshot(snapshot)))[:300],
        }

    output_path = run_dir / "trend_snapshot.json"
    command = [
        project_python_executable(),
        "-u",
        str(BASE_DIR / "discover_trending_sources.py"),
        "--output",
        str(output_path),
        "--window-hours",
        str(int(trend.get("window_hours", 72) or 72)),
        "--limit",
        str(int(trend.get("candidate_limit", 20) or 20)),
    ]
    config_path = resolve_path(trend.get("config_path"))
    if config_path and config_path.exists():
        command.extend(["--config", str(config_path)])
    completed = subprocess.run(command, cwd=BASE_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        snapshot = read_json(latest_path, {}) if latest_path.exists() else {}
        return {
            "status": "fallback_cached" if snapshot else "failed",
            "path": str(latest_path) if snapshot else "",
            "error": completed.stderr.strip() or completed.stdout.strip(),
            "snapshot": snapshot,
            "tokens": sorted(normalized_tokens(trend_text_from_snapshot(snapshot)))[:300],
        }
    snapshot = read_json(output_path, {})
    return {
        "status": "completed",
        "path": str(output_path),
        "snapshot": snapshot,
        "tokens": sorted(normalized_tokens(trend_text_from_snapshot(snapshot)))[:300],
    }


def selected_trend_candidate(trend_result: dict[str, Any]) -> dict[str, Any] | None:
    candidates = trend_candidate_attempts(trend_result)
    return candidates[0] if candidates else None


def trend_candidate_attempts(trend_result: dict[str, Any]) -> list[dict[str, Any]]:
    snapshot = trend_result.get("snapshot", {}) if isinstance(trend_result.get("snapshot"), dict) else {}
    attempts: list[dict[str, Any]] = []
    seen_video_ids: set[str] = set()
    # The monitor may still display an already used source, but production
    # must advance to a fresh long-form candidate for the next source cycle.
    history = read_json(BASE_DIR / "analysis" / "trends" / "trend_history.json", {})
    processed_sources = history.get("processed_sources", {}) if isinstance(history, dict) else {}
    processed_video_ids = set(processed_sources) if isinstance(processed_sources, dict) else set()

    def add_candidate(item) -> None:
        if not isinstance(item, dict) or not item.get("source_url"):
            return
        video_id = str(item.get("video_id") or item.get("source_url") or "").strip()
        if not video_id or video_id in seen_video_ids or video_id in processed_video_ids:
            return
        seen_video_ids.add(video_id)
        attempts.append(item)

    # The public trend list can contain finished Shorts. Keep them visible to
    # the user, but prefer the separately ranked long-form source list here.
    for item in snapshot.get("production_candidates", []) or []:
        add_candidate(item)
    selected = snapshot.get("selected") if isinstance(snapshot.get("selected"), dict) else None
    if not attempts:
        add_candidate(selected)
    for item in snapshot.get("candidates", []) or []:
        if isinstance(item, dict) and item.get("status") == "green":
            add_candidate(item)
    for item in snapshot.get("candidates", []) or []:
        add_candidate(item)
    return attempts


def trend_source_summary(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_key": source["source_key"],
        "analysis_dir": str(source["analysis_dir"]),
        "source_video": str(source["source_video"] or ""),
        "source_title": source["source_title"],
        "source_channel": str(source.get("source_channel") or ""),
        "source_url": str(source.get("source_url") or ""),
        "ready": bool(source.get("ready")),
    }


def source_safety_reason(config: dict[str, Any], source: dict[str, Any]) -> str:
    """Return a human-readable blocking reason for prohibited source material."""
    safety = config.get("source_safety", {}) if isinstance(config.get("source_safety"), dict) else {}
    if not bool(safety.get("enabled", True)):
        return ""

    metadata: dict[str, Any] = {}
    analysis_dir = source.get("analysis_dir")
    if analysis_dir:
        context = read_json(Path(analysis_dir) / "youtube_context.json", {})
        if isinstance(context, dict) and isinstance(context.get("metadata"), dict):
            metadata = context["metadata"]
    title = first_string(source.get("source_title"), metadata.get("title"))
    channel = first_string(source.get("source_channel"), metadata.get("channel_title"), metadata.get("uploader"))
    normalized_title = re.sub(r"\s+", " ", title).strip().casefold()
    normalized_channel = re.sub(r"\s+", " ", channel).strip().casefold()

    # Category 24 is not a Korean-variety-only category.  Music promotional
    # uploads regularly rank there, but they do not provide the dialogue and
    # action/reaction structure this pipeline is designed to re-edit.
    music_promo_markers = (
        "official visualizer", "official music video", "official mv",
        "official audio", "lyric video", "music video", "m/v",
    )
    if any(marker in normalized_title for marker in music_promo_markers):
        return f"music promotional source is outside the variety workflow: {title}"

    # YouTube's Entertainment category contains music and global fan content
    # as well as Korean variety.  For this Korean variety workflow, reject a
    # title with no meaningful Hangul signal before downloading it.
    minimum_korean_title_characters = max(0, int(safety.get("minimum_korean_title_characters", 0) or 0))
    if minimum_korean_title_characters and title:
        hangul_count = len(re.findall(r"[가-힣]", title))
        if hangul_count < minimum_korean_title_characters:
            return f"source title is not Korean-variety suitable: {title}"

    for value in safety.get("blocked_channels", []) or []:
        blocked = str(value or "").strip().casefold()
        if blocked and normalized_channel and (blocked in normalized_channel or normalized_channel in blocked):
            return f"blocked channel: {channel}"
    for value in safety.get("blocked_channel_keywords", []) or []:
        blocked = str(value or "").strip().casefold()
        if blocked and blocked in normalized_channel:
            return f"blocked broadcaster channel: {channel}"
    for value in safety.get("blocked_source_title_keywords", []) or []:
        blocked = str(value or "").strip().casefold()
        if blocked and blocked in normalized_title:
            return f"blocked broadcaster source title: {title}"
    # Configuration-specific exclusions are additive. A local config cannot
    # accidentally remove baseline broadcaster/politics/finance protections.
    blocked_terms = [*ALWAYS_BLOCKED_SOURCE_TERMS, *(safety.get("blocked_terms", []) or [])]
    for value in blocked_terms:
        blocked = str(value or "").strip().casefold()
        if blocked and (blocked in normalized_title or blocked in normalized_channel):
            return f"blocked source term: {value}"
    if bool(safety.get("require_verified_owned_source", False)) and not bool(source.get("owned", False)):
        return "source rights are not verified; use a rights-cleared owned library source"
    return ""


def source_visual_preflight(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Reject document/screen-share sources before expensive visual LLM analysis.

    The aim is not to judge whether a video is technically valid.  It is to
    prevent a Korean variety Shorts workflow from spending 180-frame vision
    analysis on a lecture, article page, or desktop capture that cannot become
    an actor/reaction-led vertical short.
    """
    settings = config.get("source_visual_gate", {}) if isinstance(config.get("source_visual_gate"), dict) else {}
    if not bool(settings.get("enabled", True)):
        return {"status": "disabled"}
    source_video = source.get("source_video")
    if not isinstance(source_video, Path) or not source_video.exists():
        return {"status": "unavailable", "reason": "source video is unavailable for visual preflight"}
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        return {"status": "unavailable", "reason": "OpenCV could not read source video"}
    try:
        frame_count = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        sample_count = max(6, int(settings.get("sample_count", 12) or 12))
        indices = sorted({int(round((frame_count - 1) * ratio)) for ratio in np.linspace(0.05, 0.95, sample_count)})
        bright_ratios: list[float] = []
        for frame_index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if ok and frame is not None:
                bright_ratios.append(float(np.mean(np.all(frame >= 242, axis=2))))
        if not bright_ratios:
            return {"status": "unavailable", "reason": "source samples could not be read"}
        median_ratio = float(np.median(bright_ratios))
        threshold = float(settings.get("max_bright_document_ratio", 0.35) or 0.35)
        result = {
            "sampled_frames": len(bright_ratios),
            "bright_document_ratio_median": round(median_ratio, 4),
            "bright_document_ratio_max": round(float(max(bright_ratios)), 4),
            "maximum_allowed_ratio": threshold,
        }
        if median_ratio > threshold:
            return {
                "status": "blocked",
                "reason": (
                    f"source is document/screen-share dominant ({median_ratio:.0%} bright area; "
                    f"maximum {threshold:.0%})"
                ),
                **result,
            }
        return {"status": "pass", **result}
    finally:
        cap.release()


def acquired_source_from_context(
    *,
    candidate: dict[str, Any],
    analysis_dir: Path,
    context: dict[str, Any],
) -> dict[str, Any]:
    metadata = context.get("metadata", {}) if isinstance(context.get("metadata"), dict) else {}
    try:
        duration_sec = float(metadata.get("duration_sec") or candidate.get("duration_sec") or 0.0)
    except (TypeError, ValueError):
        duration_sec = 0.0
    downloaded_video = resolve_path(context.get("downloaded_video"))
    title = first_string(
        candidate.get("title"),
        metadata.get("title") if isinstance(metadata, dict) else "",
        candidate.get("video_id"),
        analysis_dir.name,
    )
    channel = first_string(
        candidate.get("channel_title"),
        candidate.get("channel"),
        metadata.get("channel_title") if isinstance(metadata, dict) else "",
        metadata.get("uploader") if isinstance(metadata, dict) else "",
    )
    return {
        "source_key": source_key_for(analysis_dir),
        "analysis_dir": analysis_dir,
        "source_video": downloaded_video if downloaded_video and downloaded_video.exists() else None,
        "source_title": title,
        "source_channel": channel,
        "source_url": str(candidate.get("source_url") or context.get("source_url") or ""),
        "duration_sec": duration_sec,
        "source_origin": "trend_acquisition",
        "owned": False,
        "active": True,
        "priority": 100,
        "ready": (analysis_dir / "merged" / "merged_transcript.json").exists(),
    }


def acquire_selected_trend_source(
    config: dict[str, Any],
    trend_result: dict[str, Any],
    *,
    execute: bool,
) -> dict[str, Any]:
    acquisition = config.get("source_acquisition", {}) if isinstance(config.get("source_acquisition"), dict) else {}
    if not bool(acquisition.get("enabled", False)):
        return {"status": "disabled"}
    if str(acquisition.get("strategy") or "selected_trend") != "selected_trend":
        return {"status": "disabled", "reason": "unsupported acquisition strategy"}
    candidates = trend_candidate_attempts(trend_result)
    if not candidates:
        return {"status": "no_selected_trend_source"}
    max_attempts = max(1, int(acquisition.get("max_attempts", 5) or 5))
    minimum_source_duration = max(1, int(acquisition.get("minimum_source_duration_sec", 480) or 480))
    maximum_source_duration = max(
        minimum_source_duration,
        int(acquisition.get("maximum_source_duration_sec", 3600) or 3600),
    )
    attempts: list[dict[str, Any]] = []
    download_attempts = 0

    # The live chart can legitimately be filled with broadcaster uploads.
    # Do not weaken the safety rule just to keep the pipeline moving. Instead,
    # make one on-demand category/date/ranking query for other creators.
    def is_eligible_longform(candidate: dict[str, Any]) -> bool:
        try:
            duration = float(candidate.get("duration_sec") or 0)
        except (TypeError, ValueError):
            return False
        return minimum_source_duration <= duration <= maximum_source_duration and not source_safety_reason(
            config,
            {
                "source_title": candidate.get("title"),
                "source_channel": candidate.get("channel_title") or candidate.get("channel"),
            },
        )

    eligible_primary = [candidate for candidate in candidates if is_eligible_longform(candidate)]
    # One technically eligible chart result is not a resilient acquisition
    # pool: downloads can fail or the video can turn out visually unusable.
    # Top up from the broad creator/channel scan whenever the primary chart
    # cannot supply the configured number of genuine attempts.
    if len(eligible_primary) < max_attempts:
        trend = config.get("trend", {}) if isinstance(config.get("trend"), dict) else {}
        fallback_path = BASE_DIR / "analysis" / "trends" / "broad_longform_fallback.json"
        command = [
            project_python_executable(),
            "-u",
            str(BASE_DIR / "discover_trending_sources.py"),
            "--output",
            str(fallback_path),
            "--window-hours",
            str(int(trend.get("window_hours", 72) or 72)),
            "--limit",
            str(max(50, int(trend.get("candidate_limit", 20) or 20))),
            "--include-processed",
            "--playboard-channels",
        ]
        trend_config_path = resolve_path(trend.get("config_path"))
        if trend_config_path and trend_config_path.exists():
            command.extend(["--config", str(trend_config_path)])
        completed = subprocess.run(
            command,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=subprocess_environment(config),
        )
        fallback_snapshot = read_json(fallback_path, {}) if completed.returncode == 0 else {}
        fallback_candidates = trend_candidate_attempts({"snapshot": fallback_snapshot}) if isinstance(fallback_snapshot, dict) else []
        eligible_fallbacks = [candidate for candidate in fallback_candidates if is_eligible_longform(candidate)]
        attempts.append(
            {
                "status": "broad_longform_fallback",
                "eligible_candidates": len(eligible_fallbacks),
                "log_tail": completed.stdout.splitlines()[-8:],
                "error": "" if completed.returncode == 0 else (completed.stderr.strip() or completed.stdout.strip()),
            }
        )
        if eligible_fallbacks:
            primary_ids = {str(candidate.get("video_id") or "") for candidate in candidates}
            candidates = candidates + [
                candidate
                for candidate in eligible_fallbacks
                if str(candidate.get("video_id") or "") not in primary_ids
            ]

    for candidate in candidates:
        source_url = str(candidate.get("source_url") or "").strip()
        video_id = str(candidate.get("video_id") or "").strip()
        if not source_url or not video_id:
            attempts.append({"status": "invalid", "candidate": candidate})
            continue

        try:
            candidate_duration = float(candidate.get("duration_sec") or 0)
        except (TypeError, ValueError):
            candidate_duration = 0.0
        if candidate_duration and candidate_duration < minimum_source_duration:
            attempts.append(
                {
                    "status": "skipped_short_source",
                    "candidate": candidate,
                    "reason": f"source is {candidate_duration:.0f}s; requires at least {minimum_source_duration}s",
                }
            )
            continue
        if candidate_duration and candidate_duration > maximum_source_duration:
            attempts.append(
                {
                    "status": "skipped_oversized_source",
                    "candidate": candidate,
                    "reason": f"source is {candidate_duration:.0f}s; maximum is {maximum_source_duration}s",
                }
            )
            continue

        candidate_block_reason = source_safety_reason(
            config,
            {
                "source_title": candidate.get("title"),
                "source_channel": candidate.get("channel_title") or candidate.get("channel"),
            },
        )
        if candidate_block_reason:
            attempts.append({"status": "blocked", "candidate": candidate, "reason": candidate_block_reason})
            continue

        # Safety skips are not download failures.  Keep moving through the
        # ranked list until we have genuinely tried the configured number of
        # eligible creator sources.
        if download_attempts >= max_attempts:
            break

        analysis_dir = BASE_DIR / "analysis" / f"youtube_{video_id}"
        context_path = analysis_dir / "youtube_context.json"
        if not execute:
            context = read_json(context_path, {}) if context_path.exists() else {}
            source = acquired_source_from_context(candidate=candidate, analysis_dir=analysis_dir, context=context if isinstance(context, dict) else {})
            block_reason = source_safety_reason(config, source)
            if block_reason:
                return {"status": "blocked", "reason": block_reason, "summary": trend_source_summary(source)}
            return {"status": "planned", "source": source, "summary": trend_source_summary(source)}

        download_attempts += 1
        command = [
            project_python_executable(),
            "-u",
            str(BASE_DIR / "collect_youtube_context.py"),
            "--url",
            source_url,
            "--output-dir",
            str(analysis_dir),
            "--download-dir",
            str(resolve_path(acquisition.get("download_dir")) or (BASE_DIR / "downloads")),
            "--max-comments",
            str(int(acquisition.get("max_comments", 300) or 300)),
        ]
        if bool(acquisition.get("download_video", True)):
            command.append("--download-video")
        if bool(acquisition.get("prepare_transcript", True)):
            command.append("--prepare-transcript")
        if bool(acquisition.get("captions_only", False)):
            command.append("--captions-only")

        print(f"[orchestrator] source_attempt={source_url}", flush=True)
        completed = subprocess.run(command, cwd=BASE_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace")
        attempt = {
            "status": "completed" if completed.returncode == 0 else "failed",
            "candidate": candidate,
            "analysis_dir": str(analysis_dir),
            "log_tail": completed.stdout.splitlines()[-12:],
        }
        if completed.returncode != 0:
            attempt["error"] = completed.stderr.strip() or completed.stdout.strip()
            attempts.append(attempt)
            continue

        context = read_json(context_path, {})
        if not isinstance(context, dict):
            attempt["status"] = "failed"
            attempt["error"] = "youtube_context.json was not created"
            attempts.append(attempt)
            continue

        source = acquired_source_from_context(candidate=candidate, analysis_dir=analysis_dir, context=context)
        attempt["summary"] = trend_source_summary(source)
        attempts.append(attempt)
        block_reason = source_safety_reason(config, source)
        if block_reason:
            attempt["status"] = "blocked"
            attempt["reason"] = block_reason
            continue
        # Downloading/transcribing is unavoidable for a source-level visual
        # check, but vision-model analysis is not.  Reject a document-heavy
        # source here and keep trying the ranked list instead of returning it
        # to the main run where it would consume an expensive event script.
        visual_gate = source_visual_preflight(config, source)
        attempt["source_visual_gate"] = visual_gate
        if visual_gate.get("status") != "pass":
            attempt["status"] = "blocked"
            attempt["reason"] = str(visual_gate.get("reason") or "source visual preflight did not pass")
            continue
        if source.get("ready") or source.get("source_video"):
            return {
                "status": "completed",
                "source": source,
                "summary": trend_source_summary(source),
                "context_path": str(context_path),
                "source_visual_gate": visual_gate,
                "attempts": attempts,
                "log_tail": completed.stdout.splitlines()[-16:],
            }

        attempt["status"] = "unusable"
        attempt["reason"] = "missing merged transcript and source video"

    return {
        "status": "failed",
        "error": "No trend candidate could be downloaded or transcribed.",
        "attempts": attempts,
    }


def mark_acquired_source_processed(source: dict[str, Any], render: dict[str, Any], *, allow_zero: bool = False) -> None:
    source_url = str(source.get("source_url") or "")
    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})", source_url)
    video_id = match.group(1) if match else ""
    if not video_id:
        context = read_json(source["analysis_dir"] / "youtube_context.json", {})
        if isinstance(context, dict):
            video_id = str(context.get("video_id") or "")
            source_url = source_url or str(context.get("source_url") or "")
    if not video_id:
        return
    rendered_count = len(
        [
            item
            for item in render.get("rendered", []) or []
            if isinstance(item, dict) and item.get("status") == "completed"
        ]
    )
    if rendered_count <= 0 and not allow_zero:
        return
    command = [
        project_python_executable(),
        "-u",
        str(BASE_DIR / "discover_trending_sources.py"),
        "--mark-processed",
        "--video-id",
        video_id,
        "--source-url",
        source_url or f"https://www.youtube.com/watch?v={video_id}",
        "--title",
        source["source_title"],
        "--shorts-created",
        str(rendered_count),
    ]
    subprocess.run(command, cwd=BASE_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace")


def rank_sources(
    conn: sqlite3.Connection,
    sources: list[dict[str, Any]],
    trend_tokens: set[str],
    max_chars: int,
) -> list[dict[str, Any]]:
    ranked = []
    now = utc_now()
    for source in sources:
        row = conn.execute(
            "SELECT last_selected_at FROM library_sources WHERE source_key = ?", (source["source_key"],)
        ).fetchone()
        source_tokens = normalized_tokens(source_text_for_match(source, max_chars))
        overlap = len(source_tokens & trend_tokens)
        recency_penalty = 0.0
        last_selected = parse_time(row["last_selected_at"] if row else "")
        if last_selected and now - last_selected < timedelta(hours=24):
            recency_penalty = 100.0
        ranked.append(
            {
                **source,
                "trend_overlap": overlap,
                "selection_score": round(float(source["priority"]) * 10 + overlap - recency_penalty, 3),
            }
        )
    return sorted(
        ranked,
        key=lambda item: (item["selection_score"], item["ready"], item["priority"], item["source_title"]),
        reverse=True,
    )


def next_checkpoint(published_at: datetime, now: datetime) -> tuple[str, datetime | None, str]:
    for hours in CHECKPOINT_HOURS:
        due = published_at + timedelta(hours=hours)
        if due > now:
            return f"{hours}h", due, "watching"
    return "7d", None, "learned"


def checkpoint_for_observation(published_at: datetime, observed_at: datetime) -> str:
    elapsed_hours = max(0.0, (observed_at - published_at).total_seconds() / 3600.0)
    if elapsed_hours >= 24:
        return "24h"
    if elapsed_hours >= 12:
        return "12h"
    if elapsed_hours >= 6:
        return "6h"
    if elapsed_hours >= 3:
        return "3h"
    return "1h"


def register_published_short(
    conn: sqlite3.Connection,
    *,
    youtube_video_id: str,
    package_path: Path,
    published_at: datetime,
) -> dict[str, Any]:
    package = read_json(package_path, {})
    source_key = ""
    package_row = conn.execute(
        "SELECT source_key FROM generated_packages WHERE package_path = ?", (str(package_path.resolve()),)
    ).fetchone()
    if package_row:
        source_key = package_row["source_key"]
    elif isinstance(package.get("orchestration"), dict):
        source_key = str(package["orchestration"].get("source_key") or "")
    checkpoint, due, status = next_checkpoint(published_at, utc_now())
    conn.execute(
        """
        INSERT INTO published_shorts (
            youtube_video_id, package_path, source_key, published_at, status, next_check_at, last_checked_at
        ) VALUES (?, ?, ?, ?, ?, ?, NULL)
        ON CONFLICT(youtube_video_id) DO UPDATE SET
            package_path=excluded.package_path,
            source_key=excluded.source_key,
            published_at=excluded.published_at,
            status=excluded.status,
            next_check_at=excluded.next_check_at
        """,
        (
            youtube_video_id,
            str(package_path.resolve()),
            source_key or None,
            iso_time(published_at),
            status,
            iso_time(due) if due else None,
        ),
    )
    conn.commit()
    return {"youtube_video_id": youtube_video_id, "next_checkpoint": checkpoint, "status": status}


def record_metrics_snapshot(
    conn: sqlite3.Connection,
    *,
    youtube_video_id: str,
    metrics: dict[str, Any],
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    published = conn.execute(
        "SELECT published_at FROM published_shorts WHERE youtube_video_id = ?", (youtube_video_id,)
    ).fetchone()
    if not published:
        raise RuntimeError(f"Unknown published short: {youtube_video_id}")
    published_at = parse_time(published["published_at"])
    if not published_at:
        raise RuntimeError(f"Invalid published_at for {youtube_video_id}")
    observed = observed_at or utc_now()
    checkpoint = checkpoint_for_observation(published_at, observed)
    _next_label, next_due, status = next_checkpoint(published_at, observed)
    conn.execute(
        """
        INSERT INTO metrics_snapshots (youtube_video_id, observed_at, checkpoint, metrics_json)
        VALUES (?, ?, ?, ?)
        """,
        (youtube_video_id, iso_time(observed), checkpoint, json.dumps(metrics, ensure_ascii=False, sort_keys=True)),
    )
    conn.execute(
        """
        UPDATE published_shorts
        SET status = ?, next_check_at = ?, last_checked_at = ?
        WHERE youtube_video_id = ?
        """,
        (status, iso_time(next_due) if next_due else None, iso_time(observed), youtube_video_id),
    )
    conn.commit()
    return {"checkpoint": checkpoint, "status": status, "next_check_at": iso_time(next_due) if next_due else ""}


def reschedule_metric_checks(conn: sqlite3.Connection, *, now: datetime | None = None) -> int:
    """Apply the current short-form checkpoint policy to existing uploads."""
    observed = now or utc_now()
    rows = conn.execute(
        "SELECT youtube_video_id, published_at FROM published_shorts "
        "WHERE status != 'archived_previous_channel'"
    ).fetchall()
    changed = 0
    for row in rows:
        published_at = parse_time(str(row["published_at"] or ""))
        if not published_at:
            continue
        _label, next_due, status = next_checkpoint(published_at, observed)
        conn.execute(
            "UPDATE published_shorts SET status = ?, next_check_at = ? WHERE youtube_video_id = ?",
            (status, iso_time(next_due) if next_due else None, row["youtube_video_id"]),
        )
        changed += 1
    conn.commit()
    return changed


def load_youtube_credentials(settings: dict[str, Any], *, interactive: bool):
    try:
        from google.auth.transport.requests import Request as GoogleRequest
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise RuntimeError("Install Google API dependencies from requirements.txt before enabling analytics sync.") from exc

    client_secret = resolve_path(settings.get("oauth_client_secret_path"))
    token_path = resolve_path(settings.get("token_path"))
    if not client_secret or not client_secret.exists():
        raise RuntimeError("YouTube OAuth client secret file is missing.")
    if not token_path:
        raise RuntimeError("YouTube OAuth token path is missing.")

    credentials = None
    if token_path.exists():
        credentials = Credentials.from_authorized_user_file(str(token_path), ANALYTICS_SCOPES)
    if credentials and not credentials.has_scopes(ANALYTICS_SCOPES):
        credentials = None
    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(GoogleRequest())
    elif not credentials or not credentials.valid:
        if not interactive:
            raise RuntimeError("YouTube OAuth authorization is required before daily analytics sync can run.")
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), ANALYTICS_SCOPES)
        credentials = flow.run_local_server(port=0)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(credentials.to_json(), encoding="utf-8")
    return credentials


def build_youtube_analytics_service(settings: dict[str, Any], *, interactive: bool):
    try:
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError("Install Google API dependencies from requirements.txt before enabling analytics sync.") from exc
    credentials = load_youtube_credentials(settings, interactive=interactive)
    return build("youtubeAnalytics", "v2", credentials=credentials, cache_discovery=False)


def build_youtube_upload_service(settings: dict[str, Any], *, interactive: bool):
    try:
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError("Install Google API dependencies from requirements.txt before enabling YouTube upload.") from exc
    credentials = load_youtube_credentials(settings, interactive=interactive)
    return build("youtube", "v3", credentials=credentials, cache_discovery=False)


def analytics_rows_to_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    headers = [str(item.get("name") or "") for item in payload.get("columnHeaders", []) or []]
    rows = payload.get("rows", []) or []
    if not headers or not rows:
        return {"views": 0, "engagedViews": 0, "averageViewDuration": 0.0, "averageViewPercentage": 0.0}
    values = rows[0]
    return {headers[index]: values[index] for index in range(min(len(headers), len(values)))}


def sync_due_youtube_metrics(conn: sqlite3.Connection, settings: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    now = utc_now()
    rescheduled = 0 if dry_run else reschedule_metric_checks(conn, now=now)
    due_rows = conn.execute(
        """
        SELECT youtube_video_id, published_at
        FROM published_shorts
        WHERE next_check_at IS NOT NULL AND next_check_at <= ?
        ORDER BY next_check_at
        """,
        (iso_time(now),),
    ).fetchall()
    if not due_rows:
        return {"status": "nothing_due", "synced": 0, "rescheduled": rescheduled}
    if not settings.get("enabled", False):
        return {"status": "not_configured", "due": len(due_rows), "synced": 0, "rescheduled": rescheduled}
    if dry_run:
        return {"status": "dry_run", "due": len(due_rows), "synced": 0, "rescheduled": rescheduled}

    service = build_youtube_analytics_service(
        settings,
        interactive=bool(settings.get("interactive_on_first_run", False)),
    )
    synced = 0
    errors = []
    for row in due_rows:
        published_at = parse_time(row["published_at"])
        if not published_at:
            errors.append(f"{row['youtube_video_id']}: invalid published time")
            continue
        try:
            payload = (
                service.reports()
                .query(
                    ids="channel==MINE",
                    startDate=published_at.date().isoformat(),
                    endDate=now.date().isoformat(),
                    metrics="views,engagedViews,averageViewDuration,averageViewPercentage,likes,comments,shares",
                    filters=f"video=={row['youtube_video_id']}",
                )
                .execute()
            )
            record_metrics_snapshot(
                conn,
                youtube_video_id=row["youtube_video_id"],
                metrics=analytics_rows_to_metrics(payload),
                observed_at=now,
            )
            synced += 1
        except Exception as exc:
            errors.append(f"{row['youtube_video_id']}: {exc}")
    return {
        "status": "completed" if not errors else "partial",
        "due": len(due_rows),
        "synced": synced,
        "errors": errors,
        "rescheduled": rescheduled,
    }


def latest_evaluated_metrics(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT metrics_snapshots.youtube_video_id, metrics_snapshots.checkpoint,
               metrics_snapshots.metrics_json, metrics_snapshots.observed_at,
               published_shorts.package_path
        FROM metrics_snapshots
        JOIN published_shorts
          ON published_shorts.youtube_video_id = metrics_snapshots.youtube_video_id
        WHERE metrics_snapshots.checkpoint IN ('1h', '3h', '6h', '12h', '24h')
          AND published_shorts.status IN ('watching', 'learned')
        ORDER BY observed_at DESC
        """
    ).fetchall()
    results = []
    seen = set()
    for row in rows:
        video_id = row["youtube_video_id"]
        if video_id in seen:
            continue
        metrics = read_json_text(row["metrics_json"])
        if not isinstance(metrics, dict):
            continue
        seen.add(video_id)
        package = read_json(Path(str(row["package_path"] or "")), {})
        experiment = package.get("experiment", {}) if isinstance(package, dict) else {}
        results.append(
            {
                "youtube_video_id": video_id,
                "checkpoint": row["checkpoint"],
                "metrics": metrics,
                "experiment": experiment if isinstance(experiment, dict) else {},
            }
        )
    return results


def read_json_text(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def last_rule_kind(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT rule_kind FROM learning_rules ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    return str(row["rule_kind"]) if row else ""


def recent_review_feedback(conn: sqlite3.Connection, limit: int = 5) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT review_status, reviewer_note, short_id, score, output_path, decided_at, created_at
        FROM review_items
        WHERE review_status IN ('revision_requested', 'rejected')
            AND COALESCE(reviewer_note, '') != ''
            AND reviewer_note NOT LIKE 'Rejected automatically:%'
            AND reviewer_note NOT LIKE 'Paused:%'
            AND reviewer_note NOT LIKE 'Deleted after copyright claim%'
        ORDER BY COALESCE(decided_at, created_at) DESC
        LIMIT ?
        """,
        (max(1, limit),),
    ).fetchall()
    return [dict(row) for row in rows]


def derive_learning_rule(conn: sqlite3.Connection, config: dict[str, Any], run_id: str) -> dict[str, Any]:
    learning = config.get("learning", {}) if isinstance(config.get("learning"), dict) else {}
    minimum = max(1, int(learning.get("min_evaluated_videos", 3) or 3))
    low_percentage = float(learning.get("low_average_view_percentage", 60.0) or 60.0)
    evaluated = latest_evaluated_metrics(conn)
    review_feedback = recent_review_feedback(conn)
    previous_kind = last_rule_kind(conn)
    mode = "baseline"
    rule_kind = "baseline_collect_signal"
    usable_evaluated = [
        item
        for item in evaluated
        if isinstance(item.get("metrics"), dict) and float(item["metrics"].get("views") or 0) > 0
    ]
    evidence: dict[str, Any] = {
        "evaluated_video_count": len(usable_evaluated),
        "ignored_zero_view_rows": len(evaluated) - len(usable_evaluated),
    }
    if review_feedback:
        evidence["recent_review_feedback"] = review_feedback
    constraints = [
        "Use the strongest source-specific trigger in the first two cuts.",
        "Preserve a clear setup-to-payoff thread; do not pad with unrelated context.",
    ]
    change_note = "No comparable published performance set exists yet; record this run as the baseline."

    percentages = []
    for item in usable_evaluated:
        try:
            percentages.append(float(item["metrics"].get("averageViewPercentage")))
        except (TypeError, ValueError):
            continue
    experiment_results = []
    for item in usable_evaluated[:5]:
        experiment = item.get("experiment", {}) if isinstance(item.get("experiment"), dict) else {}
        if not experiment:
            continue
        metrics = item.get("metrics", {}) if isinstance(item.get("metrics"), dict) else {}
        experiment_results.append(
            {
                "checkpoint": item.get("checkpoint"),
                "hypothesis": str(experiment.get("hypothesis") or "")[:180],
                "primary_variable": str(experiment.get("primary_variable") or ""),
                "average_view_percentage": metrics.get("averageViewPercentage"),
                "views": metrics.get("views"),
            }
        )
    if experiment_results:
        evidence["recent_experiment_results"] = experiment_results
    if len(usable_evaluated) >= minimum and percentages:
        median_percentage = round(float(statistics.median(percentages)), 3)
        evidence["median_average_view_percentage"] = median_percentage
        mode = "experiment"
        if median_percentage < low_percentage and previous_kind == "hook_reaction_first":
            rule_kind = "hook_visual_question_first"
            constraints = [
                "Open with a visually legible odd image, gesture, or unresolved question within the first 1.5 seconds.",
                "Use the next two cuts to recover who, what changed, and why the payoff matters.",
                "Do not open on a generic greeting, title card, or long spoken setup.",
            ]
            change_note = "The prior reaction-first test did not clear the retention threshold, so this run tests a clearer visual question first."
        elif median_percentage < low_percentage:
            rule_kind = "hook_reaction_first"
            constraints = [
                "Open with the clearest audible or visible reaction/payoff within the first 1.5 seconds.",
                "Recover the minimum setup in the next one or two cuts.",
                "Do not open with a greeting, generic explanation, or slow chronological setup.",
            ]
            change_note = "Recent comparable uploads retained too little viewing; this run moves the reaction/payoff before context recovery."
        else:
            rule_kind = "payoff_micro_montage"
            constraints = [
                "Keep the established hook structure, then test a tighter trigger-to-payoff micro-montage.",
                "Every middle cut must either recover context, escalate the mismatch, or show reaction.",
                "Remove adjacent dialogue that does not change the payoff.",
            ]
            change_note = "Retention is above the current threshold; this run keeps the working hook and tests tighter payoff density."

    if review_feedback:
        feedback_lines = [
            f"{item.get('review_status')}: {item.get('reviewer_note')}"
            for item in review_feedback
            if str(item.get("reviewer_note") or "").strip()
        ]
        constraints.extend(
            [
                "Apply the latest human content review before scoring any candidate.",
                "If a prior render was rejected or sent for revision, avoid repeating the same content weakness.",
                "Prefer clips whose standalone situation, payoff, and title promise are obvious without needing the viewer to know the full episode.",
                "Recent human review notes: " + " | ".join(feedback_lines[:3]),
            ]
        )
        if len(usable_evaluated) < minimum:
            mode = "review_feedback"
            rule_kind = "human_review_revision"
        change_note = "Human review feedback is available; this run must correct the prior content-selection weakness before testing performance."

    if experiment_results:
        constraints.append(
            "Treat recent experiment results as evidence, not a fixed template: keep the successful mechanism, but test one clearly named editing variable per package."
        )

    created_at = utc_now()
    rule_id = f"{rule_kind}_{created_at.strftime('%Y%m%dT%H%M%SZ')}"
    rule = {
        "rule_id": rule_id,
        "run_id": run_id,
        "created_at": iso_time(created_at),
        "mode": mode,
        "rule_kind": rule_kind,
        "evidence": evidence,
        "constraints": constraints,
        "change_note": change_note,
        "previous_rule_kind": previous_kind,
    }
    conn.execute(
        """
        INSERT INTO learning_rules (rule_id, run_id, created_at, mode, rule_kind, rule_json, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (rule_id, run_id, rule["created_at"], mode, rule_kind, json.dumps(rule, ensure_ascii=False), "active"),
    )
    conn.commit()
    return rule


def ensure_source_analysis(source: dict[str, Any], *, execute: bool) -> dict[str, Any]:
    if source["ready"]:
        return {"status": "ready", "analysis_dir": str(source["analysis_dir"])}
    if not execute:
        return {"status": "planned", "analysis_dir": str(source["analysis_dir"])}
    source_video = source.get("source_video")
    if not source_video or not source_video.exists():
        return {"status": "blocked", "reason": "missing merged transcript and source video"}
    command = [
        project_python_executable(),
        "-u",
        str(BASE_DIR / "analyze_longform.py"),
        "--source-video",
        str(source_video),
        "--output-dir",
        str(source["analysis_dir"]),
    ]
    completed = subprocess.run(command, cwd=BASE_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        return {"status": "failed", "reason": completed.stderr.strip() or completed.stdout.strip()}
    source["ready"] = (source["analysis_dir"] / "merged" / "merged_transcript.json").exists()
    return {"status": "ready" if source["ready"] else "failed", "analysis_dir": str(source["analysis_dir"])}


def ensure_visual_event_script(config: dict[str, Any], source: dict[str, Any], *, execute: bool) -> dict[str, Any]:
    settings = config.get("visual_events", {}) if isinstance(config.get("visual_events"), dict) else {}
    if not bool(settings.get("enabled", True)):
        return {"status": "disabled"}
    output_path = source["analysis_dir"] / "visual_events" / "visual_event_script.json"
    if output_path.exists() and not bool(settings.get("force", False)):
        return {"status": "ready", "path": str(output_path), "cached": True}
    if not execute:
        return {"status": "planned", "path": str(output_path)}
    source_video = source.get("source_video")
    if not source_video or not source_video.exists():
        return {"status": "blocked", "reason": "missing source video for visual event script"}
    command = [
        project_python_executable(),
        "-u",
        str(BASE_DIR / "build_visual_event_script.py"),
        "--source-video",
        str(source_video),
        "--analysis-dir",
        str(source["analysis_dir"]),
        "--model",
        str(settings.get("model") or "gpt-5.4"),
        "--sample-interval-sec",
        str(max(4, int(settings.get("sample_interval_sec", 12) or 12))),
        "--frames-per-batch",
        str(max(2, int(settings.get("frames_per_batch", 6) or 6))),
        "--max-frames",
        str(max(1, int(settings.get("max_frames", 180) or 180))),
    ]
    generation = config.get("generation", {}) if isinstance(config.get("generation"), dict) else {}
    total_budget = float(generation.get("run_budget_usd", 0.0) or 0.0)
    final_reserve = float(generation.get("final_package_reserve_usd", 0.0) or 0.0)
    safety = float(generation.get("budget_safety_usd", 0.0) or 0.0)
    visual_budget = max(0.0, total_budget - final_reserve - safety)
    if visual_budget > 0:
        command.extend(["--max-estimated-usd", f"{visual_budget:.3f}"])
    if bool(settings.get("force", False)):
        command.append("--force")
    completed = subprocess.run(
        command,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=subprocess_environment(config),
    )
    if completed.returncode != 0:
        return {"status": "failed", "error": completed.stderr.strip() or completed.stdout.strip()}
    return {"status": "ready" if output_path.exists() else "failed", "path": str(output_path), "log_tail": completed.stdout.splitlines()[-8:]}


def annotate_packages(
    conn: sqlite3.Connection,
    *,
    source: dict[str, Any],
    run_id: str,
    learning_rule: dict[str, Any],
    package_root: Path,
) -> dict[str, Any]:
    aggregate_path = package_root / "final" / "shorts_packages.json"
    payload = read_json(aggregate_path, {})
    if not isinstance(payload, dict):
        return {"status": "missing_output", "path": str(aggregate_path)}
    orchestration = {
        "run_id": run_id,
        "source_key": source["source_key"],
        "learning_rule_id": learning_rule["rule_id"],
        "learning_rule_kind": learning_rule["rule_kind"],
        "change_note": learning_rule["change_note"],
    }
    payload["orchestration"] = orchestration
    write_json(aggregate_path, payload)
    recorded = 0
    for package in payload.get("shorts", []) or []:
        if not isinstance(package, dict):
            continue
        package["orchestration"] = orchestration
        rank = int(package.get("global_rank", 0) or 0)
        package_path = package_root / "final" / f"short_{rank:02d}.json"
        if package_path.exists():
            individual = read_json(package_path, {})
            if isinstance(individual, dict):
                individual["orchestration"] = orchestration
                write_json(package_path, individual)
        scorecard = package.get("genre_scorecard", {}) if isinstance(package.get("genre_scorecard"), dict) else {}
        conn.execute(
            """
            INSERT INTO generated_packages (package_path, source_key, run_id, short_id, score, decision, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(package_path) DO UPDATE SET
                source_key=excluded.source_key,
                run_id=excluded.run_id,
                short_id=excluded.short_id,
                score=excluded.score,
                decision=excluded.decision,
                created_at=excluded.created_at
            """,
            (
                str(package_path.resolve()),
                source["source_key"],
                run_id,
                str(package.get("short_id") or ""),
                int(package.get("score", 0) or 0),
                str(scorecard.get("decision") or ""),
                iso_time(),
            ),
        )
        recorded += 1
    conn.commit()
    return {"status": "completed", "path": str(aggregate_path), "package_count": recorded}


def generate_for_source(
    conn: sqlite3.Connection,
    config: dict[str, Any],
    source: dict[str, Any],
    run_id: str,
    learning_rule_path: Path,
    learning_rule: dict[str, Any],
    *,
    execute: bool,
) -> dict[str, Any]:
    if not execute:
        return {"status": "planned", "analysis_dir": str(source["analysis_dir"])}
    generation = config.get("generation", {}) if isinstance(config.get("generation"), dict) else {}
    batch_id = re.sub(r"[^0-9A-Za-z_-]+", "_", run_id).strip("_") or "batch"
    package_root = source["analysis_dir"] / "shorts_candidates" / "batches" / batch_id
    command = [
        project_python_executable(),
        "-u",
        str(BASE_DIR / "generate_shorts_packages.py"),
        "--analysis-dir",
        str(source["analysis_dir"]),
        "--source-title",
        source["source_title"],
        "--model",
        str(generation.get("candidate_model") or generation.get("model") or "gpt-4.1-mini"),
        "--final-model",
        str(generation.get("final_model") or generation.get("candidate_model") or generation.get("model") or "gpt-4.1-mini"),
        "--learning-rule",
        str(learning_rule_path),
        "--output-dir",
        str(package_root),
    ]
    if bool(generation.get("timeline_first", True)):
        command.append("--timeline-first")
        command.extend(
            [
                "--final-candidate-limit",
                str(max(1, min(8, int(generation.get("final_candidate_limit", 5) or 5)))),
            ]
        )
        refinement = generation.get("candidate_visual_refinement", {})
        refinement = refinement if isinstance(refinement, dict) else {}
        if not bool(refinement.get("enabled", True)):
            command.append("--no-candidate-visual-refinement")
        else:
            command.extend(
                [
                    "--visual-refinement-model",
                    str(refinement.get("model") or "gpt-5.4"),
                    "--visual-refinement-frames-per-clip",
                    str(max(1, min(3, int(refinement.get("frames_per_clip", 2) or 2)))),
                    "--visual-refinement-candidate-limit",
                    str(max(1, min(12, int(refinement.get("candidate_limit", 10) or 10)))),
                ]
            )
        package_budget = float(generation.get("final_package_reserve_usd", 0.0) or 0.0)
        if package_budget > 0:
            command.extend(["--package-budget-usd", f"{package_budget:.3f}"])
    benchmark_profile = resolve_path(config.get("benchmark_profile"))
    if benchmark_profile and benchmark_profile.exists():
        command.extend(["--benchmark-profile", str(benchmark_profile)])
    context_path = source["analysis_dir"] / "youtube_context.json"
    if context_path.exists():
        command.extend(["--youtube-context", str(context_path)])
    movie_info_path = source["analysis_dir"] / "movie_info.json"
    if movie_info_path.exists():
        command.extend(["--movie-info", str(movie_info_path)])
    if not bool(generation.get("include_wide_windows", True)):
        command.append("--no-wide-windows")
    if bool(generation.get("force", False)):
        command.append("--force")
    completed = subprocess.run(
        command,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=subprocess_environment(config),
    )
    if completed.returncode != 0:
        return {"status": "failed", "error": completed.stderr.strip() or completed.stdout.strip()}
    annotated = annotate_packages(
        conn,
        source=source,
        run_id=run_id,
        learning_rule=learning_rule,
        package_root=package_root,
    )
    return {
        "status": annotated["status"],
        "output": annotated,
        "batch_id": batch_id,
        "package_root": str(package_root),
        "log_tail": completed.stdout.splitlines()[-12:],
    }


def render_for_source(
    config: dict[str, Any],
    source: dict[str, Any],
    *,
    package_root: Path | None = None,
    batch_id: str = "",
    execute: bool,
) -> dict[str, Any]:
    production = config.get("production", {}) if isinstance(config.get("production"), dict) else {}
    if not bool(production.get("enabled", True)):
        return {"status": "disabled"}
    if not execute:
        return {"status": "planned"}
    source_video = source.get("source_video")
    if not source_video or not source_video.exists():
        return {"status": "blocked", "reason": "owned source video is unavailable for rendering"}

    package_root = package_root or (source["analysis_dir"] / "shorts_candidates")
    aggregate_path = package_root / "final" / "shorts_packages.json"
    aggregate = read_json(aggregate_path, {})
    packages = aggregate.get("shorts", []) if isinstance(aggregate, dict) else []
    # A cost cap or a manual stop can happen after individual final packages
    # have been safely written but before the aggregate index is emitted.
    # Those already-validated packages are renderable; do not make the user
    # pay to regenerate them merely to recreate an index file.
    if not isinstance(packages, list) or not packages:
        final_dir = package_root / "final"
        packages = [
            read_json(path, {})
            for path in sorted(final_dir.glob("short_*.json"))
            if path.is_file()
        ]
    if not isinstance(packages, list):
        return {"status": "missing_output", "path": str(aggregate_path)}

    minimum_score = int(production.get("minimum_score", 85) or 0)
    minimum_visual_score = int(production.get("minimum_visual_shortability_score", 6) or 0)
    minimum_duration = float(production.get("minimum_duration_sec_exclusive", 20.0) or 20.0)
    required_minimum = production_minimum_render_count(production, source)
    allowed_decisions = {str(value) for value in production.get("allowed_decisions", ["auto_render"]) or []}
    selected: list[dict[str, Any]] = []
    for package in sorted(
        (item for item in packages if isinstance(item, dict)),
        # The first package is the deliberate controlled experiment.  Do not
        # silently replace it with a higher-score non-experiment package, or
        # a required narration/reframe test can be generated on paper but
        # never reach a real rendered review candidate.
        key=lambda item: (
            int(item.get("global_rank", 999) or 999) == 1
            and bool(item.get("narration") or []),
            int(item.get("score", 0) or 0),
            -int(item.get("global_rank", 999) or 999),
        ),
        reverse=True,
    ):
        score = int(package.get("score", 0) or 0)
        scorecard = package.get("genre_scorecard", {}) if isinstance(package.get("genre_scorecard"), dict) else {}
        decision = str(scorecard.get("decision") or package.get("decision") or "")
        if score < minimum_score or (allowed_decisions and decision not in allowed_decisions):
            continue
        dimensions = scorecard.get("dimensions", []) if isinstance(scorecard.get("dimensions"), list) else []
        visual_score = next(
            (
                int(dimension.get("score", 0) or 0)
                for dimension in dimensions
                if isinstance(dimension, dict) and str(dimension.get("id") or "") == "visual_shortability"
            ),
            None,
        )
        if visual_score is not None and visual_score < minimum_visual_score:
            continue
        short_id = str(package.get("short_id") or "").strip()
        if not short_id:
            continue
        package_path = package_root / "final" / f"{short_id}.json"
        if not package_path.exists():
            continue
        package_detail = read_json(package_path, {})
        try:
            planned_duration = float(package_detail.get("target_duration_sec") or package.get("target_duration_sec") or 0.0)
        except (TypeError, ValueError):
            planned_duration = 0.0
        if planned_duration <= 0:
            planned_duration = sum(
                max(0.0, float(clip.get("end_sec", 0.0)) - float(clip.get("start_sec", 0.0)))
                for clip in package_detail.get("source_clips", []) or []
                if isinstance(clip, dict)
            )
        if planned_duration <= minimum_duration:
            continue
        selected.append(
            {
                "short_id": short_id,
                "package_path": str(package_path),
                "score": score,
                "decision": decision,
                "planned_duration_sec": planned_duration,
            }
        )
        if len(selected) >= required_minimum:
            break
    if not selected:
        return {
            "status": "no_auto_render_candidate",
            "minimum_score": minimum_score,
            "minimum_duration_sec_exclusive": minimum_duration,
            "required_minimum": required_minimum,
        }

    rendered: list[dict[str, Any]] = []
    for candidate in selected:
        output_base = source["analysis_dir"] / "productions"
        if batch_id:
            output_base = output_base / "batches" / batch_id
        output_path = output_base / f"{candidate['short_id']}.mp4"
        command = [
            project_python_executable(),
            "-u",
            str(BASE_DIR / "render_short.py"),
            "--source-video",
            str(source_video),
            "--package",
            str(candidate["package_path"]),
            "--output",
            str(output_path),
        ]
        completed = subprocess.run(
            command,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=subprocess_environment(config),
        )
        output_is_valid = output_path.exists() and output_path.is_file() and output_path.stat().st_size > 1024
        entry = {
            **candidate,
            "output_path": str(output_path),
            "status": "completed" if completed.returncode == 0 and output_is_valid else "failed",
            "log_tail": completed.stdout.splitlines()[-8:],
        }
        if completed.returncode != 0:
            entry["error"] = completed.stderr.strip() or completed.stdout.strip()
        elif not output_is_valid:
            entry["error"] = "Renderer returned success but did not produce a usable MP4 (missing or too small)."
        rendered.append(entry)
    completed_count = sum(1 for item in rendered if item["status"] == "completed")
    return {
        "status": "completed" if completed_count == len(rendered) else "partial",
        "rendered": rendered,
        "batch_id": batch_id,
        "package_root": str(package_root),
        "required_minimum": required_minimum,
        "minimum_met": completed_count >= required_minimum,
        "minimum_shortfall": max(0, required_minimum - completed_count),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def probe_video_output(output_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height:format=duration",
        "-of",
        "json",
        str(output_path),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except FileNotFoundError:
        # Windows installs used for rendering often ship ffmpeg through
        # imageio-ffmpeg but not a separate ffprobe executable.  Still enforce
        # the duration gate from the rendered media rather than silently
        # downgrading it to a warning.
        try:
            import imageio_ffmpeg  # type: ignore

            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            fallback = subprocess.run(
                [ffmpeg_path, "-hide_banner", "-i", str(output_path), "-f", "null", "-"],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
            )
        except (ImportError, FileNotFoundError):
            return {"status": "unavailable", "reason": "ffprobe and ffmpeg are unavailable"}
        except subprocess.TimeoutExpired:
            return {"status": "failed", "reason": "ffmpeg timed out"}
        text = "\n".join((fallback.stdout, fallback.stderr))
        duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
        size_match = re.search(r"(\d{2,5})x(\d{2,5})", text)
        if not duration_match:
            return {"status": "failed", "reason": "ffmpeg could not read the rendered media"}
        hours, minutes, seconds = duration_match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        return {
            "status": "completed",
            "width": int(size_match.group(1)) if size_match else 0,
            "height": int(size_match.group(2)) if size_match else 0,
            "duration": duration,
            "probe_backend": "ffmpeg",
        }
    except subprocess.TimeoutExpired:
        return {"status": "failed", "reason": "ffprobe timed out"}
    if completed.returncode != 0:
        return {"status": "failed", "reason": completed.stderr.strip() or completed.stdout.strip()}
    payload = read_json_text(completed.stdout)
    if not isinstance(payload, dict):
        return {"status": "failed", "reason": "ffprobe returned invalid JSON"}
    streams = payload.get("streams", []) if isinstance(payload.get("streams"), list) else []
    stream = streams[0] if streams and isinstance(streams[0], dict) else {}
    fmt = payload.get("format", {}) if isinstance(payload.get("format"), dict) else {}
    try:
        duration = float(fmt.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    return {
        "status": "completed",
        "width": int(stream.get("width", 0) or 0),
        "height": int(stream.get("height", 0) or 0),
        "duration": duration,
    }


def inspect_rendered_footage(output_path: Path, *, sample_count: int = 5) -> dict[str, Any]:
    """Measure the actual footage area, excluding the permanent header/footer.

    A technically valid 9:16 encode can still be an unusable Short when a
    browser/document slide occupies most of the visible frame.  This catches
    that class of failure before a human approval or upload is possible.
    """
    cap = cv2.VideoCapture(str(output_path))
    if not cap.isOpened():
        return {"status": "unavailable", "reason": "OpenCV could not read rendered video"}
    try:
        frame_count = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        indices = sorted({int(round((frame_count - 1) * ratio)) for ratio in np.linspace(0.12, 0.88, max(3, sample_count))})
        bright_ratios: list[float] = []
        motion_scores: list[float] = []
        previous_gray = None
        for frame_index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            top = max(0, min(height - 1, round(height * (260 / 1920))))
            bottom = max(top + 1, min(height, round(height * (1488 / 1920))))
            footage = frame[top:bottom, :]
            if footage.size == 0:
                continue
            bright_mask = np.all(footage >= 242, axis=2)
            bright_ratios.append(float(np.mean(bright_mask)))
            gray = cv2.cvtColor(footage, cv2.COLOR_BGR2GRAY)
            if previous_gray is not None and previous_gray.shape == gray.shape:
                motion_scores.append(float(np.mean(cv2.absdiff(gray, previous_gray))))
            previous_gray = gray
        if not bright_ratios:
            return {"status": "unavailable", "reason": "No representative footage frames were readable"}
        return {
            "status": "completed",
            "sampled_frames": len(bright_ratios),
            "bright_document_ratio_median": round(float(np.median(bright_ratios)), 4),
            "bright_document_ratio_max": round(float(max(bright_ratios)), 4),
            "low_motion_score_median": round(float(np.median(motion_scores)), 4) if motion_scores else 0.0,
        }
    finally:
        cap.release()


def auto_review_render(
    output_path: Path,
    *,
    minimum_duration_exclusive: float = 20.0,
    visual_qa: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks: dict[str, Any] = {"output_path": str(output_path)}
    issues: list[str] = []
    warnings: list[str] = []
    if not output_path.exists():
        issues.append("rendered file does not exist")
        return {"qa_status": "fail", "checks": checks, "issues": issues, "warnings": warnings}

    size_bytes = output_path.stat().st_size
    checks["size_bytes"] = size_bytes
    if size_bytes < 50_000:
        warnings.append("rendered file is unusually small")

    probe = probe_video_output(output_path)
    checks["probe"] = probe
    if probe.get("status") == "completed":
        if probe.get("width") != 1080 or probe.get("height") != 1920:
            issues.append(f"expected 1080x1920 video, got {probe.get('width')}x{probe.get('height')}")
        duration = float(probe.get("duration") or 0.0)
        if duration <= 0.5:
            issues.append("rendered duration is too short")
        elif duration <= minimum_duration_exclusive:
            issues.append(f"rendered duration must be longer than {minimum_duration_exclusive:g} seconds")
    elif probe.get("status") == "unavailable":
        warnings.append(str(probe.get("reason") or "ffprobe unavailable"))
    else:
        warnings.append(str(probe.get("reason") or "ffprobe failed"))

    visual_settings = visual_qa or {}
    if bool(visual_settings.get("enabled", True)):
        visual = inspect_rendered_footage(
            output_path,
            sample_count=max(3, int(visual_settings.get("sample_count", 5) or 5)),
        )
        checks["visual_footage"] = visual
        if visual.get("status") == "completed":
            maximum_ratio = float(visual_settings.get("max_bright_document_ratio", 0.25) or 0.25)
            observed_ratio = float(visual.get("bright_document_ratio_median") or 0.0)
            if observed_ratio > maximum_ratio:
                issues.append(
                    f"visible footage is {observed_ratio:.0%} bright document/empty area; maximum is {maximum_ratio:.0%}"
                )
        else:
            warnings.append(str(visual.get("reason") or "visual footage inspection unavailable"))

    qa_status = "fail" if issues else ("warning" if warnings else "pass")
    return {"qa_status": qa_status, "checks": checks, "issues": issues, "warnings": warnings}


def register_review_items(
    conn: sqlite3.Connection,
    *,
    source: dict[str, Any],
    run_id: str,
    render: dict[str, Any],
    minimum_duration_exclusive: float = 20.0,
    visual_qa: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rendered = [item for item in render.get("rendered", []) or [] if isinstance(item, dict)]
    entries: list[dict[str, Any]] = []
    for item in rendered:
        if item.get("status") != "completed" or not item.get("output_path"):
            continue
        output_path = Path(str(item["output_path"])).resolve()
        package_path = Path(str(item.get("package_path") or "")).resolve()
        qa = auto_review_render(
            output_path,
            minimum_duration_exclusive=minimum_duration_exclusive,
            visual_qa=visual_qa,
        )
        if output_path.exists():
            content_sha256 = file_sha256(output_path)
        else:
            content_sha256 = hashlib.sha256(f"{run_id}|{output_path}".encode("utf-8")).hexdigest()
        review_status = "needs_review" if qa["qa_status"] in {"pass", "warning"} else "qa_failed"
        entry = {
            "content_sha256": content_sha256,
            "output_path": str(output_path),
            "package_path": str(package_path),
            "source_key": source["source_key"],
            "run_id": run_id,
            "short_id": str(item.get("short_id") or ""),
            "score": int(item.get("score", 0) or 0),
            "decision": str(item.get("decision") or ""),
            "auto_qa_status": qa["qa_status"],
            "auto_qa": qa,
            "review_status": review_status,
        }
        # Re-rendering a candidate changes its content hash.  Keep the old
        # review record for auditability, but remove it from actionable review
        # queues so the GUI/Telegram never show two decisions for the same
        # output file.
        conn.execute(
            """
            UPDATE review_items
            SET review_status = 'superseded', reviewer_note = ?
            WHERE output_path = ?
              AND content_sha256 != ?
              AND review_status IN ('needs_review', 'qa_failed')
            """,
            ("Superseded by a newer render of the same candidate.", str(output_path), content_sha256),
        )
        conn.execute(
            """
            INSERT INTO review_items (
                content_sha256, output_path, package_path, source_key, run_id, short_id, score, decision,
                auto_qa_status, auto_qa_json, review_status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(content_sha256) DO UPDATE SET
                output_path=excluded.output_path,
                package_path=excluded.package_path,
                source_key=excluded.source_key,
                run_id=excluded.run_id,
                short_id=excluded.short_id,
                score=excluded.score,
                decision=excluded.decision,
                auto_qa_status=excluded.auto_qa_status,
                auto_qa_json=excluded.auto_qa_json,
                review_status=CASE
                    WHEN review_items.review_status IN ('approved', 'rejected', 'revision_requested')
                    THEN review_items.review_status
                    ELSE excluded.review_status
                END
            """,
            (
                entry["content_sha256"],
                entry["output_path"],
                entry["package_path"],
                entry["source_key"],
                entry["run_id"],
                entry["short_id"],
                entry["score"],
                entry["decision"],
                entry["auto_qa_status"],
                json.dumps(qa, ensure_ascii=False),
                entry["review_status"],
                iso_time(),
            ),
        )
        entries.append(entry)
    conn.commit()
    return {"status": "completed" if entries else "empty", "items": entries}


def list_review_items(conn: sqlite3.Connection, *, status: str, limit: int) -> list[dict[str, Any]]:
    if status == "all":
        rows = conn.execute(
            "SELECT * FROM review_items ORDER BY created_at DESC LIMIT ?",
            (max(1, limit),),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM review_items WHERE review_status = ? ORDER BY created_at DESC LIMIT ?",
            (status, max(1, limit)),
        ).fetchall()
    return [dict(row) for row in rows]


def record_review_decision(
    conn: sqlite3.Connection,
    config: dict[str, Any],
    *,
    output_path: Path,
    status: str,
    note: str,
) -> dict[str, Any]:
    resolved = str(output_path.resolve())
    # A batch can be re-rendered at the same output path.  Approval must bind
    # to the bytes the reviewer can currently see, never to the oldest row
    # sharing that path.
    current_hash = file_sha256(output_path) if output_path.exists() else ""
    if current_hash:
        row = conn.execute(
            "SELECT * FROM review_items WHERE content_sha256 = ? ORDER BY created_at DESC LIMIT 1",
            (current_hash,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM review_items WHERE output_path = ? ORDER BY created_at DESC LIMIT 1",
            (resolved,),
        ).fetchone()
    if not row:
        raise RuntimeError(f"No review item found for output: {resolved}")
    if status == "approved":
        source_row = conn.execute(
            "SELECT active FROM library_sources WHERE source_key = ?",
            (row["source_key"],),
        ).fetchone()
        if not source_row or not bool(source_row["active"]):
            raise RuntimeError("Approval blocked: this source has been disabled by the source-safety check.")
        production = config.get("production", {}) if isinstance(config.get("production"), dict) else {}
        qa = auto_review_render(
            Path(str(row["output_path"])).resolve(),
            minimum_duration_exclusive=float(production.get("minimum_duration_sec_exclusive", 20.0) or 20.0),
            visual_qa=production.get("rendered_visual_qa") if isinstance(production.get("rendered_visual_qa"), dict) else None,
        )
        if qa["qa_status"] == "fail":
            raise RuntimeError("Approval blocked: the rendered video failed the mandatory safety/length check.")
    conn.execute(
        """
        UPDATE review_items
        SET review_status = ?, decided_at = ?, reviewer_note = ?
        WHERE content_sha256 = ?
        """,
        (status, iso_time(), note, row["content_sha256"]),
    )
    conn.commit()
    updated = conn.execute(
        "SELECT * FROM review_items WHERE content_sha256 = ?",
        (row["content_sha256"],),
    ).fetchone()
    return dict(updated) if updated else {}


def strip_title_emoji(value: str) -> str:
    """Keep YouTube upload titles text-only, even if an older package used emoji."""
    return "".join(
        char
        for char in str(value or "")
        if not (0x1F000 <= ord(char) <= 0x1FAFF or 0x2600 <= ord(char) <= 0x27BF)
    )


def youtube_upload_metadata(package_path: Path) -> dict[str, Any]:
    package = read_json(package_path, {})
    if not isinstance(package, dict):
        raise RuntimeError(f"Invalid rendered package JSON: {package_path}")

    analysis_dir = next(
        (parent for parent in package_path.parents if (parent / "youtube_context.json").exists()),
        package_path.parent.parent.parent,
    )
    context = read_json(analysis_dir / "youtube_context.json", {})
    context_metadata = context.get("metadata", {}) if isinstance(context, dict) else {}
    source_channel = str(context_metadata.get("channel_title") or "").strip()
    source_url = str(context.get("source_url") or "").strip() if isinstance(context, dict) else ""
    if not source_channel or not source_url:
        raise RuntimeError("Upload blocked: source channel and source URL are required for attribution.")

    title = re.sub(r"(?:^|\s)#[^\s#]+", "", str(package.get("upload_title") or "")).strip()
    if not title:
        title = " ".join(
            part.strip()
            for part in (str(package.get("title_line1") or ""), str(package.get("title_line2") or ""))
            if part.strip()
        )[:100]
    if not title:
        title = package_path.stem[:100]
    pitch = str(package.get("selection_pitch") or "").strip()
    tags = [str(tag).strip().lstrip("#") for tag in package.get("fun_tags", []) or [] if str(tag).strip()]
    if source_channel and source_channel.casefold() not in {tag.casefold() for tag in tags}:
        tags.append(source_channel)
    emoji_by_tag = {
        "웃김": "😂",
        "반전": "🤯",
        "긴장": "😮",
        "감동": "🥹",
        "관계": "🤝",
        "캐릭터": "✨",
    }
    lead_emoji = next((emoji_by_tag[tag] for tag in tags if tag in emoji_by_tag), "🎬")
    # Upload titles must be plain human-written text.  Do not inherit an
    # emoji from fun-tags, and strip one from older package metadata too.
    title = strip_title_emoji(title).strip()[:100]
    hashtag_line = " ".join(f"#{re.sub(r'[^0-9A-Za-z가-힣_]', '', tag)}" for tag in tags[:8])
    attribution = f"📺 원본 전체 영상은 {source_channel}에서 확인하세요.\n🔗 원본 링크: {source_url}"
    description = "\n\n".join(part for part in (pitch, attribution, hashtag_line, "#shorts") if part)[:5000]
    return {
        "title": title,
        "description": description,
        "tags": tags[:20],
        "source_channel": source_channel,
        "source_url": source_url,
        "source_title": str(context_metadata.get("title") or ""),
    }


def latest_channel_publish_anchor(service: Any, conn: sqlite3.Connection) -> datetime | None:
    """Find the latest actual or scheduled publication across the connected channel."""
    anchors: list[datetime] = []
    for row in conn.execute(
        "SELECT published_at FROM published_shorts "
        "WHERE published_at != '' AND status != 'archived_previous_channel'"
    ).fetchall():
        recorded = parse_time(str(row["published_at"] or ""))
        if recorded:
            anchors.append(recorded)

    channels = service.channels().list(part="contentDetails", mine=True).execute().get("items", [])
    if not channels:
        return max(anchors) if anchors else None
    uploads_playlist = str(
        channels[0].get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads") or ""
    )
    if not uploads_playlist:
        return max(anchors) if anchors else None
    playlist_items = service.playlistItems().list(
        part="contentDetails",
        playlistId=uploads_playlist,
        maxResults=50,
    ).execute().get("items", [])
    video_ids = [
        str(item.get("contentDetails", {}).get("videoId") or "")
        for item in playlist_items
        if item.get("contentDetails", {}).get("videoId")
    ]
    for start in range(0, len(video_ids), 50):
        videos = service.videos().list(
            part="snippet,status",
            id=",".join(video_ids[start : start + 50]),
        ).execute().get("items", [])
        for video in videos:
            status = video.get("status", {}) if isinstance(video.get("status"), dict) else {}
            snippet = video.get("snippet", {}) if isinstance(video.get("snippet"), dict) else {}
            scheduled = parse_time(str(status.get("publishAt") or ""))
            if scheduled:
                anchors.append(scheduled)
                continue
            if str(status.get("privacyStatus") or "").lower() in {"public", "unlisted"}:
                published = parse_time(str(snippet.get("publishedAt") or ""))
                if published:
                    anchors.append(published)
    return max(anchors) if anchors else None


def recent_channel_title_keys(service: Any) -> set[str]:
    """Return normalized titles already present on the connected channel."""
    channels = service.channels().list(part="contentDetails", mine=True).execute().get("items", [])
    if not channels:
        return set()
    uploads_playlist = str(channels[0].get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads") or "")
    if not uploads_playlist:
        return set()
    playlist_items = service.playlistItems().list(
        part="contentDetails",
        playlistId=uploads_playlist,
        maxResults=50,
    ).execute().get("items", [])
    video_ids = [
        str(item.get("contentDetails", {}).get("videoId") or "")
        for item in playlist_items
        if item.get("contentDetails", {}).get("videoId")
    ]
    if not video_ids:
        return set()
    videos = service.videos().list(part="snippet", id=",".join(video_ids)).execute().get("items", [])
    return {
        normalized_upload_title(str(video.get("snippet", {}).get("title") or ""))
        for video in videos
        if normalized_upload_title(str(video.get("snippet", {}).get("title") or ""))
    }


KST = timezone(timedelta(hours=9), name="KST")


def parse_daily_publish_slots(schedule: dict[str, Any]) -> list[tuple[int, int]]:
    """Parse unique local-time publishing slots such as ``09:00``."""
    raw_slots = schedule.get("daily_slots", [])
    if not isinstance(raw_slots, list):
        raise RuntimeError("publish_schedule.daily_slots must be a list such as ['09:00', '12:00'].")
    slots: set[tuple[int, int]] = set()
    for raw_slot in raw_slots:
        match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(raw_slot))
        if not match:
            raise RuntimeError(f"Invalid daily publish slot: {raw_slot!r}. Use HH:MM.")
        hour, minute = int(match.group(1)), int(match.group(2))
        if hour > 23 or minute > 59:
            raise RuntimeError(f"Invalid daily publish slot: {raw_slot!r}. Use a valid 24-hour time.")
        slots.add((hour, minute))
    if not slots:
        raise RuntimeError("At least one publish_schedule.daily_slots value is required for daily_slots mode.")
    return sorted(slots)


def parse_explicit_kst_publish_time(value: str) -> datetime:
    """Parse an operator-requested KST publishing time for a one-off upload."""
    raw = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError("--schedule-at must be a KST time such as '2026-08-03 18:00'.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=KST)
    scheduled = parsed.astimezone(timezone.utc)
    if scheduled <= utc_now():
        raise RuntimeError("--schedule-at must be in the future.")
    return scheduled


def fixed_daily_publish_schedule(
    *,
    anchor: datetime | None,
    now: datetime,
    count: int,
    slots: list[tuple[int, int]],
) -> list[datetime]:
    """Return the next local KST slots strictly after the latest channel event."""
    cutoff = max(now, anchor) if anchor else now
    cutoff = cutoff.astimezone(KST)
    day = cutoff.date()
    publish_times: list[datetime] = []
    while len(publish_times) < count:
        selected = False
        for hour, minute in slots:
            slot = datetime.combine(day, time(hour, minute), tzinfo=KST)
            if slot <= cutoff:
                continue
            publish_times.append(slot.astimezone(timezone.utc))
            cutoff = slot
            selected = True
            break
        if not selected:
            day += timedelta(days=1)
            cutoff = datetime.combine(day, time.min, tzinfo=KST)
    return publish_times


def approved_upload_schedule(
    conn: sqlite3.Connection,
    upload: dict[str, Any],
    count: int,
    service: Any,
) -> tuple[list[datetime | None], dict[str, str]]:
    """Schedule approved uploads after the latest channel publish or reservation."""
    schedule = upload.get("publish_schedule", {}) if isinstance(upload.get("publish_schedule"), dict) else {}
    if not bool(schedule.get("enabled", False)):
        return [None] * count, {"anchor_at": "", "first_publish_at": ""}
    if str(upload.get("privacy_status") or "private").lower() != "public":
        raise RuntimeError("Scheduled publishing requires youtube_upload.privacy_status to be 'public'.")
    now = utc_now()
    anchor = latest_channel_publish_anchor(service, conn)
    mode = str(schedule.get("mode") or "interval").strip().lower()
    if mode == "daily_slots":
        slots = parse_daily_publish_slots(schedule)
        publish_times = fixed_daily_publish_schedule(
            anchor=anchor,
            now=now,
            count=count,
            slots=slots,
        )
        return publish_times, {
            "anchor_at": iso_time(anchor) if anchor else "",
            "first_publish_at": iso_time(publish_times[0]) if publish_times else "",
            "mode": "daily_slots",
            "daily_slots": ", ".join(f"{hour:02d}:{minute:02d}" for hour, minute in slots),
        }

    if mode != "interval":
        raise RuntimeError("publish_schedule.mode must be 'interval' or 'daily_slots'.")
    interval_hours = max(1, int(schedule.get("interval_hours", 4) or 4))
    next_allowed = anchor + timedelta(hours=interval_hours) if anchor else now
    first_publish = max(now, next_allowed)
    publish_times = [first_publish + timedelta(hours=interval_hours * index) for index in range(count)]
    return [None if index == 0 and first_publish <= now else value for index, value in enumerate(publish_times)], {
        "anchor_at": iso_time(anchor) if anchor else "",
        "first_publish_at": iso_time(first_publish),
        "mode": "interval",
    }


def upload_rendered_outputs(
    conn: sqlite3.Connection,
    config: dict[str, Any],
    render: dict[str, Any],
    *,
    execute: bool,
    youtube_service: Any | None = None,
) -> dict[str, Any]:
    upload = config.get("youtube_upload", {}) if isinstance(config.get("youtube_upload"), dict) else {}
    if not bool(upload.get("enabled", False)):
        return {"status": "disabled"}
    if not execute:
        return {"status": "planned"}
    rendered = [item for item in render.get("rendered", []) or [] if isinstance(item, dict) and item.get("status") == "completed"]
    if not rendered:
        return {"status": "nothing_to_upload"}

    privacy_status = str(upload.get("privacy_status") or "private").lower()
    if privacy_status not in {"private", "unlisted", "public"}:
        raise RuntimeError("youtube_upload.privacy_status must be private, unlisted, or public.")
    try:
        from googleapiclient.http import MediaFileUpload
    except ImportError as exc:
        raise RuntimeError("Install Google API dependencies from requirements.txt before enabling YouTube upload.") from exc
    analytics = config.get("youtube_analytics", {}) if isinstance(config.get("youtube_analytics"), dict) else {}
    service = youtube_service or build_youtube_upload_service(
        analytics,
        interactive=bool(analytics.get("interactive_on_first_run", False)),
    )
    existing_title_keys = recent_channel_title_keys(service)

    results: list[dict[str, Any]] = []
    for item in rendered:
        output_path = Path(str(item["output_path"])).resolve()
        package_path = Path(str(item["package_path"])).resolve()
        if not output_path.exists() or not package_path.exists():
            results.append({"status": "missing_input", "output_path": str(output_path)})
            continue
        content_sha256 = file_sha256(output_path)
        if bool(upload.get("require_review_approval", True)):
            review = conn.execute(
                "SELECT review_status, auto_qa_status FROM review_items WHERE content_sha256 = ?",
                (content_sha256,),
            ).fetchone()
            if not review or review["review_status"] != "approved":
                results.append(
                    {
                        "status": "waiting_for_review",
                        "output_path": str(output_path),
                        "review_status": review["review_status"] if review else "missing",
                    }
                )
                continue
        existing = conn.execute(
            "SELECT youtube_video_id FROM rendered_uploads WHERE content_sha256 = ?",
            (content_sha256,),
        ).fetchone()
        if existing:
            results.append(
                {
                    "status": "already_uploaded",
                    "output_path": str(output_path),
                    "youtube_video_id": existing["youtube_video_id"],
                }
            )
            continue
        metadata = youtube_upload_metadata(package_path)
        source_reason = source_safety_reason(
            config,
            {
                "source_title": metadata.get("source_title"),
                "source_channel": metadata.get("source_channel"),
                "source_url": metadata.get("source_url"),
            },
        )
        if source_reason:
            conn.execute(
                "UPDATE review_items SET review_status = 'qa_failed', reviewer_note = ? WHERE content_sha256 = ?",
                (f"Upload blocked by source policy: {source_reason}", content_sha256),
            )
            conn.commit()
            results.append({"status": "blocked_source", "output_path": str(output_path), "reason": source_reason})
            continue
        title_key = normalized_upload_title(str(metadata.get("title") or ""))
        if title_key and title_key in existing_title_keys:
            conn.execute(
                "UPDATE review_items SET review_status = 'qa_failed', reviewer_note = ? WHERE content_sha256 = ?",
                ("Upload blocked: an identical title already exists on the connected channel.", content_sha256),
            )
            conn.commit()
            results.append({"status": "blocked_duplicate", "output_path": str(output_path), "title": metadata.get("title")})
            continue
        scheduled_publish_at = parse_time(str(item.get("scheduled_publish_at") or ""))
        effective_privacy_status = "private" if scheduled_publish_at else privacy_status
        body = {
            "snippet": {
                "title": metadata["title"],
                "description": metadata["description"],
                "tags": metadata["tags"],
                "categoryId": str(upload.get("category_id") or "24"),
            },
            "status": {
                "privacyStatus": effective_privacy_status,
                "selfDeclaredMadeForKids": bool(upload.get("made_for_kids", False)),
            },
        }
        if scheduled_publish_at:
            body["status"]["publishAt"] = iso_time(scheduled_publish_at)
        try:
            response = (
                service.videos()
                .insert(
                    part="snippet,status",
                    body=body,
                    media_body=MediaFileUpload(str(output_path), mimetype="video/mp4", resumable=True),
                )
                .execute()
            )
            video_id = str(response.get("id") or "")
            if not video_id:
                raise RuntimeError("YouTube upload returned no video id.")
            conn.execute(
                """
                INSERT INTO rendered_uploads (content_sha256, output_path, package_path, youtube_video_id, privacy_status, uploaded_at, response_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    content_sha256,
                    str(output_path),
                    str(package_path),
                    video_id,
                    effective_privacy_status,
                    iso_time(),
                    json.dumps(response, ensure_ascii=False),
                ),
            )
            conn.commit()
            if title_key:
                existing_title_keys.add(title_key)
            result = {"status": "uploaded", "output_path": str(output_path), "youtube_video_id": video_id, "privacy_status": effective_privacy_status}
            if scheduled_publish_at:
                register_published_short(
                    conn,
                    youtube_video_id=video_id,
                    package_path=package_path,
                    published_at=scheduled_publish_at,
                )
                result["scheduled_publish_at"] = iso_time(scheduled_publish_at)
                result["metrics_registered"] = True
            elif privacy_status == "public":
                register_published_short(
                    conn,
                    youtube_video_id=video_id,
                    package_path=package_path,
                    published_at=utc_now(),
                )
                result["metrics_registered"] = True
            conn.execute(
                "UPDATE review_items SET review_status = 'uploaded' WHERE content_sha256 = ?",
                (content_sha256,),
            )
            conn.commit()
            results.append(result)
        except Exception as exc:
            results.append({"status": "failed", "output_path": str(output_path), "error": str(exc)})
    return {
        "status": "completed" if results and all(item["status"] in {"uploaded", "already_uploaded"} for item in results) else "partial",
        "uploads": results,
    }


def upload_approved_reviews(
    conn: sqlite3.Connection,
    config: dict[str, Any],
    *,
    execute: bool,
    limit: int,
    output_path: Path | None = None,
    publish_now: bool = False,
    schedule_at: str = "",
) -> dict[str, Any]:
    filters = [
        "review_items.review_status = 'approved'",
        "rendered_uploads.content_sha256 IS NULL",
    ]
    parameters: list[Any] = []
    if output_path is not None:
        filters.append("review_items.output_path = ?")
        parameters.append(str(output_path.resolve()))
    parameters.append(max(1, limit))
    rows = conn.execute(
        f"""
        SELECT review_items.*
        FROM review_items
        LEFT JOIN rendered_uploads
            ON rendered_uploads.content_sha256 = review_items.content_sha256
        JOIN library_sources
            ON library_sources.source_key = review_items.source_key
        WHERE {' AND '.join(filters)}
            AND library_sources.active = 1
        ORDER BY COALESCE(review_items.decided_at, review_items.created_at) DESC
        LIMIT ?
        """,
        parameters,
    ).fetchall()
    if not rows:
        return {"status": "nothing_to_upload"}
    eligible_rows = []
    blocked_outputs: list[str] = []
    for row in rows:
        production = config.get("production", {}) if isinstance(config.get("production"), dict) else {}
        qa = auto_review_render(
            Path(str(row["output_path"])).resolve(),
            minimum_duration_exclusive=float(production.get("minimum_duration_sec_exclusive", 20.0) or 20.0),
            visual_qa=production.get("rendered_visual_qa") if isinstance(production.get("rendered_visual_qa"), dict) else None,
        )
        if qa["qa_status"] == "fail":
            blocked_outputs.append(str(row["output_path"]))
            conn.execute(
                "UPDATE review_items SET review_status = 'qa_failed', reviewer_note = ? WHERE content_sha256 = ?",
                ("Upload blocked by mandatory source-safety/length check.", row["content_sha256"]),
            )
            continue
        eligible_rows.append(row)
    conn.commit()
    if not eligible_rows:
        return {"status": "blocked_by_safety", "blocked_outputs": blocked_outputs}
    rows = eligible_rows
    analytics = config.get("youtube_analytics", {}) if isinstance(config.get("youtube_analytics"), dict) else {}
    service = build_youtube_upload_service(
        analytics,
        interactive=bool(analytics.get("interactive_on_first_run", False)),
    )
    if publish_now and schedule_at:
        raise RuntimeError("Use either publish_now or schedule_at, not both.")
    if schedule_at:
        if len(rows) != 1:
            raise RuntimeError("--schedule-at requires exactly one approved output. Pass --output.")
        publish_times = [parse_explicit_kst_publish_time(schedule_at)]
        schedule_info = {"anchor_at": "", "first_publish_at": iso_time(publish_times[0]), "mode": "explicit_kst"}
    elif publish_now:
        # Explicit operator choice for a one-off immediate release.  The
        # normal daily-slot policy remains active for every later upload.
        publish_times = [None] * len(rows)
        schedule_info = {"anchor_at": "", "first_publish_at": "", "mode": "immediate"}
    else:
        publish_times, schedule_info = approved_upload_schedule(
            conn,
            config.get("youtube_upload", {}) if isinstance(config.get("youtube_upload"), dict) else {},
            len(rows),
            service,
        )
    render = {
        "rendered": [
            {
                "status": "completed",
                "output_path": row["output_path"],
                "package_path": row["package_path"],
                "scheduled_publish_at": iso_time(publish_times[index]) if publish_times[index] else "",
            }
            for index, row in enumerate(rows)
        ]
    }
    result = upload_rendered_outputs(conn, config, render, execute=execute, youtube_service=service)
    result["schedule"] = schedule_info
    return result


def youtube_connection_status(conn: sqlite3.Connection, config: dict[str, Any], *, sync_metrics: bool) -> dict[str, Any]:
    analytics = config.get("youtube_analytics", {}) if isinstance(config.get("youtube_analytics"), dict) else {}
    if not bool(analytics.get("enabled", False)):
        return {"status": "disabled"}
    service = build_youtube_upload_service(analytics, interactive=False)
    channels = service.channels().list(part="snippet", mine=True).execute().get("items", [])
    if not channels:
        return {"status": "no_channel"}
    channel = channels[0]
    result = {
        "status": "connected",
        "channel_title": str(channel.get("snippet", {}).get("title") or ""),
        "channel_id": str(channel.get("id") or ""),
    }
    tracking = conn.execute(
        """
        SELECT COUNT(*) AS tracked_count, MIN(next_check_at) AS next_check_at
        FROM published_shorts
        WHERE status = 'watching' AND next_check_at IS NOT NULL
        """
    ).fetchone()
    result["metrics_tracking"] = {
        "tracked_count": int(tracking["tracked_count"] or 0),
        "next_check_at": str(tracking["next_check_at"] or ""),
    }
    if sync_metrics:
        result["metrics_sync"] = sync_due_youtube_metrics(conn, analytics, dry_run=False)
    return result


def recent_youtube_performance(config: dict[str, Any], *, limit: int) -> dict[str, Any]:
    """Read the newest public channel videos and their current visible statistics."""
    analytics = config.get("youtube_analytics", {}) if isinstance(config.get("youtube_analytics"), dict) else {}
    if not bool(analytics.get("enabled", False)):
        return {"status": "disabled", "videos": []}
    service = build_youtube_upload_service(analytics, interactive=False)
    channels = service.channels().list(part="snippet,contentDetails", mine=True).execute().get("items", [])
    if not channels:
        return {"status": "no_channel", "videos": []}
    channel = channels[0]
    uploads_playlist = str(channel.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads") or "")
    if not uploads_playlist:
        return {"status": "no_uploads_playlist", "videos": []}
    playlist_items = service.playlistItems().list(
        part="contentDetails",
        playlistId=uploads_playlist,
        maxResults=50,
    ).execute().get("items", [])
    video_ids = [
        str(item.get("contentDetails", {}).get("videoId") or "")
        for item in playlist_items
        if item.get("contentDetails", {}).get("videoId")
    ]
    if not video_ids:
        return {"status": "connected", "channel_title": str(channel.get("snippet", {}).get("title") or ""), "videos": []}
    videos: list[dict[str, Any]] = []
    for start in range(0, len(video_ids), 50):
        payload = service.videos().list(
            part="snippet,status,statistics",
            id=",".join(video_ids[start : start + 50]),
        ).execute().get("items", [])
        for video in payload:
            status = video.get("status", {}) if isinstance(video.get("status"), dict) else {}
            if str(status.get("privacyStatus") or "").lower() != "public":
                continue
            snippet = video.get("snippet", {}) if isinstance(video.get("snippet"), dict) else {}
            statistics = video.get("statistics", {}) if isinstance(video.get("statistics"), dict) else {}
            published_at = parse_time(str(snippet.get("publishedAt") or ""))
            if not published_at:
                continue
            videos.append(
                {
                    "youtube_video_id": str(video.get("id") or ""),
                    "title": str(snippet.get("title") or ""),
                    "published_at": iso_time(published_at),
                    "views": int(statistics.get("viewCount") or 0),
                    "likes": int(statistics.get("likeCount") or 0),
                    "comments": int(statistics.get("commentCount") or 0),
                }
            )
    videos.sort(key=lambda item: item["published_at"], reverse=True)
    return {
        "status": "connected",
        "channel_title": str(channel.get("snippet", {}).get("title") or ""),
        "videos": videos[: max(1, min(limit, 20))],
    }


def youtube_video_id_from_url(value: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})", value or "")
    return match.group(1) if match else ""


def normalized_upload_title(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", str(value or "").casefold())


def source_key_for_source_url(conn: sqlite3.Connection, source_url: str) -> str:
    source_video_id = youtube_video_id_from_url(source_url)
    if not source_video_id:
        return ""
    for row in conn.execute("SELECT source_key, analysis_dir FROM library_sources").fetchall():
        context = read_json(Path(str(row["analysis_dir"] or "")) / "youtube_context.json", {})
        context_url = str(context.get("source_url") or "") if isinstance(context, dict) else ""
        if youtube_video_id_from_url(context_url) == source_video_id:
            return str(row["source_key"] or "")
    return ""


def audit_recent_channel_uploads(
    conn: sqlite3.Connection,
    config: dict[str, Any],
    *,
    dry_run: bool,
    limit: int = 30,
) -> dict[str, Any]:
    """Audit recently uploaded channel videos before a new production cycle.

    The Data API does not expose every Content ID detail, but it does expose
    public/processing state and region restrictions.  We also follow the
    credited original link to enforce the local broadcaster block list.
    """
    analytics = config.get("youtube_analytics", {}) if isinstance(config.get("youtube_analytics"), dict) else {}
    if not bool(analytics.get("enabled", False)):
        return {"status": "disabled", "issues": []}
    service = build_youtube_upload_service(analytics, interactive=False)
    channels = service.channels().list(part="snippet,contentDetails", mine=True).execute().get("items", [])
    if not channels:
        return {"status": "no_channel", "issues": []}
    channel = channels[0]
    uploads_playlist = str(channel.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads") or "")
    if not uploads_playlist:
        return {"status": "no_uploads_playlist", "issues": []}
    playlist_items = service.playlistItems().list(
        part="contentDetails",
        playlistId=uploads_playlist,
        maxResults=max(1, min(int(limit or 30), 50)),
    ).execute().get("items", [])
    video_ids = [
        str(item.get("contentDetails", {}).get("videoId") or "")
        for item in playlist_items
        if item.get("contentDetails", {}).get("videoId")
    ]
    videos = service.videos().list(
        part="snippet,status,contentDetails,processingDetails",
        id=",".join(video_ids),
    ).execute().get("items", []) if video_ids else []
    known_ids = {
        str(row["youtube_video_id"] or "")
        for row in conn.execute(
            "SELECT youtube_video_id FROM published_shorts WHERE status != 'archived_previous_channel'"
        ).fetchall()
    }
    source_video_ids = {
        youtube_video_id_from_url(str(video.get("snippet", {}).get("description") or ""))
        for video in videos
    }
    source_video_ids.discard("")
    source_metadata: dict[str, dict[str, Any]] = {}
    if source_video_ids:
        for source in service.videos().list(part="snippet", id=",".join(sorted(source_video_ids))).execute().get("items", []):
            source_metadata[str(source.get("id") or "")] = source.get("snippet", {}) if isinstance(source.get("snippet"), dict) else {}

    issues: list[dict[str, Any]] = []
    title_groups: dict[tuple[str, str], list[str]] = {}
    for video in videos:
        video_id = str(video.get("id") or "")
        snippet = video.get("snippet", {}) if isinstance(video.get("snippet"), dict) else {}
        status = video.get("status", {}) if isinstance(video.get("status"), dict) else {}
        details = video.get("contentDetails", {}) if isinstance(video.get("contentDetails"), dict) else {}
        title = str(snippet.get("title") or "")
        source_url = str(snippet.get("description") or "")
        source_id = youtube_video_id_from_url(source_url)
        source_snippet = source_metadata.get(source_id, {})
        source_channel = str(source_snippet.get("channelTitle") or "")
        source_title = str(source_snippet.get("title") or "")
        source_key = source_key_for_source_url(conn, source_url)
        source_reason = source_safety_reason(
            config,
            {"source_title": source_title, "source_channel": source_channel, "source_url": source_url},
        )
        restricted_regions = details.get("regionRestriction", {}).get("blocked", []) if isinstance(details.get("regionRestriction"), dict) else []
        if restricted_regions:
            issues.append({
                "type": "region_or_rights_restricted",
                "youtube_video_id": video_id,
                "title": title,
                "restricted_region_count": len(restricted_regions),
                "source_key": source_key,
            })
            if not dry_run:
                conn.execute("UPDATE published_shorts SET status = 'restricted', next_check_at = NULL WHERE youtube_video_id = ?", (video_id,))
        if source_reason:
            issues.append({
                "type": "blocked_source_detected_after_upload",
                "youtube_video_id": video_id,
                "title": title,
                "source_channel": source_channel,
                "reason": source_reason,
                "source_key": source_key,
            })
            if not dry_run and source_key:
                conn.execute("UPDATE library_sources SET active = 0 WHERE source_key = ?", (source_key,))
                conn.execute(
                    "UPDATE review_items SET review_status = 'qa_failed', reviewer_note = ? WHERE source_key = ? AND review_status IN ('needs_review', 'approved')",
                    (f"Upload audit blocked future use: {source_reason}", source_key),
                )
        if video_id not in known_ids:
            issues.append({"type": "untracked_channel_upload", "youtube_video_id": video_id, "title": title})
        title_key = (normalized_upload_title(title), str(details.get("duration") or ""))
        if title_key[0]:
            title_groups.setdefault(title_key, []).append(video_id)
    for (title_key, duration), ids in title_groups.items():
        if len(ids) > 1:
            issues.append({"type": "duplicate_upload", "video_ids": ids, "duration": duration, "normalized_title": title_key})
    if not dry_run:
        conn.commit()
    return {
        "status": "completed",
        "channel_title": str(channel.get("snippet", {}).get("title") or ""),
        "checked": len(videos),
        "issues": issues,
        "blocked_source_count": len({item.get("source_key") for item in issues if item.get("source_key")}),
    }


def delete_audited_problem_uploads(
    conn: sqlite3.Connection,
    config: dict[str, Any],
    *,
    limit: int,
) -> dict[str, Any]:
    """Delete only videos the current audit identifies as unsafe or duplicate."""
    audit = audit_recent_channel_uploads(conn, config, dry_run=True, limit=limit)
    issues = audit.get("issues", []) if isinstance(audit.get("issues"), list) else []
    target_ids: set[str] = set()
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        issue_type = str(issue.get("type") or "")
        if issue_type in {"region_or_rights_restricted", "blocked_source_detected_after_upload"}:
            video_id = str(issue.get("youtube_video_id") or "")
            if video_id:
                target_ids.add(video_id)
        elif issue_type == "duplicate_upload":
            target_ids.update(str(video_id) for video_id in issue.get("video_ids", []) or [] if str(video_id))
    if not target_ids:
        return {"status": "nothing_to_delete", "audit": audit, "deleted": []}

    analytics = config.get("youtube_analytics", {}) if isinstance(config.get("youtube_analytics"), dict) else {}
    service = build_youtube_upload_service(analytics, interactive=False)
    deleted: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for video_id in sorted(target_ids):
        try:
            service.videos().delete(id=video_id).execute()
            conn.execute(
                "UPDATE published_shorts SET status = 'deleted_by_audit', next_check_at = NULL WHERE youtube_video_id = ?",
                (video_id,),
            )
            review = conn.execute(
                "SELECT content_sha256 FROM rendered_uploads WHERE youtube_video_id = ?",
                (video_id,),
            ).fetchone()
            if review:
                conn.execute(
                    "UPDATE review_items SET review_status = 'rejected', reviewer_note = ? WHERE content_sha256 = ?",
                    ("Deleted after broadcaster/rights/duplicate upload audit.", review["content_sha256"]),
                )
            deleted.append({"youtube_video_id": video_id})
        except Exception as exc:
            errors.append({"youtube_video_id": video_id, "error": str(exc)})
    conn.commit()
    return {
        "status": "completed" if not errors else "partial",
        "audit": audit,
        "deleted": deleted,
        "errors": errors,
    }


def run_daily(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    db_path = args.db.resolve()
    config = load_config(config_path)
    conn = connect(db_path)
    initialize_database(conn)
    run_id = run_id_for()
    run_dir = AUTOMATION_DIR / "runs" / run_id
    summary_path = run_dir / "run_summary.json"
    usage_path = run_dir / "openai_usage.jsonl"
    # This key is runtime-only; it is never written into automation_config.
    config["_usage_log_path"] = str(usage_path)
    dry_run = not bool(args.execute)
    conn.execute(
        "INSERT INTO runs (run_id, requested_at, status, dry_run, summary_path) VALUES (?, ?, ?, ?, ?)",
        (run_id, iso_time(), "running", int(dry_run), str(summary_path)),
    )
    conn.commit()

    summary: dict[str, Any] = {
        "run_id": run_id,
        "requested_at": iso_time(),
        "mode": "execute" if args.execute else "dry_run",
        "status": "running",
        "config_path": str(config_path),
        "db_path": str(db_path),
        "usage_accounting": {"status": "collecting", "path": str(usage_path)},
    }
    try:
        analytics = config.get("youtube_analytics", {}) if isinstance(config.get("youtube_analytics"), dict) else {}
        print("[orchestrator] phase=channel_audit", flush=True)
        summary["channel_audit"] = audit_recent_channel_uploads(conn, config, dry_run=dry_run)
        print("[orchestrator] phase=metrics_sync", flush=True)
        summary["metrics_sync"] = sync_due_youtube_metrics(conn, analytics, dry_run=dry_run)
        print("[orchestrator] phase=learning_rule", flush=True)
        learning_rule = derive_learning_rule(conn, config, run_id)
        learning_rule_path = run_dir / "learning_rule.json"
        write_json(learning_rule_path, learning_rule)
        summary["learning_rule"] = learning_rule

        if args.library_dir and not args.confirm_owned_library_dirs:
            raise RuntimeError("Pass --confirm-owned-library-dirs when using --library-dir.")
        library_dirs = [path.resolve() for path in args.library_dir]
        sources = configured_sources(config, library_dirs)

        if library_dirs:
            # An explicitly supplied analysis directory is an intentional
            # rerun/revision target. Do not spend time discovering or
            # downloading a different trending source before processing it.
            trend = {"status": "skipped_explicit_source", "tokens": []}
            summary["trend"] = {"status": "skipped_explicit_source"}
            summary["source_acquisition"] = {"status": "skipped_explicit_source"}
        else:
            print("[orchestrator] phase=trend_research", flush=True)
            trend = run_trend_research(config, run_dir, dry_run=dry_run)
            summary["trend"] = {key: value for key, value in trend.items() if key != "snapshot"}

            print("[orchestrator] phase=source_acquisition", flush=True)
            acquisition_result = acquire_selected_trend_source(
                config,
                trend,
                execute=bool(args.execute),
            )
            summary["source_acquisition"] = {
                key: value
                for key, value in acquisition_result.items()
                if key not in {"source"}
            }
            acquired_source = acquisition_result.get("source") if isinstance(acquisition_result.get("source"), dict) else None
            if acquired_source:
                sources.insert(0, acquired_source)

        permitted_sources: list[dict[str, Any]] = []
        blocked_sources: list[dict[str, str]] = []
        for source in sources:
            block_reason = source_safety_reason(config, source)
            if block_reason:
                blocked_sources.append(
                    {
                        "source_key": str(source.get("source_key") or ""),
                        "source_title": str(source.get("source_title") or ""),
                        "reason": block_reason,
                    }
                )
                conn.execute("UPDATE library_sources SET active = 0 WHERE source_key = ?", (source["source_key"],))
                continue
            permitted_sources.append(source)
        conn.commit()
        sources = permitted_sources
        summary["source_preflight"] = {
            "checked": len(permitted_sources) + len(blocked_sources),
            "blocked": blocked_sources,
        }

        if not sources:
            raise RuntimeError(
                "No permitted source is available after the source-safety preflight."
            )
        for source in sources:
            upsert_library_source(conn, source)

        library_config = config.get("library", {}) if isinstance(config.get("library"), dict) else {}
        maximum = args.max_sources or int(library_config.get("max_sources_per_run", 1) or 1)
        ranked_sources = rank_sources(
            conn,
            sources,
            set(trend.get("tokens", [])),
            int(library_config.get("max_source_transcript_chars", 12000) or 12000),
        )
        selected_sources = ranked_sources[: max(1, maximum)]
        summary["selected_sources"] = [
            {
                "source_key": source["source_key"],
                "source_title": source["source_title"],
                "analysis_dir": str(source["analysis_dir"]),
                "selection_score": source["selection_score"],
                "trend_overlap": source["trend_overlap"],
            }
            for source in selected_sources
        ]

        source_results = []
        for source in selected_sources:
            print(f"[orchestrator] phase=source_analysis source={source['source_title']}", flush=True)
            analysis_result = ensure_source_analysis(source, execute=bool(args.execute))
            item = {"source_key": source["source_key"], "analysis": analysis_result}
            if analysis_result["status"] == "ready":
                print(f"[orchestrator] phase=source_visual_gate source={source['source_title']}", flush=True)
                item["source_visual_gate"] = source_visual_preflight(config, source)
                if item["source_visual_gate"].get("status") == "pass":
                    print(f"[orchestrator] phase=visual_event_script source={source['source_title']}", flush=True)
                    item["visual_events"] = ensure_visual_event_script(config, source, execute=bool(args.execute))
                else:
                    item["visual_events"] = {
                        "status": "skipped",
                        "reason": item["source_visual_gate"].get("reason", "source visual preflight did not pass"),
                    }
                if item["visual_events"].get("status") == "ready":
                    print(f"[orchestrator] phase=package_generation source={source['source_title']}", flush=True)
                    item["generation"] = generate_for_source(
                        conn,
                        config,
                        source,
                        run_id,
                        learning_rule_path,
                        learning_rule,
                        execute=bool(args.execute),
                    )
                else:
                    item["generation"] = {
                        "status": "failed",
                        "error": item["visual_events"].get(
                            "reason", "visual event script was not ready; candidate generation was intentionally skipped"
                        ),
                    }
                if item["generation"].get("status") == "completed":
                    print(f"[orchestrator] phase=render source={source['source_title']}", flush=True)
                    generation_output = item["generation"].get("output", {})
                    package_root_text = item["generation"].get("package_root") or generation_output.get("package_root")
                    item["render"] = render_for_source(
                        config,
                        source,
                        package_root=Path(package_root_text) if package_root_text else None,
                        batch_id=str(item["generation"].get("batch_id") or ""),
                        execute=bool(args.execute),
                    )
                    if item["render"].get("status") in {"completed", "partial"}:
                        review_config = config.get("review", {}) if isinstance(config.get("review"), dict) else {}
                        if bool(review_config.get("enabled", True)):
                            print(f"[orchestrator] phase=review source={source['source_title']}", flush=True)
                            item["review"] = register_review_items(
                                conn,
                                source=source,
                                run_id=run_id,
                                render=item["render"],
                                minimum_duration_exclusive=float(
                                    (config.get("production", {}) if isinstance(config.get("production"), dict) else {}).get(
                                        "minimum_duration_sec_exclusive", 20.0
                                    )
                                    or 20.0
                                ),
                                visual_qa=(config.get("production", {}) if isinstance(config.get("production"), dict) else {}).get(
                                    "rendered_visual_qa"
                                ),
                            )
                        print(f"[orchestrator] phase=upload source={source['source_title']}", flush=True)
                        item["upload"] = upload_rendered_outputs(
                            conn,
                            config,
                            item["render"],
                            execute=bool(args.execute),
                        )
                        acquisition_config = config.get("source_acquisition", {}) if isinstance(config.get("source_acquisition"), dict) else {}
                        if (
                            source.get("source_origin") == "trend_acquisition"
                            and bool(acquisition_config.get("mark_processed_after_render", True))
                        ):
                            mark_acquired_source_processed(source, item["render"])
                elif (
                    item["generation"].get("status") == "failed"
                    and source.get("source_origin") == "trend_acquisition"
                    and "All selected candidates failed dense visual verification" in str(item["generation"].get("error") or "")
                ):
                    # This source passed download/safety checks but offered no
                    # complete, visually proven Shorts story.  Do not spend
                    # another run re-testing the same weak longform.
                    mark_acquired_source_processed(source, {"rendered": []}, allow_zero=True)
                    conn.execute("UPDATE library_sources SET active = 0 WHERE source_key = ?", (source["source_key"],))
                conn.execute(
                    "UPDATE library_sources SET last_selected_at = ? WHERE source_key = ?",
                    (iso_time(), source["source_key"]),
                )
                conn.commit()
            source_results.append(item)
        summary["sources"] = source_results
        generation_failures = [
            item.get("generation", {}).get("error", "candidate generation failed")
            for item in source_results
            if isinstance(item.get("generation"), dict) and item["generation"].get("status") == "failed"
        ]
        generated_any = any(
            isinstance(item.get("generation"), dict) and item["generation"].get("status") == "completed"
            for item in source_results
        )
        if source_results and not generated_any and generation_failures:
            summary["status"] = "no_candidates"
            summary["candidate_failure"] = generation_failures[0]
            conn.execute(
                "UPDATE runs SET status = ?, learning_rule_id = ?, error_text = ? WHERE run_id = ?",
                ("no_candidates", learning_rule["rule_id"], generation_failures[0], run_id),
            )
        else:
            summary["status"] = "completed"
            conn.execute(
                "UPDATE runs SET status = ?, learning_rule_id = ? WHERE run_id = ?",
                ("completed", learning_rule["rule_id"], run_id),
            )
        conn.commit()
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        summary["traceback"] = traceback.format_exc()
        conn.execute("UPDATE runs SET status = ?, error_text = ? WHERE run_id = ?", ("failed", str(exc), run_id))
        conn.commit()
    finally:
        summary["usage_accounting"] = summarize_usage(usage_path)
        write_json(summary_path, summary)
        conn.close()
    return summary


def print_daily_summary(summary: dict[str, Any]) -> None:
    print(f"[orchestrator] run_id={summary.get('run_id', '')}", flush=True)
    print(f"[orchestrator] status={summary.get('status', '')}", flush=True)
    rule = summary.get("learning_rule", {}) if isinstance(summary.get("learning_rule"), dict) else {}
    if rule:
        print(f"[orchestrator] learning_rule={rule.get('rule_id', '')}", flush=True)
        print(f"[orchestrator] change_note={rule.get('change_note', '')}", flush=True)
    metrics = summary.get("metrics_sync", {}) if isinstance(summary.get("metrics_sync"), dict) else {}
    print(f"[orchestrator] metrics_sync={metrics.get('status', '')}", flush=True)
    usage = summary.get("usage_accounting", {}) if isinstance(summary.get("usage_accounting"), dict) else {}
    usage_totals = usage.get("totals", {}) if isinstance(usage.get("totals"), dict) else {}
    if usage.get("status") == "completed":
        print(
            "[orchestrator] "
            f"usage_tokens={usage_totals.get('total_tokens', 0)} "
            f"estimated_usd={float(usage_totals.get('estimated_usd', 0.0) or 0.0):.4f} "
            f"unpriced_events={usage_totals.get('unpriced_events', 0)}",
            flush=True,
        )
    acquisition = summary.get("source_acquisition", {}) if isinstance(summary.get("source_acquisition"), dict) else {}
    if acquisition:
        acquisition_summary = acquisition.get("summary", {}) if isinstance(acquisition.get("summary"), dict) else {}
        print(
            "[orchestrator] "
            f"source_acquisition={acquisition.get('status', '')} "
            f"source={acquisition_summary.get('source_title', '')}",
            flush=True,
        )
    for source in summary.get("selected_sources", []) or []:
        print(
            "[orchestrator] "
            f"selected_source={source.get('source_title', '')} "
            f"trend_overlap={source.get('trend_overlap', 0)}",
            flush=True,
        )
    for source in summary.get("sources", []) or []:
        generation = source.get("generation", {}) if isinstance(source.get("generation"), dict) else {}
        output = generation.get("output", {}) if isinstance(generation.get("output"), dict) else {}
        if output.get("path"):
            print(f"[orchestrator] package_output={output['path']}", flush=True)
        render = source.get("render", {}) if isinstance(source.get("render"), dict) else {}
        for item in render.get("rendered", []) or []:
            if isinstance(item, dict) and item.get("output_path"):
                print(f"[orchestrator] rendered_output={item['output_path']}", flush=True)
        review = source.get("review", {}) if isinstance(source.get("review"), dict) else {}
        for item in review.get("items", []) or []:
            if isinstance(item, dict) and item.get("output_path"):
                print(
                    "[orchestrator] "
                    f"review_item={item['output_path']} "
                    f"status={item.get('review_status', '')} "
                    f"qa={item.get('auto_qa_status', '')}",
                    flush=True,
                )
        upload = source.get("upload", {}) if isinstance(source.get("upload"), dict) else {}
        for item in upload.get("uploads", []) or []:
            if isinstance(item, dict) and item.get("youtube_video_id"):
                print(f"[orchestrator] uploaded_video={item['youtube_video_id']}", flush=True)
    print(f"[orchestrator] summary={AUTOMATION_DIR / 'runs' / str(summary.get('run_id', '')) / 'run_summary.json'}", flush=True)


def status_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    latest_run = conn.execute("SELECT * FROM runs ORDER BY requested_at DESC LIMIT 1").fetchone()
    due = conn.execute(
        "SELECT COUNT(*) AS count FROM published_shorts "
        "WHERE status != 'archived_previous_channel' AND next_check_at IS NOT NULL AND next_check_at <= ?",
        (iso_time(),),
    ).fetchone()
    published = conn.execute(
        "SELECT COUNT(*) AS count FROM published_shorts WHERE status != 'archived_previous_channel'"
    ).fetchone()
    snapshots = conn.execute("SELECT COUNT(*) AS count FROM metrics_snapshots").fetchone()
    pending_reviews = conn.execute(
        "SELECT COUNT(*) AS count FROM review_items WHERE review_status = 'needs_review'"
    ).fetchone()
    qa_failed_reviews = conn.execute(
        "SELECT COUNT(*) AS count FROM review_items WHERE review_status = 'qa_failed'"
    ).fetchone()
    return {
        "latest_run": dict(latest_run) if latest_run else {},
        "published_shorts": int(published["count"]),
        "due_metric_checks": int(due["count"]),
        "metric_snapshots": int(snapshots["count"]),
        "pending_reviews": int(pending_reviews["count"]),
        "qa_failed_reviews": int(qa_failed_reviews["count"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the persistent longform-to-shorts daily orchestration loop.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize the orchestration database and optional config file.")
    init_parser.add_argument("--write-config", action="store_true")

    daily_parser = subparsers.add_parser("daily-run", help="Sync due metrics, choose a learning rule, research trends, and produce packages.")
    daily_parser.add_argument("--execute", action="store_true", help="Run analysis/package generation. Without this flag, only plan the run.")
    daily_parser.add_argument("--library-dir", action="append", type=Path, default=[], help="Explicit owned analysis directory for this run.")
    daily_parser.add_argument(
        "--confirm-owned-library-dirs",
        action="store_true",
        help="Required confirmation when using --library-dir outside the persisted owned library.",
    )
    daily_parser.add_argument("--max-sources", type=int, default=0)

    publish_parser = subparsers.add_parser("register-published", help="Register a YouTube upload for scheduled metric checks.")
    publish_parser.add_argument("--youtube-video-id", required=True)
    publish_parser.add_argument("--package", type=Path, required=True)
    publish_parser.add_argument("--published-at", default="")

    metrics_parser = subparsers.add_parser("record-metrics", help="Record a manual/exported metrics snapshot.")
    metrics_parser.add_argument("--youtube-video-id", required=True)
    metrics_parser.add_argument("--views", type=float, default=0.0)
    metrics_parser.add_argument("--engaged-views", type=float, default=0.0)
    metrics_parser.add_argument("--average-view-duration", type=float, default=0.0)
    metrics_parser.add_argument("--average-view-percentage", type=float, default=0.0)
    metrics_parser.add_argument("--likes", type=float, default=0.0)
    metrics_parser.add_argument("--comments", type=float, default=0.0)
    metrics_parser.add_argument("--shares", type=float, default=0.0)

    source_parser = subparsers.add_parser("register-source", help="Register a permitted owned longform source for daily runs.")
    source_parser.add_argument("--source-video", type=Path, required=True)
    source_parser.add_argument("--analysis-dir", type=Path, required=True)
    source_parser.add_argument("--source-title", default="")
    source_parser.add_argument("--priority", type=int, default=10)
    source_parser.add_argument(
        "--confirm-owned",
        action="store_true",
        help="Required confirmation that you have permission to create Shorts from this source.",
    )

    rerender_parser = subparsers.add_parser(
        "rerender-batch",
        help="Render an existing isolated batch again without discovery, generation, or upload.",
    )
    rerender_parser.add_argument("--analysis-dir", type=Path, required=True)
    rerender_parser.add_argument("--batch-id", required=True)
    rerender_parser.add_argument("--register-review", action="store_true")

    reviews_parser = subparsers.add_parser("list-reviews", help="List rendered Shorts waiting for final review.")
    reviews_parser.add_argument("--status", default="needs_review", choices=["needs_review", "approved", "rejected", "revision_requested", "qa_failed", "all"])
    reviews_parser.add_argument("--limit", type=int, default=20)

    review_parser = subparsers.add_parser("review-decision", help="Approve, reject, or request revision for a rendered Short.")
    review_parser.add_argument("--output", type=Path, required=True)
    review_parser.add_argument("--status", required=True, choices=["approved", "rejected", "revision_requested"])
    review_parser.add_argument("--note", default="")

    upload_parser = subparsers.add_parser("upload-approved", help="Upload approved review items using the configured YouTube upload settings.")
    upload_parser.add_argument("--execute", action="store_true")
    upload_parser.add_argument("--limit", type=int, default=5)
    upload_parser.add_argument("--output", type=Path, help="Upload only this approved rendered MP4.")
    upload_parser.add_argument("--publish-now", action="store_true", help="Publish immediately instead of using the configured schedule. Use only for an explicit one-off release.")
    upload_parser.add_argument("--schedule-at", default="", help="Schedule one --output at this KST time, e.g. '2026-08-03 18:00'.")

    youtube_status_parser = subparsers.add_parser("youtube-status", help="Verify the connected YouTube channel and optionally sync due performance checks.")
    youtube_status_parser.add_argument("--sync-metrics", action="store_true")

    recent_performance_parser = subparsers.add_parser("recent-performance", help="Show current visible statistics for recent public channel videos.")
    recent_performance_parser.add_argument("--limit", type=int, default=5)

    audit_parser = subparsers.add_parser("youtube-audit", help="Audit recent channel uploads for blocked sources, restrictions, duplicates, and missing local records.")
    audit_parser.add_argument("--limit", type=int, default=30)
    audit_parser.add_argument("--apply", action="store_true", help="Mark affected local sources and pending reviews as blocked. Never changes YouTube visibility.")

    cleanup_parser = subparsers.add_parser("youtube-cleanup-problems", help="Delete only current audit findings: blocked broadcaster sources, rights-restricted videos, and exact duplicate uploads.")
    cleanup_parser.add_argument("--execute", action="store_true", help="Required because this permanently deletes the selected YouTube videos.")
    cleanup_parser.add_argument("--limit", type=int, default=50)

    subparsers.add_parser("authorize-youtube", help="Run the one-time local OAuth flow for YouTube Analytics syncing.")

    subparsers.add_parser("status", help="Show persisted orchestration status.")
    return parser


def main() -> None:
    configure_stdout()
    load_project_env()
    args = build_parser().parse_args()
    conn = connect(args.db.resolve())
    initialize_database(conn)
    if args.command == "init":
        if args.write_config:
            write_default_config(args.config.resolve())
        print(f"[orchestrator] db={args.db.resolve()}", flush=True)
        print(f"[orchestrator] config={args.config.resolve()}", flush=True)
        conn.close()
        return
    if args.command == "daily-run":
        conn.close()
        summary = run_daily(args)
        print_daily_summary(summary)
        if summary.get("status") != "completed":
            raise SystemExit(1)
        return
    if args.command == "rerender-batch":
        config = load_config(args.config.resolve())
        analysis_dir = args.analysis_dir.resolve()
        sources = configured_sources(config, [analysis_dir])
        if not sources:
            raise RuntimeError(f"No usable source was found for: {analysis_dir}")
        source = sources[0]
        block_reason = source_safety_reason(config, source)
        if block_reason:
            raise RuntimeError(f"Render blocked by source safety: {block_reason}")
        batch_id = re.sub(r"[^0-9A-Za-z_-]+", "_", str(args.batch_id)).strip("_")
        if not batch_id:
            raise RuntimeError("batch-id is empty after validation")
        package_root = analysis_dir / "shorts_candidates" / "batches" / batch_id
        render = render_for_source(
            config,
            source,
            package_root=package_root,
            batch_id=batch_id,
            execute=True,
        )
        result: dict[str, Any] = {"render": render}
        if bool(args.register_review) and render.get("status") in {"completed", "partial"}:
            # A re-render can be started independently of daily-run.  Review rows
            # still reference a run, so persist this lightweight execution record
            # before registering the rendered packages.
            conn.execute(
                """
                INSERT INTO runs (run_id, requested_at, status, dry_run, summary_path, learning_rule_id, error_text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO NOTHING
                """,
                (
                    batch_id,
                    iso_time(),
                    "completed",
                    0,
                    str(package_root / "rerender_summary.json"),
                    None,
                    None,
                ),
            )
            conn.commit()
            result["review"] = register_review_items(
                conn,
                source=source,
                run_id=batch_id,
                render=render,
                minimum_duration_exclusive=float(
                    (config.get("production", {}) if isinstance(config.get("production"), dict) else {}).get(
                        "minimum_duration_sec_exclusive", 20.0
                    )
                    or 20.0
                ),
                visual_qa=(config.get("production", {}) if isinstance(config.get("production"), dict) else {}).get(
                    "rendered_visual_qa"
                ),
            )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        conn.close()
        return
    if args.command == "register-source":
        if not args.confirm_owned:
            raise RuntimeError("Pass --confirm-owned to register a source in the owned library.")
        entry = register_owned_source(
            args.config.resolve(),
            source_video=args.source_video.resolve(),
            analysis_dir=args.analysis_dir.resolve(),
            source_title=args.source_title,
            priority=args.priority,
        )
        configured = configured_sources(load_config(args.config.resolve()), [])
        matching = next(
            (source for source in configured if source["analysis_dir"] == args.analysis_dir.resolve()),
            None,
        )
        if matching:
            upsert_library_source(conn, matching)
        print(json.dumps({"status": "registered", "source": entry}, ensure_ascii=False), flush=True)
    elif args.command == "authorize-youtube":
        config = load_config(args.config.resolve())
        settings = config.get("youtube_analytics", {}) if isinstance(config.get("youtube_analytics"), dict) else {}
        build_youtube_analytics_service(settings, interactive=True)
        print("[orchestrator] youtube_analytics_authorized=true", flush=True)
    elif args.command == "list-reviews":
        print(
            json.dumps(
                list_review_items(conn, status=args.status, limit=args.limit),
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    elif args.command == "review-decision":
        config = load_config(args.config.resolve())
        result = record_review_decision(
            conn,
            config,
            output_path=args.output,
            status=args.status,
            note=args.note,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    elif args.command == "upload-approved":
        config = load_config(args.config.resolve())
        result = upload_approved_reviews(
            conn,
            config,
            execute=bool(args.execute),
            limit=args.limit,
            output_path=args.output,
            publish_now=bool(args.publish_now),
            schedule_at=str(args.schedule_at or ""),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    elif args.command == "youtube-status":
        config = load_config(args.config.resolve())
        result = youtube_connection_status(conn, config, sync_metrics=bool(args.sync_metrics))
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    elif args.command == "recent-performance":
        config = load_config(args.config.resolve())
        result = recent_youtube_performance(config, limit=int(args.limit or 5))
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    elif args.command == "youtube-audit":
        config = load_config(args.config.resolve())
        result = audit_recent_channel_uploads(
            conn,
            config,
            dry_run=not bool(args.apply),
            limit=int(args.limit or 30),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    elif args.command == "youtube-cleanup-problems":
        if not args.execute:
            raise RuntimeError("Pass --execute to permanently delete the current audited problem uploads.")
        config = load_config(args.config.resolve())
        result = delete_audited_problem_uploads(conn, config, limit=int(args.limit or 50))
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    elif args.command == "register-published":
        result = register_published_short(
            conn,
            youtube_video_id=args.youtube_video_id,
            package_path=args.package.resolve(),
            published_at=parse_time(args.published_at) or utc_now(),
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
    elif args.command == "record-metrics":
        metrics = {
            "views": args.views,
            "engagedViews": args.engaged_views,
            "averageViewDuration": args.average_view_duration,
            "averageViewPercentage": args.average_view_percentage,
            "likes": args.likes,
            "comments": args.comments,
            "shares": args.shares,
        }
        result = record_metrics_snapshot(conn, youtube_video_id=args.youtube_video_id, metrics=metrics)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    elif args.command == "status":
        print(json.dumps(status_summary(conn), ensure_ascii=False, indent=2), flush=True)
    conn.close()


if __name__ == "__main__":
    main()
