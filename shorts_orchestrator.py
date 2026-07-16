from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import sqlite3
import statistics
import subprocess
import sys
from typing import Any

from env_loader import load_project_env


BASE_DIR = Path(__file__).resolve().parent
AUTOMATION_DIR = BASE_DIR / "analysis" / "automation"
DEFAULT_DB_PATH = AUTOMATION_DIR / "orchestrator.sqlite3"
DEFAULT_CONFIG_PATH = BASE_DIR / "automation_config.json"
DEFAULT_BENCHMARK_PROFILE = "templates/benchmark_profiles/rescene_gyaru_variety.json"
CHECKPOINT_HOURS = (1, 24, 72, 168)
ANALYTICS_SCOPES = [
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.upload",
]

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
        "mark_processed_after_render": True,
    },
    "benchmark_profile": DEFAULT_BENCHMARK_PROFILE,
    "generation": {
        "model": "gpt-4.1-mini",
        "include_wide_windows": True,
        "force": False,
    },
    "production": {
        "enabled": True,
        "max_packages_per_source": 1,
        "minimum_score": 85,
        "allowed_decisions": ["auto_render"],
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
    configured = resolve_path(entry.get("source_video"))
    if configured and configured.exists():
        return configured
    metadata = read_json(analysis_dir / "job_metadata.json", {})
    if isinstance(metadata, dict):
        source_video = resolve_path(metadata.get("source_video"))
        if source_video and source_video.exists():
            return source_video
    return None


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

    latest_path = BASE_DIR / "analysis" / "trends" / "latest_trend_candidates.json"
    if dry_run:
        snapshot = read_json(latest_path, {}) if latest_path.exists() else {}
        return {
            "status": "dry_run_cached" if snapshot else "dry_run_unavailable",
            "path": str(latest_path) if snapshot else "",
            "snapshot": snapshot,
            "tokens": sorted(normalized_tokens(trend_text_from_snapshot(snapshot)))[:300],
        }

    output_path = run_dir / "trend_snapshot.json"
    command = [
        sys.executable,
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
    snapshot = trend_result.get("snapshot", {}) if isinstance(trend_result.get("snapshot"), dict) else {}
    selected = snapshot.get("selected") if isinstance(snapshot.get("selected"), dict) else None
    if selected and selected.get("source_url"):
        return selected
    for item in snapshot.get("candidates", []) or []:
        if isinstance(item, dict) and item.get("status") == "green" and item.get("source_url"):
            return item
    for item in snapshot.get("candidates", []) or []:
        if isinstance(item, dict) and item.get("source_url"):
            return item
    return None


def trend_source_summary(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_key": source["source_key"],
        "analysis_dir": str(source["analysis_dir"]),
        "source_video": str(source["source_video"] or ""),
        "source_title": source["source_title"],
        "source_url": str(source.get("source_url") or ""),
        "ready": bool(source.get("ready")),
    }


def acquired_source_from_context(
    *,
    candidate: dict[str, Any],
    analysis_dir: Path,
    context: dict[str, Any],
) -> dict[str, Any]:
    metadata = context.get("metadata", {}) if isinstance(context.get("metadata"), dict) else {}
    downloaded_video = resolve_path(context.get("downloaded_video"))
    title = first_string(
        candidate.get("title"),
        metadata.get("title") if isinstance(metadata, dict) else "",
        candidate.get("video_id"),
        analysis_dir.name,
    )
    return {
        "source_key": source_key_for(analysis_dir),
        "analysis_dir": analysis_dir,
        "source_video": downloaded_video if downloaded_video and downloaded_video.exists() else None,
        "source_title": title,
        "source_url": str(candidate.get("source_url") or context.get("source_url") or ""),
        "source_origin": "trend_acquisition",
        "owned": True,
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
    candidate = selected_trend_candidate(trend_result)
    if not candidate:
        return {"status": "no_selected_trend_source"}
    source_url = str(candidate.get("source_url") or "").strip()
    video_id = str(candidate.get("video_id") or "").strip()
    if not source_url or not video_id:
        return {"status": "invalid_selected_trend_source", "candidate": candidate}

    analysis_dir = BASE_DIR / "analysis" / f"youtube_{video_id}"
    context_path = analysis_dir / "youtube_context.json"
    if not execute:
        context = read_json(context_path, {}) if context_path.exists() else {}
        source = acquired_source_from_context(candidate=candidate, analysis_dir=analysis_dir, context=context if isinstance(context, dict) else {})
        return {"status": "planned", "source": source, "summary": trend_source_summary(source)}

    command = [
        sys.executable,
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

    completed = subprocess.run(command, cwd=BASE_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        return {
            "status": "failed",
            "error": completed.stderr.strip() or completed.stdout.strip(),
            "candidate": candidate,
        }
    context = read_json(context_path, {})
    if not isinstance(context, dict):
        return {"status": "failed", "error": "youtube_context.json was not created", "candidate": candidate}
    source = acquired_source_from_context(candidate=candidate, analysis_dir=analysis_dir, context=context)
    return {
        "status": "completed",
        "source": source,
        "summary": trend_source_summary(source),
        "context_path": str(context_path),
        "log_tail": completed.stdout.splitlines()[-16:],
    }


def mark_acquired_source_processed(source: dict[str, Any], render: dict[str, Any]) -> None:
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
    if rendered_count <= 0:
        return
    command = [
        sys.executable,
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
    if elapsed_hours >= 168:
        return "7d"
    if elapsed_hours >= 72:
        return "72h"
    if elapsed_hours >= 24:
        return "24h"
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
        return {"status": "nothing_due", "synced": 0}
    if not settings.get("enabled", False):
        return {"status": "not_configured", "due": len(due_rows), "synced": 0}
    if dry_run:
        return {"status": "dry_run", "due": len(due_rows), "synced": 0}

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
    return {"status": "completed" if not errors else "partial", "due": len(due_rows), "synced": synced, "errors": errors}


def latest_evaluated_metrics(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT youtube_video_id, checkpoint, metrics_json, observed_at
        FROM metrics_snapshots
        WHERE checkpoint IN ('24h', '72h', '7d')
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
        results.append({"youtube_video_id": video_id, "checkpoint": row["checkpoint"], "metrics": metrics})
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
    evidence: dict[str, Any] = {"evaluated_video_count": len(evaluated)}
    if review_feedback:
        evidence["recent_review_feedback"] = review_feedback
    constraints = [
        "Use the strongest source-specific trigger in the first two cuts.",
        "Preserve a clear setup-to-payoff thread; do not pad with unrelated context.",
    ]
    change_note = "No comparable published performance set exists yet; record this run as the baseline."

    percentages = []
    for item in evaluated:
        try:
            percentages.append(float(item["metrics"].get("averageViewPercentage")))
        except (TypeError, ValueError):
            continue
    if len(evaluated) >= minimum and percentages:
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
        if len(evaluated) < minimum:
            mode = "review_feedback"
            rule_kind = "human_review_revision"
        change_note = "Human review feedback is available; this run must correct the prior content-selection weakness before testing performance."

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
        sys.executable,
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


def annotate_packages(
    conn: sqlite3.Connection,
    *,
    source: dict[str, Any],
    run_id: str,
    learning_rule: dict[str, Any],
) -> dict[str, Any]:
    aggregate_path = source["analysis_dir"] / "shorts_candidates" / "final" / "shorts_packages.json"
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
        package_path = source["analysis_dir"] / "shorts_candidates" / "final" / f"short_{rank:02d}.json"
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
    command = [
        sys.executable,
        "-u",
        str(BASE_DIR / "generate_shorts_packages.py"),
        "--analysis-dir",
        str(source["analysis_dir"]),
        "--source-title",
        source["source_title"],
        "--model",
        str(generation.get("model") or "gpt-4.1-mini"),
        "--learning-rule",
        str(learning_rule_path),
    ]
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
    completed = subprocess.run(command, cwd=BASE_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        return {"status": "failed", "error": completed.stderr.strip() or completed.stdout.strip()}
    annotated = annotate_packages(
        conn,
        source=source,
        run_id=run_id,
        learning_rule=learning_rule,
    )
    return {"status": annotated["status"], "output": annotated, "log_tail": completed.stdout.splitlines()[-12:]}


def render_for_source(
    config: dict[str, Any],
    source: dict[str, Any],
    *,
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

    aggregate_path = source["analysis_dir"] / "shorts_candidates" / "final" / "shorts_packages.json"
    aggregate = read_json(aggregate_path, {})
    packages = aggregate.get("shorts", []) if isinstance(aggregate, dict) else []
    if not isinstance(packages, list):
        return {"status": "missing_output", "path": str(aggregate_path)}

    minimum_score = int(production.get("minimum_score", 85) or 0)
    maximum = max(1, int(production.get("max_packages_per_source", 1) or 1))
    allowed_decisions = {str(value) for value in production.get("allowed_decisions", ["auto_render"]) or []}
    selected: list[dict[str, Any]] = []
    for package in sorted(
        (item for item in packages if isinstance(item, dict)),
        key=lambda item: (int(item.get("score", 0) or 0), -int(item.get("global_rank", 999) or 999)),
        reverse=True,
    ):
        score = int(package.get("score", 0) or 0)
        scorecard = package.get("genre_scorecard", {}) if isinstance(package.get("genre_scorecard"), dict) else {}
        decision = str(scorecard.get("decision") or package.get("decision") or "")
        if score < minimum_score or (allowed_decisions and decision not in allowed_decisions):
            continue
        short_id = str(package.get("short_id") or "").strip()
        if not short_id:
            continue
        package_path = source["analysis_dir"] / "shorts_candidates" / "final" / f"{short_id}.json"
        if not package_path.exists():
            continue
        selected.append({"short_id": short_id, "package_path": str(package_path), "score": score, "decision": decision})
        if len(selected) >= maximum:
            break
    if not selected:
        return {"status": "no_auto_render_candidate", "minimum_score": minimum_score}

    rendered: list[dict[str, Any]] = []
    for candidate in selected:
        output_path = source["analysis_dir"] / "productions" / f"{candidate['short_id']}.mp4"
        command = [
            sys.executable,
            "-u",
            str(BASE_DIR / "render_short.py"),
            "--source-video",
            str(source_video),
            "--package",
            str(candidate["package_path"]),
            "--output",
            str(output_path),
        ]
        completed = subprocess.run(command, cwd=BASE_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace")
        entry = {
            **candidate,
            "output_path": str(output_path),
            "status": "completed" if completed.returncode == 0 else "failed",
            "log_tail": completed.stdout.splitlines()[-8:],
        }
        if completed.returncode != 0:
            entry["error"] = completed.stderr.strip() or completed.stdout.strip()
        rendered.append(entry)
    return {
        "status": "completed" if all(item["status"] == "completed" for item in rendered) else "partial",
        "rendered": rendered,
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
        return {"status": "unavailable", "reason": "ffprobe not found"}
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


def auto_review_render(output_path: Path) -> dict[str, Any]:
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
        if float(probe.get("duration") or 0.0) <= 0.5:
            issues.append("rendered duration is too short")
    elif probe.get("status") == "unavailable":
        warnings.append(str(probe.get("reason") or "ffprobe unavailable"))
    else:
        warnings.append(str(probe.get("reason") or "ffprobe failed"))

    qa_status = "fail" if issues else ("warning" if warnings else "pass")
    return {"qa_status": qa_status, "checks": checks, "issues": issues, "warnings": warnings}


def register_review_items(
    conn: sqlite3.Connection,
    *,
    source: dict[str, Any],
    run_id: str,
    render: dict[str, Any],
) -> dict[str, Any]:
    rendered = [item for item in render.get("rendered", []) or [] if isinstance(item, dict)]
    entries: list[dict[str, Any]] = []
    for item in rendered:
        if item.get("status") != "completed" or not item.get("output_path"):
            continue
        output_path = Path(str(item["output_path"])).resolve()
        package_path = Path(str(item.get("package_path") or "")).resolve()
        qa = auto_review_render(output_path)
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


def record_review_decision(conn: sqlite3.Connection, *, output_path: Path, status: str, note: str) -> dict[str, Any]:
    resolved = str(output_path.resolve())
    row = conn.execute("SELECT * FROM review_items WHERE output_path = ?", (resolved,)).fetchone()
    if not row and output_path.exists():
        row = conn.execute("SELECT * FROM review_items WHERE content_sha256 = ?", (file_sha256(output_path),)).fetchone()
    if not row:
        raise RuntimeError(f"No review item found for output: {resolved}")
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


def youtube_upload_metadata(package_path: Path) -> dict[str, Any]:
    package = read_json(package_path, {})
    if not isinstance(package, dict):
        raise RuntimeError(f"Invalid rendered package JSON: {package_path}")
    title = " ".join(
        part.strip()
        for part in (str(package.get("title_line1") or ""), str(package.get("title_line2") or ""))
        if part.strip()
    )[:100]
    if not title:
        title = package_path.stem[:100]
    pitch = str(package.get("selection_pitch") or "").strip()
    tags = [str(tag).strip().lstrip("#") for tag in package.get("fun_tags", []) or [] if str(tag).strip()]
    hashtag_line = " ".join(f"#{tag.replace(' ', '')}" for tag in tags[:8])
    description = "\n\n".join(part for part in (pitch, hashtag_line, "#shorts") if part)[:5000]
    return {"title": title, "description": description, "tags": tags[:20]}


def upload_rendered_outputs(
    conn: sqlite3.Connection,
    config: dict[str, Any],
    render: dict[str, Any],
    *,
    execute: bool,
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
    service = build_youtube_upload_service(
        analytics,
        interactive=bool(analytics.get("interactive_on_first_run", False)),
    )

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
        body = {
            "snippet": {
                "title": metadata["title"],
                "description": metadata["description"],
                "tags": metadata["tags"],
                "categoryId": str(upload.get("category_id") or "24"),
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": bool(upload.get("made_for_kids", False)),
            },
        }
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
                    privacy_status,
                    iso_time(),
                    json.dumps(response, ensure_ascii=False),
                ),
            )
            conn.commit()
            result = {"status": "uploaded", "output_path": str(output_path), "youtube_video_id": video_id, "privacy_status": privacy_status}
            if privacy_status == "public":
                register_published_short(
                    conn,
                    youtube_video_id=video_id,
                    package_path=package_path,
                    published_at=utc_now(),
                )
                result["metrics_registered"] = True
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
) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT review_items.*
        FROM review_items
        LEFT JOIN rendered_uploads
            ON rendered_uploads.content_sha256 = review_items.content_sha256
        WHERE review_items.review_status = 'approved'
            AND rendered_uploads.content_sha256 IS NULL
        ORDER BY COALESCE(review_items.decided_at, review_items.created_at) DESC
        LIMIT ?
        """,
        (max(1, limit),),
    ).fetchall()
    if not rows:
        return {"status": "nothing_to_upload"}
    render = {
        "rendered": [
            {
                "status": "completed",
                "output_path": row["output_path"],
                "package_path": row["package_path"],
            }
            for row in rows
        ]
    }
    return upload_rendered_outputs(conn, config, render, execute=execute)


def run_daily(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    db_path = args.db.resolve()
    config = load_config(config_path)
    conn = connect(db_path)
    initialize_database(conn)
    run_id = run_id_for()
    run_dir = AUTOMATION_DIR / "runs" / run_id
    summary_path = run_dir / "run_summary.json"
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
    }
    try:
        analytics = config.get("youtube_analytics", {}) if isinstance(config.get("youtube_analytics"), dict) else {}
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

        if not sources:
            raise RuntimeError(
                "No source is available. Enable source_acquisition or add permitted library sources."
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
                if item["generation"].get("status") == "completed":
                    print(f"[orchestrator] phase=render source={source['source_title']}", flush=True)
                    item["render"] = render_for_source(config, source, execute=bool(args.execute))
                    if item["render"].get("status") in {"completed", "partial"}:
                        review_config = config.get("review", {}) if isinstance(config.get("review"), dict) else {}
                        if bool(review_config.get("enabled", True)):
                            print(f"[orchestrator] phase=review source={source['source_title']}", flush=True)
                            item["review"] = register_review_items(
                                conn,
                                source=source,
                                run_id=run_id,
                                render=item["render"],
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
                conn.execute(
                    "UPDATE library_sources SET last_selected_at = ? WHERE source_key = ?",
                    (iso_time(), source["source_key"]),
                )
                conn.commit()
            source_results.append(item)
        summary["sources"] = source_results
        summary["status"] = "completed"
        conn.execute(
            "UPDATE runs SET status = ?, learning_rule_id = ? WHERE run_id = ?",
            ("completed", learning_rule["rule_id"], run_id),
        )
        conn.commit()
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        conn.execute("UPDATE runs SET status = ?, error_text = ? WHERE run_id = ?", ("failed", str(exc), run_id))
        conn.commit()
    finally:
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
        "SELECT COUNT(*) AS count FROM published_shorts WHERE next_check_at IS NOT NULL AND next_check_at <= ?",
        (iso_time(),),
    ).fetchone()
    published = conn.execute("SELECT COUNT(*) AS count FROM published_shorts").fetchone()
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
        result = record_review_decision(
            conn,
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
        )
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
