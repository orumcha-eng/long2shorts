"""Private Telegram control bot for Long2Shorts.

The bot uses Telegram long polling, so it needs no public IP address.  It is
deliberately restricted to the chat IDs configured in ``.env`` and delegates
all production work to the existing orchestrator CLI.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from env_loader import load_project_env


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "automation_config.json"
ORCHESTRATOR_PATH = BASE_DIR / "shorts_orchestrator.py"
STATE_PATH = BASE_DIR / "analysis" / "automation" / "telegram_bot_state.json"
POLL_TIMEOUT_SECONDS = 25
MAX_TELEGRAM_MESSAGE = 3900
PREVIEW_DIR = BASE_DIR / "analysis" / "automation" / "telegram_previews"


def python_executable() -> Path:
    venv_python = BASE_DIR / ".venv" / "Scripts" / "python.exe"
    return venv_python if venv_python.exists() else Path(sys.executable)


def parse_chat_ids(value: str) -> set[int]:
    chat_ids: set[int] = set()
    for token in value.replace(",", " ").split():
        try:
            chat_ids.add(int(token))
        except ValueError as exc:
            raise RuntimeError("TELEGRAM_ALLOWED_CHAT_IDS must contain only numeric chat IDs.") from exc
    return chat_ids


def read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def compact_text(value: str, *, limit: int = MAX_TELEGRAM_MESSAGE) -> str:
    value = value.strip()
    return value if len(value) <= limit else value[: limit - 20] + "\n…(뒤 내용 생략)"


class Long2ShortsTelegramBot:
    def __init__(self, token: str, allowed_chat_ids: set[int]) -> None:
        self.token = token
        self.allowed_chat_ids = allowed_chat_ids
        self.state = read_json(STATE_PATH, {})
        if not isinstance(self.state, dict):
            self.state = {}
        self.offset = int(self.state.get("offset", 0) or 0)
        self.active_process: subprocess.Popen[str] | None = None
        self.active_name = ""
        self.active_chat_id = 0
        self.active_log: list[str] = []
        self.active_stage = ""
        self.active_last_progress_at = 0.0
        self.lock = threading.Lock()

    def save_state(self) -> None:
        self.state["offset"] = self.offset
        write_json(STATE_PATH, self.state)

    def source_rows(self, source_key: str) -> list[dict[str, Any]]:
        return [
            row
            for row in self.review_rows("all")
            if str(row.get("source_key") or "") == source_key
        ]

    def source_review_is_resolved(self, source_key: str) -> bool:
        rows = self.source_rows(source_key)
        return bool(rows) and not any(str(row.get("review_status") or "") == "needs_review" for row in rows)

    def offer_next_source(self, chat_id: int, source_key: str) -> None:
        """Ask exactly once when every candidate for one source is settled."""
        if not source_key or not self.source_review_is_resolved(source_key):
            return
        offered = self.state.setdefault("next_source_offered", {})
        if not isinstance(offered, dict):
            offered = {}
            self.state["next_source_offered"] = offered
        if source_key in offered:
            return
        offered[source_key] = datetime.now(timezone.utc).isoformat()
        self.save_state()
        self.send(
            chat_id,
            "이 롱폼의 쇼츠 후보 처리가 모두 끝났어요.\n\n다음 롱폼 후보로 추가 쇼츠를 만들까요?",
            reply_markup={
                "inline_keyboard": [[
                    {"text": "▶ 다음 롱폼으로 추가 제작", "callback_data": f"more_source:{source_key}"},
                    {"text": "🛑 이번 회차 종료", "callback_data": f"end_session:{source_key}"},
                ]]
            },
        )

    def api(self, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload or {}).encode("utf-8")
        request = Request(
            f"https://api.telegram.org/bot{self.token}/{method}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=POLL_TIMEOUT_SECONDS + 10) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"Telegram API connection failed: {exc}") from exc
        if not body.get("ok"):
            raise RuntimeError(f"Telegram API error: {body.get('description', 'unknown error')}")
        return body

    def api_photo(self, *, chat_id: int, photo_path: Path, caption: str, reply_markup: dict[str, Any]) -> dict[str, Any]:
        boundary = f"----Long2Shorts{os.urandom(12).hex()}"
        body = bytearray()

        def add_field(name: str, value: str) -> None:
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            body.extend(value.encode("utf-8"))
            body.extend(b"\r\n")

        add_field("chat_id", str(chat_id))
        add_field("caption", compact_text(caption, limit=1000))
        add_field("reply_markup", json.dumps(reply_markup, ensure_ascii=False))
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="photo"; filename="{photo_path.name}"\r\n'.encode()
        )
        body.extend(b"Content-Type: image/jpeg\r\n\r\n")
        body.extend(photo_path.read_bytes())
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode())
        request = Request(
            f"https://api.telegram.org/bot{self.token}/sendPhoto",
            data=bytes(body),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=POLL_TIMEOUT_SECONDS + 10) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"Telegram photo upload failed: {exc}") from exc
        if not result.get("ok"):
            raise RuntimeError(f"Telegram API error: {result.get('description', 'unknown error')}")
        return result

    def send(self, chat_id: int, text: str, *, reply_markup: dict[str, Any] | None = None) -> None:
        if not text:
            return
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": compact_text(text),
            "disable_web_page_preview": True,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        self.api(
            "sendMessage",
            payload,
        )

    def run_orchestrator(self, *args: str, timeout: int = 90) -> tuple[int, str]:
        command = [
            str(python_executable()),
            str(ORCHESTRATOR_PATH),
            "--config",
            str(CONFIG_PATH),
            *args,
        ]
        completed = subprocess.run(
            command,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
        return completed.returncode, output

    def review_rows(self, status: str) -> list[dict[str, Any]]:
        code, output = self.run_orchestrator("list-reviews", "--status", status, "--limit", "30")
        if code != 0:
            raise RuntimeError(output or "검토 목록을 읽지 못했습니다.")
        try:
            rows = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError("검토 목록 응답을 읽지 못했습니다.") from exc
        return rows if isinstance(rows, list) else []

    def find_review(self, short_id: str) -> dict[str, Any] | None:
        for row in self.review_rows("all"):
            if str(row.get("short_id") or "") == short_id:
                return row
        return None

    def find_review_by_token(self, token: str) -> dict[str, Any] | None:
        matches = [
            row
            for row in self.review_rows("all")
            if str(row.get("content_sha256") or "").startswith(token)
        ]
        return matches[0] if len(matches) == 1 else None

    def has_active_job(self) -> bool:
        with self.lock:
            return bool(self.active_process and self.active_process.poll() is None)

    def review_title(self, row: dict[str, Any]) -> str:
        package = read_json(Path(str(row.get("package_path") or "")), {})
        if not isinstance(package, dict):
            package = {}
        title_lines = [
            str(package.get(key) or "").strip()
            for key in ("title_line1", "title_line2")
            if str(package.get(key) or "").strip()
        ]
        title = "\n".join(title_lines)
        return title or str(package.get("upload_title") or row.get("short_id") or "쇼츠 후보")

    def review_source_summary(self, row: dict[str, Any]) -> str:
        package_path = Path(str(row.get("package_path") or "")).resolve()
        analysis_dir = next(
            (parent for parent in package_path.parents if (parent / "youtube_context.json").exists()),
            package_path.parent.parent.parent,
        )
        context = read_json(analysis_dir / "youtube_context.json", {})
        if not isinstance(context, dict):
            return ""
        metadata = context.get("metadata", {}) if isinstance(context.get("metadata"), dict) else {}
        title = str(metadata.get("title") or "").strip()
        channel = str(metadata.get("channel_title") or metadata.get("uploader") or "").strip()
        source_url = str(context.get("source_url") or "").strip()
        lines = []
        if title:
            lines.append(f"원본: {title}")
        if channel:
            lines.append(f"출처 채널: {channel}")
        if source_url:
            lines.append(f"원본 링크: {source_url}")
        return "\n".join(lines)

    def review_experiment_summary(self, row: dict[str, Any]) -> str:
        package = read_json(Path(str(row.get("package_path") or "")), {})
        if not isinstance(package, dict):
            return ""
        experiment = package.get("experiment") if isinstance(package.get("experiment"), dict) else {}
        if not experiment:
            return ""
        hypothesis = str(experiment.get("hypothesis") or "").strip()
        variable = str(experiment.get("primary_variable") or "").strip()
        choices = experiment.get("choices") if isinstance(experiment.get("choices"), list) else []
        first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        choice_text = str(first_choice.get("decision") or "").strip()
        lines = ["🧪 이번 편집 실험"]
        if hypothesis:
            lines.append(f"가설: {hypothesis}")
        if variable:
            lines.append(f"핵심: {variable}")
        if choice_text:
            lines.append(f"적용: {choice_text}")
        return "\n".join(lines)

    def review_screenshot(self, row: dict[str, Any]) -> Path:
        output_path = Path(str(row.get("output_path") or "")).resolve()
        if not output_path.exists():
            raise RuntimeError(f"렌더 영상을 찾지 못했습니다: {output_path.name}")
        token = str(row.get("content_sha256") or output_path.stem)[:16]
        preview_path = PREVIEW_DIR / f"{token}.jpg"
        if preview_path.exists() and preview_path.stat().st_mtime >= output_path.stat().st_mtime:
            return preview_path
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError("첫 장면 미리보기에 imageio-ffmpeg가 필요합니다.") from exc
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "0.4",
            "-i",
            str(output_path),
            "-frames:v",
            "1",
            "-vf",
            "scale=720:-2",
            str(preview_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        if completed.returncode != 0 or not preview_path.exists():
            raise RuntimeError(completed.stderr.strip() or "첫 장면 스크린샷 생성에 실패했습니다.")
        return preview_path

    def send_review_cards(self, chat_id: int, source_key: str = "") -> None:
        rows = self.review_rows("needs_review")
        if source_key:
            rows = [row for row in rows if str(row.get("source_key") or "") == source_key]
        elif rows:
            # A production cycle handles one source at a time.  Keep stale
            # pending items from earlier cycles out of this review round.
            source_key = str(rows[0].get("source_key") or "")
            rows = [row for row in rows if str(row.get("source_key") or "") == source_key]
        if not rows:
            self.send(chat_id, "새 검토 후보가 없습니다.")
            return
        self.send(chat_id, f"🎬 새 쇼츠 후보 {len(rows)}개입니다. 각 후보 아래 버튼으로 결정하세요.")
        for row in rows[:10]:
            token = str(row.get("content_sha256") or "")[:16]
            if not token:
                continue
            keyboard = {
                "inline_keyboard": [
                    [
                        {"text": "✅ 승인 후 업로드", "callback_data": f"approve_upload:{token}"},
                        {"text": "🗑 거부", "callback_data": f"reject:{token}"},
                    ]
                ]
            }
            title = self.review_title(row)
            source_summary = self.review_source_summary(row)
            experiment_summary = self.review_experiment_summary(row)
            self.send(
                chat_id,
                "\n".join(
                    part
                    for part in (f"🎬 {row.get('short_id') or '후보'}", title, source_summary, experiment_summary)
                    if part
                ),
            )
            caption = f"점수 {row.get('score') or '?'} · 첫 장면 미리보기"
            try:
                self.api_photo(
                    chat_id=chat_id,
                    photo_path=self.review_screenshot(row),
                    caption=caption,
                    reply_markup=keyboard,
                )
            except Exception as exc:
                self.send(chat_id, f"{caption}\n(스크린샷 생성 실패: {exc})", reply_markup=keyboard)

    @staticmethod
    def has_verified_owned_source() -> bool:
        try:
            config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return False
        safety = config.get("source_safety", {}) if isinstance(config.get("source_safety"), dict) else {}
        if not (bool(safety.get("enabled", True)) and bool(safety.get("require_verified_owned_source", False))):
            return True
        library = config.get("library", {}) if isinstance(config.get("library"), dict) else {}
        for source in library.get("sources", []) or []:
            if not isinstance(source, dict) or source.get("owned") is not True:
                continue
            source_video = Path(str(source.get("source_video") or ""))
            if not source_video.is_absolute():
                source_video = (CONFIG_PATH.parent / source_video).resolve()
            if source_video.exists():
                return True
        return False

    @staticmethod
    def phase_message(phase: str) -> str:
        messages = {
            "channel_audit": "\ucd5c\uadfc \uc5c5\ub85c\ub4dc\uc758 \uc800\uc791\uad8c\u00b7\uc911\ubcf5\u00b7\ucc28\ub2e8 \uc0c1\ud0dc\ub97c \uc810\uac80\ud558\uace0 \uc788\uc5b4\uc694.",
            "metrics_sync": "\uc774\uc804 \uc5c5\ub85c\ub4dc \uae30\ub85d\uc744 \ud655\uc778\ud558\uace0 \uc788\uc5b4\uc694.",
            "learning_rule": "\uc774\uc804 \uac80\ud1a0 \uc758\uacac\uc744 \uc815\ub9ac\ud558\uace0 \uc788\uc5b4\uc694.",
            "trend_research": "\uc624\ub298\uc758 \uc6d0\ubcf8 \ud6c4\ubcf4\ub97c \ucc3e\uace0 \uc788\uc5b4\uc694.",
            "source_acquisition": "\uc6d0\ubcf8\uc744 \uc900\ube44\ud558\uace0 \uc788\uc5b4\uc694.",
            "source_analysis": "\uc601\uc0c1 \ub0b4\uc6a9\uc744 \ubd84\uc11d\ud558\uace0 \uc788\uc5b4\uc694.",
            "package_generation": "\uc1fc\uce20 \ud6c4\ubcf4\ub97c \uad6c\uc131\ud558\uace0 \uc788\uc5b4\uc694.",
            "render": "\ud6c4\ubcf4 \uc601\uc0c1\uc744 \ub9cc\ub4e4\uace0 \uc788\uc5b4\uc694.",
            "review": "\uc5c5\ub85c\ub4dc \uc804 \uac80\ud1a0\ub97c \uc900\ube44\ud558\uace0 \uc788\uc5b4\uc694.",
            "upload": "\uc2b9\uc778\ubcf8 \uc5c5\ub85c\ub4dc\ub97c \uc900\ube44\ud558\uace0 \uc788\uc5b4\uc694.",
        }
        return messages.get(phase, "\uc791\uc5c5\uc744 \uc9c4\ud589\ud558\uace0 \uc788\uc5b4\uc694.")

    def watch_heartbeat(self, process: subprocess.Popen[str], chat_id: int) -> None:
        last_notice_stage = ""
        while process.poll() is None:
            time.sleep(75)
            with self.lock:
                if self.active_process is not process:
                    return
                stage = self.active_stage
                idle_seconds = time.monotonic() - self.active_last_progress_at
            if stage and idle_seconds >= 60 and stage != last_notice_stage:
                try:
                    self.send(chat_id, f"\u23f3 {self.phase_message(stage)} \uc870\uae08 \ub354 \uac78\ub9b4 \uc218 \uc788\uc5b4\uc694.")
                    last_notice_stage = stage
                except RuntimeError:
                    pass

    def start_job(self, chat_id: int, name: str, *args: str, followup_source_key: str = "") -> None:
        if args and args[0] == "daily-run" and not self.has_verified_owned_source():
            self.send(
                chat_id,
                "⛔ 권리 확인된 내 롱폼이 아직 등록되지 않아 제작을 시작하지 않았어요. "
                "GUI에서 ‘내 롱폼’ → ‘라이브러리 등록’을 먼저 해주세요.",
            )
            return
        with self.lock:
            if self.active_process and self.active_process.poll() is None:
                self.send(chat_id, f"이미 '{self.active_name}' 작업이 실행 중입니다. /status 또는 /logs 로 확인하세요.")
                return
            command = [
                str(python_executable()),
                str(ORCHESTRATOR_PATH),
                "--config",
                str(CONFIG_PATH),
                *args,
            ]
            self.active_process = subprocess.Popen(
                command,
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self.active_name = name
            self.active_chat_id = chat_id
            self.active_log = []
            self.active_stage = ""
            self.active_last_progress_at = time.monotonic()
            process = self.active_process
        self.send(chat_id, f"\u25b6 {name} \uc2dc\uc791\ud588\uc5b4\uc694. \uc9c4\ud589 \uc0c1\ud0dc\ub9cc \uac04\ub2e8\ud788 \uc54c\ub824\ub4dc\ub9b4\uac8c\uc694.")
        threading.Thread(
            target=self.watch_job,
            args=(process, name, chat_id, followup_source_key),
            daemon=True,
        ).start()
        threading.Thread(target=self.watch_heartbeat, args=(process, chat_id), daemon=True).start()

    def watch_job(
        self,
        process: subprocess.Popen[str],
        name: str,
        chat_id: int,
        followup_source_key: str = "",
    ) -> None:
        last_phase = ""
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            with self.lock:
                self.active_log.append(line)
                self.active_log = self.active_log[-30:]
            if line.startswith("[orchestrator] phase="):
                phase = line.split("phase=", 1)[1].split(maxsplit=1)[0]
                if phase == last_phase:
                    continue
                last_phase = phase
                friendly_message = self.phase_message(phase)
                with self.lock:
                    self.active_stage = phase
                    self.active_last_progress_at = time.monotonic()
                try:
                    self.send(chat_id, f"\u00b7 {friendly_message}")
                except RuntimeError:
                    pass
        exit_code = process.wait()
        with self.lock:
            if self.active_process is process:
                self.active_process = None
                self.active_name = ""
                self.active_chat_id = 0
                self.active_stage = ""
        log_text = "\n".join(self.active_log)
        if exit_code == 0:
            if "status=no_candidates" in log_text:
                message = "\u26a0\ufe0f \uc1fc\uce20 \ud6c4\ubcf4\ub97c \ub9cc\ub4e4\uc9c0 \ubabb\ud588\uc5b4\uc694. \uc120\ud0dd\ud55c \ub86f\ud3fc\uc5d0\uc11c \uc5b4\uc0c9\ud558\uc9c0 \uc54a\uc740 \uc810\ud504\ucef7 \uad6c\uc131\uc744 \ub9cc\ub4e4\uc9c0 \ubabb\ud55c \uacbd\uc6b0\uc608\uc694. \ub2e4\uc74c \uc2dc\ub3c4\uc5d0\uc11c\ub294 \ub2e4\ub978 \uc7a5\uba74\ubc30\uc5f4\ub85c \uc7ac\uc2dc\ub3c4\ud569\ub2c8\ub2e4."
            else:
                message = f"\u2705 {name} \uc644\ub8cc"
        else:
            if "blocked source term" in log_text or "blocked channel" in log_text:
                message = "\u26d4 \uae08\uc9c0\ub41c \uc6d0\ubcf8\uc774 \ud0d0\uc9c0\ub418\uc5b4 \uc81c\uc791\uc744 \uba48\ucdc4\uc5b4\uc694. \uc6d0\ubcf8\uc744 \ubc14\uafd4\uc8fc\uc138\uc694."
            elif "rights are not verified" in log_text or "No permitted source" in log_text:
                message = "\u26d4 \uad8c\ub9ac\uac00 \ud655\uc778\ub41c \uc6d0\ubcf8\uc774 \uc5c6\uc5b4 \uc81c\uc791\uc744 \uba48\ucdc4\uc5b4\uc694. \uc18c\uc720\ud588\uac70\ub098 \uc0ac\uc6a9 \ud5c8\uac00\ub97c \ubc1b\uc740 \uc601\uc0c1\uc744 \ub4f1\ub85d\ud574 \uc8fc\uc138\uc694."
            else:
                message = f"\u274c {name}\uc5d0 \ubb38\uc81c\uac00 \uc0dd\uacbc어요. /logs\ub97c \ubcf4\ub0b4\uba74 \uc0c1\uc138 \ub0b4\uc6a9을 \ud655\uc778\ud560 \uc218 \uc788\uc5b4\uc694."
        try:
            self.send(chat_id, message)
        except RuntimeError:
            pass
        if exit_code == 0 and name == "오늘 쇼츠 제작" and "status=no_candidates" not in log_text:
            try:
                self.send_review_cards(chat_id)
            except Exception as exc:
                try:
                    self.send(chat_id, f"검토 후보 카드 전송 실패: {exc}")
                except RuntimeError:
                    pass
        if exit_code == 0 and followup_source_key:
            try:
                self.offer_next_source(chat_id, followup_source_key)
            except Exception as exc:
                try:
                    self.send(chat_id, f"다음 롱폼 확인 실패: {exc}")
                except RuntimeError:
                    pass

    def command_help(self) -> str:
        return (
            "/recent - Recent 5 videos: views / likes / comments\n"
            "Long2Shorts Telegram 명령어\n\n"
            "/status - 제작/검토/업로드 상태\n"
            "/reviews - 검토 대기 쇼츠 목록\n"
            "/approve short_01 - 해당 쇼츠 승인\n"
            "/revise short_01 피드백 - 수정 요청\n"
            "/create - 오늘 쇼츠 제작 시작\n"
            "/upload - 승인본 수 확인\n"
            "/upload confirm - 승인본 실제 업로드/예약\n"
            "/logs - 현재 작업의 최근 로그\n"
            "/stop - 현재 Telegram 작업 중지\n"
            "/id - 이 대화의 Telegram chat ID"
        )

    def send_recent_performance(self, chat_id: int) -> None:
        code, output = self.run_orchestrator("recent-performance", "--limit", "5")
        if code != 0:
            self.send(chat_id, f"최근 성과 확인 실패\n{output}")
            return
        try:
            result = json.loads(output)
        except json.JSONDecodeError:
            self.send(chat_id, "최근 성과 응답을 읽지 못했습니다.")
            return
        videos = result.get("videos", []) if isinstance(result.get("videos"), list) else []
        if result.get("status") != "connected":
            self.send(chat_id, f"최근 성과를 불러오지 못했습니다: {result.get('status', 'unknown')}")
            return
        if not videos:
            self.send(chat_id, "공개된 최근 영상이 없습니다.")
            return
        lines = [f"📊 {result.get('channel_title') or '채널'} 최근 공개 영상 {len(videos)}개"]
        for index, video in enumerate(videos, start=1):
            published_at = str(video.get("published_at") or "")
            try:
                local_time = datetime.fromisoformat(published_at.replace("Z", "+00:00")).astimezone(timezone(timedelta(hours=9)))
                date_label = local_time.strftime("%m/%d %H:%M")
            except ValueError:
                date_label = published_at
            title = " ".join(str(video.get("title") or "제목 없음").split())[:70]
            lines.append(
                f"{index}. {title}\n"
                f"   {date_label} KST · 조회 {int(video.get('views') or 0):,} · 좋아요 {int(video.get('likes') or 0):,} · 댓글 {int(video.get('comments') or 0):,}"
            )
        self.send(chat_id, "\n".join(lines))

    def handle_conversation(self, chat_id: int, text: str) -> None:
        """Route common Korean requests without allowing ambiguous uploads."""
        normalized = re.sub(r"\s+", "", text).lower()
        if any(word in normalized for word in ("조회수", "좋아요", "댓글", "성과", "최근올린")):
            self.send_recent_performance(chat_id)
            return
        if any(word in normalized for word in ("도움", "명령", "사용법", "뭐할수")):
            self.send(chat_id, self.command_help())
            return
        if "쇼츠" in normalized and any(word in normalized for word in ("제작", "만들", "생성", "돌려")):
            self.start_job(chat_id, "오늘 쇼츠 제작", "daily-run", "--execute")
            return
        if any(word in normalized for word in ("상태", "진행", "현황")):
            self.handle_command(chat_id, "/status")
            return
        if any(word in normalized for word in ("검토", "후보", "리뷰")):
            self.handle_command(chat_id, "/reviews")
            return
        if "업로드" in normalized or "예약" in normalized:
            self.send(
                chat_id,
                "업로드 대기 수를 확인할게요. 실제 업로드는 안전을 위해 /upload confirm 을 보내야 진행합니다.",
            )
            self.handle_command(chat_id, "/upload")
            return
        if any(word in normalized for word in ("로그", "어디까지", "에러")):
            self.handle_command(chat_id, "/logs")
            return
        self.send(
            chat_id,
            "네, 이렇게 말해도 돼요: ‘쇼츠 제작해줘’, ‘현재 상태 알려줘’, ‘검토할 후보 보여줘’.\n"
            "업로드는 /upload confirm 으로 마지막 확인을 받아요.",
        )

    def handle_command(self, chat_id: int, text: str) -> None:
        command, _, remainder = text.strip().partition(" ")
        command = command.split("@", 1)[0].lower()
        argument = remainder.strip()
        if command in {"/start", "/help"}:
            self.send(chat_id, self.command_help())
            return
        if command == "/id":
            self.send(chat_id, f"Telegram chat ID: {chat_id}")
            return
        if command == "/status":
            code, output = self.run_orchestrator("status")
            with self.lock:
                running = bool(self.active_process and self.active_process.poll() is None)
                job = self.active_name if running else ""
                stage = self.active_stage if running else ""
            try:
                status = json.loads(output) if output else {}
            except json.JSONDecodeError:
                status = {}
            if not code or not isinstance(status, dict):
                self.send(chat_id, "현재 상태를 읽지 못했습니다. 잠시 후 다시 시도해 주세요.")
                return
            latest_run = status.get("latest_run", {}) if isinstance(status.get("latest_run"), dict) else {}
            lines = [
                f"📍 현재 작업: {job or '진행 중인 작업 없음'}",
            ]
            if stage:
                lines.append(f"진행 단계: {self.phase_message(stage)}")
            elif latest_run:
                run_status = str(latest_run.get("status") or "")
                run_label = {"completed": "완료", "running": "진행 중", "failed": "실패"}.get(run_status, "확인 중")
                lines.append(f"최근 제작 작업: {run_label}")
            lines.extend(
                [
                    f"검토 대기 후보: {int(status.get('pending_reviews') or 0)}개",
                    f"자동 검사 보류: {int(status.get('qa_failed_reviews') or 0)}개",
                    f"성과 데이터 수집본: {int(status.get('metric_snapshots') or 0)}건",
                ]
            )
            self.send(chat_id, "\n".join(lines))
            return
        if command == "/recent":
            self.send_recent_performance(chat_id)
            return
        if command == "/reviews":
            self.send_review_cards(chat_id)
            return
        if command in {"/approve", "/revise"}:
            short_id, _, note = argument.partition(" ")
            if not short_id:
                self.send(chat_id, "예: /approve short_01 또는 /revise short_01 첫 장면을 더 강하게")
                return
            row = self.find_review(short_id)
            if not row:
                self.send(chat_id, f"'{short_id}' 검토 항목을 찾지 못했습니다. /reviews 로 확인하세요.")
                return
            status = "approved" if command == "/approve" else "revision_requested"
            review_note = note or ("telegram approval" if status == "approved" else "telegram revision request")
            code, output = self.run_orchestrator(
                "review-decision",
                "--output",
                str(row.get("output_path") or ""),
                "--status",
                status,
                "--note",
                review_note,
            )
            self.send(chat_id, ("✅ " if code == 0 else "❌ ") + compact_text(output or f"{short_id}: {status}"))
            return
        if command == "/create":
            self.start_job(chat_id, "오늘 쇼츠 제작", "daily-run", "--execute")
            return
        if command == "/upload":
            approved = self.review_rows("approved")
            if argument.lower() != "confirm":
                self.send(
                    chat_id,
                    f"승인된 업로드 대기본: {len(approved)}개\n실제 업로드와 예약을 하려면 /upload confirm 을 보내세요.",
                )
                return
            if not approved:
                self.send(chat_id, "업로드할 승인본이 없습니다.")
                return
            self.start_job(chat_id, "승인본 업로드/예약", "upload-approved", "--execute")
            return
        if command == "/logs":
            with self.lock:
                name = self.active_name
                lines = list(self.active_log[-12:])
            self.send(chat_id, f"{name or '실행 중인 작업 없음'}\n\n" + ("\n".join(lines) or "로그 없음"))
            return
        if command == "/stop":
            with self.lock:
                process = self.active_process
                name = self.active_name
            if not process or process.poll() is not None:
                self.send(chat_id, "중지할 Telegram 작업이 없습니다.")
                return
            process.terminate()
            self.send(chat_id, f"{name} 중지 요청을 보냈습니다.")
            return
        self.send(chat_id, "알 수 없는 명령입니다. /help 를 보내세요.")

    def handle_callback(self, callback: dict[str, Any]) -> None:
        callback_id = str(callback.get("id") or "")
        message = callback.get("message") if isinstance(callback.get("message"), dict) else {}
        chat = message.get("chat") if isinstance(message.get("chat"), dict) else {}
        chat_id = chat.get("id")
        data = str(callback.get("data") or "")
        if not isinstance(chat_id, int) or chat_id not in self.allowed_chat_ids:
            if callback_id:
                self.api("answerCallbackQuery", {"callback_query_id": callback_id, "text": "권한이 없습니다."})
            return
        action, separator, token = data.partition(":")
        if not separator or action not in {"approve_upload", "reject", "more_source", "end_session"}:
            self.api("answerCallbackQuery", {"callback_query_id": callback_id, "text": "알 수 없는 버튼입니다."})
            return
        if action == "more_source":
            self.api("answerCallbackQuery", {"callback_query_id": callback_id, "text": "다음 롱폼 제작을 시작합니다."})
            self.start_job(chat_id, "오늘 쇼츠 제작", "daily-run", "--execute")
            return
        if action == "end_session":
            self.api("answerCallbackQuery", {"callback_query_id": callback_id, "text": "이번 회차를 종료했습니다."})
            ended = self.state.setdefault("ended_source_sessions", {})
            if not isinstance(ended, dict):
                ended = {}
                self.state["ended_source_sessions"] = ended
            ended[token] = datetime.now(timezone.utc).isoformat()
            self.save_state()
            try:
                self.api(
                    "editMessageReplyMarkup",
                    {
                        "chat_id": chat_id,
                        "message_id": int(message.get("message_id") or 0),
                        "reply_markup": {"inline_keyboard": []},
                    },
                )
            except Exception:
                pass
            self.send(
                chat_id,
                "이번 회차의 다음 롱폼 후보는 폐기했어요. 이미 승인·업로드·거부한 쇼츠는 그대로 유지됩니다.",
            )
            return
        if self.has_active_job():
            self.api("answerCallbackQuery", {"callback_query_id": callback_id, "text": "현재 작업이 끝난 뒤 다시 눌러주세요."})
            return
        row = self.find_review_by_token(token)
        if not row or str(row.get("review_status") or "") != "needs_review":
            self.api("answerCallbackQuery", {"callback_query_id": callback_id, "text": "이미 처리됐거나 찾을 수 없는 후보입니다."})
            return
        status = "approved" if action == "approve_upload" else "rejected"
        self.api("answerCallbackQuery", {"callback_query_id": callback_id, "text": "처리 중입니다."})
        code, output = self.run_orchestrator(
            "review-decision",
            "--output",
            str(row.get("output_path") or ""),
            "--status",
            status,
            "--note",
            "telegram inline decision",
        )
        if code != 0:
            self.send(chat_id, f"처리 실패\n{output}")
            return
        try:
            self.api(
                "editMessageReplyMarkup",
                {
                    "chat_id": chat_id,
                    "message_id": int(message.get("message_id") or 0),
                    "reply_markup": {"inline_keyboard": []},
                },
            )
        except Exception:
            pass
        short_id = str(row.get("short_id") or "후보")
        if action == "reject":
            self.send(chat_id, f"🗑 {short_id} 거부 처리했습니다.")
            self.offer_next_source(chat_id, str(row.get("source_key") or ""))
            return
        self.send(chat_id, f"✅ {short_id} 승인 완료. 지금 바로 업로드/예약을 시작합니다.")
        self.start_job(
            chat_id,
            f"{short_id} 업로드/예약",
            "upload-approved",
            "--execute",
            "--limit",
            "1",
            "--output",
            str(row.get("output_path") or ""),
            followup_source_key=str(row.get("source_key") or ""),
        )

    def process_update(self, update: dict[str, Any]) -> None:
        callback = update.get("callback_query") if isinstance(update.get("callback_query"), dict) else None
        if callback is not None:
            try:
                self.handle_callback(callback)
            except Exception as exc:
                callback_id = str(callback.get("id") or "")
                if callback_id:
                    try:
                        self.api("answerCallbackQuery", {"callback_query_id": callback_id, "text": f"처리 실패: {exc}"})
                    except Exception:
                        pass
            return
        message = update.get("message") if isinstance(update.get("message"), dict) else {}
        chat = message.get("chat") if isinstance(message.get("chat"), dict) else {}
        chat_id = chat.get("id")
        text = str(message.get("text") or "").strip()
        if not isinstance(chat_id, int) or not text:
            return
        if not self.allowed_chat_ids:
            self.send(
                chat_id,
                f"초기 설정 중입니다. 이 chat ID는 {chat_id} 입니다. .env의 TELEGRAM_ALLOWED_CHAT_IDS에 이 숫자를 넣고 봇을 재시작하세요.",
            )
            return
        if chat_id not in self.allowed_chat_ids:
            return
        try:
            if text.startswith("/"):
                self.handle_command(chat_id, text)
            else:
                self.handle_conversation(chat_id, text)
        except Exception as exc:
            self.send(chat_id, f"❌ 명령 처리 실패: {exc}")

    def run_forever(self) -> None:
        print("[telegram] bot started; use Ctrl+C to stop", flush=True)
        while True:
            try:
                response = self.api("getUpdates", {"offset": self.offset, "timeout": POLL_TIMEOUT_SECONDS})
                for update in response.get("result", []):
                    update_id = int(update.get("update_id", 0))
                    self.offset = max(self.offset, update_id + 1)
                    self.save_state()
                    self.process_update(update)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"[telegram] polling error: {exc}", flush=True)
                time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the private Long2Shorts Telegram control bot.")
    parser.add_argument("--check", action="store_true", help="Validate local Telegram configuration without calling Telegram.")
    args = parser.parse_args()
    load_project_env()
    token = str(os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    allowed_chat_ids = parse_chat_ids(str(os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS") or ""))
    if args.check:
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is missing from .env.")
        print(f"Telegram configuration OK; allowed chat IDs: {len(allowed_chat_ids)}")
        return
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing from .env. See docs/TELEGRAM_BOT.md.")
    Long2ShortsTelegramBot(token, allowed_chat_ids).run_forever()


if __name__ == "__main__":
    main()
