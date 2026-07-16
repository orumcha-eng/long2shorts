import json
import os
import queue
import re
import subprocess
import sys
import threading
from io import BytesIO
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from urllib.request import Request, urlopen

from PIL import Image, ImageTk

from discover_trending_sources import mark_processed_source


BASE_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
REFERENCE_CHANNELS_PATH = BASE_DIR / "reference_channels.json"
ORCHESTRATOR_CONFIG_PATH = BASE_DIR / "automation_config.json"
BENCHMARK_PROFILE_OPTIONS: dict[str, Path | None] = {
    "Korean variety (RESCENE)": BASE_DIR / "templates" / "benchmark_profiles" / "rescene_gyaru_variety.json",
    "No benchmark": None,
}

PROCESS_STAGES = [
    ("trend", "0. 트렌드 후보 탐색", "최근 3일 예능/연예 롱폼 후보를 찾고 자동 처리할 원본을 고릅니다."),
    ("link", "1. 링크/메타데이터", "YouTube URL, 제목, 길이, 댓글 가능 여부를 확인합니다."),
    ("comments", "2. 댓글 반응 분석", "타임스탬프 댓글, 좋아요, 답글, 감정 키워드를 수집합니다."),
    ("media", "3. 영상 파일 확보", "yt-dlp로 YouTube 영상을 다운로드합니다."),
    ("transcript", "4. 오디오/STT 전사", "음성을 자막 타임라인으로 바꿉니다."),
    ("candidates", "5. AI 후보 선정", "대본과 댓글 신호로 쇼츠 후보를 고릅니다."),
    ("output", "6. 미리보기/CapCut", "선택 후보를 실제 영상 또는 편집 패키지로 만듭니다."),
]
YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/|youtube_)([A-Za-z0-9_-]{11})")
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm"}
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".aac", ".wav", ".flac", ".ogg", ".opus"}


def get_python_exe() -> Path:
    venv_python = BASE_DIR / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def analysis_dir_for_video(video_path: Path) -> Path:
    video_id = extract_youtube_id(video_path)
    if video_id:
        return BASE_DIR / "analysis" / f"youtube_{video_id}"
    return BASE_DIR / "analysis" / video_path.stem


def movie_info_path_for_video(video_path: Path) -> Path:
    return analysis_dir_for_video(video_path) / "movie_info.json"


def packages_json_path(video_path: Path) -> Path:
    return analysis_dir_for_video(video_path) / "shorts_candidates" / "final" / "shorts_packages.json"


def youtube_context_path_for_video(video_path: Path) -> Path:
    return analysis_dir_for_video(video_path) / "youtube_context.json"


def youtube_analysis_dir_for_id(video_id: str) -> Path:
    return BASE_DIR / "analysis" / f"youtube_{video_id}"


def youtube_context_path_for_id(video_id: str) -> Path:
    return youtube_analysis_dir_for_id(video_id) / "youtube_context.json"


def packages_json_path_for_youtube_id(video_id: str) -> Path:
    return youtube_analysis_dir_for_id(video_id) / "shorts_candidates" / "final" / "shorts_packages.json"


def extract_youtube_id(value: str | Path | None) -> str | None:
    if not value:
        return None
    match = YOUTUBE_ID_RE.search(str(value))
    return match.group(1) if match else None


def path_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def companion_video_path(path: Path) -> Path | None:
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return path

    video_id = extract_youtube_id(path)
    if not video_id:
        return None

    expected = [
        path.parent / f"youtube_{video_id}_video.mp4",
        path.parent / f"youtube_{video_id}.mp4",
    ]
    for candidate in expected:
        if candidate.exists():
            return candidate

    matches: list[Path] = []
    for pattern in [f"youtube_{video_id}_video.*", f"youtube_{video_id}.*"]:
        for candidate in path.parent.glob(pattern):
            if candidate == path or "_audio." in candidate.name.lower():
                continue
            if candidate.suffix.lower() in VIDEO_EXTENSIONS:
                matches.append(candidate)
    if not matches:
        return None
    return sorted(set(matches), key=lambda item: (-path_size(item), item.name))[0]


def package_file_path(video_path: Path, short_id: str) -> Path:
    return analysis_dir_for_video(video_path) / "shorts_candidates" / "final" / f"{short_id}.json"


def preview_file_path(video_path: Path, short_id: str) -> Path:
    return analysis_dir_for_video(video_path) / "previews" / f"{short_id}_preview.mp4"


def runtime_package_path(video_path: Path, short_id: str) -> Path:
    return analysis_dir_for_video(video_path) / "runtime_packages" / f"{short_id}.json"


def safe_list(value, fallback=None):
    if isinstance(value, list) and value:
        return value
    return fallback or []


def first_nonempty_string(*values) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def format_seconds(value) -> str:
    try:
        seconds = int(float(value))
    except Exception:
        return "-"
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes}:{sec:02d}"


def configure_tcl_library_paths() -> None:
    candidates = [
        Path(sys.base_prefix) / "tcl",
        Path(sys.prefix) / "tcl",
        Path(r"C:\Users\user\AppData\Local\Programs\Python\Python313\tcl"),
        Path(r"C:\Users\user\AppData\Local\Programs\Python\Python311\tcl"),
    ]
    for root in candidates:
        tcl_dir = root / "tcl8.6"
        tk_dir = root / "tk8.6"
        if tcl_dir.exists() and "TCL_LIBRARY" not in os.environ:
            os.environ["TCL_LIBRARY"] = str(tcl_dir)
        if tk_dir.exists() and "TK_LIBRARY" not in os.environ:
            os.environ["TK_LIBRARY"] = str(tk_dir)
        if "TCL_LIBRARY" in os.environ and "TK_LIBRARY" in os.environ:
            return


class Long2ShortsApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Long2Shorts")
        self.root.geometry("1480x980")

        self.python_exe = get_python_exe()
        self.log_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.current_process: subprocess.Popen | None = None
        self.cancel_requested = False

        self.movie_info: dict | None = None
        self.youtube_context: dict | None = None
        self.youtube_context_path: Path | None = None
        self.packages: list[dict] = []
        self.action_buttons: list[ttk.Button] = []
        self.reference_channels: list[dict] = []
        self.reference_video_by_item: dict[str, dict] = {}

        self.video_path_var = tk.StringVar()
        self.youtube_url_var = tk.StringVar()
        self.title_var = tk.StringVar()
        self.analysis_dir_var = tk.StringVar()
        self.reference_channel_var = tk.StringVar()
        self.benchmark_profile_var = tk.StringVar(value="Korean variety (RESCENE)")
        self.reference_status_var = tk.StringVar(value="참고 채널을 추가하면 최신 영상이 표시됩니다.")
        self.force_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="대기 중")
        self.phase_var = tk.StringVar(value="아직 작업이 없습니다.")
        self.stage_vars: dict[str, tk.StringVar] = {}
        self.stage_note_vars: dict[str, tk.StringVar] = {}

        self.package_tree: ttk.Treeview | None = None
        self.movie_info_text: tk.Text | None = None
        self.youtube_context_text: tk.Text | None = None
        self.detail_text: tk.Text | None = None
        self.log_text: tk.Text | None = None
        self.progressbar: ttk.Progressbar | None = None
        self.stop_button: ttk.Button | None = None
        self.reference_tree: ttk.Treeview | None = None
        self.reference_thumbnail_label: ttk.Label | None = None
        self.reference_title_label: ttk.Label | None = None
        self.reference_thumbnail_image: ImageTk.PhotoImage | None = None
        self.max_auto_capcut_var = tk.IntVar(value=10)

        self._build_ui()
        self.load_reference_channels()
        self.root.after(120, self._poll_log_queue)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        source_frame = ttk.LabelFrame(outer, text="YouTube 소스", padding=10)
        source_frame.pack(fill="x")

        ttk.Label(source_frame, text="YouTube 링크").grid(row=0, column=0, sticky="w")
        ttk.Entry(source_frame, textvariable=self.youtube_url_var).grid(
            row=0, column=1, sticky="ew", padx=8
        )
        url_actions = ttk.Frame(source_frame)
        url_actions.grid(row=0, column=2, sticky="ew")
        self._make_button(url_actions, "내 롱폼", self.on_choose_owned_source_video).pack(side="left")
        self._make_button(url_actions, "라이브러리 등록", self.on_register_owned_source).pack(side="left", padx=(6, 0))
        self._make_button(url_actions, "댓글/반응 분석", self.on_collect_youtube_context).pack(side="left")
        self._make_button(url_actions, "다운로드+분석", self.on_download_youtube_video).pack(side="left", padx=(6, 0))

        ttk.Label(source_frame, text="분석 폴더").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(source_frame, textvariable=self.analysis_dir_var, state="readonly").grid(
            row=1, column=1, sticky="ew", padx=8, pady=(8, 0)
        )
        ttk.Checkbutton(source_frame, text="강제 재생성", variable=self.force_var).grid(
            row=1, column=2, sticky="w", pady=(8, 0)
        )
        source_frame.columnconfigure(1, weight=1)

        process_frame = ttk.LabelFrame(outer, text="전체 프로세싱", padding=10)
        process_frame.pack(fill="x", pady=(10, 0))
        self._build_process_board(process_frame)

        action_frame = ttk.Frame(outer, padding=(0, 10, 0, 10))
        action_frame.pack(fill="x")
        self._make_button(action_frame, "오늘 쇼츠 제작", self.on_daily_shorts_run).pack(side="left")
        self._make_button(action_frame, "트렌드 자동 생성", self.on_auto_trend_generate).pack(side="left")
        ttk.Label(action_frame, text="원본당 최대").pack(side="left", padx=(8, 2))
        ttk.Spinbox(action_frame, from_=1, to=10, width=3, textvariable=self.max_auto_capcut_var).pack(side="left")
        ttk.Label(action_frame, text="Benchmark").pack(side="left", padx=(8, 2))
        ttk.Combobox(
            action_frame,
            textvariable=self.benchmark_profile_var,
            values=list(BENCHMARK_PROFILE_OPTIONS),
            state="readonly",
            width=24,
        ).pack(side="left")
        ttk.Label(action_frame, text="개").pack(side="left", padx=(2, 10))
        self._make_button(action_frame, "쇼츠 시놉시스 생성", self.on_generate_packages).pack(side="left")
        self._make_button(action_frame, "목록 새로고침", self.on_refresh_packages).pack(side="left", padx=8)
        self._make_button(action_frame, "미리보기 열기", self.on_open_preview).pack(side="left")
        self._make_button(action_frame, "선택 항목 CapCut 만들기", self.on_create_capcut).pack(side="left", padx=(8, 0))
        self.stop_button = ttk.Button(action_frame, text="작업 중지", command=self.on_cancel_worker, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))

        status_frame = ttk.Frame(action_frame)
        status_frame.pack(side="right", fill="x", expand=True)
        ttk.Label(status_frame, textvariable=self.phase_var).pack(side="top", anchor="e")
        row = ttk.Frame(status_frame)
        row.pack(side="top", fill="x")
        ttk.Label(row, textvariable=self.status_var).pack(side="right")
        self.progressbar = ttk.Progressbar(row, mode="indeterminate", length=220)
        self.progressbar.pack(side="right", padx=(0, 8))

        top_pane = ttk.PanedWindow(outer, orient="horizontal")
        top_pane.pack(fill="both", expand=True)

        left = ttk.Frame(top_pane, padding=(0, 0, 8, 0))
        right = ttk.Frame(top_pane)
        top_pane.add(left, weight=3)
        top_pane.add(right, weight=2)

        candidate_frame = ttk.LabelFrame(left, text="쇼츠 후보", padding=10)
        candidate_frame.pack(fill="both", expand=True)

        columns = ("rank", "score", "decision", "title", "reaction", "tags", "clarity", "protagonist")
        self.package_tree = ttk.Treeview(candidate_frame, columns=columns, show="headings", height=14)
        self.package_tree.heading("score", text="Score")
        self.package_tree.heading("decision", text="Decision")
        self.package_tree.heading("rank", text="순위")
        self.package_tree.heading("title", text="제목")
        self.package_tree.heading("reaction", text="댓글 신호")
        self.package_tree.heading("tags", text="재미 태그")
        self.package_tree.heading("clarity", text="독립 이해도")
        self.package_tree.heading("protagonist", text="주인공")
        self.package_tree.column("rank", width=55, anchor="center")
        self.package_tree.column("score", width=65, anchor="center")
        self.package_tree.column("decision", width=90, anchor="center")
        self.package_tree.column("title", width=360)
        self.package_tree.column("reaction", width=120, anchor="center")
        self.package_tree.column("tags", width=190)
        self.package_tree.column("clarity", width=90, anchor="center")
        self.package_tree.column("protagonist", width=90, anchor="center")
        self.package_tree.pack(fill="both", expand=True)
        self.package_tree.bind("<<TreeviewSelect>>", self.on_package_selected)

        context_tabs = ttk.Notebook(right)
        context_tabs.pack(fill="both", expand=True)

        reference_frame = ttk.Frame(context_tabs, padding=10)
        comment_frame = ttk.Frame(context_tabs, padding=10)
        context_tabs.add(reference_frame, text="참고 채널")
        context_tabs.add(comment_frame, text="댓글 반응")

        self.build_reference_channel_tab(reference_frame)
        self.youtube_context_text = tk.Text(comment_frame, height=18, wrap="word")
        self.youtube_context_text.pack(fill="both", expand=True)

        bottom_pane = ttk.PanedWindow(outer, orient="horizontal")
        bottom_pane.pack(fill="both", expand=True, pady=(10, 0))

        detail_frame = ttk.LabelFrame(bottom_pane, text="선택 후보 상세", padding=10)
        log_frame = ttk.LabelFrame(bottom_pane, text="로그", padding=10)
        bottom_pane.add(detail_frame, weight=3)
        bottom_pane.add(log_frame, weight=2)

        self.detail_text = tk.Text(detail_frame, height=18, wrap="word")
        self.detail_text.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, height=18, wrap="word")
        self.log_text.pack(fill="both", expand=True)

    def build_reference_channel_tab(self, parent: ttk.Frame) -> None:
        input_row = ttk.Frame(parent)
        input_row.pack(fill="x")
        ttk.Label(input_row, text="채널 URL/핸들").pack(side="left")
        ttk.Entry(input_row, textvariable=self.reference_channel_var).pack(side="left", fill="x", expand=True, padx=8)
        self._make_button(input_row, "채널 추가", self.on_add_reference_channel).pack(side="left")
        self._make_button(input_row, "목록 새로고침", self.on_refresh_reference_channels).pack(side="left", padx=(6, 0))

        ttk.Label(parent, textvariable=self.reference_status_var).pack(anchor="w", pady=(8, 6))

        columns = ("channel", "title")
        self.reference_tree = ttk.Treeview(parent, columns=columns, show="headings", height=7)
        self.reference_tree.heading("channel", text="채널")
        self.reference_tree.heading("title", text="최신 영상 제목")
        self.reference_tree.column("channel", width=150)
        self.reference_tree.column("title", width=360)
        self.reference_tree.pack(fill="x")
        self.reference_tree.bind("<<TreeviewSelect>>", self.on_reference_video_selected)

        preview = ttk.Frame(parent)
        preview.pack(fill="both", expand=True, pady=(10, 0))
        self.reference_thumbnail_label = ttk.Label(preview, text="썸네일")
        self.reference_thumbnail_label.pack(anchor="center")
        self.reference_title_label = ttk.Label(preview, text="", wraplength=420, justify="center")
        self.reference_title_label.pack(fill="x", pady=(8, 0))

        action_row = ttk.Frame(parent)
        action_row.pack(fill="x", pady=(10, 0))
        self._make_button(action_row, "선택 링크 사용", self.on_use_reference_video).pack(side="left")
        self._make_button(action_row, "선택 후 다운로드+분석", self.on_use_reference_video_and_download).pack(side="left", padx=(6, 0))

    def _build_process_board(self, parent: ttk.Frame) -> None:
        for col, label in enumerate(["단계", "상태", "무엇을 보는가"]):
            ttk.Label(parent, text=label).grid(row=0, column=col, sticky="w", padx=(0, 10), pady=(0, 6))

        for row, (key, label, note) in enumerate(PROCESS_STAGES, start=1):
            status_var = tk.StringVar(value="대기")
            note_var = tk.StringVar(value=note)
            self.stage_vars[key] = status_var
            self.stage_note_vars[key] = note_var
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=2)
            ttk.Label(parent, textvariable=status_var, width=12).grid(row=row, column=1, sticky="w", padx=(0, 10), pady=2)
            ttk.Label(parent, textvariable=note_var).grid(row=row, column=2, sticky="ew", pady=2)
        parent.columnconfigure(2, weight=1)

    def _make_button(self, parent, text: str, command) -> ttk.Button:
        button = ttk.Button(parent, text=text, command=command)
        self.action_buttons.append(button)
        return button

    def set_busy(self, is_busy: bool, phase: str | None = None) -> None:
        state = "disabled" if is_busy else "normal"
        for button in self.action_buttons:
            button.configure(state=state)
        if self.stop_button:
            self.stop_button.configure(state="normal" if is_busy else "disabled")
        if self.progressbar:
            if is_busy:
                self.progressbar.start(10)
            else:
                self.progressbar.stop()
        if phase:
            self.phase_var.set(phase)
        elif not is_busy:
            self.phase_var.set("아직 작업이 없습니다.")

    def append_text(self, widget: tk.Text | None, text: str) -> None:
        if not widget:
            return
        widget.delete("1.0", "end")
        widget.insert("1.0", text)

    def append_log(self, text: str) -> None:
        if not self.log_text:
            return
        self.log_text.insert("end", text.rstrip() + "\n")
        self.log_text.see("end")

    def clear_log(self) -> None:
        if self.log_text:
            self.log_text.delete("1.0", "end")

    def set_stage_status(self, key: str, status: str, note: str | None = None) -> None:
        if key in self.stage_vars:
            self.stage_vars[key].set(status)
        if note is not None and key in self.stage_note_vars:
            self.stage_note_vars[key].set(note)

    def reset_process_board(self) -> None:
        for key, _label, note in PROCESS_STAGES:
            self.set_stage_status(key, "대기", note)

    def load_reference_channels(self) -> None:
        try:
            data = json.loads(REFERENCE_CHANNELS_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = []
        self.reference_channels = data if isinstance(data, list) else []
        self.refresh_reference_tree()

    def save_reference_channels(self) -> None:
        REFERENCE_CHANNELS_PATH.write_text(
            json.dumps(self.reference_channels, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def refresh_reference_tree(self) -> None:
        if not self.reference_tree:
            return
        self.reference_video_by_item.clear()
        for item in self.reference_tree.get_children():
            self.reference_tree.delete(item)
        for idx, video in enumerate(self.reference_channels):
            item_id = str(idx)
            self.reference_video_by_item[item_id] = video
            self.reference_tree.insert(
                "",
                "end",
                iid=item_id,
                values=(video.get("channel_title", ""), video.get("title", "")),
            )
        count = len(self.reference_channels)
        self.reference_status_var.set(f"참고 채널 최신 영상 {count}개" if count else "참고 채널을 추가하면 최신 영상이 표시됩니다.")

    def normalize_reference_channel_url(self, raw: str) -> str:
        value = raw.strip()
        if not value:
            raise ValueError("채널 URL 또는 @핸들을 입력해주세요.")
        if value.startswith("@"):
            return f"https://www.youtube.com/{value}/videos"
        if value.startswith("UC") and "/" not in value:
            return f"https://www.youtube.com/channel/{value}/videos"
        if not value.startswith(("http://", "https://")):
            if value.startswith("youtube.com/") or value.startswith("www.youtube.com/"):
                value = "https://" + value
            else:
                value = f"https://www.youtube.com/@{value.lstrip('@')}/videos"
        if "youtube.com" in value and not any(part in value for part in ["/videos", "/shorts", "/streams"]):
            value = value.rstrip("/") + "/videos"
        return value

    def fetch_latest_reference_video(self, raw_channel: str) -> dict:
        channel_url = self.normalize_reference_channel_url(raw_channel)
        command = [
            str(self.python_exe),
            "-m",
            "yt_dlp",
            "--dump-single-json",
            "--flat-playlist",
            "--playlist-end",
            "1",
            "--no-warnings",
            channel_url,
        ]
        self.log_queue.put(("log", "$ " + " ".join(command)))
        process = subprocess.Popen(
            command,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.current_process = process
        try:
            stdout, stderr = process.communicate()
        finally:
            if self.current_process is process:
                self.current_process = None

        if self.cancel_requested:
            raise RuntimeError("작업이 중지되었습니다.")
        if process.returncode != 0:
            raise RuntimeError((stderr or stdout or "최신 영상을 가져오지 못했습니다.").strip())

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            start = stdout.find("{")
            end = stdout.rfind("}")
            if start < 0 or end < start:
                raise RuntimeError("yt-dlp 결과를 JSON으로 읽지 못했습니다.")
            data = json.loads(stdout[start : end + 1])

        entries = [entry for entry in data.get("entries", []) or [] if isinstance(entry, dict)]
        if not entries:
            raise RuntimeError("이 채널에서 공개 최신 영상을 찾지 못했습니다.")
        entry = entries[0]
        video_id = first_nonempty_string(entry.get("id"), entry.get("url"))
        video_url = first_nonempty_string(entry.get("webpage_url"), entry.get("url"))
        if video_id and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            video_url = f"https://www.youtube.com/watch?v={video_id}"
        elif video_url and video_url.startswith("/"):
            video_url = "https://www.youtube.com" + video_url

        thumbnail = first_nonempty_string(entry.get("thumbnail"))
        if not thumbnail and video_id and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            thumbnail = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

        return {
            "input": raw_channel.strip(),
            "channel_url": channel_url,
            "channel_title": first_nonempty_string(
                entry.get("channel"),
                entry.get("uploader"),
                data.get("channel"),
                data.get("uploader"),
                data.get("title"),
            ),
            "title": first_nonempty_string(entry.get("title"), "제목 없음"),
            "video_id": video_id,
            "video_url": video_url,
            "thumbnail": thumbnail,
        }

    def upsert_reference_video(self, video: dict) -> None:
        key = first_nonempty_string(video.get("channel_url"), video.get("input"))
        updated = False
        for idx, existing in enumerate(self.reference_channels):
            existing_key = first_nonempty_string(existing.get("channel_url"), existing.get("input"))
            if existing_key == key:
                self.reference_channels[idx] = video
                updated = True
                break
        if not updated:
            self.reference_channels.append(video)
        self.save_reference_channels()
        self.refresh_reference_tree()

    def on_add_reference_channel(self) -> None:
        raw = self.reference_channel_var.get().strip()
        if not raw:
            messagebox.showerror("오류", "참고 채널 URL 또는 @핸들을 입력해주세요.")
            return

        def task() -> None:
            self.log_queue.put(("status", "참고 채널 최신 영상 확인 중"))
            self.log_queue.put(("phase", "참고 채널 최신 영상 확인 중"))
            try:
                video = self.fetch_latest_reference_video(raw)
                self.log_queue.put(("reference_channel_added", video))
                self.log_queue.put(("status", "대기 중"))
            except Exception as exc:
                self.log_queue.put(("error", f"참고 채널 추가 실패: {exc}"))
                self.log_queue.put(("status", "대기 중"))

        self.start_worker(task, "참고 채널 최신 영상 확인 중")

    def on_refresh_reference_channels(self) -> None:
        if not self.reference_channels:
            self.reference_status_var.set("새로고침할 참고 채널이 없습니다.")
            return
        channels = [first_nonempty_string(item.get("input"), item.get("channel_url")) for item in self.reference_channels]

        def task() -> None:
            refreshed = []
            self.log_queue.put(("status", "참고 채널 새로고침 중"))
            try:
                for channel in channels:
                    if not channel:
                        continue
                    refreshed.append(self.fetch_latest_reference_video(channel))
                self.log_queue.put(("reference_channels_refreshed", refreshed))
                self.log_queue.put(("status", "대기 중"))
            except Exception as exc:
                self.log_queue.put(("error", f"참고 채널 새로고침 실패: {exc}"))
                self.log_queue.put(("status", "대기 중"))

        self.start_worker(task, "참고 채널 새로고침 중")

    def selected_reference_video(self) -> dict | None:
        if not self.reference_tree:
            return None
        selection = self.reference_tree.selection()
        if not selection:
            return None
        return self.reference_video_by_item.get(selection[0])

    def on_reference_video_selected(self, _event=None) -> None:
        video = self.selected_reference_video()
        if not video:
            return
        title = video.get("title", "")
        channel = video.get("channel_title", "")
        self.reference_title_label.configure(text=f"{title}\n{channel}") if self.reference_title_label else None
        self.reference_status_var.set(video.get("video_url", ""))
        self.load_reference_thumbnail(video.get("thumbnail", ""))

    def load_reference_thumbnail(self, thumbnail_url: str) -> None:
        if not self.reference_thumbnail_label:
            return
        if not thumbnail_url:
            self.reference_thumbnail_label.configure(text="썸네일 없음", image="")
            self.reference_thumbnail_image = None
            return
        try:
            request = Request(thumbnail_url, headers={"User-Agent": "Long2Shorts/0.1"})
            with urlopen(request, timeout=10) as response:
                image = Image.open(BytesIO(response.read())).convert("RGB")
            image.thumbnail((360, 202))
            self.reference_thumbnail_image = ImageTk.PhotoImage(image)
            self.reference_thumbnail_label.configure(image=self.reference_thumbnail_image, text="")
        except Exception:
            self.reference_thumbnail_label.configure(text="썸네일을 불러오지 못했습니다.", image="")
            self.reference_thumbnail_image = None

    def apply_reference_video_to_input(self) -> bool:
        video = self.selected_reference_video()
        if not video:
            messagebox.showerror("오류", "사용할 참고 영상을 먼저 선택해주세요.")
            return False
        video_url = video.get("video_url", "")
        if not video_url:
            messagebox.showerror("오류", "선택한 영상의 URL이 없습니다.")
            return False
        self.youtube_url_var.set(video_url)
        self.title_var.set(video.get("title", ""))
        video_id = extract_youtube_id(video_url)
        if video_id:
            self.analysis_dir_var.set(str(youtube_analysis_dir_for_id(video_id)))
        self.append_log(f"[gui] 참고 영상 선택: {video.get('title', '')} -> {video_url}")
        return True

    def on_use_reference_video(self) -> None:
        self.apply_reference_video_to_input()

    def on_use_reference_video_and_download(self) -> None:
        if self.apply_reference_video_to_input():
            self.on_download_youtube_video()

    def current_youtube_video_id(self, video_path: Path | None = None) -> str | None:
        path_id = extract_youtube_id(video_path)
        if path_id:
            return path_id
        url_id = extract_youtube_id(self.get_youtube_url())
        if url_id:
            return url_id
        if self.youtube_context:
            video_id = self.youtube_context.get("video_id")
            if isinstance(video_id, str) and video_id.strip():
                return video_id.strip()
        return (
            extract_youtube_id(self.analysis_dir_var.get().strip())
            or extract_youtube_id(self.youtube_context_path)
        )

    def resolve_youtube_context_path(self, video_path: Path | None = None) -> Path | None:
        candidates: list[Path] = []
        if video_path:
            candidates.append(youtube_context_path_for_video(video_path))
        video_id = self.current_youtube_video_id(video_path)
        if video_id:
            candidates.append(youtube_context_path_for_id(video_id))
        if self.youtube_context_path:
            candidates.append(self.youtube_context_path)
        analysis_dir = self.analysis_dir_var.get().strip()
        if analysis_dir:
            candidates.append(Path(analysis_dir) / "youtube_context.json")

        seen: set[Path] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            if resolved in seen:
                continue
            seen.add(resolved)
            if candidate.exists():
                return candidate
        return None

    def load_youtube_context_path(self, context_path: Path) -> bool:
        try:
            self.youtube_context = json.loads(context_path.read_text(encoding="utf-8"))
            self.youtube_context_path = context_path
            return True
        except Exception as exc:
            self.append_log(f"[gui] 댓글 반응 파일 로드 실패: {exc}")
            return False

    def load_youtube_context_for_video(self, video_path: Path | None = None) -> None:
        context_path = self.resolve_youtube_context_path(video_path)
        if context_path and self.load_youtube_context_path(context_path):
            self.render_youtube_context()
            return
        if not self.youtube_context:
            self.youtube_context_path = None
        self.render_youtube_context()

    def resolve_packages_json_path(self, video_path: Path) -> Path | None:
        candidates = [packages_json_path(video_path)]
        video_id = self.current_youtube_video_id(video_path)
        if video_id:
            candidates.append(packages_json_path_for_youtube_id(video_id))
        package_output = ((self.youtube_context or {}).get("package_generation") or {}).get("output")
        if package_output:
            candidates.append(Path(package_output))
        if self.youtube_context_path:
            candidates.append(self.youtube_context_path.parent / "shorts_candidates" / "final" / "shorts_packages.json")

        seen: set[Path] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            if resolved in seen:
                continue
            seen.add(resolved)
            if candidate.exists():
                return candidate
        return None

    def render_youtube_context(self) -> None:
        if not self.youtube_context:
            self.append_text(
                self.youtube_context_text,
                "댓글 반응 데이터가 아직 없습니다.\n\n"
                "- YouTube 링크를 입력하고 `댓글/반응 분석`을 누르면 댓글 타임스탬프와 반응 키워드를 모읍니다.\n"
                "- API 키가 없으면 댓글 수집은 건너뛰고, GUI에는 그 상태가 표시됩니다.",
            )
            return

        metadata = self.youtube_context.get("metadata", {}) or {}
        comments = self.youtube_context.get("comments", {}) or {}
        transcript_prep = self.youtube_context.get("transcript_preparation", {}) or {}
        package_gen = self.youtube_context.get("package_generation", {}) or {}
        insights = self.youtube_context.get("comment_insights", {}) or {}
        moments = insights.get("timecode_moments", []) or []
        top_comments = insights.get("top_reaction_comments", []) or []
        keyword_counts = insights.get("keyword_counts", {}) or {}

        lines = [
            f"링크: {self.youtube_context.get('source_url', '')}",
            f"영상 ID: {self.youtube_context.get('video_id', '')}",
            f"제목: {metadata.get('title', '')}",
            f"채널: {metadata.get('channel_title', '')}",
            f"조회수/좋아요/댓글: {metadata.get('view_count', 0)} / {metadata.get('like_count', 0)} / {metadata.get('comment_count', 0)}",
            f"수집 댓글: {comments.get('fetched_count', 0)}",
            f"전사 준비: {transcript_prep.get('status', 'unknown')} ({transcript_prep.get('source', '-')})",
            f"후보 생성: {package_gen.get('status', 'unknown')}",
        ]
        if comments.get("error"):
            lines.append(f"댓글 수집 메모: {comments.get('error')}")
        if transcript_prep.get("merged_transcript"):
            lines.append(f"전사 파일: {transcript_prep.get('merged_transcript')}")
        if transcript_prep.get("error"):
            lines.append(f"전사 준비 메모: {transcript_prep.get('error')}")
        if package_gen.get("output"):
            lines.append(f"후보 파일: {package_gen.get('output')}")
        if package_gen.get("error"):
            lines.append(f"후보 생성 메모: {package_gen.get('error')}")
        lines.extend(["", "타임스탬프 반응 TOP:"])
        if moments:
            for item in moments[:8]:
                time_label = format_seconds(item.get("representative_time_sec", item.get("bucket_start_sec")))
                sample = ""
                samples = item.get("samples", []) or []
                if samples:
                    sample = samples[0].get("text", "")
                lines.append(
                    f"- {time_label} | 점수 {item.get('reaction_score', 0)} | 댓글 {item.get('comment_count', 0)} | {sample}"
                )
        else:
            lines.append("- 타임스탬프 댓글이 아직 없습니다.")

        lines.extend(["", "반응 키워드:"])
        if keyword_counts:
            lines.append(", ".join(f"{key}({value})" for key, value in list(keyword_counts.items())[:16]))
        else:
            lines.append("- 없음")

        lines.extend(["", "좋은 반응 댓글 TOP:"])
        if top_comments:
            for item in top_comments[:6]:
                timecodes = ", ".join(format_seconds(sec) for sec in item.get("timecodes", [])[:3])
                time_part = f" [{timecodes}]" if timecodes else ""
                lines.append(
                    f"- 점수 {item.get('reaction_score', 0)} 좋아요 {item.get('like_count', 0)}{time_part}: {item.get('text', '')}"
                )
        else:
            lines.append("- 없음")

        self.append_text(self.youtube_context_text, "\n".join(lines).strip())

    def set_video(self, path: Path) -> None:
        replacement = companion_video_path(path)
        if replacement and replacement != path:
            self.append_log(f"[gui] 오디오 파일 대신 영상 파일로 전환: {replacement.name}")
            path = replacement
        self.video_path_var.set(str(path))
        self.title_var.set(path.stem)
        self.analysis_dir_var.set(str(analysis_dir_for_video(path)))
        self.movie_info = None
        saved_movie_info = movie_info_path_for_video(path)
        if saved_movie_info.exists():
            try:
                self.movie_info = json.loads(saved_movie_info.read_text(encoding="utf-8"))
            except Exception:
                self.movie_info = None
        self.load_youtube_context_for_video(path)
        self.render_movie_info()
        self.packages = []
        self.clear_package_tree()
        self.append_text(self.detail_text, self.describe_cache_status(path))
        self.set_stage_status("media", "완료", f"영상 파일 준비됨: {path.name}")
        if self.youtube_context:
            fetched = (self.youtube_context.get("comments") or {}).get("fetched_count", 0)
            moments = ((self.youtube_context.get("comment_insights") or {}).get("timecode_moments") or [])
            metadata = self.youtube_context.get("metadata", {}) or {}
            if metadata.get("title"):
                self.title_var.set(metadata.get("title", path.stem))
            self.set_stage_status("link", "완료", "YouTube 컨텍스트 파일을 불러왔습니다.")
            self.set_stage_status("comments", "완료", f"댓글 {fetched}개, 타임스탬프 반응 {len(moments)}개")

    def on_choose_owned_source_video(self) -> None:
        selected = filedialog.askopenfilename(
            title="내 롱폼 원본 선택",
            filetypes=[("Video files", "*.mp4 *.mkv *.mov *.webm"), ("All files", "*.*")],
        )
        if selected:
            self.set_video(Path(selected))

    def on_register_owned_source(self) -> None:
        video_path = self.ensure_ready_video()
        if not video_path:
            return
        analysis_dir = analysis_dir_for_video(video_path)
        source_title = self.title_var.get().strip() or video_path.stem

        def task() -> None:
            try:
                self.log_queue.put(("status", "내 롱폼 라이브러리 등록 중"))
                self.run_subprocess(
                    [
                        str(self.python_exe),
                        "shorts_orchestrator.py",
                        "--config",
                        str(ORCHESTRATOR_CONFIG_PATH),
                        "register-source",
                        "--source-video",
                        str(video_path),
                        "--analysis-dir",
                        str(analysis_dir),
                        "--source-title",
                        source_title,
                        "--confirm-owned",
                    ]
                )
                self.log_queue.put(("log", "[gui] owned source registered"))
                self.log_queue.put(("status", "라이브러리 등록 완료"))
            except Exception as exc:
                self.log_queue.put(("error", f"라이브러리 등록 실패: {exc}"))
                self.log_queue.put(("status", "대기 중"))

        self.start_worker(task, "내 롱폼 라이브러리 등록 중")

    def get_video_path(self) -> Path | None:
        value = self.video_path_var.get().strip()
        if not value:
            return None
        return Path(value)

    def ensure_ready_video(self) -> Path | None:
        video_path = self.get_video_path()
        if not video_path:
            messagebox.showerror("오류", "먼저 영상 파일을 선택해주세요.")
            return None
        if not video_path.exists():
            messagebox.showerror("오류", f"영상 파일을 찾을 수 없습니다.\n{video_path}")
            return None
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            replacement = companion_video_path(video_path)
            if replacement and replacement.exists():
                self.set_video(replacement)
                return replacement
            messagebox.showerror(
                "오류",
                "현재 선택된 파일은 영상 파일이 아니라 오디오 파일입니다.\n"
                "쇼츠 렌더링까지 하려면 mp4 같은 영상 파일이 필요합니다.",
            )
            return None
        return video_path

    def selected_package(self) -> dict | None:
        if not self.package_tree:
            return None
        selection = self.package_tree.selection()
        if not selection:
            return None
        index = int(selection[0])
        if index < 0 or index >= len(self.packages):
            return None
        return self.packages[index]

    def ensure_runtime_package(self, video_path: Path, pkg: dict) -> Path:
        short_id = pkg.get("short_id")
        if not short_id:
            raise RuntimeError("선택한 패키지에 short_id가 없습니다.")
        runtime_path = runtime_package_path(video_path, short_id)
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_pkg = dict(pkg)
        source_label = self.resolve_source_credit_label(video_path)
        if source_label:
            runtime_pkg.setdefault("source_label", source_label)
        runtime_path.write_text(json.dumps(runtime_pkg, ensure_ascii=False, indent=2), encoding="utf-8")
        return runtime_path

    def clear_package_tree(self) -> None:
        if not self.package_tree:
            return
        for item in self.package_tree.get_children():
            self.package_tree.delete(item)

    def format_package_comment_signal(self, pkg: dict) -> str:
        for key in ["comment_signal", "comment_evidence", "human_reaction_signal", "audience_signal"]:
            value = pkg.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:28]
            if isinstance(value, dict) and value:
                score = value.get("score") or value.get("reaction_score")
                time_sec = value.get("time_sec") or value.get("timestamp_sec")
                if score and time_sec is not None:
                    return f"{format_seconds(time_sec)} / {score}"
                if score:
                    return f"점수 {score}"

        clips = pkg.get("source_clips", []) or []
        if not self.youtube_context or not clips:
            return "-"

        insights = self.youtube_context.get("comment_insights", {}) or {}
        moments = insights.get("timecode_moments", []) or []
        if not moments:
            return "-"

        clip_ranges = []
        for clip in clips:
            try:
                clip_ranges.append((float(clip.get("source_start", 0)), float(clip.get("source_end", 0))))
            except Exception:
                continue

        best = None
        for moment in moments:
            try:
                sec = float(moment.get("representative_time_sec", moment.get("bucket_start_sec")))
            except Exception:
                continue
            if any(start - 5 <= sec <= end + 5 for start, end in clip_ranges):
                if not best or float(moment.get("reaction_score", 0)) > float(best.get("reaction_score", 0)):
                    best = moment
        if not best:
            return "-"
        return f"{format_seconds(best.get('representative_time_sec'))} / {best.get('reaction_score', 0)}"

    def describe_cache_status(self, video_path: Path) -> str:
        package_path = self.resolve_packages_json_path(video_path)
        if package_path:
            return (
                "이 영상에 대한 기존 결과가 있습니다.\n\n"
                f"- 캐시 파일: {package_path}\n"
                "- `목록 새로고침`을 누르면 기존 결과를 불러옵니다.\n"
                "- `쇼츠 시놉시스 생성`을 누르면 새로 분석하거나, 강제 재생성 옵션에 따라 다시 만듭니다."
            )
        return (
            "이 영상에 대한 기존 결과가 없습니다.\n\n"
            "- `쇼츠 시놉시스 생성`을 눌러 새로 분석을 시작하세요."
        )

    def render_movie_info(self) -> None:
        if not self.movie_info:
            self.append_text(self.movie_info_text, "영화 정보가 아직 없습니다.")
            return

        lines = [
            f"조회어: {self.movie_info.get('query', '')}",
            f"매칭 제목: {self.movie_info.get('matched_title', '')}",
            f"언어: {self.movie_info.get('language', '')}",
            f"링크: {self.movie_info.get('url', '')}",
            "",
            f"설명: {self.movie_info.get('description', '')}",
            "",
            self.movie_info.get("summary", ""),
        ]
        self.append_text(self.movie_info_text, "\n".join(lines).strip())

    def resolve_source_credit_label(self, video_path: Path) -> str:
        context = self.youtube_context
        if not context:
            context_path = self.resolve_youtube_context_path(video_path)
            if context_path:
                try:
                    context = json.loads(context_path.read_text(encoding="utf-8"))
                except Exception:
                    context = None

        metadata = (context or {}).get("metadata", {}) or {}
        channel_name = first_nonempty_string(metadata.get("channel_title"))
        if channel_name:
            return channel_name

        movie_info = self.movie_info
        if not movie_info:
            saved_info = movie_info_path_for_video(video_path)
            if saved_info.exists():
                try:
                    movie_info = json.loads(saved_info.read_text(encoding="utf-8"))
                except Exception:
                    movie_info = None

        if isinstance(movie_info, dict):
            movie_title = first_nonempty_string(
                movie_info.get("matched_title"),
                movie_info.get("query"),
                movie_info.get("title"),
            )
            if movie_title:
                return movie_title

        return first_nonempty_string(self.title_var.get(), video_path.stem)

    def get_youtube_url(self) -> str:
        return self.youtube_url_var.get().strip()

    def selected_benchmark_profile_path(self) -> Path | None:
        path = BENCHMARK_PROFILE_OPTIONS.get(self.benchmark_profile_var.get())
        return path if path and path.exists() else None

    @staticmethod
    def package_genre_score(pkg: dict) -> tuple[str, str]:
        scorecard = pkg.get("genre_scorecard") if isinstance(pkg.get("genre_scorecard"), dict) else {}
        if not scorecard:
            return "-", "-"
        return str(scorecard.get("total", "-")), str(scorecard.get("decision", "-"))

    @staticmethod
    def package_auto_render_allowed(pkg: dict) -> bool:
        scorecard = pkg.get("genre_scorecard") if isinstance(pkg.get("genre_scorecard"), dict) else {}
        return not scorecard or scorecard.get("decision") == "auto_render"

    def first_log_value(self, lines: list[str], prefix: str) -> str:
        for line in lines:
            if line.startswith(prefix):
                return line.split("=", 1)[1].strip()
        return ""

    def load_package_items_for_auto(self, package_path: Path, source_label: str) -> list[dict]:
        data = json.loads(package_path.read_text(encoding="utf-8"))
        source_title = first_nonempty_string(data.get("source_title"), source_label)
        movie_info = data.get("movie_info")
        packages: list[dict] = []
        for item in data.get("shorts", []) or []:
            if not isinstance(item, dict):
                continue
            pkg = dict(item)
            if source_label:
                pkg.setdefault("source_label", source_label)
            if source_title:
                pkg.setdefault("source_title", source_title)
            if isinstance(movie_info, dict):
                pkg.setdefault("movie_info", movie_info)
            packages.append(pkg)
        return packages

    def write_auto_runtime_package(self, video_path: Path, pkg: dict, source_label: str) -> Path:
        short_id = pkg.get("short_id")
        if not short_id:
            raise RuntimeError("자동 생성 패키지에 short_id가 없습니다.")
        runtime_path = runtime_package_path(video_path, short_id)
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_pkg = dict(pkg)
        if source_label:
            runtime_pkg.setdefault("source_label", source_label)
        runtime_path.write_text(json.dumps(runtime_pkg, ensure_ascii=False, indent=2), encoding="utf-8")
        return runtime_path

    @staticmethod
    def has_registered_owned_sources() -> bool:
        if not ORCHESTRATOR_CONFIG_PATH.exists():
            return False
        try:
            config = json.loads(ORCHESTRATOR_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return False
        library = config.get("library", {}) if isinstance(config.get("library"), dict) else {}
        sources = library.get("sources", []) if isinstance(library.get("sources"), list) else []
        return any(
            isinstance(source, dict)
            and source.get("owned") is True
            and bool(source.get("source_video"))
            for source in sources
        )

    @staticmethod
    def has_daily_source_strategy() -> bool:
        if Long2ShortsApp.has_registered_owned_sources():
            return True
        if not ORCHESTRATOR_CONFIG_PATH.exists():
            return False
        try:
            config = json.loads(ORCHESTRATOR_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return False
        acquisition = config.get("source_acquisition", {}) if isinstance(config.get("source_acquisition"), dict) else {}
        return bool(acquisition.get("enabled", False))

    def on_daily_shorts_run(self) -> None:
        if not self.has_daily_source_strategy():
            messagebox.showwarning(
                "소스 설정 필요",
                "트렌드 원본 자동 선정 또는 내 롱폼 라이브러리 중 하나가 활성화되어야 합니다.",
            )
            self.log_queue.put(("status", "소스 설정 대기 중"))
            return
        self.reset_process_board()

        def task() -> None:
            try:
                self.log_queue.put(("status", "오늘 쇼츠 제작 준비 중"))
                self.run_subprocess(
                    [
                        str(self.python_exe),
                        "shorts_orchestrator.py",
                        "--config",
                        str(ORCHESTRATOR_CONFIG_PATH),
                        "daily-run",
                        "--execute",
                    ]
                )
                self.log_queue.put(("log", "[gui] daily orchestration completed"))
                self.log_queue.put(("status", "패키지 검토 대기"))
            except Exception as exc:
                self.log_queue.put(("error", f"오늘 쇼츠 제작 실패: {exc}"))
                self.log_queue.put(("status", "대기 중"))

        self.start_worker(task, "오늘 쇼츠 제작 중")

    def on_auto_trend_generate(self) -> None:
        try:
            max_capcut = int(self.max_auto_capcut_var.get())
        except Exception:
            max_capcut = 10
        max_capcut = max(1, min(10, max_capcut))
        self.reset_process_board()
        benchmark_profile_path = self.selected_benchmark_profile_path()

        def task() -> None:
            selected_url = ""
            selected_video_id = ""
            selected_title = ""
            selected_channel = ""
            context_path = ""
            downloaded_video = ""
            package_output = ""
            capcut_created = 0

            try:
                self.log_queue.put(("status", "트렌드 후보 탐색 중"))
                self.log_queue.put(("phase", "최근 3일 예능/연예 롱폼 후보 탐색 중"))
                self.log_queue.put(("stage_status", {"key": "trend", "status": "진행", "note": "최근 72시간 후보를 수집하고 green 원본을 고릅니다."}))

                trend_lines = self.run_subprocess(
                    [
                        str(self.python_exe),
                        "discover_trending_sources.py",
                        "--window-hours",
                        "72",
                        "--limit",
                        "20",
                    ]
                )
                selected_url = self.first_log_value(trend_lines, "[trend] selected_url=")
                selected_video_id = self.first_log_value(trend_lines, "[trend] selected_video_id=")
                selected_title = self.first_log_value(trend_lines, "[trend] selected_title=")
                selected_channel = self.first_log_value(trend_lines, "[trend] selected_channel=")
                if not selected_url or not selected_video_id:
                    raise RuntimeError("최근 72시간 안에서 자동 처리 가능한 green 후보를 찾지 못했습니다.")

                self.log_queue.put(("set_youtube_url", selected_url))
                self.log_queue.put(("set_title", selected_title))
                self.log_queue.put(("stage_status", {"key": "trend", "status": "완료", "note": f"{selected_channel} / {selected_title}"}))
                self.log_queue.put(("status", "선택된 트렌드 원본 다운로드/분석 중"))

                collect_command = [
                    str(self.python_exe),
                    "collect_youtube_context.py",
                    "--url",
                    selected_url,
                    "--max-comments",
                    "300",
                    "--download-video",
                    "--prepare-transcript",
                    "--generate-packages",
                    "--download-dir",
                    str(DOWNLOADS_DIR),
                ]
                if benchmark_profile_path:
                    collect_command.extend(["--benchmark-profile", str(benchmark_profile_path)])
                collect_lines = self.run_subprocess(collect_command)

                for line in collect_lines:
                    if line.startswith("[youtube] context_packages="):
                        context_path = line.split("=", 1)[1].strip()
                    elif line.startswith("[youtube] context_transcript=") and not context_path:
                        context_path = line.split("=", 1)[1].strip()
                    elif line.startswith("[youtube] context=") and not context_path:
                        context_path = line.split("=", 1)[1].strip()
                    elif line.startswith("[youtube] downloaded_video="):
                        downloaded_video = line.split("=", 1)[1].strip()
                    elif line.startswith("[youtube] package_generation_done="):
                        package_output = line.split("=", 1)[1].strip()

                if context_path:
                    self.log_queue.put(("youtube_context_loaded", context_path))
                if downloaded_video:
                    self.log_queue.put(("video_downloaded", downloaded_video))

                if not package_output and context_path:
                    context = json.loads(Path(context_path).read_text(encoding="utf-8"))
                    package_output = ((context.get("package_generation") or {}).get("output") or "")
                    downloaded_video = downloaded_video or context.get("downloaded_video", "")
                    selected_channel = selected_channel or ((context.get("metadata") or {}).get("channel_title") or "")
                    selected_title = selected_title or ((context.get("metadata") or {}).get("title") or "")

                video_path = Path(downloaded_video)
                package_path = Path(package_output) if package_output else None
                if not video_path.exists():
                    raise RuntimeError("트렌드 원본 다운로드 파일을 찾지 못했습니다.")
                if not package_path or not package_path.is_file():
                    raise RuntimeError("쇼츠 패키지 결과 파일을 찾지 못했습니다.")

                packages = self.load_package_items_for_auto(package_path, selected_channel)
                auto_render_packages = [pkg for pkg in packages if self.package_auto_render_allowed(pkg)]
                if len(auto_render_packages) != len(packages):
                    self.log_queue.put(
                        (
                            "log",
                            f"[gui] benchmark gate: {len(packages) - len(auto_render_packages)} review/reject packages skipped",
                        )
                    )
                packages = auto_render_packages
                if not packages:
                    raise RuntimeError("품질 기준을 통과한 쇼츠 패키지가 없습니다.")

                self.log_queue.put(("status", "CapCut draft 자동 생성 중"))
                self.log_queue.put(("phase", f"CapCut draft 자동 생성 중: {min(len(packages), max_capcut)}개"))
                self.log_queue.put(("stage_status", {"key": "output", "status": "진행", "note": f"CapCut draft 최대 {max_capcut}개 생성"}))

                for pkg in packages[:max_capcut]:
                    short_id = pkg.get("short_id", "")
                    runtime_path = self.write_auto_runtime_package(video_path, pkg, selected_channel)
                    command = [
                        str(self.python_exe),
                        "build_capcut_from_package.py",
                        "--source-video",
                        str(video_path),
                        "--package",
                        str(runtime_path),
                    ]
                    if selected_channel:
                        command.extend(["--channel-name", selected_channel])
                    self.run_subprocess(command)
                    capcut_created += 1
                    self.log_queue.put(("log", f"[gui] CapCut draft 자동 생성 완료: {short_id}"))

                mark_processed_source(
                    video_id=selected_video_id,
                    source_url=selected_url,
                    title=selected_title,
                    channel_title=selected_channel,
                    shorts_created=len(packages),
                    capcut_drafts_created=capcut_created,
                )
                self.log_queue.put(("refresh_packages", str(video_path)))
                self.log_queue.put(("stage_status", {"key": "output", "status": "완료", "note": f"CapCut draft {capcut_created}개 생성 완료"}))
                self.log_queue.put(("status", "대기 중"))
                self.log_queue.put(("phase", f"트렌드 자동 생성 완료: CapCut draft {capcut_created}개"))
            except Exception as exc:
                self.log_queue.put(("error", f"트렌드 자동 생성 실패: {exc}"))
                self.log_queue.put(("status", "대기 중"))

        self.start_worker(task, "트렌드 자동 생성 중")

    def on_collect_youtube_context(self) -> None:
        self.collect_youtube_context(download_video=False)

    def on_download_youtube_video(self) -> None:
        self.collect_youtube_context(download_video=True)

    def collect_youtube_context(self, download_video: bool) -> None:
        url = self.get_youtube_url()
        if not url:
            messagebox.showerror("오류", "YouTube 링크를 먼저 입력해주세요.")
            return

        if not self.youtube_context or self.youtube_context.get("source_url") != url:
            self.reset_process_board()

        video_path = self.get_video_path()
        output_dir = None
        if not download_video and video_path and video_path.exists():
            output_dir = analysis_dir_for_video(video_path)

        def task() -> None:
            self.log_queue.put(("status", "YouTube 링크 분석 중"))
            self.log_queue.put(("phase", "댓글 반응 수집 준비"))
            self.log_queue.put(("stage_status", {"key": "link", "status": "진행", "note": "YouTube 메타데이터 확인 중"}))
            self.log_queue.put(("stage_status", {"key": "comments", "status": "진행", "note": "댓글과 타임스탬프 반응 수집 중"}))
            if download_video:
                self.log_queue.put(("stage_status", {"key": "media", "status": "진행", "note": "링크 영상 다운로드 준비"}))
                self.log_queue.put(("stage_status", {"key": "transcript", "status": "진행", "note": "자막 우선, 없으면 오디오+STT 준비"}))

            command = [
                str(self.python_exe),
                "collect_youtube_context.py",
                "--url",
                url,
                "--max-comments",
                "300",
            ]
            if output_dir:
                command.extend(["--output-dir", str(output_dir)])
            if download_video:
                command.extend(
                    [
                        "--download-video",
                        "--prepare-transcript",
                        "--generate-packages",
                        "--download-dir",
                        str(DOWNLOADS_DIR),
                    ]
                )

            try:
                lines = self.run_subprocess(command)
                context_path = ""
                downloaded_video = ""
                for line in lines:
                    if line.startswith("[youtube] context_preliminary="):
                        context_path = line.split("=", 1)[1].strip()
                        self.log_queue.put(("youtube_context_loaded", context_path))
                    elif line.startswith("[youtube] context_transcript="):
                        context_path = line.split("=", 1)[1].strip()
                        self.log_queue.put(("youtube_context_loaded", context_path))
                    elif line.startswith("[youtube] context_packages="):
                        context_path = line.split("=", 1)[1].strip()
                        self.log_queue.put(("youtube_context_loaded", context_path))
                    elif line.startswith("[youtube] context="):
                        context_path = line.split("=", 1)[1].strip()
                    elif line.startswith("[youtube] downloaded_video="):
                        downloaded_video = line.split("=", 1)[1].strip()

                if context_path:
                    self.log_queue.put(("youtube_context_loaded", context_path))
                if downloaded_video:
                    self.log_queue.put(("video_downloaded", downloaded_video))
                self.log_queue.put(("status", "대기 중"))
            except Exception as exc:
                self.log_queue.put(("error", f"YouTube 링크 분석 실패: {exc}"))
                self.log_queue.put(("status", "대기 중"))

        phase = "YouTube 다운로드/댓글 분석 중" if download_video else "YouTube 댓글 반응 분석 중"
        self.start_worker(task, phase)

    def terminate_current_process(self) -> None:
        process = self.current_process
        if not process or process.poll() is not None:
            return

        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return

        process.terminate()

    def on_cancel_worker(self) -> None:
        if not self.worker_thread or not self.worker_thread.is_alive():
            return
        self.cancel_requested = True
        self.append_log("[gui] 현재 작업 중지 요청")
        self.status_var.set("작업 중지 중")
        self.phase_var.set("작업 중지 중")
        self.terminate_current_process()

    def run_subprocess(self, args: list[str], cwd: Path | None = None) -> list[str]:
        command = list(args)
        if command and Path(command[0]).name.lower().startswith("python") and "-u" not in command[1:3]:
            command.insert(1, "-u")

        self.log_queue.put(("log", "$ " + " ".join(command)))
        output_lines: list[str] = []
        process = subprocess.Popen(
            command,
            cwd=str(cwd or BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.current_process = process
        try:
            assert process.stdout is not None
            for line in process.stdout:
                clean = line.rstrip()
                output_lines.append(clean)
                self.log_queue.put(("log", clean))
                if clean.startswith("[youtube]"):
                    self.update_stages_from_log(clean)
                if clean.startswith("[trend]") or clean.startswith("[analyze]") or clean.startswith("[package]") or clean.startswith("[preview]") or clean.startswith("[capcut]") or clean.startswith("[orchestrator]"):
                    self.log_queue.put(("phase", clean))
                    self.update_stages_from_log(clean)
            code = process.wait()
        finally:
            if self.current_process is process:
                self.current_process = None

        if self.cancel_requested:
            raise RuntimeError("작업이 중지되었습니다.")
        if code != 0:
            raise RuntimeError(f"명령 실행 실패 (exit {code})")
        return output_lines

    def update_stages_from_log(self, line: str) -> None:
        stage_payload = None
        if line.startswith("[orchestrator] phase=trend_research"):
            stage_payload = {"key": "trend", "status": "진행", "note": "트렌드 신호 조사 중"}
        elif line.startswith("[orchestrator] phase=source_acquisition"):
            stage_payload = {"key": "media", "status": "진행", "note": "선정 롱폼 다운로드/전사 준비 중"}
        elif line.startswith("[orchestrator] phase=source_analysis"):
            stage_payload = {"key": "transcript", "status": "진행", "note": line.split("source=", 1)[-1]}
        elif line.startswith("[orchestrator] phase=package_generation"):
            stage_payload = {"key": "candidates", "status": "진행", "note": line.split("source=", 1)[-1]}
        elif line.startswith("[orchestrator] phase=render"):
            stage_payload = {"key": "output", "status": "진행", "note": "세로형 MP4 렌더 중"}
        elif line.startswith("[orchestrator] phase=review"):
            stage_payload = {"key": "output", "status": "진행", "note": "검토 항목 등록 중"}
        elif line.startswith("[orchestrator] phase=upload"):
            stage_payload = {"key": "output", "status": "진행", "note": "YouTube 업로드 중"}
        elif line.startswith("[orchestrator] learning_rule="):
            stage_payload = {"key": "candidates", "status": "진행", "note": f"학습 규칙: {line.split('=', 1)[1]}"}
        elif line.startswith("[orchestrator] change_note="):
            stage_payload = {"key": "candidates", "status": "진행", "note": line.split("=", 1)[1]}
        elif line.startswith("[orchestrator] selected_source="):
            stage_payload = {"key": "trend", "status": "완료", "note": line.split("=", 1)[1]}
        elif line.startswith("[orchestrator] source_acquisition="):
            stage_payload = {"key": "media", "status": "완료", "note": line}
        elif line.startswith("[orchestrator] package_output="):
            stage_payload = {"key": "output", "status": "완료", "note": Path(line.split("=", 1)[1]).name}
        elif line.startswith("[orchestrator] rendered_output="):
            stage_payload = {"key": "output", "status": "완료", "note": Path(line.split("=", 1)[1]).name}
        elif line.startswith("[orchestrator] review_item="):
            review_path = line.split("review_item=", 1)[1].split(" status=", 1)[0].strip()
            stage_payload = {"key": "output", "status": "검토 대기", "note": Path(review_path).name}
        elif line.startswith("[orchestrator] uploaded_video="):
            stage_payload = {"key": "output", "status": "완료", "note": f"YouTube: {line.split('=', 1)[1]}"}
        elif line.startswith("[trend] candidates="):
            stage_payload = {"key": "trend", "status": "진행", "note": f"후보 {line.split('=', 1)[1]}개 수집"}
        elif line.startswith("[trend] rank="):
            stage_payload = {"key": "trend", "status": "진행", "note": line}
        elif line.startswith("[trend] selected_url="):
            value = line.split("=", 1)[1].strip()
            stage_payload = {"key": "trend", "status": "완료" if value else "보류", "note": value or "자동 처리 가능한 green 후보 없음"}
        elif line.startswith("[trend] output="):
            stage_payload = {"key": "trend", "status": "완료", "note": f"후보 목록 저장: {Path(line.split('=', 1)[1]).name}"}
        elif line.startswith("[youtube] video_id=") or line.startswith("[youtube] metadata"):
            stage_payload = {"key": "link", "status": "진행", "note": line}
        elif line.startswith("[youtube] parallel_download_started="):
            stage_payload = {"key": "media", "status": "진행", "note": "영상 다운로드를 먼저 백그라운드로 시작"}
        elif line.startswith("[youtube] transcript_preparation_started="):
            stage_payload = {"key": "transcript", "status": "진행", "note": "자막 우선 다운로드와 오디오 전사 준비 시작"}
        elif line.startswith("[youtube] collecting_context_while_download="):
            stage_payload = {"key": "comments", "status": "진행", "note": "다운로드 중 댓글/메타데이터 분석 병렬 진행"}
        elif line.startswith("[youtube] comments_fetched="):
            count = line.split("=", 1)[1]
            stage_payload = {"key": "comments", "status": "완료", "note": f"댓글 {count}개 수집"}
        elif line.startswith("[youtube] timecode_moments="):
            count = line.split("=", 1)[1]
            stage_payload = {"key": "comments", "status": "완료", "note": f"타임스탬프 반응 {count}개 발견"}
        elif line.startswith("[youtube] downloading_video"):
            stage_payload = {"key": "media", "status": "진행", "note": "YouTube 영상 다운로드 중"}
        elif line.startswith("[youtube] waiting_for_download="):
            stage_payload = {"key": "media", "status": "진행", "note": "댓글 분석 완료, 영상 다운로드 완료 대기"}
        elif line.startswith("[youtube] downloaded_video="):
            stage_payload = {"key": "media", "status": "완료", "note": line.split("=", 1)[1]}
        elif line.startswith("[youtube] download_error="):
            stage_payload = {"key": "media", "status": "실패", "note": line.split("=", 1)[1]}
        elif line.startswith("[youtube] subtitle_download_started="):
            stage_payload = {"key": "transcript", "status": "진행", "note": "YouTube 자막 먼저 확인 중"}
        elif line.startswith("[youtube] subtitle_file="):
            value = line.split("=", 1)[1]
            status = "건너뜀" if value == "none" else "진행"
            note = "사용 가능한 자막 없음, 오디오 전사로 전환" if value == "none" else f"자막 확보: {Path(value).name}"
            stage_payload = {"key": "transcript", "status": status, "note": note}
        elif line.startswith("[youtube] audio_download_started="):
            stage_payload = {"key": "transcript", "status": "진행", "note": "자막 대체용 오디오 다운로드 중"}
        elif line.startswith("[youtube] audio_file="):
            stage_payload = {"key": "transcript", "status": "진행", "note": f"오디오 확보: {Path(line.split('=', 1)[1]).name}"}
        elif line.startswith("[youtube] audio_transcription_started="):
            stage_payload = {"key": "transcript", "status": "진행", "note": "오디오 Whisper 전사 시작"}
        elif line.startswith("[youtube] transcript_from_subtitles="):
            stage_payload = {"key": "transcript", "status": "완료", "note": "자막으로 전사 파일 생성 완료"}
        elif line.startswith("[youtube] transcript_from_audio="):
            stage_payload = {"key": "transcript", "status": "완료", "note": "오디오 전사 파일 생성 완료"}
        elif line.startswith("[youtube] waiting_for_transcript="):
            stage_payload = {"key": "transcript", "status": "진행", "note": "전사 준비 완료 대기"}
        elif line.startswith("[youtube] prepared_transcript="):
            stage_payload = {"key": "transcript", "status": "완료", "note": line.split("=", 1)[1]}
        elif line.startswith("[youtube] prepared_transcript_failed=") or line.startswith("[youtube] transcript_error="):
            stage_payload = {"key": "transcript", "status": "실패", "note": line.split("=", 1)[1]}
        elif line.startswith("[youtube] context_transcript="):
            stage_payload = {"key": "transcript", "status": "완료", "note": "전사 준비 상태 저장 완료"}
        elif line.startswith("[youtube] package_generation_started="):
            stage_payload = {"key": "candidates", "status": "진행", "note": "전사 파일 기반으로 쇼츠 후보 먼저 생성"}
        elif line.startswith("[youtube] package_generation_done="):
            stage_payload = {"key": "candidates", "status": "완료", "note": line.split("=", 1)[1]}
        elif line.startswith("[youtube] package_generation_failed="):
            stage_payload = {"key": "candidates", "status": "실패", "note": line.split("=", 1)[1]}
        elif line.startswith("[youtube] context_packages="):
            stage_payload = {"key": "candidates", "status": "완료", "note": "쇼츠 후보 생성 상태 저장 완료"}
        elif line.startswith("[youtube] context="):
            stage_payload = {"key": "link", "status": "완료", "note": "YouTube 컨텍스트 저장 완료"}
        elif line.startswith("[youtube] context_preliminary="):
            stage_payload = {"key": "comments", "status": "완료", "note": "댓글 반응 먼저 저장, 다운로드는 계속 진행"}
        elif line.startswith("[analyze] chunk") or line.startswith("[analyze] transcribing"):
            stage_payload = {"key": "transcript", "status": "진행", "note": line}
        elif line.startswith("[analyze] merging") or line.startswith("[analyze] output_dir="):
            stage_payload = {"key": "transcript", "status": "완료", "note": line}
        elif line.startswith("[package] local") or line.startswith("[package] requesting global"):
            stage_payload = {"key": "candidates", "status": "진행", "note": line}
        elif line.startswith("[package] output_dir="):
            stage_payload = {"key": "candidates", "status": "완료", "note": "쇼츠 후보 패키지 생성 완료"}
        elif line.startswith("[preview]") or line.startswith("[capcut]"):
            stage_payload = {"key": "output", "status": "완료", "note": line}

        if stage_payload:
            self.log_queue.put(("stage_status", stage_payload))

    def on_generate_packages(self) -> None:
        video_path = self.ensure_ready_video()
        if not video_path:
            return

        analysis_dir = analysis_dir_for_video(video_path)
        source_title = self.title_var.get().strip() or video_path.stem
        movie_info_path = movie_info_path_for_video(video_path) if self.movie_info else None
        youtube_context_path = self.resolve_youtube_context_path(video_path)
        force = self.force_var.get()
        benchmark_profile_path = self.selected_benchmark_profile_path()

        def task() -> None:
            self.log_queue.put(("status", "전사/시놉시스 생성 중"))
            self.log_queue.put(("phase", "전사 시작"))
            self.log_queue.put(("stage_status", {"key": "media", "status": "완료", "note": f"영상 파일: {video_path.name}"}))
            if youtube_context_path:
                self.log_queue.put(("stage_status", {"key": "comments", "status": "완료", "note": f"댓글 반응 JSON 반영: {youtube_context_path.name}"}))
            else:
                self.log_queue.put(("stage_status", {"key": "comments", "status": "건너뜀", "note": "댓글 반응 파일 없이 대본만 사용"}))
            try:
                self.run_subprocess(
                    [
                        str(self.python_exe),
                        "analyze_longform.py",
                        "--source-video",
                        str(video_path),
                        "--output-dir",
                        str(analysis_dir),
                    ]
                )
                self.log_queue.put(("phase", "쇼츠 패키지 생성 중"))
                command = [
                    str(self.python_exe),
                    "generate_shorts_packages.py",
                    "--analysis-dir",
                    str(analysis_dir),
                    "--source-title",
                    source_title,
                ]
                if movie_info_path and movie_info_path.exists():
                    command.extend(["--movie-info", str(movie_info_path)])
                if youtube_context_path:
                    command.extend(["--youtube-context", str(youtube_context_path)])
                if benchmark_profile_path:
                    command.extend(["--benchmark-profile", str(benchmark_profile_path)])
                if force:
                    command.append("--force")
                self.run_subprocess(command)
                self.log_queue.put(("refresh_packages", str(video_path)))
                self.log_queue.put(("log", "[gui] 쇼츠 시놉시스 생성 완료"))
                self.log_queue.put(("status", "대기 중"))
            except Exception as exc:
                self.log_queue.put(("error", f"쇼츠 시놉시스 생성 실패: {exc}"))
                self.log_queue.put(("status", "대기 중"))

        self.start_worker(task, "쇼츠 시놉시스 생성 중")

    def on_refresh_packages(self) -> None:
        video_path = self.get_video_path()
        if not video_path:
            return

        saved_info = movie_info_path_for_video(video_path)
        if saved_info.exists() and not self.movie_info:
            try:
                self.movie_info = json.loads(saved_info.read_text(encoding="utf-8"))
            except Exception:
                self.movie_info = None
        self.render_movie_info()
        self.load_youtube_context_for_video(video_path)

        package_path = self.resolve_packages_json_path(video_path)
        if not package_path:
            self.packages = []
            self.clear_package_tree()
            self.append_text(self.detail_text, "아직 생성된 쇼츠 시놉시스가 없습니다.")
            return

        data = json.loads(package_path.read_text(encoding="utf-8"))
        source_label = self.resolve_source_credit_label(video_path)
        source_title = first_nonempty_string(data.get("source_title"), source_label)
        movie_info = data.get("movie_info")
        self.packages = []
        for item in data.get("shorts", []):
            if not isinstance(item, dict):
                continue
            pkg = dict(item)
            if source_label:
                pkg.setdefault("source_label", source_label)
            if source_title:
                pkg.setdefault("source_title", source_title)
            if isinstance(movie_info, dict):
                pkg.setdefault("movie_info", movie_info)
            self.packages.append(pkg)
        self.clear_package_tree()

        for idx, pkg in enumerate(self.packages):
            tags = ", ".join(safe_list(pkg.get("fun_tags")))
            title = f"{pkg.get('title_line1', '')} / {pkg.get('title_line2', '')}"
            reaction = self.format_package_comment_signal(pkg)
            clarity = pkg.get("standalone_clarity", "-")
            protagonist = pkg.get("protagonist_presence", "-")
            genre_score, genre_decision = self.package_genre_score(pkg)
            self.package_tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(pkg.get("global_rank", idx + 1), genre_score, genre_decision, title, reaction, tags, clarity, protagonist),
            )

        self.append_log(f"[gui] 쇼츠 패키지 {len(self.packages)}개를 불러왔습니다.")
        if self.packages:
            first_id = "0"
            self.package_tree.selection_set(first_id)
            self.package_tree.focus(first_id)
            self.on_package_selected()

    def describe_capcut_settings(self, pkg: dict) -> str:
        capcut = pkg.get("capcut")
        if not isinstance(capcut, dict) or not capcut:
            return "기본값 사용"

        sections = []
        labels = {
            "draft": "draft",
            "template": "템플릿",
            "text_overlay": "제목/출처 오버레이",
            "title": "제목 스타일",
            "channel": "출처명",
            "point": "포인트 자막 스타일",
            "narration": "나레이션",
        }
        for key, label in labels.items():
            value = capcut.get(key)
            if isinstance(value, dict) and value:
                sections.append(label)

        item_overrides = 0
        for caption in pkg.get("point_captions", []) or []:
            if isinstance(caption.get("capcut"), dict) and caption.get("capcut"):
                item_overrides += 1

        if item_overrides:
            sections.append(f"개별 자막 {item_overrides}개")

        return ", ".join(sections) if sections else "기본값 사용"

    def upload_hashtag(self, text: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z가-힣_]", "", text or "")
        return f"#{cleaned}" if cleaned else ""

    def suggest_upload_title(self, pkg: dict) -> str:
        explicit = first_nonempty_string(pkg.get("upload_title"), pkg.get("youtube_upload_title"))
        if explicit:
            return explicit

        title_line1 = first_nonempty_string(pkg.get("title_line1"))
        title_line2 = first_nonempty_string(pkg.get("title_line2"))
        if title_line1 and title_line2:
            title = f"{title_line1}, {title_line2}"
        else:
            title = first_nonempty_string(title_line1, title_line2, pkg.get("hook_line"), pkg.get("core_event"))

        hashtags: list[str] = []
        source_label = first_nonempty_string(pkg.get("source_label"), pkg.get("source_title"))
        if source_label:
            source_tag = self.upload_hashtag(source_label.split()[0])
            if source_tag:
                hashtags.append(source_tag)
        for tag_text in safe_list(pkg.get("fun_tags")):
            tag = self.upload_hashtag(str(tag_text))
            if tag and tag not in hashtags:
                hashtags.append(tag)
            if len(hashtags) >= 3:
                break

        suffix = f" {' '.join(hashtags)}" if hashtags else ""
        return f"{title}{suffix}".strip()

    def format_package_detail(self, pkg: dict) -> str:
        tags = ", ".join(safe_list(pkg.get("fun_tags"), ["미정"]))
        characters = ", ".join(safe_list(pkg.get("main_characters"), ["미정"]))
        selection_pitch = pkg.get("selection_pitch") or "아직 선택 설명이 없습니다."
        hook_line = pkg.get("hook_line") or "아직 핵심 대사가 없습니다."
        clips = pkg.get("source_clips", []) or []
        point_captions = pkg.get("point_captions", []) or []
        narration = pkg.get("narration", []) or []
        capcut_summary = self.describe_capcut_settings(pkg)
        comment_signal = self.format_package_comment_signal(pkg)
        clip_total_sec = sum(max(0.0, float(clip.get("source_end", 0.0)) - float(clip.get("source_start", 0.0))) for clip in clips)
        scorecard = pkg.get("genre_scorecard") if isinstance(pkg.get("genre_scorecard"), dict) else {}

        lines = [
            f"업로드 제목: {self.suggest_upload_title(pkg)}",
            "",
            f"제목 1: {pkg.get('title_line1', '')}",
            f"제목 2: {pkg.get('title_line2', '')}",
            f"출처명: {pkg.get('source_label', '') or pkg.get('source_title', '')}",
            "",
            f"왜 볼만한가: {selection_pitch}",
            f"재미 태그: {tags}",
            f"핵심 대사/포인트: {hook_line}",
            f"중심 인물: {characters}",
            f"주인공 포함도: {pkg.get('protagonist_presence', 'unknown')}",
            f"맥락 없이 이해도: {pkg.get('standalone_clarity', 'unknown')}",
            f"목표 길이: {pkg.get('target_duration_sec', '')}초",
            f"실제 컷 합계: {clip_total_sec:.1f}초",
            f"컷 수: {len(clips)}",
            f"점수: {pkg.get('global_score', pkg.get('score', ''))}",
            f"댓글 신호: {comment_signal}",
            f"CapCut 설정: {capcut_summary}",
            "",
            f"핵심 사건: {pkg.get('core_event', '')}",
            f"감정 흐름: {pkg.get('emotion_arc', '')}",
            "",
            "컷 구성:",
        ]

        if scorecard:
            lines.extend(
                [
                    "",
                    f"Genre score: {scorecard.get('total', '-')}/100 ({scorecard.get('decision', '-')})",
                ]
            )
            for item in scorecard.get("dimensions", []) or []:
                if isinstance(item, dict):
                    lines.append(
                        f"  - {item.get('id', '')}: {item.get('score', '')}/{item.get('max_score', '')} | {item.get('reason', '')}"
                    )
            for item in scorecard.get("penalties", []) or []:
                if isinstance(item, dict):
                    lines.append(
                        f"  - penalty {item.get('id', '')}: -{item.get('deduction', '')} | {item.get('reason', '')}"
                    )

        for index, clip in enumerate(clips, start=1):
            lines.append(
                f"  {index}. {clip.get('purpose', '')} | {float(clip.get('source_start', 0.0)):.1f}s - {float(clip.get('source_end', 0.0)):.1f}s"
            )

        lines.append("")
        lines.append("포인트 자막:")
        if point_captions:
            for item in point_captions:
                lines.append(
                    f"  - {float(item.get('target_start', 0.0)):.1f}s ~ {float(item.get('target_end', 0.0)):.1f}s | {item.get('text', '')}"
                )
        else:
            lines.append("  - 없음")

        lines.append("")
        lines.append("나레이션:")
        if narration:
            for item in narration:
                lines.append(f"  - {float(item.get('target_start', 0.0)):.1f}s | {item.get('text', '')}")
        else:
            lines.append("  - 없음")

        lines.append("")
        lines.append("편집 메모:")
        for note in safe_list(pkg.get("edit_notes")):
            lines.append(f"  - {note}")

        if self.youtube_context and clips:
            insights = self.youtube_context.get("comment_insights", {}) or {}
            moments = insights.get("timecode_moments", []) or []
            nearby = []
            for moment in moments:
                try:
                    sec = float(moment.get("representative_time_sec", moment.get("bucket_start_sec")))
                except Exception:
                    continue
                for clip in clips:
                    try:
                        start = float(clip.get("source_start", 0.0))
                        end = float(clip.get("source_end", 0.0))
                    except Exception:
                        continue
                    if start - 5 <= sec <= end + 5:
                        nearby.append(moment)
                        break
            if nearby:
                lines.append("")
                lines.append("근처 댓글 반응:")
                for moment in sorted(nearby, key=lambda item: float(item.get("reaction_score", 0)), reverse=True)[:5]:
                    samples = moment.get("samples", []) or []
                    sample_text = samples[0].get("text", "") if samples else ""
                    lines.append(
                        f"  - {format_seconds(moment.get('representative_time_sec'))} | 점수 {moment.get('reaction_score', 0)} | {sample_text}"
                    )

        return "\n".join(lines)

    def on_package_selected(self, _event=None) -> None:
        pkg = self.selected_package()
        if not pkg:
            self.append_text(self.detail_text, "")
            return
        self.append_text(self.detail_text, self.format_package_detail(pkg))

    def on_open_preview(self) -> None:
        video_path = self.ensure_ready_video()
        if not video_path:
            return
        pkg = self.selected_package()
        if not pkg:
            messagebox.showerror("오류", "먼저 후보를 하나 선택해주세요.")
            return

        short_id = pkg.get("short_id")
        if not short_id:
            messagebox.showerror("오류", "선택한 패키지에 short_id가 없습니다.")
            return

        preview_path = preview_file_path(video_path, short_id)

        def task() -> None:
            self.log_queue.put(("status", "미리보기 준비 중"))
            self.log_queue.put(("phase", "러프 미리보기 생성 중"))
            self.log_queue.put(("stage_status", {"key": "output", "status": "진행", "note": "러프 미리보기 MP4 생성/열기"}))
            try:
                package_path = self.ensure_runtime_package(video_path, pkg)
                if not preview_path.exists():
                    self.run_subprocess(
                        [
                            str(self.python_exe),
                            "render_package_preview.py",
                            "--source-video",
                            str(video_path),
                            "--package",
                            str(package_path),
                            "--output",
                            str(preview_path),
                        ]
                    )
                os.startfile(str(preview_path))
                self.log_queue.put(("log", f"[gui] 미리보기 열기: {preview_path}"))
                self.log_queue.put(("status", "대기 중"))
            except Exception as exc:
                self.log_queue.put(("error", f"미리보기 열기 실패: {exc}"))
                self.log_queue.put(("status", "대기 중"))

        self.start_worker(task, "미리보기 준비 중")

    def on_create_capcut(self) -> None:
        video_path = self.ensure_ready_video()
        if not video_path:
            return
        pkg = self.selected_package()
        if not pkg:
            messagebox.showerror("오류", "CapCut으로 만들 후보를 먼저 선택해주세요.")
            return

        short_id = pkg.get("short_id")
        if not short_id:
            messagebox.showerror("오류", "선택한 패키지에 short_id가 없습니다.")
            return

        def task() -> None:
            self.log_queue.put(("status", "CapCut draft 생성 중"))
            self.log_queue.put(("phase", "CapCut draft 생성 중"))
            self.log_queue.put(("stage_status", {"key": "output", "status": "진행", "note": "CapCut draft 생성 중"}))
            try:
                package_path = self.ensure_runtime_package(video_path, pkg)
                command = [
                    str(self.python_exe),
                    "build_capcut_from_package.py",
                    "--source-video",
                    str(video_path),
                    "--package",
                    str(package_path),
                ]
                source_label = self.resolve_source_credit_label(video_path)
                if source_label:
                    command.extend(["--channel-name", source_label])
                self.run_subprocess(command)
                self.log_queue.put(("log", f"[gui] CapCut draft 생성 완료: {short_id}"))
                self.log_queue.put(("status", "대기 중"))
            except Exception as exc:
                self.log_queue.put(("error", f"CapCut draft 생성 실패: {exc}"))
                self.log_queue.put(("status", "대기 중"))

        self.start_worker(task, "CapCut draft 생성 중")

    def start_worker(self, func, busy_phase: str) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            self.append_log("[gui] 이미 작업이 진행 중입니다.")
            self.phase_var.set("이미 작업 진행 중")
            return

        self.clear_log()
        self.cancel_requested = False
        self.set_busy(True, busy_phase)

        def runner() -> None:
            try:
                func()
            finally:
                self.log_queue.put(("busy_done", ""))

        self.worker_thread = threading.Thread(target=runner, daemon=True)
        self.worker_thread.start()

    def _poll_log_queue(self) -> None:
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self.append_log(str(payload))
                elif kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "phase":
                    self.phase_var.set(str(payload))
                elif kind == "set_youtube_url":
                    self.youtube_url_var.set(str(payload))
                elif kind == "set_title":
                    self.title_var.set(str(payload))
                elif kind == "error":
                    self.append_log(str(payload))
                    if "작업이 중지" not in str(payload):
                        messagebox.showerror("오류", str(payload))
                elif kind == "movie_info":
                    self.render_movie_info()
                elif kind == "reference_channel_added":
                    if isinstance(payload, dict):
                        self.upsert_reference_video(payload)
                        self.reference_channel_var.set("")
                        self.append_log(
                            f"[gui] 참고 채널 최신 영상 추가: {payload.get('channel_title', '')} / {payload.get('title', '')}"
                        )
                elif kind == "reference_channels_refreshed":
                    if isinstance(payload, list):
                        self.reference_channels = [item for item in payload if isinstance(item, dict)]
                        self.save_reference_channels()
                        self.refresh_reference_tree()
                        self.append_log(f"[gui] 참고 채널 {len(self.reference_channels)}개 새로고침 완료")
                elif kind == "refresh_packages":
                    self.on_refresh_packages()
                elif kind == "stage_status":
                    if isinstance(payload, dict):
                        self.set_stage_status(
                            str(payload.get("key", "")),
                            str(payload.get("status", "")),
                            str(payload.get("note", "")),
                        )
                elif kind == "youtube_context_loaded":
                    context_path = Path(str(payload))
                    if context_path.exists():
                        if self.load_youtube_context_path(context_path):
                            self.render_youtube_context()
                            metadata = self.youtube_context.get("metadata", {}) or {}
                            if metadata.get("title") and not self.title_var.get().strip():
                                self.title_var.set(metadata.get("title", ""))
                            if not self.analysis_dir_var.get().strip():
                                self.analysis_dir_var.set(str(context_path.parent))
                            fetched = (self.youtube_context.get("comments") or {}).get("fetched_count", 0)
                            moments = ((self.youtube_context.get("comment_insights") or {}).get("timecode_moments") or [])
                            self.set_stage_status("link", "완료", f"컨텍스트 저장: {context_path.name}")
                            self.set_stage_status("comments", "완료", f"댓글 {fetched}개, 타임스탬프 반응 {len(moments)}개")
                elif kind == "video_downloaded":
                    downloaded_path = Path(str(payload))
                    if downloaded_path.exists():
                        self.set_video(downloaded_path)
                        self.set_stage_status("media", "완료", f"다운로드 완료: {downloaded_path.name}")
                        self.append_log(f"[gui] 다운로드 영상 선택: {downloaded_path}")
                        if self.resolve_packages_json_path(downloaded_path):
                            self.on_refresh_packages()
                elif kind == "busy_done":
                    self.set_busy(False)
        except queue.Empty:
            pass
        finally:
            self.root.after(120, self._poll_log_queue)


class ShortsDashboardApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Long2Shorts Studio")
        self.root.geometry("1280x860")
        self.root.minsize(1080, 760)
        self.python_exe = get_python_exe()
        self.log_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.current_process: subprocess.Popen | None = None
        self.review_rows: dict[str, dict] = {}

        self.status_var = tk.StringVar(value="대기 중")
        self.run_note_var = tk.StringVar(value="오늘 쇼츠 제작을 누르면 트렌드 조사부터 검토 대기 등록까지 자동으로 진행합니다.")
        self.review_note_var = tk.StringVar(value="검토 대기 영상을 선택하세요.")
        self.stage_vars: dict[str, tk.StringVar] = {}
        self.stage_note_vars: dict[str, tk.StringVar] = {}

        self.review_tree: ttk.Treeview | None = None
        self.feedback_text: tk.Text | None = None
        self.log_text: tk.Text | None = None
        self.primary_button: tk.Button | None = None
        self.busy_buttons: list[tk.Widget] = []

        self.configure_style()
        self.build_ui()
        self.root.after(100, self.poll_log_queue)
        self.refresh_reviews()

    def configure_style(self) -> None:
        self.root.configure(bg="#f4f6fb")
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Studio.TFrame", background="#f4f6fb")
        style.configure("Panel.TFrame", background="#ffffff", relief="flat")
        style.configure("Muted.TLabel", background="#ffffff", foreground="#64748b", font=("Malgun Gothic", 10))
        style.configure("Title.TLabel", background="#ffffff", foreground="#111827", font=("Malgun Gothic", 18, "bold"))
        style.configure("Section.TLabel", background="#ffffff", foreground="#111827", font=("Malgun Gothic", 12, "bold"))
        style.configure("Status.TLabel", background="#111827", foreground="#ffffff", font=("Malgun Gothic", 10, "bold"), padding=(10, 5))
        style.configure("Treeview", font=("Malgun Gothic", 10), rowheight=28)
        style.configure("Treeview.Heading", font=("Malgun Gothic", 10, "bold"))

    def build_ui(self) -> None:
        outer = ttk.Frame(self.root, style="Studio.TFrame", padding=18)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=3)
        outer.columnconfigure(1, weight=2)
        outer.rowconfigure(1, weight=1)

        hero = ttk.Frame(outer, style="Panel.TFrame", padding=22)
        hero.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        hero.columnconfigure(0, weight=1)
        ttk.Label(hero, text="Long2Shorts Studio", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(hero, textvariable=self.run_note_var, style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.primary_button = tk.Button(
            hero,
            text="오늘 쇼츠 제작",
            command=self.on_daily_run,
            bg="#111827",
            fg="#ffffff",
            activebackground="#1f2937",
            activeforeground="#ffffff",
            relief="flat",
            padx=28,
            pady=14,
            font=("Malgun Gothic", 15, "bold"),
            cursor="hand2",
        )
        self.primary_button.grid(row=0, column=1, rowspan=2, sticky="e", padx=(18, 0))
        self.busy_buttons.append(self.primary_button)

        work = ttk.Frame(outer, style="Panel.TFrame", padding=18)
        work.grid(row=1, column=0, sticky="nsew", padx=(0, 14))
        work.columnconfigure(0, weight=1)
        ttk.Label(work, text="진행 상황", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(work, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=1, sticky="e")
        stage_frame = ttk.Frame(work, style="Panel.TFrame")
        stage_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(14, 18))
        stage_frame.columnconfigure(1, weight=1)
        stages = [
            ("metrics", "성과 확인", "지난 업로드 성과를 확인합니다."),
            ("learning", "학습 규칙", "성과와 피드백으로 오늘의 편집 기준을 정합니다."),
            ("trend", "트렌드 조사", "최근 예능 롱폼 후보를 고릅니다."),
            ("source", "원본 확보", "선정 롱폼을 다운로드하고 전사를 준비합니다."),
            ("package", "후보 생성", "AI가 쇼츠 후보를 만들고 점수화합니다."),
            ("render", "렌더/검토", "MP4를 만들고 검토 대기에 올립니다."),
        ]
        for row, (key, title, note) in enumerate(stages):
            badge = tk.Label(stage_frame, text="대기", width=8, bg="#e5e7eb", fg="#374151", font=("Malgun Gothic", 9, "bold"))
            badge.grid(row=row, column=0, sticky="nw", pady=5)
            self.stage_vars[key] = tk.StringVar(value="대기")
            self.stage_note_vars[key] = tk.StringVar(value=note)
            label = ttk.Label(stage_frame, text=title, style="Section.TLabel")
            label.grid(row=row, column=1, sticky="w", padx=(10, 0), pady=(3, 0))
            note_label = ttk.Label(stage_frame, textvariable=self.stage_note_vars[key], style="Muted.TLabel", wraplength=660)
            note_label.grid(row=row, column=1, sticky="w", padx=(10, 0), pady=(25, 5))
            self.stage_vars[key].trace_add("write", self.make_badge_updater(badge, self.stage_vars[key]))

        log_panel = ttk.Frame(work, style="Panel.TFrame")
        log_panel.grid(row=2, column=0, columnspan=2, sticky="nsew")
        work.rowconfigure(2, weight=1)
        ttk.Label(log_panel, text="실행 로그", style="Section.TLabel").pack(anchor="w")
        self.log_text = tk.Text(log_panel, height=12, wrap="word", bg="#0f172a", fg="#dbeafe", insertbackground="#ffffff", relief="flat", padx=12, pady=10)
        self.log_text.pack(fill="both", expand=True, pady=(8, 0))

        review = ttk.Frame(outer, style="Panel.TFrame", padding=18)
        review.grid(row=1, column=1, sticky="nsew")
        review.columnconfigure(0, weight=1)
        review.rowconfigure(2, weight=1)
        ttk.Label(review, text="검토 대기", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(review, textvariable=self.review_note_var, style="Muted.TLabel").grid(row=1, column=0, sticky="ew", pady=(6, 10))
        columns = ("score", "status", "short", "file")
        self.review_tree = ttk.Treeview(review, columns=columns, show="headings", height=10)
        for col, title, width in [
            ("score", "점수", 58),
            ("status", "상태", 96),
            ("short", "ID", 86),
            ("file", "파일", 260),
        ]:
            self.review_tree.heading(col, text=title)
            self.review_tree.column(col, width=width, anchor="center" if col != "file" else "w")
        self.review_tree.grid(row=2, column=0, sticky="nsew")
        self.review_tree.bind("<<TreeviewSelect>>", self.on_review_selected)

        actions = ttk.Frame(review, style="Panel.TFrame")
        actions.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        for text, command in [
            ("새로고침", self.refresh_reviews),
            ("영상 열기", self.open_selected_review),
            ("승인", lambda: self.review_decision("approved")),
            ("내용 수정요청", lambda: self.review_decision("revision_requested")),
            ("폐기", lambda: self.review_decision("rejected")),
            ("승인본 업로드", self.upload_approved),
        ]:
            button = ttk.Button(actions, text=text, command=command)
            button.pack(side="left", padx=(0, 6), pady=2)
            self.busy_buttons.append(button)

        ttk.Label(review, text="내용 피드백", style="Section.TLabel").grid(row=4, column=0, sticky="w", pady=(18, 6))
        self.feedback_text = tk.Text(review, height=8, wrap="word", bg="#f8fafc", relief="flat", padx=10, pady=8)
        self.feedback_text.grid(row=5, column=0, sticky="ew")
        self.feedback_text.insert(
            "1.0",
            "예: 소재가 약함, 왜 봐야 하는지 불명확함, 제목이 약속한 장면이 늦게 나옴, 리액션/반전이 부족함",
        )

        footer = ttk.Frame(outer, style="Studio.TFrame")
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        ttk.Label(
            footer,
            text="고급 작업은 기존 스크립트와 설정을 그대로 사용합니다. 이 화면은 매일 제작과 검토만 빠르게 처리하도록 정리한 대시보드입니다.",
            background="#f4f6fb",
            foreground="#64748b",
        ).pack(anchor="w")

    def make_badge_updater(self, badge: tk.Label, var: tk.StringVar):
        def update(*_args) -> None:
            value = var.get()
            colors = {
                "대기": ("#e5e7eb", "#374151"),
                "진행": ("#dbeafe", "#1d4ed8"),
                "완료": ("#dcfce7", "#166534"),
                "검토": ("#fef3c7", "#92400e"),
                "실패": ("#fee2e2", "#991b1b"),
            }
            bg, fg = colors.get(value, ("#f3f4f6", "#374151"))
            badge.configure(text=value, bg=bg, fg=fg)
        return update

    def set_stage(self, key: str, status: str, note: str | None = None) -> None:
        if key in self.stage_vars:
            self.stage_vars[key].set(status)
        if note is not None and key in self.stage_note_vars:
            self.stage_note_vars[key].set(note)

    def reset_stages(self) -> None:
        defaults = {
            "metrics": "지난 업로드 성과를 확인합니다.",
            "learning": "성과와 피드백으로 오늘의 편집 기준을 정합니다.",
            "trend": "최근 예능 롱폼 후보를 고릅니다.",
            "source": "선정 롱폼을 다운로드하고 전사를 준비합니다.",
            "package": "AI가 쇼츠 후보를 만들고 점수화합니다.",
            "render": "MP4를 만들고 검토 대기에 올립니다.",
        }
        for key, note in defaults.items():
            self.set_stage(key, "대기", note)

    def set_busy(self, busy: bool) -> None:
        for button in self.busy_buttons:
            try:
                button.configure(state="disabled" if busy else "normal")
            except tk.TclError:
                pass

    def append_log(self, text: str) -> None:
        if not self.log_text:
            return
        self.log_text.insert("end", text.rstrip() + "\n")
        self.log_text.see("end")

    def run_worker(self, args: list[str], label: str, on_done=None) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("작업 중", "이미 실행 중인 작업이 있습니다.")
            return

        def task() -> None:
            self.log_queue.put(("busy", True))
            self.log_queue.put(("status", label))
            lines: list[str] = []
            try:
                process = subprocess.Popen(
                    args,
                    cwd=BASE_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                self.current_process = process
                assert process.stdout is not None
                for line in process.stdout:
                    clean = line.rstrip()
                    if clean:
                        lines.append(clean)
                        self.log_queue.put(("log", clean))
                        self.log_queue.put(("stage", clean))
                code = process.wait()
                if code != 0:
                    raise RuntimeError(f"{label} 실패 (exit {code})")
                if on_done:
                    self.log_queue.put(("done_callback", (on_done, lines)))
                self.log_queue.put(("status", "완료"))
            except Exception as exc:
                self.log_queue.put(("error", str(exc)))
            finally:
                self.current_process = None
                self.log_queue.put(("busy", False))

        self.worker_thread = threading.Thread(target=task, daemon=True)
        self.worker_thread.start()

    def on_daily_run(self) -> None:
        self.reset_stages()
        self.run_note_var.set("트렌드 조사부터 렌더/검토 등록까지 실행 중입니다.")
        self.run_worker(
            [str(self.python_exe), "shorts_orchestrator.py", "--config", str(ORCHESTRATOR_CONFIG_PATH), "daily-run", "--execute"],
            "오늘 쇼츠 제작 중",
            on_done=lambda _lines: self.refresh_reviews(),
        )

    def refresh_reviews(self) -> None:
        def done(lines: list[str]) -> None:
            raw = "\n".join(lines)
            try:
                items = json.loads(raw or "[]")
            except json.JSONDecodeError:
                items = []
            self.populate_reviews(items if isinstance(items, list) else [])

        self.run_worker(
            [str(self.python_exe), "shorts_orchestrator.py", "--config", str(ORCHESTRATOR_CONFIG_PATH), "list-reviews", "--status", "needs_review", "--limit", "30"],
            "검토 목록 새로고침 중",
            on_done=done,
        )

    def populate_reviews(self, items: list[dict]) -> None:
        if not self.review_tree:
            return
        self.review_rows.clear()
        for row in self.review_tree.get_children():
            self.review_tree.delete(row)
        for item in items:
            if not isinstance(item, dict):
                continue
            output_path = str(item.get("output_path") or "")
            iid = output_path or str(item.get("content_sha256") or len(self.review_rows))
            self.review_rows[iid] = item
            self.review_tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    item.get("score", ""),
                    item.get("review_status", ""),
                    item.get("short_id", ""),
                    Path(output_path).name if output_path else "",
                ),
            )
        self.review_note_var.set(f"검토 대기 {len(items)}개")

    def selected_review(self) -> dict | None:
        if not self.review_tree:
            return None
        selection = self.review_tree.selection()
        if not selection:
            return None
        return self.review_rows.get(selection[0])

    def on_review_selected(self, _event=None) -> None:
        item = self.selected_review()
        if not item:
            return
        self.review_note_var.set(f"{Path(str(item.get('output_path') or '')).name} 선택됨")

    def open_selected_review(self) -> None:
        item = self.selected_review()
        if not item:
            messagebox.showinfo("선택 필요", "검토할 영상을 먼저 선택하세요.")
            return
        path = Path(str(item.get("output_path") or ""))
        if not path.exists():
            messagebox.showerror("파일 없음", str(path))
            return
        os.startfile(str(path))

    def review_decision(self, status: str) -> None:
        item = self.selected_review()
        if not item:
            messagebox.showinfo("선택 필요", "검토할 영상을 먼저 선택하세요.")
            return
        note = self.feedback_text.get("1.0", "end").strip() if self.feedback_text else ""
        placeholder = "예: 소재가 약함"
        if status in {"revision_requested", "rejected"} and (not note or placeholder in note):
            messagebox.showinfo("피드백 필요", "내용 수정요청/폐기에는 다음 제작에 반영할 피드백을 적어주세요.")
            return
        if status == "approved" and placeholder in note:
            note = "approved"
        output_path = str(item.get("output_path") or "")
        self.run_worker(
            [
                str(self.python_exe),
                "shorts_orchestrator.py",
                "--config",
                str(ORCHESTRATOR_CONFIG_PATH),
                "review-decision",
                "--output",
                output_path,
                "--status",
                status,
                "--note",
                note or status,
            ],
            "검토 결과 저장 중",
            on_done=lambda _lines: self.refresh_reviews(),
        )

    def upload_approved(self) -> None:
        self.run_worker(
            [str(self.python_exe), "shorts_orchestrator.py", "--config", str(ORCHESTRATOR_CONFIG_PATH), "upload-approved", "--execute"],
            "승인본 업로드 중",
            on_done=lambda _lines: self.refresh_reviews(),
        )

    def update_stage_from_log(self, line: str) -> None:
        if line.startswith("[orchestrator] phase=metrics_sync"):
            self.set_stage("metrics", "진행", "성과 체크 중")
        elif line.startswith("[orchestrator] phase=learning_rule"):
            self.set_stage("metrics", "완료")
            self.set_stage("learning", "진행", "오늘 적용할 편집 규칙 계산 중")
        elif line.startswith("[orchestrator] phase=trend_research"):
            self.set_stage("learning", "완료")
            self.set_stage("trend", "진행", "최근 트렌드 원본 탐색 중")
        elif line.startswith("[orchestrator] phase=source_acquisition"):
            self.set_stage("trend", "완료")
            self.set_stage("source", "진행", "선정 롱폼 다운로드/전사 준비 중")
        elif line.startswith("[orchestrator] phase=source_analysis"):
            self.set_stage("source", "진행", line.split("source=", 1)[-1])
        elif line.startswith("[orchestrator] phase=package_generation"):
            self.set_stage("source", "완료")
            self.set_stage("package", "진행", line.split("source=", 1)[-1])
        elif line.startswith("[orchestrator] phase=render"):
            self.set_stage("package", "완료")
            self.set_stage("render", "진행", "템플릿 MP4 렌더 중")
        elif line.startswith("[orchestrator] phase=review"):
            self.set_stage("render", "진행", "검토 큐 등록 중")
        elif line.startswith("[orchestrator] review_item="):
            self.set_stage("render", "검토", Path(line.split("review_item=", 1)[1].split(" status=", 1)[0]).name)
        elif line.startswith("[orchestrator] status=completed"):
            self.status_var.set("완료")
        elif line.startswith("[orchestrator] status=failed"):
            self.status_var.set("실패")

    def poll_log_queue(self) -> None:
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "busy":
                    self.set_busy(bool(payload))
                elif kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "log":
                    self.append_log(str(payload))
                elif kind == "stage":
                    self.update_stage_from_log(str(payload))
                elif kind == "error":
                    self.status_var.set("실패")
                    self.append_log(str(payload))
                    messagebox.showerror("오류", str(payload))
                elif kind == "done_callback":
                    callback, lines = payload
                    callback(lines)
        except queue.Empty:
            pass
        finally:
            self.root.after(120, self.poll_log_queue)


def main() -> None:
    configure_tcl_library_paths()
    root = tk.Tk()
    ShortsDashboardApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
