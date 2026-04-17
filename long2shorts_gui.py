import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from movie_info_lookup import lookup_movie_info


BASE_DIR = Path(__file__).resolve().parent


def get_python_exe() -> Path:
    venv_python = BASE_DIR / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def analysis_dir_for_video(video_path: Path) -> Path:
    return BASE_DIR / "analysis" / video_path.stem


def movie_info_path_for_video(video_path: Path) -> Path:
    return analysis_dir_for_video(video_path) / "movie_info.json"


def packages_json_path(video_path: Path) -> Path:
    return analysis_dir_for_video(video_path) / "shorts_candidates" / "final" / "shorts_packages.json"


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


class Long2ShortsApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Long2Shorts")
        self.root.geometry("1340x900")

        self.python_exe = get_python_exe()
        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None

        self.movie_info: dict | None = None
        self.packages: list[dict] = []
        self.action_buttons: list[ttk.Button] = []

        self.video_path_var = tk.StringVar()
        self.title_var = tk.StringVar()
        self.analysis_dir_var = tk.StringVar()
        self.force_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="대기 중")
        self.phase_var = tk.StringVar(value="아직 작업이 없습니다.")

        self.package_tree: ttk.Treeview | None = None
        self.movie_info_text: tk.Text | None = None
        self.detail_text: tk.Text | None = None
        self.log_text: tk.Text | None = None
        self.progressbar: ttk.Progressbar | None = None

        self._build_ui()
        self.root.after(120, self._poll_log_queue)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        source_frame = ttk.LabelFrame(outer, text="소스 영상", padding=10)
        source_frame.pack(fill="x")

        ttk.Label(source_frame, text="파일").grid(row=0, column=0, sticky="w")
        ttk.Entry(source_frame, textvariable=self.video_path_var).grid(row=0, column=1, sticky="ew", padx=8)
        self._make_button(source_frame, "불러오기", self.on_browse_video).grid(row=0, column=2)

        ttk.Label(source_frame, text="제목").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(source_frame, textvariable=self.title_var).grid(row=1, column=1, sticky="ew", padx=8, pady=(8, 0))
        self._make_button(source_frame, "영화 정보 조회", self.on_lookup_movie_info).grid(row=1, column=2, pady=(8, 0))

        ttk.Label(source_frame, text="분석 폴더").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(source_frame, textvariable=self.analysis_dir_var, state="readonly").grid(
            row=2, column=1, sticky="ew", padx=8, pady=(8, 0)
        )
        ttk.Checkbutton(source_frame, text="강제 재생성", variable=self.force_var).grid(
            row=2, column=2, sticky="w", pady=(8, 0)
        )
        source_frame.columnconfigure(1, weight=1)

        action_frame = ttk.Frame(outer, padding=(0, 10, 0, 10))
        action_frame.pack(fill="x")
        self._make_button(action_frame, "쇼츠 시놉시스 생성", self.on_generate_packages).pack(side="left")
        self._make_button(action_frame, "목록 새로고침", self.on_refresh_packages).pack(side="left", padx=8)
        self._make_button(action_frame, "미리보기 열기", self.on_open_preview).pack(side="left")
        self._make_button(action_frame, "선택 항목 CapCut 만들기", self.on_create_capcut).pack(side="left", padx=(8, 0))

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

        columns = ("rank", "title", "tags", "clarity", "protagonist")
        self.package_tree = ttk.Treeview(candidate_frame, columns=columns, show="headings", height=14)
        self.package_tree.heading("rank", text="순위")
        self.package_tree.heading("title", text="제목")
        self.package_tree.heading("tags", text="재미 태그")
        self.package_tree.heading("clarity", text="독립 이해도")
        self.package_tree.heading("protagonist", text="주인공")
        self.package_tree.column("rank", width=55, anchor="center")
        self.package_tree.column("title", width=420)
        self.package_tree.column("tags", width=190)
        self.package_tree.column("clarity", width=90, anchor="center")
        self.package_tree.column("protagonist", width=90, anchor="center")
        self.package_tree.pack(fill="both", expand=True)
        self.package_tree.bind("<<TreeviewSelect>>", self.on_package_selected)

        info_frame = ttk.LabelFrame(right, text="영화 정보", padding=10)
        info_frame.pack(fill="both", expand=True)
        self.movie_info_text = tk.Text(info_frame, height=18, wrap="word")
        self.movie_info_text.pack(fill="both", expand=True)

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

    def _make_button(self, parent, text: str, command) -> ttk.Button:
        button = ttk.Button(parent, text=text, command=command)
        self.action_buttons.append(button)
        return button

    def set_busy(self, is_busy: bool, phase: str | None = None) -> None:
        state = "disabled" if is_busy else "normal"
        for button in self.action_buttons:
            button.configure(state=state)
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

    def set_video(self, path: Path) -> None:
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
        self.render_movie_info()
        self.packages = []
        self.clear_package_tree()
        self.append_text(self.detail_text, self.describe_cache_status(path))

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
        runtime_path.write_text(json.dumps(pkg, ensure_ascii=False, indent=2), encoding="utf-8")
        return runtime_path

    def clear_package_tree(self) -> None:
        if not self.package_tree:
            return
        for item in self.package_tree.get_children():
            self.package_tree.delete(item)

    def on_browse_video(self) -> None:
        file_path = filedialog.askopenfilename(
            title="영상 파일 선택",
            filetypes=[
                ("Video Files", "*.mp4 *.mkv *.avi *.mov *.wmv"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return
        self.set_video(Path(file_path))
        self.append_log(f"[gui] 파일 선택: {file_path}")
        self.status_var.set("파일 선택 완료")

    def describe_cache_status(self, video_path: Path) -> str:
        package_path = packages_json_path(video_path)
        if package_path.exists():
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

    def on_lookup_movie_info(self) -> None:
        title = self.title_var.get().strip()
        if not title:
            messagebox.showerror("오류", "영화/드라마 제목을 먼저 입력해주세요.")
            return

        def task() -> None:
            self.log_queue.put(("status", "영화 정보 조회 중"))
            self.log_queue.put(("phase", "영화 정보 찾는 중"))
            self.log_queue.put(("log", f"[gui] 영화 정보 조회 시작: {title}"))
            try:
                info = lookup_movie_info(title)
            except Exception as exc:
                self.log_queue.put(("error", f"영화 정보 조회 실패: {exc}"))
                self.log_queue.put(("status", "대기 중"))
                return

            self.movie_info = info
            video_path = self.get_video_path()
            if video_path:
                output_path = movie_info_path_for_video(video_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

            self.log_queue.put(("movie_info", "updated"))
            self.log_queue.put(("log", f"[gui] 영화 정보 조회 완료: {info.get('matched_title') or title}"))
            self.log_queue.put(("status", "대기 중"))

        self.start_worker(task, "영화 정보 조회 중")

    def run_subprocess(self, args: list[str], cwd: Path | None = None) -> None:
        command = list(args)
        if command and Path(command[0]).name.lower().startswith("python") and "-u" not in command[1:3]:
            command.insert(1, "-u")

        self.log_queue.put(("log", "$ " + " ".join(command)))
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
        assert process.stdout is not None
        for line in process.stdout:
            clean = line.rstrip()
            self.log_queue.put(("log", clean))
            if clean.startswith("[analyze]") or clean.startswith("[package]") or clean.startswith("[preview]") or clean.startswith("[capcut]"):
                self.log_queue.put(("phase", clean))
        code = process.wait()
        if code != 0:
            raise RuntimeError(f"명령 실행 실패 (exit {code})")

    def on_generate_packages(self) -> None:
        video_path = self.ensure_ready_video()
        if not video_path:
            return

        analysis_dir = analysis_dir_for_video(video_path)
        source_title = self.title_var.get().strip() or video_path.stem
        movie_info_path = movie_info_path_for_video(video_path) if self.movie_info else None
        force = self.force_var.get()

        def task() -> None:
            self.log_queue.put(("status", "전사/시놉시스 생성 중"))
            self.log_queue.put(("phase", "전사 시작"))
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

        package_path = packages_json_path(video_path)
        if not package_path.exists():
            self.packages = []
            self.clear_package_tree()
            self.append_text(self.detail_text, "아직 생성된 쇼츠 시놉시스가 없습니다.")
            return

        data = json.loads(package_path.read_text(encoding="utf-8"))
        self.packages = data.get("shorts", [])
        self.clear_package_tree()

        for idx, pkg in enumerate(self.packages):
            tags = ", ".join(safe_list(pkg.get("fun_tags")))
            title = f"{pkg.get('title_line1', '')} / {pkg.get('title_line2', '')}"
            clarity = pkg.get("standalone_clarity", "-")
            protagonist = pkg.get("protagonist_presence", "-")
            self.package_tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(pkg.get("global_rank", idx + 1), title, tags, clarity, protagonist),
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
            "title": "제목 스타일",
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

    def format_package_detail(self, pkg: dict) -> str:
        tags = ", ".join(safe_list(pkg.get("fun_tags"), ["미정"]))
        characters = ", ".join(safe_list(pkg.get("main_characters"), ["미정"]))
        selection_pitch = pkg.get("selection_pitch") or "아직 선택 설명이 없습니다."
        hook_line = pkg.get("hook_line") or "아직 핵심 대사가 없습니다."
        clips = pkg.get("source_clips", []) or []
        point_captions = pkg.get("point_captions", []) or []
        narration = pkg.get("narration", []) or []
        capcut_summary = self.describe_capcut_settings(pkg)
        clip_total_sec = sum(max(0.0, float(clip.get("source_end", 0.0)) - float(clip.get("source_start", 0.0))) for clip in clips)

        lines = [
            f"제목 1: {pkg.get('title_line1', '')}",
            f"제목 2: {pkg.get('title_line2', '')}",
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
            f"CapCut 설정: {capcut_summary}",
            "",
            f"핵심 사건: {pkg.get('core_event', '')}",
            f"감정 흐름: {pkg.get('emotion_arc', '')}",
            "",
            "컷 구성:",
        ]

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
            try:
                package_path = self.ensure_runtime_package(video_path, pkg)
                self.run_subprocess(
                    [
                        str(self.python_exe),
                        "build_capcut_from_package.py",
                        "--source-video",
                        str(video_path),
                        "--package",
                        str(package_path),
                    ]
                )
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
                    self.append_log(payload)
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "phase":
                    self.phase_var.set(payload)
                elif kind == "error":
                    self.append_log(payload)
                    messagebox.showerror("오류", payload)
                elif kind == "movie_info":
                    self.render_movie_info()
                elif kind == "refresh_packages":
                    self.on_refresh_packages()
                elif kind == "busy_done":
                    self.set_busy(False)
        except queue.Empty:
            pass
        finally:
            self.root.after(120, self._poll_log_queue)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style()
    if "vista" in style.theme_names():
        style.theme_use("vista")
    app = Long2ShortsApp(root)
    app.append_text(app.movie_info_text, "영화 정보가 아직 없습니다.")
    app.append_text(
        app.detail_text,
        "영상 파일을 선택하면 이 영역에 캐시 상태가 표시됩니다.\n\n"
        "- `목록 새로고침`: 기존 결과 불러오기\n"
        "- `쇼츠 시놉시스 생성`: 새 분석 시작",
    )
    root.mainloop()


if __name__ == "__main__":
    main()
