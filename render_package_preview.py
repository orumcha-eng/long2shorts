import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from pymediainfo import MediaInfo


BASE_DIR = Path(__file__).resolve().parent


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def has_audio_track(path: Path) -> bool:
    media = MediaInfo.parse(str(path))
    return any(track.track_type == "Audio" for track in media.tracks)


def default_output_path(source_video: Path, package_path: Path) -> Path:
    short_id = package_path.stem
    previews_dir = BASE_DIR / "analysis" / source_video.stem / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    return previews_dir / f"{short_id}_preview.mp4"


def build_filter_complex(clips: list[dict], include_audio: bool) -> str:
    parts = []
    concat_inputs = []
    for idx, clip in enumerate(clips):
        start = float(clip["source_start"])
        end = float(clip["source_end"])
        parts.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{idx}]")
        concat_inputs.append(f"[v{idx}]")
        if include_audio:
            parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{idx}]")
            concat_inputs.append(f"[a{idx}]")
    if include_audio:
        parts.append(f"{''.join(concat_inputs)}concat=n={len(clips)}:v=1:a=1[vout][aout]")
    else:
        parts.append(f"{''.join(concat_inputs)}concat=n={len(clips)}:v=1:a=0[vout]")
    return ";".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a rough preview mp4 from a shorts package.")
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
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

    package = json.loads(package_path.read_text(encoding="utf-8"))
    clips = package.get("source_clips", [])
    if not clips:
        raise RuntimeError("Package has no source_clips.")

    output_path = args.output.resolve() if args.output else default_output_path(source_video, package_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    include_audio = has_audio_track(source_video)
    filter_complex = build_filter_complex(clips, include_audio)

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source_video),
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
    ]
    if include_audio:
        command.extend(["-map", "[aout]"])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if include_audio:
        command.extend(["-c:a", "aac", "-b:a", "128k"])
    command.append(str(output_path))

    subprocess.run(command, check=True)
    print(f"[preview] output={output_path}", flush=True)


if __name__ == "__main__":
    main()
