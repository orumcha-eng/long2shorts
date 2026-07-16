from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from openai import OpenAI
from pymediainfo import MediaInfo

from env_loader import format_checked_env_paths, load_project_env


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHUNK_SECONDS = 600
TRANSCRIBE_MODEL = "whisper-1"
TRANSCRIBE_LANGUAGE = "ko"
TRANSCRIBE_PROMPT = (
    "This is Korean YouTube variety, celebrity talk, movie, or drama dialogue. "
    "Preserve Korean wording, names, slang, audible speaker turns, laughter, and tense. "
    "Return faithful timestamps for dialogue and narration."
)


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a long-form video into transcript chunks.")
    parser.add_argument("--source-video", type=Path, required=True, help="Source video path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Analysis output directory. Defaults to analysis/<source_stem>.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
        help="Chunk size in seconds.",
    )
    parser.add_argument("--max-chunks", type=int, default=0, help="Limit chunk count for the current run.")
    parser.add_argument("--skip-transcribe", action="store_true", help="Only extract audio chunks.")
    return parser


def default_output_dir(source_video: Path) -> Path:
    return BASE_DIR / "analysis" / source_video.stem


def ensure_dirs(output_dir: Path) -> tuple[Path, Path, Path]:
    chunks_dir = output_dir / "chunks"
    transcripts_dir = output_dir / "transcripts"
    merged_dir = output_dir / "merged"
    for path in [output_dir, chunks_dir, transcripts_dir, merged_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return chunks_dir, transcripts_dir, merged_dir


def get_duration_seconds(path: Path) -> float:
    media = MediaInfo.parse(str(path))
    general = next(track for track in media.tracks if track.track_type == "General")
    return float(general.duration) / 1000.0


def load_client() -> OpenAI:
    load_project_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY not found. Checked: {format_checked_env_paths()}")
    return OpenAI(api_key=api_key)


def chunk_path(chunks_dir: Path, index: int) -> Path:
    return chunks_dir / f"chunk_{index:03d}.mp3"


def transcript_path(transcripts_dir: Path, index: int) -> Path:
    return transcripts_dir / f"chunk_{index:03d}.json"


def extract_audio_chunk(source: Path, start_sec: float, duration_sec: float, output_path: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration_sec:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "48k",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def transcribe_chunk(client: OpenAI, audio_path: Path, transcript_json_path: Path, start_offset_sec: float) -> dict:
    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=f,
            language=TRANSCRIBE_LANGUAGE,
            prompt=TRANSCRIBE_PROMPT,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    raw = transcription.model_dump()
    raw["chunk_file"] = str(audio_path)
    raw["chunk_start_sec"] = start_offset_sec
    raw["chunk_end_sec"] = start_offset_sec + raw.get("duration", 0)

    with open(transcript_json_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    return raw


def merge_transcripts(source_video: Path, merged_dir: Path, chunk_seconds: int, chunk_records: list[dict]) -> None:
    merged_segments = []
    merged_lines = []

    for record in chunk_records:
        offset = float(record["chunk_start_sec"])
        for segment in record.get("segments", []) or []:
            merged_segment = {
                "start": round(float(segment["start"]) + offset, 3),
                "end": round(float(segment["end"]) + offset, 3),
                "text": segment["text"].strip(),
            }
            merged_segments.append(merged_segment)
            merged_lines.append(
                f"[{merged_segment['start']:08.3f} - {merged_segment['end']:08.3f}] {merged_segment['text']}"
            )

    merged_json = {
        "source_video": str(source_video),
        "chunk_seconds": chunk_seconds,
        "segment_count": len(merged_segments),
        "segments": merged_segments,
    }

    with open(merged_dir / "merged_transcript.json", "w", encoding="utf-8") as f:
        json.dump(merged_json, f, ensure_ascii=False, indent=2)

    with open(merged_dir / "merged_transcript.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(merged_lines))


def write_metadata(
    source_video: Path,
    output_dir: Path,
    duration_sec: float,
    chunk_seconds: int,
    processed_chunks: int,
    max_chunks: int,
) -> None:
    metadata = {
        "source_video": str(source_video),
        "duration_sec": round(duration_sec, 3),
        "chunk_seconds": chunk_seconds,
        "processed_chunks": processed_chunks,
        "max_chunks": max_chunks,
        "transcribe_model": TRANSCRIBE_MODEL,
        "language": TRANSCRIBE_LANGUAGE,
    }
    with open(output_dir / "job_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    configure_stdout()
    args = build_parser().parse_args()
    source_video = args.source_video.resolve()
    if not source_video.exists():
        raise FileNotFoundError(f"Source video not found: {source_video}")

    output_dir = args.output_dir.resolve() if args.output_dir else default_output_dir(source_video)
    chunk_seconds = max(30, int(args.chunk_seconds))
    chunks_dir, transcripts_dir, merged_dir = ensure_dirs(output_dir)

    duration_sec = get_duration_seconds(source_video)
    client = None if args.skip_transcribe else load_client()
    chunk_records = []
    total_chunks = int((duration_sec + chunk_seconds - 1) // chunk_seconds)
    requested_max = args.max_chunks if args.max_chunks and args.max_chunks > 0 else total_chunks
    max_chunks = min(requested_max, total_chunks)

    print(f"[analyze] source={source_video}", flush=True)
    print(f"[analyze] duration_sec={duration_sec:.3f}", flush=True)
    print(f"[analyze] total_chunks={total_chunks}, processing={max_chunks}", flush=True)

    for index in range(max_chunks):
        chunk_index = index + 1
        start_sec = index * chunk_seconds
        chunk_duration = min(chunk_seconds, duration_sec - start_sec)
        audio_output = chunk_path(chunks_dir, chunk_index)
        transcript_output = transcript_path(transcripts_dir, chunk_index)
        print(
            f"[analyze] chunk {chunk_index}/{max_chunks} start={start_sec:.1f}s duration={chunk_duration:.1f}s",
            flush=True,
        )

        if not args.skip_transcribe and transcript_output.exists():
            print(f"[analyze] transcript exists -> {transcript_output.name}", flush=True)
            with open(transcript_output, "r", encoding="utf-8") as f:
                record = json.load(f)
            chunk_records.append(record)
            continue

        if not audio_output.exists():
            print(f"[analyze] extracting audio -> {audio_output.name}", flush=True)
            extract_audio_chunk(source_video, start_sec, chunk_duration, audio_output)
        else:
            print(f"[analyze] audio exists -> {audio_output.name}", flush=True)

        if args.skip_transcribe:
            continue

        print(f"[analyze] transcribing -> {transcript_output.name}", flush=True)
        record = transcribe_chunk(client, audio_output, transcript_output, start_sec)
        print(f"[analyze] transcribed -> {transcript_output.name}", flush=True)
        chunk_records.append(record)

    write_metadata(source_video, output_dir, duration_sec, chunk_seconds, max_chunks, requested_max)

    if not args.skip_transcribe and chunk_records:
        print("[analyze] merging transcripts", flush=True)
        merge_transcripts(source_video, merged_dir, chunk_seconds, chunk_records)

    print(f"[analyze] processed_chunks={max_chunks}/{total_chunks}", flush=True)
    print(f"[analyze] output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
