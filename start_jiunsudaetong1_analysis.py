import argparse
import json
import os
import subprocess
from pathlib import Path

import imageio_ffmpeg
from dotenv import load_dotenv
from openai import OpenAI
from pymediainfo import MediaInfo


BASE_DIR = Path(__file__).resolve().parent
SOURCE_VIDEO = BASE_DIR / "지은수대통1회.mp4"
ENV_PATH = BASE_DIR.parent / "auto_Youtube" / "shorts" / ".env"
OUTPUT_DIR = BASE_DIR / "analysis" / "jiunsudaetong1"
CHUNKS_DIR = OUTPUT_DIR / "chunks"
TRANSCRIPTS_DIR = OUTPUT_DIR / "transcripts"
MERGED_DIR = OUTPUT_DIR / "merged"

CHUNK_SECONDS = 600
TRANSCRIBE_MODEL = "whisper-1"
TRANSCRIBE_LANGUAGE = "ko"
TRANSCRIBE_PROMPT = (
    "This is Korean movie dialogue. Preserve Korean wording, names, slang, and tense. "
    "Return faithful timestamps for dialogue and narration."
)


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, CHUNKS_DIR, TRANSCRIPTS_DIR, MERGED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def get_duration_seconds(path: Path) -> float:
    media = MediaInfo.parse(str(path))
    general = next(track for track in media.tracks if track.track_type == "General")
    return float(general.duration) / 1000.0


def load_client() -> OpenAI:
    load_dotenv(ENV_PATH)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY not found in {ENV_PATH}")
    return OpenAI(api_key=api_key)


def chunk_path(index: int) -> Path:
    return CHUNKS_DIR / f"chunk_{index:03d}.mp3"


def transcript_path(index: int) -> Path:
    return TRANSCRIPTS_DIR / f"chunk_{index:03d}.json"


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


def merge_transcripts(chunk_records: list[dict]) -> None:
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
        "source_video": str(SOURCE_VIDEO),
        "chunk_seconds": CHUNK_SECONDS,
        "segment_count": len(merged_segments),
        "segments": merged_segments,
    }

    with open(MERGED_DIR / "merged_transcript.json", "w", encoding="utf-8") as f:
        json.dump(merged_json, f, ensure_ascii=False, indent=2)

    with open(MERGED_DIR / "merged_transcript.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(merged_lines))


def write_metadata(duration_sec: float, processed_chunks: int, max_chunks: int | None) -> None:
    metadata = {
        "source_video": str(SOURCE_VIDEO),
        "duration_sec": round(duration_sec, 3),
        "chunk_seconds": CHUNK_SECONDS,
        "processed_chunks": processed_chunks,
        "max_chunks": max_chunks,
        "transcribe_model": TRANSCRIBE_MODEL,
        "language": TRANSCRIBE_LANGUAGE,
    }
    with open(OUTPUT_DIR / "job_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chunks", type=int, default=1, help="Limit chunk count for the current run.")
    parser.add_argument("--skip-transcribe", action="store_true", help="Only extract audio chunks.")
    args = parser.parse_args()

    if not SOURCE_VIDEO.exists():
        raise FileNotFoundError(f"Source video not found: {SOURCE_VIDEO}")

    ensure_dirs()
    duration_sec = get_duration_seconds(SOURCE_VIDEO)
    client = None if args.skip_transcribe else load_client()

    chunk_records = []
    total_chunks = int((duration_sec + CHUNK_SECONDS - 1) // CHUNK_SECONDS)
    max_chunks = total_chunks if args.max_chunks is None or args.max_chunks <= 0 else min(args.max_chunks, total_chunks)

    for index in range(max_chunks):
        chunk_index = index + 1
        start_sec = index * CHUNK_SECONDS
        chunk_duration = min(CHUNK_SECONDS, duration_sec - start_sec)
        audio_output = chunk_path(chunk_index)
        transcript_output = transcript_path(chunk_index)

        if not audio_output.exists():
            extract_audio_chunk(SOURCE_VIDEO, start_sec, chunk_duration, audio_output)

        if args.skip_transcribe:
            continue

        if transcript_output.exists():
            with open(transcript_output, "r", encoding="utf-8") as f:
                record = json.load(f)
        else:
            record = transcribe_chunk(client, audio_output, transcript_output, start_sec)
        chunk_records.append(record)

    write_metadata(duration_sec, max_chunks, args.max_chunks)

    if not args.skip_transcribe and chunk_records:
        merge_transcripts(chunk_records)

    print(f"source={SOURCE_VIDEO}")
    print(f"duration_sec={duration_sec:.3f}")
    print(f"processed_chunks={max_chunks}/{total_chunks}")
    print(f"output_dir={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
