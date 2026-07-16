from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


BASE_DIR = Path(__file__).resolve().parent


def load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected an object in {path}")
    return raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect a reusable public YouTube reference set for shorts-selection benchmarking."
    )
    parser.add_argument(
        "--collection",
        type=Path,
        required=True,
        help="Collection JSON with public YouTube URLs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for reference contexts and the collection summary.",
    )
    parser.add_argument("--max-comments", type=int, default=300)
    parser.add_argument(
        "--prepare-transcripts",
        action="store_true",
        help="Fetch public captions, or audio plus local transcription when captions are unavailable.",
    )
    parser.add_argument(
        "--captions-only",
        action="store_true",
        help="Collect only downloadable public captions; never download audio or invoke transcription.",
    )
    parser.add_argument(
        "--download-video",
        action="store_true",
        help="Also download source video files for visual annotation. Keep false for metadata/comment-only collection.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write a partial summary instead of stopping after a failed source.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Do not call YouTube again when the source already has a youtube_context.json file.",
    )
    return parser


def context_summary(context_path: Path, item: dict[str, Any]) -> dict[str, Any]:
    if not context_path.exists():
        return {
            "label": item.get("label", ""),
            "source_url": item.get("url", ""),
            "status": "failed",
            "error": "youtube_context.json was not created",
        }

    context = load_json(context_path)
    metadata = context.get("metadata") if isinstance(context.get("metadata"), dict) else {}
    comments = context.get("comments") if isinstance(context.get("comments"), dict) else {}
    insights = context.get("comment_insights") if isinstance(context.get("comment_insights"), dict) else {}
    transcript = context.get("transcript_preparation") if isinstance(context.get("transcript_preparation"), dict) else {}
    transcript_status = str(transcript.get("status", "not_requested"))
    status = "ready" if transcript_status in {"ready", "not_requested"} else "partial"
    return {
        "label": item.get("label", ""),
        "source_url": item.get("url", ""),
        "video_id": context.get("video_id", ""),
        "expected_engine": item.get("expected_engine", ""),
        "status": status,
        "metadata": {
            key: metadata.get(key, "")
            for key in ("title", "channel_title", "published_at", "duration", "view_count", "like_count", "comment_count")
        },
        "comment_signal": {
            "fetched_count": comments.get("fetched_count", 0),
            "timecode_moment_count": len(insights.get("timecode_moments", []) or []),
            "top_topics": [
                item.get("topic", "")
                for item in (insights.get("top_comment_topics", []) or [])[:8]
                if isinstance(item, dict) and item.get("topic")
            ],
        },
        "transcript": {
            "status": transcript.get("status", "not_requested"),
            "source": transcript.get("source", ""),
            "merged_transcript": transcript.get("merged_transcript", ""),
        },
        "context_path": str(context_path),
    }


def main() -> None:
    args = build_parser().parse_args()
    collection_path = args.collection.resolve()
    collection = load_json(collection_path)
    videos = collection.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("Collection must contain a non-empty videos array.")

    output_dir = (args.output_dir or BASE_DIR / "analysis" / "reference_collections" / collection_path.stem).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for index, item in enumerate(videos, start=1):
        if not isinstance(item, dict) or not item.get("url"):
            raise ValueError(f"Invalid collection video at index {index}")
        label = str(item.get("label") or f"reference_{index:02d}")
        item_dir = output_dir / label
        command = [
            sys.executable,
            str(BASE_DIR / "collect_youtube_context.py"),
            "--url",
            str(item["url"]),
            "--output-dir",
            str(item_dir),
            "--max-comments",
            str(max(0, args.max_comments)),
        ]
        if args.prepare_transcripts or args.captions_only:
            command.append("--prepare-transcript")
        if args.captions_only:
            command.append("--captions-only")
        if args.download_video:
            command.append("--download-video")

        reuse_existing = args.reuse_existing and (item_dir / "youtube_context.json").exists()
        if reuse_existing:
            print(f"[reference] reusing {index}/{len(videos)} -> {label}", flush=True)
            result = subprocess.CompletedProcess(command, returncode=0)
        else:
            print(f"[reference] collecting {index}/{len(videos)} -> {label}", flush=True)
            result = subprocess.run(command, cwd=BASE_DIR, check=False)
        summary = context_summary(item_dir / "youtube_context.json", item)
        if result.returncode != 0:
            summary["status"] = "failed"
            summary["error"] = f"collector exited with code {result.returncode}"
        summaries.append(summary)
        if result.returncode != 0 and not args.continue_on_error:
            break

    payload = {
        "collection_id": collection.get("collection_id", collection_path.stem),
        "profile_path": collection.get("profile_path", ""),
        "purpose": collection.get("purpose", ""),
        "sources": summaries,
    }
    summary_path = output_dir / "reference_collection_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ready_count = sum(item.get("status") == "ready" for item in summaries)
    partial_count = sum(item.get("status") == "partial" for item in summaries)
    print(f"[reference] collected={ready_count}/{len(summaries)} partial={partial_count}", flush=True)
    print(f"[reference] summary={summary_path}", flush=True)
    if ready_count != len(videos) and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
