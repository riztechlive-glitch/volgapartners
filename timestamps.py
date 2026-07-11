"""
timestamps.py -- Transcribe audio and output timestamps per segment.

This script focuses exclusively on timestamped transcription output.
It accepts an audio file and produces segment-level timestamps in
multiple formats suitable for video editing, data analysis, or display.

Usage:
    python timestamps.py <audio_file>
    python timestamps.py meeting.wav --format srt
    python timestamps.py podcast.mp3 --format vtt --output-dir subs/
    python timestamps.py interview.m4a --format json labeled
    python timestamps.py recording.wav --mock --format tsv

Output formats:
    json     -- Structured JSON with start/end in seconds + formatted strings
    srt      -- SubRip (.srt) -- universal subtitle format
    vtt      -- WebVTT (.vtt) -- W3C standard for HTML5 video
    labeled  -- [HH:MM:SS] Spoken text... (human-readable)
    tsv      -- Tab-separated values (Excel/Sheets import)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from transcription_pipeline import (
    OutputFormat,
    PipelineConfig,
    TranscriptionBackend,
    run_pipeline,
)
from transcription_pipeline.audio_utils import (
    AudioValidationError,
    SUPPORTED_EXTENSIONS,
    format_duration,
    format_file_size,
    probe_audio,
    validate_audio_file,
)
from transcription_pipeline.timestamp_formats import (
    SUPPORTED_TIMESTAMP_FORMATS,
    format_timestamps,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="timestamps",
        description="Transcribe audio with timestamps per segment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python timestamps.py meeting.wav\n"
            "  python timestamps.py meeting.wav --format srt vtt\n"
            "  python timestamps.py podcast.mp3 --format labeled\n"
            "  python timestamps.py interview.m4a --format tsv --output-dir data/\n"
            "  python timestamps.py recording.wav --mock --format json\n"
        ),
    )

    p.add_argument(
        "audio_file",
        type=str,
        help="Path to audio file (WAV, MP3, M4A, FLAC, OGG, WEBM, WMA, AAC)",
    )
    p.add_argument(
        "--format", "-f",
        nargs="+",
        choices=SUPPORTED_TIMESTAMP_FORMATS,
        default=["json"],
        help="Output format(s) (default: json). Can specify multiple.",
    )
    p.add_argument(
        "--backend", "-b",
        choices=["whisper", "mock"],
        default="whisper",
        help="Transcription backend (default: whisper). Use --mock for dry run.",
    )
    p.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base).",
    )
    p.add_argument(
        "--language", "-l",
        default=None,
        help="ISO 639-1 language code. Default: auto-detect.",
    )
    p.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Output directory (default: ./output/).",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Print to stdout only; do not write files.",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Shortcut for --backend mock (dry run).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress header/info output.",
    )

    return p


def get_extension(fmt: str) -> str:
    """Map format name to file extension."""
    return {"json": "json", "srt": "srt", "vtt": "vtt", "labeled": "txt", "tsv": "tsv"}[fmt]


def main() -> int:
    args = build_parser().parse_args()

    # Logging
    level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    # Backend
    backend = TranscriptionBackend.MOCK if args.mock else TranscriptionBackend(args.backend)

    # Validate
    audio_path = Path(args.audio_file)
    try:
        validated = validate_audio_file(audio_path)
    except AudioValidationError as e:
        print(f"ERROR: {e.reason}", file=sys.stderr)
        return 1

    # Probe
    meta = probe_audio(validated)
    if not args.quiet:
        print(f"\n  File:     {validated.name}")
        print(f"  Format:   {validated.suffix.upper()[1:]} | {format_file_size(meta.file_size_bytes)}")
        if meta.duration_seconds is not None:
            print(f"  Duration: {format_duration(meta.duration_seconds)}")
        print(f"  Backend:  {backend.value}")
        print(f"  Output:   {', '.join(args.format)}")
        print()

    # Run pipeline (reuse existing pipeline, we just reformat the segments)
    config = PipelineConfig(
        backend=backend,
        whisper_model_size=args.whisper_model,
        language=args.language,
        output_formats=[],  # we handle output ourselves
        write_files=False,
    )

    t0 = time.perf_counter()
    try:
        result = run_pipeline(validated, config)
    except AudioValidationError as e:
        print(f"ERROR: {e.reason}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Pipeline error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    elapsed = time.perf_counter() - t0

    segments = result.processed.raw.segments

    if not args.quiet:
        print(f"  Transcribed {len(segments)} segments in {elapsed:.2f}s\n")

    # Format and output
    output_dir = Path(args.output_dir)

    for fmt in args.format:
        formatted = format_timestamps(segments, fmt)

        if not args.quiet:
            header = f"--- {fmt.upper()} {'(' + get_extension(fmt) + ')' if not args.no_write else ''} ---"
            print(header)
            print()

        print(formatted)
        print()

        if not args.no_write:
            output_dir.mkdir(parents=True, exist_ok=True)
            ext = get_extension(fmt)
            out_path = output_dir / f"transcript.{ext}"
            out_path.write_text(formatted, encoding="utf-8")
            if not args.quiet:
                print(f"  Saved: {out_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
