"""
transcribe.py -- Accept an audio file and produce a transcription.

This is the primary entry point for the pipeline. It validates,
probes, transcribes, processes, and outputs the result.

Usage:
    python transcribe.py <audio_file>
    python transcribe.py meeting.wav --format json srt
    python transcribe.py podcast.mp3 --language en --output-dir results
    python transcribe.py interview.m4a --backend whisper --whisper-model small
    python transcribe.py recording.wav --mock   # dry-run with mock transcriber

Supported formats: WAV, MP3, M4A, FLAC, OGG, WEBM, WMA, AAC
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe an audio file to text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python transcribe.py meeting.wav\n"
            "  python transcribe.py podcast.mp3 --format json\n"
            "  python transcribe.py interview.m4a --language en --output-dir out/\n"
            "  python transcribe.py recording.wav --mock --verbose\n"
        ),
    )

    p.add_argument(
        "audio_file",
        type=str,
        help="Path to audio file (WAV, MP3, M4A, FLAC, OGG, WEBM, WMA, AAC)",
    )
    p.add_argument(
        "--backend", "-b",
        choices=["whisper", "mock"],
        default="whisper",
        help="Transcription backend (default: whisper). Use --mock for a dry run.",
    )
    p.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base). Larger = slower but more accurate.",
    )
    p.add_argument(
        "--language", "-l",
        default=None,
        help="ISO 639-1 language code, e.g. 'en', 'es', 'fr'. Default: auto-detect.",
    )
    p.add_argument(
        "--format", "-f",
        nargs="+",
        choices=["json", "srt", "txt"],
        default=["json", "srt", "txt"],
        help="Output formats (default: all three).",
    )
    p.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Directory for output files (default: ./output/).",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Print results to stdout only; do not write files.",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Shortcut for --backend mock (dry run without real transcription).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress informational output; errors only.",
    )

    return p


def print_header(audio_path: Path, meta) -> None:
    """Print a clean summary of the audio file before processing."""
    print()
    print("=" * 60)
    print("  AUDIO FILE")
    print("=" * 60)
    print(f"  Path:      {audio_path}")
    print(f"  Format:    {audio_path.suffix.upper()[1:]}")
    print(f"  Size:      {format_file_size(meta.file_size_bytes)}")
    if meta.duration_seconds is not None:
        print(f"  Duration:  {format_duration(meta.duration_seconds)}")
    if meta.sample_rate:
        print(f"  Sample:    {meta.sample_rate} Hz, {meta.channels} ch")
    if meta.codec:
        print(f"  Codec:     {meta.codec}")
    print(f"  Probed:    {meta.probe_method}")
    print("=" * 60)
    print()


def print_result(result, elapsed: float) -> None:
    """Print the transcription results to stdout."""
    p = result.processed

    print()
    print("=" * 60)
    print("  TRANSCRIPTION RESULT")
    print("=" * 60)
    print(f"  Language:      {p.raw.detected_language}")
    print(f"  Duration:      {p.raw.duration_seconds:.1f}s")
    print(f"  Words:         {p.word_count}")
    print(f"  Speaking rate: {p.speaking_rate_wpm:.0f} wpm")
    print(f"  Sentences:     {len(p.sentences)}")
    print(f"  Keywords:      {', '.join(p.keywords)}")
    print(f"  Backend:       {p.raw.backend.value}")
    print(f"  Processing:    {elapsed:.2f}s")
    print("=" * 60)

    print()
    print("FULL TEXT:")
    print("-" * 60)
    print(p.cleaned_text)
    print("-" * 60)

    print()
    print("SENTENCES:")
    for i, sent in enumerate(p.sentences, 1):
        print(f"  {i:3d}. {sent}")

    if result.output_paths:
        print()
        print("OUTPUT FILES:")
        for fmt, path in result.output_paths.items():
            print(f"  {fmt.value:>5s}  {path}")

    print()


def main() -> int:
    args = build_parser().parse_args()

    # Logging
    if args.quiet:
        level = logging.WARNING
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,  # stdout is reserved for results
    )

    # Backend override
    backend = TranscriptionBackend.MOCK if args.mock else TranscriptionBackend(args.backend)

    # Validate audio file
    audio_path = Path(args.audio_file)
    try:
        validated = validate_audio_file(audio_path)
    except AudioValidationError as e:
        print(f"ERROR: {e.reason}", file=sys.stderr)
        print(f"  File: {e.file_path}", file=sys.stderr)
        print(f"  Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}", file=sys.stderr)
        return 1

    # Probe metadata
    meta = probe_audio(validated)
    if not args.quiet:
        print_header(validated, meta)

    # Build config
    config = PipelineConfig(
        backend=backend,
        whisper_model_size=args.whisper_model,
        language=args.language,
        output_dir=Path(args.output_dir),
        output_formats=[OutputFormat(f) for f in args.format],
        write_files=not args.no_write,
    )

    # Run pipeline
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

    # Output -- quiet only suppresses the file-info header, not the result
    print_result(result, elapsed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
