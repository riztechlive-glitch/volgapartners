"""
Demo script — runs the full pipeline with mock data.

Usage:
    python demo.py              # mock mode (no audio needed)
    python demo.py --file audio.mp3  # real audio (requires openai-whisper)

Output files are written to ./output/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from transcription_pipeline import (
    OutputFormat,
    PipelineConfig,
    TranscriptionBackend,
    run_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcription Pipeline Demo")
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to audio file. If omitted, uses mock data.",
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["mock", "whisper"],
        default="mock",
        help="Transcription backend (default: mock)",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--language", "-l",
        default=None,
        help="ISO 639-1 language code (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["json", "srt", "txt"],
        default=["json", "srt", "txt"],
        help="Output formats (default: all)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # ── Logging ───────────────────────────────────────────────────────────
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    # ── Config ────────────────────────────────────────────────────────────
    output_formats = [OutputFormat(f) for f in args.formats]
    backend = TranscriptionBackend(args.backend)

    config = PipelineConfig(
        backend=backend,
        whisper_model_size=args.whisper_model,
        language=args.language,
        output_dir=Path(args.output_dir),
        output_formats=output_formats,
    )

    # ── Validate input ────────────────────────────────────────────────────
    if args.file:
        audio_path = Path(args.file)
        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
            sys.exit(1)
        if backend == TranscriptionBackend.MOCK:
            config = PipelineConfig(
                backend=TranscriptionBackend.WHISPER,
                whisper_model_size=args.whisper_model,
                language=args.language,
                output_dir=Path(args.output_dir),
                output_formats=output_formats,
            )
    else:
        audio_path = Path("mock_audio.wav")  # dummy path for mock mode

    # ── Run pipeline ──────────────────────────────────────────────────────
    print()
    result = run_pipeline(audio_path, config)

    # ── Print results ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nSummary:")
    for key, val in result.processed.summary.items():
        print(f"   {key:.<25} {val}")

    print(f"\nKeywords: {', '.join(result.processed.keywords)}")

    print(f"\nSentences ({len(result.processed.sentences)}):")
    for i, sent in enumerate(result.processed.sentences, 1):
        print(f"   {i}. {sent[:80]}{'...' if len(sent) > 80 else ''}")

    print(f"\nOutput files:")
    for fmt, path in result.output_paths.items():
        print(f"   {fmt.value:.<10} {path}")

    # Show JSON preview
    if OutputFormat.JSON in result.outputs:
        data = json.loads(result.outputs[OutputFormat.JSON])
        print(f"\nJSON preview (summary):")
        print(json.dumps(data["summary"], indent=4))

    print()


if __name__ == "__main__":
    main()
