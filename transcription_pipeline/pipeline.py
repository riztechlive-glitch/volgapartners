"""
Pipeline orchestrator — wires together transcription, processing, and output.

This is the single entry point for running the full pipeline.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .audio_utils import (
    AudioValidationError,
    format_duration,
    format_file_size,
    probe_audio,
    validate_audio_file,
)
from .config import PipelineConfig
from .models import AudioInput, AudioMetadata, OutputFormat, PipelineResult
from .output import format_all, write_outputs
from .processor import process_transcript
from .transcriber import get_transcriber

logger = logging.getLogger(__name__)


def run_pipeline(
    audio_path: str | Path,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """
    Execute the full transcription pipeline:

        Audio file -> [Validate] -> [Probe] -> [Transcribe] -> [Process] -> [Format] -> [Write]

    Returns a PipelineResult with all outputs in memory and on disk.

    Raises AudioValidationError if the file is invalid.
    """
    config = config or PipelineConfig()

    # ── Stage 0: Validate + Probe ─────────────────────────────────────────
    validated_path = validate_audio_file(audio_path)
    probed = probe_audio(validated_path)

    metadata = AudioMetadata(
        file_size_bytes=probed.file_size_bytes,
        duration_seconds=probed.duration_seconds,
        sample_rate=probed.sample_rate,
        channels=probed.channels,
        codec=probed.codec,
        probe_method=probed.probe_method,
    )

    audio = AudioInput(
        file_path=validated_path,
        language=config.language,
        metadata=metadata,
    )

    logger.info("=" * 60)
    logger.info("TRANSCRIPTION PIPELINE START")
    logger.info("  File:    %s (%s)", audio.file_path.name, metadata.file_size_display)
    logger.info("  Format:  %s | Duration: %s", metadata.codec or audio.file_path.suffix, metadata.duration_display)
    logger.info("  Audio:   %dHz, %dch (probed via %s)", metadata.sample_rate or 0, metadata.channels or 0, metadata.probe_method)
    logger.info("  Backend: %s", config.backend.value)
    logger.info("  Outputs: %s", [f.value for f in config.output_formats])
    logger.info("=" * 60)

    # ── Stage 1: Transcribe ───────────────────────────────────────────────
    t0 = time.perf_counter()
    transcriber = get_transcriber(config)
    raw_transcript = transcriber.transcribe(audio)
    t_transcribe = time.perf_counter() - t0
    logger.info("Stage 1 [Transcribe]: %.2fs -> %d segments, %d chars",
                t_transcribe, len(raw_transcript.segments), len(raw_transcript.full_text))

    # ── Stage 2: Process ──────────────────────────────────────────────────
    t1 = time.perf_counter()
    processed = process_transcript(raw_transcript, config)
    t_process = time.perf_counter() - t1
    logger.info("Stage 2 [Process]:    %.2fs -> %d words, %d keywords",
                t_process, processed.word_count, len(processed.keywords))

    # ── Stage 3: Format & Write ───────────────────────────────────────────
    t2 = time.perf_counter()
    outputs = format_all(processed, config)
    result = PipelineResult(processed=processed, outputs=outputs)

    if config.write_files:
        output_paths = write_outputs(result, config)
        result.output_paths = output_paths
    else:
        output_paths = {}

    t_format = time.perf_counter() - t2
    logger.info("Stage 3 [Output]:     %.2fs -> %d formats, %d files written",
                t_format, len(outputs), len(output_paths))

    # ── Done ──────────────────────────────────────────────────────────────
    t_total = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE in %.2fs", t_total)
    logger.info("  Summary: %s", processed.summary)
    logger.info("=" * 60)

    return result
