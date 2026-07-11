"""
Output formatter — serializes processed transcripts to downstream formats.

Supported formats:
- JSON:  Machine-readable, ideal for APIs and data pipelines.
- SRT:   Subtitle format for video editors and media players.
- TXT:   Plain text, human-readable.

All formatters are pure functions (input → string) for easy testing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import PipelineConfig
from .models import OutputFormat, PipelineResult, ProcessedTranscript

logger = logging.getLogger(__name__)


# ── JSON ──────────────────────────────────────────────────────────────────────

def format_json(processed: ProcessedTranscript) -> str:
    """Full structured output as JSON."""
    data = {
        "summary": processed.summary,
        "cleaned_text": processed.cleaned_text,
        "sentences": processed.sentences,
        "keywords": processed.keywords,
        "segments": [
            {
                "start": seg.start.total_seconds(),
                "end": seg.end.total_seconds(),
                "text": seg.text,
                "confidence": seg.confidence,
            }
            for seg in processed.raw.segments
        ],
        "metadata": {
            "backend": processed.raw.backend.value,
            "language": processed.raw.detected_language,
            "duration_seconds": processed.raw.duration_seconds,
            **processed.raw.metadata,
        },
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


# ── SRT ───────────────────────────────────────────────────────────────────────

def _format_srt_time(td) -> str:
    """Format timedelta as SRT timestamp: HH:MM:SS,mmm"""
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    millis = int((td.total_seconds() - int(td.total_seconds())) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def format_srt(processed: ProcessedTranscript) -> str:
    """SRT subtitle format — compatible with most video players."""
    lines: list[str] = []
    for i, seg in enumerate(processed.raw.segments, 1):
        lines.append(str(i))
        lines.append(f"{_format_srt_time(seg.start)} --> {_format_srt_time(seg.end)}")
        lines.append(seg.text)
        lines.append("")  # blank line separator
    return "\n".join(lines)


# ── TXT ───────────────────────────────────────────────────────────────────────

def format_txt(processed: ProcessedTranscript) -> str:
    """Human-readable plain text with summary header."""
    header = (
        f"=== TRANSCRIPTION RESULT ===\n"
        f"Language:  {processed.raw.detected_language}\n"
        f"Duration:  {processed.raw.duration_seconds:.1f}s\n"
        f"Words:     {processed.word_count}\n"
        f"Rate:      {processed.speaking_rate_wpm:.0f} wpm\n"
        f"Keywords:  {', '.join(processed.keywords)}\n"
        f"{'=' * 29}\n\n"
    )
    return header + processed.cleaned_text


# ── Dispatcher ────────────────────────────────────────────────────────────────

_FORMATTERS: dict[OutputFormat, callable] = {
    OutputFormat.JSON: format_json,
    OutputFormat.SRT: format_srt,
    OutputFormat.TXT: format_txt,
}


def format_all(processed: ProcessedTranscript, config: PipelineConfig) -> dict[OutputFormat, str]:
    """Run all configured formatters and return {format: content}."""
    results: dict[OutputFormat, str] = {}
    for fmt in config.output_formats:
        formatter = _FORMATTERS.get(fmt)
        if formatter:
            results[fmt] = formatter(processed)
            logger.info("Formatted output: %s (%d bytes)", fmt.value, len(results[fmt]))
    return results


def write_outputs(result: PipelineResult, config: PipelineConfig) -> dict[OutputFormat, Path]:
    """Write formatted outputs to disk. Returns {format: file_path}."""
    config.ensure_output_dir()
    paths: dict[OutputFormat, Path] = {}

    for fmt, content in result.outputs.items():
        file_path = config.output_dir / f"transcript.{fmt.value}"
        file_path.write_text(content, encoding="utf-8")
        paths[fmt] = file_path
        logger.info("Wrote %s -> %s", fmt.value, file_path)

    return paths
