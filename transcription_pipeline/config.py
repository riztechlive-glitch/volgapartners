"""
Pipeline configuration.

Uses environment variables with sensible defaults.
No secrets — this is a local-only pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .models import OutputFormat, TranscriptionBackend


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable config for the full pipeline."""

    # Transcription
    backend: TranscriptionBackend = TranscriptionBackend.MOCK
    whisper_model_size: str = "base"  # tiny | base | small | medium | large
    language: str | None = None       # None = auto-detect

    # Post-processing
    min_confidence: float = 0.5
    extract_keywords: bool = True
    max_keywords: int = 10

    # Output
    output_dir: Path = field(default_factory=lambda: Path("output"))
    output_formats: list[OutputFormat] = field(
        default_factory=lambda: [OutputFormat.JSON, OutputFormat.SRT, OutputFormat.TXT]
    )
    write_files: bool = True          # False = in-memory only, skip disk writes

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
