"""
Data models for the transcription pipeline.

Uses Pydantic for runtime validation and serialization.
Every stage of the pipeline has a clear input/output contract.
"""

from __future__ import annotations

from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TranscriptionBackend(str, Enum):
    """Available speech-to-text backends."""
    WHISPER = "whisper"
    MOCK = "mock"


class AudioMetadata(BaseModel):
    """Probed metadata about an audio file (populated before transcription)."""
    file_size_bytes: int = 0
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    codec: str | None = None
    probe_method: str = "unknown"

    @property
    def duration_display(self) -> str:
        if self.duration_seconds is None:
            return "unknown"
        m, s = divmod(int(self.duration_seconds), 60)
        return f"{m}m {s}s" if m else f"{s}s"

    @property
    def file_size_display(self) -> str:
        size = self.file_size_bytes
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"


class AudioInput(BaseModel):
    """Represents an audio file entering the pipeline."""
    file_path: Path
    language: str | None = Field(default=None, description="ISO 639-1 code, e.g. 'en'. None = auto-detect.")
    sample_rate: int = Field(default=16000, description="Expected sample rate in Hz.")
    metadata: AudioMetadata = Field(default_factory=AudioMetadata)

    model_config = {"arbitrary_types_allowed": True}


class TranscriptSegment(BaseModel):
    """A single timed segment of transcribed text."""
    start: timedelta
    end: timedelta
    text: str
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class RawTranscript(BaseModel):
    """Output of the transcription stage — unprocessed text + metadata."""
    segments: list[TranscriptSegment]
    full_text: str
    detected_language: str
    duration_seconds: float
    backend: TranscriptionBackend
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessedTranscript(BaseModel):
    """Post-processed transcript with cleaned text, keywords, and summaries."""
    raw: RawTranscript
    cleaned_text: str
    sentences: list[str]
    keywords: list[str]
    word_count: int
    speaking_rate_wpm: float  # words per minute

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "word_count": self.word_count,
            "speaking_rate_wpm": round(self.speaking_rate_wpm, 1),
            "num_sentences": len(self.sentences),
            "num_keywords": len(self.keywords),
            "language": self.raw.detected_language,
            "duration_seconds": round(self.raw.duration_seconds, 2),
        }


class OutputFormat(str, Enum):
    """Supported downstream output formats."""
    JSON = "json"
    SRT = "srt"
    TXT = "txt"


class PipelineResult(BaseModel):
    """Final result bundle — the processed transcript plus all output artifacts."""
    processed: ProcessedTranscript
    outputs: dict[OutputFormat, str] = Field(default_factory=dict)
    output_paths: dict[OutputFormat, Path] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}
