"""Transcription Pipeline -- audio -> text -> structured output."""

from .config import PipelineConfig
from .pipeline import run_pipeline
from .models import (
    AudioInput,
    AudioMetadata,
    OutputFormat,
    PipelineResult,
    ProcessedTranscript,
    RawTranscript,
    TranscriptSegment,
    TranscriptionBackend,
)
from .audio_utils import (
    AudioValidationError,
    SUPPORTED_EXTENSIONS,
    probe_audio,
    validate_audio_file,
)
from .service import TranscriptionService, TranscriptionResult

__all__ = [
    "AudioInput",
    "AudioMetadata",
    "AudioValidationError",
    "OutputFormat",
    "PipelineConfig",
    "PipelineResult",
    "ProcessedTranscript",
    "RawTranscript",
    "SUPPORTED_EXTENSIONS",
    "TranscriptionBackend",
    "TranscriptionResult",
    "TranscriptionService",
    "probe_audio",
    "run_pipeline",
    "validate_audio_file",
]
