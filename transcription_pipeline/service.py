"""
TranscriptionService -- programmatic API for spoken-language-to-text.

This is the primary importable interface. It wraps the pipeline internals
into a clean, stateful service that:

- Lazily loads the transcription backend (Whisper model stays in memory)
- Accepts file paths or raw bytes
- Returns typed result objects (no side effects unless configured)
- Can be used from any Python code or HTTP framework

Usage:
    from transcription_pipeline.service import TranscriptionService

    svc = TranscriptionService(backend="whisper", whisper_model_size="base")
    result = svc.transcribe_file("meeting.wav")
    print(result.text)

    # Or from raw bytes (e.g., uploaded file):
    result = svc.transcribe_bytes(audio_bytes, filename="clip.mp3")
"""

from __future__ import annotations

import io
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from .audio_utils import AudioValidationError, probe_audio, validate_audio_file
from .config import PipelineConfig
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
from .output import format_all
from .processor import process_transcript
from .transcriber import Transcriber, get_transcriber

logger = logging.getLogger(__name__)


class TranscriptionResult:
    """
    Clean result object returned by TranscriptionService.

    Exposes the most useful fields directly, with full data
    available via .raw, .processed, and .pipeline_result.
    """

    __slots__ = (
        "text", "sentences", "keywords", "language",
        "duration_seconds", "word_count", "speaking_rate_wpm",
        "segments", "elapsed_seconds",
        "raw", "processed", "pipeline_result",
    )

    def __init__(
        self,
        text: str,
        sentences: list[str],
        keywords: list[str],
        language: str,
        duration_seconds: float,
        word_count: int,
        speaking_rate_wpm: float,
        segments: list[TranscriptSegment],
        elapsed_seconds: float,
        raw: RawTranscript,
        processed: ProcessedTranscript,
        pipeline_result: PipelineResult,
    ) -> None:
        self.text = text
        self.sentences = sentences
        self.keywords = keywords
        self.language = language
        self.duration_seconds = duration_seconds
        self.word_count = word_count
        self.speaking_rate_wpm = speaking_rate_wpm
        self.segments = segments
        self.elapsed_seconds = elapsed_seconds
        self.raw = raw
        self.processed = processed
        self.pipeline_result = pipeline_result

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        return {
            "text": self.text,
            "sentences": self.sentences,
            "keywords": self.keywords,
            "language": self.language,
            "duration_seconds": round(self.duration_seconds, 2),
            "word_count": self.word_count,
            "speaking_rate_wpm": round(self.speaking_rate_wpm, 1),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "segments": [
                {
                    "start": round(s.start.total_seconds(), 3),
                    "end": round(s.end.total_seconds(), 3),
                    "text": s.text,
                    "confidence": round(s.confidence, 4),
                }
                for s in self.segments
            ],
        }

    def to_timestamps_dict(self) -> dict[str, Any]:
        """Serialize segments with formatted timestamp strings for display."""
        from .timestamp_formats import fmt_vtt_time

        return {
            "language": self.language,
            "duration_seconds": round(self.duration_seconds, 2),
            "segment_count": len(self.segments),
            "segments": [
                {
                    "index": i,
                    "start": round(s.start.total_seconds(), 3),
                    "end": round(s.end.total_seconds(), 3),
                    "start_time": fmt_vtt_time(s.start),
                    "end_time": fmt_vtt_time(s.end),
                    "text": s.text,
                    "confidence": round(s.confidence, 4),
                }
                for i, s in enumerate(self.segments)
            ],
        }

    def __repr__(self) -> str:
        return (
            f"TranscriptionResult(text={self.text[:50]!r}..., "
            f"language={self.language!r}, words={self.word_count})"
        )


class TranscriptionService:
    """
    Stateful service that transcribes spoken language into text.

    The transcription backend (Whisper model) is loaded once on first use
    and kept in memory for subsequent calls. This avoids the ~2-5s model
    load penalty on every request.

    Parameters:
        backend: "whisper" for real STT, "mock" for testing.
        whisper_model_size: tiny|base|small|medium|large (whisper only).
        language: ISO 639-1 code or None for auto-detect.
        output_formats: which formats to generate (json/srt/txt).
        write_files: if True, also write output files to output_dir.
        output_dir: where to write output files.
    """

    def __init__(
        self,
        backend: str | TranscriptionBackend = "mock",
        whisper_model_size: str = "base",
        language: str | None = None,
        output_formats: list[str] | None = None,
        write_files: bool = False,
        output_dir: str | Path = "output",
    ) -> None:
        if isinstance(backend, str):
            backend = TranscriptionBackend(backend)

        formats = [OutputFormat(f) for f in output_formats] if output_formats else []

        self._config = PipelineConfig(
            backend=backend,
            whisper_model_size=whisper_model_size,
            language=language,
            output_formats=formats or [OutputFormat.JSON],
            write_files=write_files,
            output_dir=Path(output_dir),
            extract_keywords=True,
        )

        # Lazy-loaded transcriber (avoids loading model at __init__ time)
        self._transcriber: Transcriber | None = None

    def _get_transcriber(self) -> Transcriber:
        """Lazy-load the transcription backend."""
        if self._transcriber is None:
            logger.info("Initializing %s transcriber...", self._config.backend.value)
            self._transcriber = get_transcriber(self._config)
        return self._transcriber

    def transcribe_file(
        self,
        file_path: str | Path,
        language: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file on disk.

        Args:
            file_path: path to WAV, MP3, M4A, FLAC, OGG, WEBM, etc.
            language: override ISO 639-1 language code for this call.

        Returns:
            TranscriptionResult with text, segments, metadata.

        Raises:
            AudioValidationError: if the file is invalid.
            FileNotFoundError: if the file does not exist.
        """
        t0 = time.perf_counter()

        # Validate
        validated = validate_audio_file(file_path)
        probed = probe_audio(validated)

        metadata = AudioMetadata(
            file_size_bytes=probed.file_size_bytes,
            duration_seconds=probed.duration_seconds,
            sample_rate=probed.sample_rate,
            channels=probed.channels,
            codec=probed.codec,
            probe_method=probed.probe_method,
        )

        audio = AudioInput(
            file_path=validated,
            language=language or self._config.language,
            metadata=metadata,
        )

        logger.info("Transcribing file: %s (%s, %s)",
                     validated.name, metadata.file_size_display,
                     metadata.duration_display)

        return self._transcribe(audio, t0)

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio from raw bytes (e.g., an uploaded file).

        Writes to a temp file, transcribes, then cleans up.
        The temp file uses the original extension so Whisper
        can detect the format correctly.

        Args:
            audio_bytes: raw audio file content.
            filename: original filename (used for extension detection).
            language: override ISO 639-1 language code for this call.

        Returns:
            TranscriptionResult with text, segments, metadata.
        """
        t0 = time.perf_counter()

        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)

        try:
            return self.transcribe_file(tmp_path, language=language)
        finally:
            tmp_path.unlink(missing_ok=True)

    def transcribe(self, source: str | Path | bytes, **kwargs: Any) -> TranscriptionResult:
        """
        Universal entry point -- auto-detects the source type.

        - str/Path on disk  -> transcribe_file()
        - bytes              -> transcribe_bytes()
        """
        if isinstance(source, bytes):
            return self.transcribe_bytes(source, **kwargs)
        return self.transcribe_file(source, **kwargs)

    def _transcribe(self, audio: AudioInput, t0: float) -> TranscriptionResult:
        """Internal: run transcribe -> process -> format."""
        # Stage 1: Transcribe
        transcriber = self._get_transcriber()
        raw = transcriber.transcribe(audio)

        # Stage 2: Process
        processed = process_transcript(raw, self._config)

        # Stage 3: Format (in-memory, no disk writes unless configured)
        outputs = format_all(processed, self._config)
        result = PipelineResult(processed=processed, outputs=outputs)

        elapsed = time.perf_counter() - t0
        logger.info("Transcription complete in %.2fs: %d words, %d sentences, %d keywords",
                     elapsed, processed.word_count, len(processed.sentences), len(processed.keywords))

        return TranscriptionResult(
            text=processed.cleaned_text,
            sentences=processed.sentences,
            keywords=processed.keywords,
            language=raw.detected_language,
            duration_seconds=raw.duration_seconds,
            word_count=processed.word_count,
            speaking_rate_wpm=processed.speaking_rate_wpm,
            segments=raw.segments,
            elapsed_seconds=elapsed,
            raw=raw,
            processed=processed,
            pipeline_result=result,
        )

    @property
    def backend(self) -> str:
        return self._config.backend.value

    @property
    def is_loaded(self) -> bool:
        return self._transcriber is not None
