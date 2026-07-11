"""
Transcription engine — converts audio to text.

Supports two backends:
1. OpenAI Whisper (local, open-source) — real speech-to-text
2. Mock — generates realistic fake transcripts for testing/development

The abstract base class makes it trivial to swap in a new backend
(e.g., Faster-Whisper, speech_recognition, a cloud API).
"""

from __future__ import annotations

import abc
import logging
import re
from pathlib import Path

from .models import (
    AudioInput,
    RawTranscript,
    TranscriptSegment,
    TranscriptionBackend,
)
from .config import PipelineConfig

logger = logging.getLogger(__name__)


class Transcriber(abc.ABC):
    """Abstract base for all transcription backends."""

    @abc.abstractmethod
    def transcribe(self, audio: AudioInput) -> RawTranscript:
        ...


class WhisperTranscriber(Transcriber):
    """
    Real transcription using OpenAI's open-source Whisper model.

    Requires: pip install openai-whisper
    Runs entirely locally — no API key needed.
    """

    def __init__(self, model_size: str = "base") -> None:
        try:
            import whisper  # type: ignore[import-untyped]

            logger.info("Loading Whisper model (size=%s)...", model_size)
            self._model = whisper.load_model(model_size)
            logger.info("Whisper model loaded.")
        except ImportError:
            raise ImportError(
                "whisper not installed. Run: pip install openai-whisper"
            ) from None

    def transcribe(self, audio: AudioInput) -> RawTranscript:
        logger.info("Transcribing %s with Whisper...", audio.file_path)

        options: dict = {"fp16": False}
        if audio.language:
            options["language"] = audio.language

        result = self._model.transcribe(str(audio.file_path), **options)

        segments = []
        for seg in result.get("segments", []):
            segments.append(
                TranscriptSegment(
                    start=__import__("datetime").timedelta(seconds=seg["start"]),
                    end=__import__("datetime").timedelta(seconds=seg["end"]),
                    text=seg["text"].strip(),
                    confidence=seg.get("avg_logprob", 0.0),
                )
            )

        full_text = " ".join(s.text for s in segments)
        # Approximate duration from last segment end
        duration = segments[-1].end.total_seconds() if segments else 0.0

        return RawTranscript(
            segments=segments,
            full_text=full_text,
            detected_language=result.get("language", audio.language or "en"),
            duration_seconds=duration,
            backend=TranscriptionBackend.WHISPER,
            metadata={"model_size": self._model.model_dims},
        )


class MockTranscriber(Transcriber):
    """
    Deterministic mock for testing the pipeline without real audio.

    Produces realistic-looking transcripts with timestamps so the
    full pipeline can be exercised end-to-end.
    """

    MOCK_TRANSCRIPTS: list[str] = [
        "Welcome everyone to today's quarterly business review. "
        "We'll be covering our key performance metrics and strategic initiatives.",

        "Our revenue grew by fifteen percent year over year, reaching "
        "four point two million dollars. Customer acquisition cost decreased "
        "by eight percent, which is a strong signal for our growth efficiency.",

        "On the product side, we shipped three major features this quarter: "
        "real-time collaboration, automated reporting, and the new dashboard. "
        "User engagement increased by twenty two percent across all features.",

        "Looking ahead, our priorities for next quarter include expanding "
        "into the European market, launching our mobile application, and "
        "improving our machine learning capabilities for content recommendations.",

        "Before we wrap up, I want to highlight our team's incredible work. "
        "Employee satisfaction scores are at an all-time high of ninety one percent. "
        "Thank you all for your dedication and hard work.",
    ]

    def __init__(self) -> None:
        logger.info("Using mock transcriber (no real audio processing).")

    def transcribe(self, audio: AudioInput) -> RawTranscript:
        """Generate a mock transcript as if we processed real audio."""
        logger.info("Generating mock transcript for %s", audio.file_path)

        segments: list[TranscriptSegment] = []
        current_time = 0.0

        for i, text in enumerate(self.MOCK_TRANSCRIPTS):
            # ~150 words per minute speaking rate → ~4 sec per sentence
            words = len(text.split())
            duration = (words / 150) * 60
            confidence = 0.85 + (i * 0.025)  # slight variation

            segments.append(
                TranscriptSegment(
                    start=__import__("datetime").timedelta(seconds=current_time),
                    end=__import__("datetime").timedelta(seconds=current_time + duration),
                    text=text,
                    confidence=min(confidence, 1.0),
                )
            )
            current_time += duration

        full_text = " ".join(s.text for s in segments)

        return RawTranscript(
            segments=segments,
            full_text=full_text,
            detected_language="en",
            duration_seconds=current_time,
            backend=TranscriptionBackend.MOCK,
            metadata={"mock": True, "num_mock_segments": len(segments)},
        )


# ---------- Factory ----------

def get_transcriber(config: PipelineConfig) -> Transcriber:
    """Create the appropriate transcriber based on config."""
    match config.backend:
        case TranscriptionBackend.WHISPER:
            return WhisperTranscriber(model_size=config.whisper_model_size)
        case TranscriptionBackend.MOCK:
            return MockTranscriber()
        case _:
            raise ValueError(f"Unknown backend: {config.backend}")
