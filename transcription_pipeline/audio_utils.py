"""
Audio utilities -- validation, probing, and format detection.

Probing strategy (graceful degradation):
1. ffprobe  -- best: duration, sample rate, channels, codec, bitrate
2. wave     -- built-in Python: duration + sample rate for WAV only
3. fallback -- file extension + size check (no duration info)

Design rationale:
- ffprobe requires ffmpeg, which is common but not guaranteed.
- Python's wave module covers WAV natively with zero dependencies.
- For MP3/other formats without ffprobe, we still validate the file
  exists and is non-empty, then let the transcription backend
  (Whisper) handle the actual decoding via its own ffmpeg dependency.
"""

from __future__ import annotations

import json
import logging
import os
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Supported formats ─────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".wma", ".aac",
})

# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AudioMetadata:
    """Probed metadata about an audio file."""
    file_path: Path
    file_size_bytes: int
    extension: str
    duration_seconds: float | None  # None if probing failed
    sample_rate: int | None
    channels: int | None
    codec: str | None
    probe_method: str  # "ffprobe" | "wave" | "file_check"


@dataclass
class AudioValidationError(Exception):
    """Raised when an audio file fails validation."""
    file_path: str
    reason: str


# ── Validation ────────────────────────────────────────────────────────────────


def validate_audio_file(file_path: str | Path) -> Path:
    """
    Validate that an audio file exists, is readable, and has a
    supported extension. Raises AudioValidationError on failure.
    """
    path = Path(file_path)

    if not path.exists():
        raise AudioValidationError(str(path), "File does not exist.")

    if not path.is_file():
        raise AudioValidationError(str(path), "Path is not a file.")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise AudioValidationError(
            str(path),
            f"Unsupported format '{ext}'. Supported: {supported}",
        )

    file_size = path.stat().st_size
    if file_size == 0:
        raise AudioValidationError(str(path), "File is empty (0 bytes).")

    # Check read permissions
    try:
        with open(path, "rb") as f:
            f.read(1)
    except PermissionError:
        raise AudioValidationError(str(path), "File is not readable (permission denied).")

    return path


# ── Probing ───────────────────────────────────────────────────────────────────


def _probe_with_ffprobe(path: Path) -> AudioMetadata | None:
    """Use ffprobe (ships with ffmpeg) for full metadata extraction."""
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)

        # Find the first audio stream
        audio_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                audio_stream = stream
                break

        fmt = data.get("format", {})
        duration = float(fmt.get("duration", 0)) or None
        sample_rate = int(audio_stream.get("sample_rate", 0)) or None if audio_stream else None
        channels = int(audio_stream.get("channels", 0)) or None if audio_stream else None
        codec = audio_stream.get("codec_name") if audio_stream else None

        return AudioMetadata(
            file_path=path,
            file_size_bytes=path.stat().st_size,
            extension=path.suffix.lower(),
            duration_seconds=duration,
            sample_rate=sample_rate,
            channels=channels,
            codec=codec,
            probe_method="ffprobe",
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        return None


def _probe_wav_file(path: Path) -> AudioMetadata | None:
    """Use Python's built-in wave module for WAV files."""
    if path.suffix.lower() != ".wav":
        return None

    try:
        import wave

        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
            duration = num_frames / sample_rate if sample_rate > 0 else None

            return AudioMetadata(
                file_path=path,
                file_size_bytes=path.stat().st_size,
                extension=path.suffix.lower(),
                duration_seconds=duration,
                sample_rate=sample_rate,
                channels=channels,
                codec="pcm_s16le",  # standard WAV codec
                probe_method="wave",
            )
    except Exception:
        return None


def _probe_file_check(path: Path) -> AudioMetadata:
    """Minimal fallback: just file size + extension."""
    return AudioMetadata(
        file_path=path,
        file_size_bytes=path.stat().st_size,
        extension=path.suffix.lower(),
        duration_seconds=None,
        sample_rate=None,
        channels=None,
        codec=None,
        probe_method="file_check",
    )


def probe_audio(file_path: str | Path) -> AudioMetadata:
    """
    Probe an audio file for metadata. Tries ffprobe, then wave,
    then falls back to basic file info.
    """
    path = Path(file_path)

    # Try ffprobe first (richest metadata)
    meta = _probe_with_ffprobe(path)
    if meta:
        logger.info("Probed via ffprobe: %.1fs, %dHz, %dch",
                     meta.duration_seconds or 0, meta.sample_rate or 0, meta.channels or 0)
        return meta

    # Try Python wave module (WAV only)
    meta = _probe_wav_file(path)
    if meta:
        logger.info("Probed via wave module: %.1fs, %dHz, %dch",
                     meta.duration_seconds or 0, meta.sample_rate or 0, meta.channels or 0)
        return meta

    # Fallback
    meta = _probe_file_check(path)
    logger.warning("Limited probe (no ffprobe): file=%s, size=%d bytes. "
                   "Install ffmpeg for full metadata.", path.name, meta.file_size_bytes)
    return meta


# ── Helpers ───────────────────────────────────────────────────────────────────


def format_duration(seconds: float | None) -> str:
    """Human-readable duration: '1m 23s' or 'N/A'."""
    if seconds is None:
        return "N/A"
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def format_file_size(size_bytes: int) -> str:
    """Human-readable file size: '1.5 MB'."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
