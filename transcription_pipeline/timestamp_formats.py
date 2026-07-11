"""
Timestamped transcription formats.

Each formatter takes a list of TranscriptSegment objects and returns
a string in the target format. All are pure functions for easy testing.

Supported formats:
- json:      Structured JSON with segments array (timestamps in seconds + formatted)
- srt:       SubRip -- universally supported by video players/editors
- vtt:       WebVTT -- W3C standard for HTML5 video captions
- labeled:   [HH:MM:SS] Spoken text... (human-readable, one line per segment)
- tsv:       Tab-separated values (importable into Excel/Sheets)

Design rationale:
- Separate from output.py because output.py handles the full-pipeline
  JSON/TXT/SRT which bundles metadata, keywords, etc. This module is
  focused exclusively on segment-level timestamped output.
- Pure functions: take segments in, return strings out. No I/O, no config.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from .models import TranscriptSegment


# ── Shared timestamp formatters ───────────────────────────────────────────────


def fmt_srt_time(td: timedelta) -> str:
    """HH:MM:SS,mmm (SRT standard)"""
    total = int(td.total_seconds())
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    ms = int((td.total_seconds() - total) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_vtt_time(td: timedelta) -> str:
    """HH:MM:SS.mmm (WebVTT standard -- dot instead of comma)"""
    total = int(td.total_seconds())
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    ms = int((td.total_seconds() - total) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def fmt_label_time(td: timedelta) -> str:
    """[HH:MM:SS] -- short label format for readability."""
    total = int(td.total_seconds())
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    return f"[{h:02d}:{m:02d}:{s:02d}]"


# ── JSON (segments-only) ─────────────────────────────────────────────────────


def format_json_segments(segments: list[TranscriptSegment]) -> str:
    """
    JSON array of segment objects with both raw seconds and formatted strings.

    Output:
    [
      {
        "index": 0,
        "start": 0.0,
        "end": 6.8,
        "start_time": "00:00:00.000",
        "end_time": "00:00:06.800",
        "text": "Welcome everyone...",
        "confidence": 0.85
      },
      ...
    ]
    """
    data = []
    for i, seg in enumerate(segments):
        data.append({
            "index": i,
            "start": round(seg.start.total_seconds(), 3),
            "end": round(seg.end.total_seconds(), 3),
            "start_time": fmt_vtt_time(seg.start),
            "end_time": fmt_vtt_time(seg.end),
            "text": seg.text,
            "confidence": round(seg.confidence, 4),
        })
    return json.dumps(data, indent=2, ensure_ascii=False)


# ── SRT ───────────────────────────────────────────────────────────────────────


def format_srt(segments: list[TranscriptSegment]) -> str:
    """
    SubRip (.srt) -- the most widely supported subtitle format.

    1
    00:00:00,000 --> 00:00:06,800
    Welcome everyone to today's quarterly business review.

    2
    00:00:06,800 --> 00:00:19,200
    Our revenue grew by fifteen percent year over year...
    """
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{fmt_srt_time(seg.start)} --> {fmt_srt_time(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


# ── WebVTT ────────────────────────────────────────────────────────────────────


def format_vtt(segments: list[TranscriptSegment]) -> str:
    """
    WebVTT (.vtt) -- W3C standard for HTML5 video captions.

    Nearly identical to SRT but with:
    - "WEBVTT" header
    - Dot instead of comma for milliseconds
    - No sequence numbers required
    - Optional per-cue styling

    WEBVTT

    00:00:00.000 --> 00:00:06.800
    Welcome everyone to today's quarterly business review.

    00:00:06.800 --> 00:00:19.200
    Our revenue grew by fifteen percent year over year...
    """
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{fmt_vtt_time(seg.start)} --> {fmt_vtt_time(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


# ── Labeled transcript ────────────────────────────────────────────────────────


def format_labeled(segments: list[TranscriptSegment]) -> str:
    """
    Human-readable labeled format -- timestamp prefix per segment.

    [00:00:00] Welcome everyone to today's quarterly business review.
    We'll be covering our key performance metrics and strategic initiatives.
    [00:00:06] Our revenue grew by fifteen percent year over year, reaching
    four point two million dollars...
    """
    lines: list[str] = []
    for seg in segments:
        lines.append(f"{fmt_label_time(seg.start)} {seg.text}")
    return "\n".join(lines)


# ── Tab-separated values ──────────────────────────────────────────────────────


def format_tsv(segments: list[TranscriptSegment]) -> str:
    """
    Tab-separated values -- importable into Excel, Google Sheets, pandas.

    index\tstart\tend\tstart_time\tend_time\ttext\tconfidence
    0\t0.0\t6.8\t00:00:00.000\t00:00:06.800\tWelcome everyone...\t0.85
    """
    header = "index\tstart\tend\tstart_time\tend_time\ttext\tconfidence"
    rows = [header]
    for i, seg in enumerate(segments):
        row = "\t".join([
            str(i),
            str(round(seg.start.total_seconds(), 3)),
            str(round(seg.end.total_seconds(), 3)),
            fmt_vtt_time(seg.start),
            fmt_vtt_time(seg.end),
            seg.text.replace("\t", " "),  # escape tabs in text
            str(round(seg.confidence, 4)),
        ])
        rows.append(row)
    return "\n".join(rows)


# ── Registry ──────────────────────────────────────────────────────────────────

TIMESTAMP_FORMATTERS: dict[str, Any] = {
    "json": format_json_segments,
    "srt": format_srt,
    "vtt": format_vtt,
    "labeled": format_labeled,
    "tsv": format_tsv,
}

SUPPORTED_TIMESTAMP_FORMATS = list(TIMESTAMP_FORMATTERS.keys())


def format_timestamps(
    segments: list[TranscriptSegment],
    fmt: str,
) -> str:
    """
    Format segments into a specific timestamp format.

    Args:
        segments: list of TranscriptSegment with start/end times.
        fmt: one of "json", "srt", "vtt", "labeled", "tsv".

    Returns:
        Formatted string.
    """
    formatter = TIMESTAMP_FORMATTERS.get(fmt)
    if formatter is None:
        raise ValueError(
            f"Unknown timestamp format '{fmt}'. "
            f"Supported: {SUPPORTED_TIMESTAMP_FORMATS}"
        )
    return formatter(segments)
