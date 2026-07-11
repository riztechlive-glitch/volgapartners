# Transcription Pipeline

A modular speech-to-text pipeline that converts audio into structured text output. Built with OpenAI's open-source Whisper model, runs entirely locally with no API keys required.

## Quick Start

```bash
pip install pydantic scikit-learn
python transcribe.py test_audio.wav --mock
```

For real transcription:
```bash
pip install openai-whisper
python transcribe.py meeting.mp3
```

## Features

- **Multiple backends** -- OpenAI Whisper (local, open-source) or mock for testing
- **8 audio formats** -- WAV, MP3, M4A, FLAC, OGG, WEBM, WMA, AAC
- **5 timestamp formats** -- JSON, SRT, WebVTT, labeled, TSV
- **3 output formats** -- JSON (structured), SRT (subtitles), TXT (plain text)
- **3 interfaces** -- CLI script, programmatic Python API, HTTP REST API
- **Audio probing** -- Automatic metadata extraction (duration, sample rate, channels) via ffprobe, Python wave module, or file-level fallback
- **Post-processing** -- Text cleaning, sentence segmentation, TF-IDF keyword extraction, speaking rate calculation

## Project Structure

```
transcription_pipeline/
    __init__.py           # Public API exports
    models.py             # Pydantic data models (AudioInput, TranscriptSegment, etc.)
    config.py             # PipelineConfig (frozen dataclass)
    transcriber.py        # Whisper + Mock backends (ABC pattern)
    processor.py          # Text cleaning, sentence splitting, keyword extraction
    output.py             # JSON / SRT / TXT formatters (full-pipeline output)
    audio_utils.py        # Audio validation + metadata probing
    pipeline.py           # Pipeline orchestrator (validate -> probe -> transcribe -> process -> format)
    service.py            # TranscriptionService (stateful, lazy-loading, typed API)
    timestamp_formats.py  # Segment-level timestamped formatters (JSON, SRT, VTT, labeled, TSV)
    api.py                # FastAPI HTTP server with REST endpoints
transcribe.py             # CLI: audio file -> text transcription
timestamps.py             # CLI: audio file -> timestamped segments
demo.py                   # Demo with mock data
requirements.txt          # Dependencies
```

## Installation

### Core (mock mode)

```bash
pip install pydantic scikit-learn
```

### Full (real transcription)

```bash
pip install pydantic scikit-learn openai-whisper
```

### HTTP API

```bash
pip install pydantic scikit-learn openai-whisper fastapi uvicorn python-multipart
```

### Requirements

| Package | Purpose | Required |
|---|---|---|
| `pydantic>=2.0` | Data validation and models | Yes |
| `scikit-learn>=1.3` | TF-IDF keyword extraction | Yes |
| `openai-whisper` | Speech-to-text (local, no API key) | For real transcription |
| `fastapi` | HTTP API server | For REST API |
| `uvicorn` | ASGI server for FastAPI | For REST API |
| `python-multipart` | File upload support | For REST API |

## Usage

### CLI -- Transcribe an audio file

```bash
python transcribe.py meeting.wav
python transcribe.py podcast.mp3 --format json
python transcribe.py interview.m4a --language en --output-dir results/
python transcribe.py recording.wav --backend whisper --whisper-model small
python transcribe.py recording.wav --mock   # dry run
```

Options:
```
--backend whisper|mock     Transcription backend (default: whisper)
--whisper-model            Model size: tiny|base|small|medium|large (default: base)
--language LANG            ISO 639-1 code, e.g. 'en' (default: auto-detect)
--format json srt txt      Output formats (default: all three)
--output-dir DIR           Output directory (default: ./output/)
--no-write                 Print to stdout only, skip file writes
--mock                     Shortcut for --backend mock
--verbose                  Debug-level logging
--quiet                    Suppress informational output
```

### CLI -- Timestamped output

```bash
python timestamps.py meeting.wav --format srt
python timestamps.py podcast.mp3 --format vtt labeled
python timestamps.py interview.m4a --format tsv --output-dir data/
```

Formats:
```
json      Structured JSON with start/end in seconds + formatted strings
srt       SubRip (.srt) -- universal subtitle format
vtt       WebVTT (.vtt) -- W3C standard for HTML5 video
labeled   [HH:MM:SS] Spoken text... (human-readable)
tsv       Tab-separated values (Excel/Sheets import)
```

### Programmatic API

```python
from transcription_pipeline.service import TranscriptionService

svc = TranscriptionService(backend="mock")  # or "whisper"

# From a file
result = svc.transcribe_file("meeting.wav")
print(result.text)          # plain text
print(result.sentences)     # list of sentences
print(result.keywords)      # extracted keywords
print(result.language)      # detected language

# From raw bytes (e.g., uploaded file)
audio_bytes = open("clip.mp3", "rb").read()
result = svc.transcribe_bytes(audio_bytes, filename="clip.mp3")

# Get timestamped segments
timestamps = result.to_timestamps_dict()
for seg in timestamps["segments"]:
    print(f"[{seg['start_time']} -> {seg['end_time']}] {seg['text']}")
```

#### TranscriptionResult fields

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Full cleaned transcript |
| `sentences` | `list[str]` | Sentence-split text |
| `keywords` | `list[str]` | TF-IDF extracted keywords |
| `language` | `str` | Detected ISO 639-1 code |
| `duration_seconds` | `float` | Audio duration |
| `word_count` | `int` | Total words |
| `speaking_rate_wpm` | `float` | Words per minute |
| `segments` | `list[TranscriptSegment]` | Timed segments with start/end |
| `elapsed_seconds` | `float` | Processing time |
| `raw` | `RawTranscript` | Full raw output from backend |
| `processed` | `ProcessedTranscript` | Full processed output |

### HTTP API

```bash
python -m transcription_pipeline.api --backend mock --port 8000
```

#### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | API documentation |
| `GET` | `/health` | Health check + model status |
| `POST` | `/transcribe` | Upload audio -> plain text |
| `POST` | `/transcribe/json` | Upload audio -> full JSON (segments, keywords, metadata) |
| `POST` | `/transcribe/timestamps` | Upload audio -> segments with formatted timestamps |

#### Timestamp format parameter

The `/transcribe/timestamps` endpoint accepts a `format` form field:
- `json` (default) -- structured JSON with formatted timestamp strings
- `srt` -- SubRip subtitle format
- `vtt` -- WebVTT format
- `labeled` -- `[HH:MM:SS] Text...`
- `tsv` -- tab-separated values

#### Example: Upload audio via curl

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@meeting.wav"

curl -X POST http://localhost:8000/transcribe/json \
  -F "file=@meeting.wav"

curl -X POST http://localhost:8000/transcribe/timestamps \
  -F "file=@meeting.wav" \
  -F "format=srt"
```

Interactive API docs are available at `http://localhost:8000/docs` when running with FastAPI.

## Audio Format Handling

### Supported formats

WAV, MP3, M4A, FLAC, OGG, WEBM, WMA, AAC

### Validation pipeline

```
File exists? -> Supported extension? -> Non-empty? -> Readable?
```

### Probing strategy (graceful degradation)

| Priority | Method | Metadata | Requires |
|---|---|---|---|
| 1 | `ffprobe` | Duration, sample rate, channels, codec | ffmpeg |
| 2 | Python `wave` | Duration, sample rate, channels | Nothing (WAV only) |
| 3 | File check | File size + extension | Nothing |

### Decoding

Audio decoding is handled by Whisper via ffmpeg internally:

```
mp3/m4a/flac/ogg/webm/wma/aac
         |
     ffmpeg (inside whisper)
         |
    16kHz mono float32
         |
    Whisper model
         |
    text + timestamps
```

WAV files at 16kHz mono bypass re-encoding entirely.

## Architecture

### Data flow

```
Audio File
    |
    v
[validate_audio_file]  -- extension, existence, size, permissions
    |
    v
[probe_audio]          -- duration, sample rate, channels, codec
    |
    v
[Transcriber]          -- Whisper or Mock backend -> RawTranscript
    |
    v
[processor]            -- clean text, split sentences, extract keywords
    |
    v
[output formatters]    -- JSON, SRT, TXT, VTT, labeled, TSV
    |
    v
TranscriptionResult    -- typed result with all data
```

### Key design decisions

| Decision | Rationale |
|---|---|
| **Pydantic models** | Runtime validation and clear data contracts between pipeline stages |
| **ABC for transcriber** | Swappable backends (Whisper, mock, or future cloud APIs) without touching the pipeline |
| **Lazy model loading** | Whisper model loads on first transcription call, reused after (~0.05s vs ~5s) |
| **Frozen config** | `PipelineConfig` is immutable -- no hidden state mutations |
| **TF-IDF keywords** | Uses scikit-learn for ngram-aware extraction; falls back to frequency-based if unavailable |
| **Three-tier probing** | ffprobe -> wave -> file check ensures WAV files always get full metadata |
| **Separate timestamp module** | `timestamp_formats.py` is independent of the full pipeline -- pure functions for segment formatting |

### Pipeline stages

1. **Validate** -- File exists, readable, supported extension, non-empty
2. **Probe** -- Extract metadata (duration, sample rate, channels, codec)
3. **Transcribe** -- Convert audio to text with timed segments
4. **Process** -- Clean fillers, split sentences, extract keywords, compute stats
5. **Format** -- Serialize to JSON/SRT/TXT/VTT/labeled/TSV
6. **Write** -- Optionally write output files to disk

## Configuration

`PipelineConfig` (frozen dataclass):

```python
PipelineConfig(
    backend="whisper",          # "whisper" | "mock"
    whisper_model_size="base",  # "tiny" | "base" | "small" | "medium" | "large"
    language=None,              # ISO 639-1 code or None for auto-detect
    min_confidence=0.5,         # Minimum segment confidence threshold
    extract_keywords=True,      # Enable TF-IDF keyword extraction
    max_keywords=10,            # Maximum keywords to extract
    output_dir="output",        # Output directory
    output_formats=[...],       # List of OutputFormat enums
    write_files=True,           # Write output files to disk
)
```

## Output Formats

### JSON (structured)

```json
{
  "summary": {
    "word_count": 136,
    "speaking_rate_wpm": 150.0,
    "num_sentences": 10,
    "language": "en",
    "duration_seconds": 54.4
  },
  "cleaned_text": "Welcome everyone...",
  "sentences": ["Welcome everyone...", "Our revenue grew..."],
  "keywords": ["percent", "work", "features"],
  "segments": [
    {
      "start": 0.0,
      "end": 6.8,
      "text": "Welcome everyone...",
      "confidence": 0.85
    }
  ]
}
```

### SRT (subtitles)

```
1
00:00:00,000 --> 00:00:06,799
Welcome everyone to today's quarterly business review.

2
00:00:06,799 --> 00:00:19,199
Our revenue grew by fifteen percent year over year...
```

### WebVTT

```
WEBVTT

00:00:00.000 --> 00:00:06.799
Welcome everyone to today's quarterly business review.

00:00:06.799 --> 00:00:19.199
Our revenue grew by fifteen percent year over year...
```

### Labeled

```
[00:00:00] Welcome everyone to today's quarterly business review.
[00:00:06] Our revenue grew by十五 percent year over year...
```

### TSV (tab-separated)

```
index	start	end	start_time	end_time	text	confidence
0	0.0	6.8	00:00:00.000	00:00:06.799	Welcome everyone...	0.85
```

## Extending

### Adding a new transcription backend

Implement the `Transcriber` ABC:

```python
from transcription_pipeline.transcriber import Transcriber
from transcription_pipeline.models import AudioInput, RawTranscript

class MyCustomTranscriber(Transcriber):
    def transcribe(self, audio: AudioInput) -> RawTranscript:
        # Your implementation here
        ...
```

Register it in `get_transcriber()` in `transcriber.py` and add the enum value to `TranscriptionBackend`.

### Adding a new output format

Add a formatter function in `output.py` or `timestamp_formats.py`, then register it in the `_FORMATTERS` or `TIMESTAMP_FORMATTERS` dict.

### Adding a new API endpoint

Add a route handler in `api.py` inside the `create_app()` function. The `TranscriptionService` instance is shared across all handlers.

## Limitations

- **No long-audio chunking** -- Whisper handles 30s internal windows, but the pipeline has no explicit chunking, progress reporting, or timeout for files over ~30 minutes
- **All in-memory** -- Segments, text, and TF-IDF matrix are held in RAM; no streaming for very large transcripts
- **No cancellation** -- Synchronous pipeline; interrupting requires killing the process
- **No max duration guard** -- A multi-hour file will be attempted in full

## License

This project uses open-source components:
- [OpenAI Whisper](https://github.com/openai/whisper) (MIT License)
- [Pydantic](https://github.com/pydantic/pydantic) (MIT License)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) (BSD License)
- [FastAPI](https://github.com/tiangolo/fastapi) (MIT License)
