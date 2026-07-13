# Design Document: Transcription Pipeline

## Problem Statement

Build a service or script that transcribes spoken language into text, with timestamps per segment. The focus is on engineering decisions, not training a model from scratch.

## How I Approached the Problem

The brief had three requirements buried in sequence: accept audio, transcribe it, return timestamps. Rather than build three disconnected scripts, I designed a **layered system** where each layer adds capability without breaking the one below:

1. **Core pipeline** -- the minimum viable path from audio file to text
2. **Service layer** -- a stateful, importable API that any Python code can use
3. **HTTP layer** -- a REST API that any client can call
4. **CLI layer** -- human-friendly entry points for each use case

The reasoning was simple: if I built the HTTP API first and then tried to extract a Python import from it, I would end up with framework-coupled code that is hard to test. By building bottom-up, each layer is independently testable and replaceable.

I also chose to **mock first and wire Whisper second**. The mock transcriber produces realistic output with proper timestamps, which means I could build and verify the entire pipeline -- validation, probing, processing, formatting, error handling -- before ever needing a GPU or a real audio file. This is not a shortcut; it is a deliberate testing strategy. If the mock passes end-to-end, I know the pipeline logic is correct, and Whisper integration becomes a single point of change.

## Key Decisions

### Why Pydantic models as stage contracts

Every pipeline stage has an explicit input and output type:

```
AudioInput -> RawTranscript -> ProcessedTranscript -> PipelineResult
```

This is the single most important architectural decision. It means:

- The keyword extractor cannot accidentally read from the audio probe
- The output formatter cannot accidentally modify the raw transcript
- Adding a new stage requires declaring its input/output types before writing any logic
- The types serve as documentation -- you can read `models.py` and understand the entire data flow without reading any implementation

### Why an abstract base class for transcribers

The `Transcriber` ABC with a single `transcribe(AudioInput) -> RawTranscript` method means:

- Adding Faster-Whisper, a cloud API, or a custom model requires implementing one method
- The pipeline, processor, and formatter never know which backend ran
- The mock backend can be injected for testing without patching

The factory function `get_transcriber()` maps config to implementation. This is the only place that knows about concrete classes.

### Why lazy model loading in the service

Whisper models take 2-5 seconds to load and consume 1-4GB of RAM. If the model loaded at `__init__` time, every `TranscriptionService()` construction would pay that cost even if the instance was never used. By loading on first `transcribe_file()` call:

- Construction is instant (~0.1ms)
- First transcription pays the load cost (~5s)
- Subsequent transcriptions reuse the loaded model (~0.05s)
- The model stays warm for the lifetime of the service instance

This is critical for the HTTP API: one model serves all requests.

### Why three-tier audio probing

Not every machine has ffmpeg. The probing strategy degrades gracefully:

| Tier | Method | What it covers | Dependencies |
|------|--------|---------------|-------------|
| 1 | `ffprobe` | All formats, full metadata | ffmpeg |
| 2 | Python `wave` | WAV only, full metadata | None |
| 3 | File check | Extension + size only | None |

This means WAV files always get duration, sample rate, and channels on any Python install. MP3/M4A files still pass through -- they just skip to the fallback and let Whisper handle decoding.

### Why separate timestamp formatters

The `timestamp_formats.py` module contains pure functions: segments go in, formatted strings come out. No config, no I/O, no state. This separation means:

- The HTTP API can return SRT with the correct Content-Type without importing pipeline config
- The CLI can format labeled output without touching the service layer
- Each formatter can be unit-tested in isolation
- Adding a new format (e.g., ASS subtitles) requires adding one function to one dict

### Why the mock backend is not a stub

The mock transcriber produces five segments with realistic text, proper timestamps (calculated from word count at 150 wpm), and varied confidence scores. This is deliberate:

- It exercises the sentence splitter (multiple sentences per segment)
- It exercises the keyword extractor (real English text with domain-specific terms)
- It produces enough data to verify SRT/VTT/TSV formatting
- It runs in ~0.001s, making the full pipeline testable in seconds

## How the Code Works

### Data flow

```
Audio file on disk
        |
  [validate_audio_file]  -- exists? readable? supported extension? non-empty?
        |
  [probe_audio]          -- ffprobe -> wave -> file check
        |
  [AudioInput]           -- path + metadata + language hint
        |
  [Transcriber]          -- Whisper or Mock -> RawTranscript
        |                   (segments with start/end/text/confidence)
  [process_transcript]   -- clean fillers, split sentences, extract keywords
        |
  [ProcessedTranscript]  -- cleaned text + sentences + keywords + stats
        |
  [format_timestamps]    -- JSON / SRT / VTT / labeled / TSV
        |
  [write_outputs]        -- optional: write files to disk
```

### The three interfaces

**CLI (`transcribe.py`, `timestamps.py`)** parses arguments, sets up logging, validates the file, calls `run_pipeline()`, and prints results. The logging goes to stderr so stdout can be piped. Error messages go to stderr with clear explanations and supported format lists.

**Programmatic (`TranscriptionService`)** is a stateful class that owns the transcriber lifecycle. It accepts file paths or raw bytes (writing to a temp file for the latter). It returns `TranscriptionResult` objects with flat fields (`text`, `sentences`, `keywords`) for convenience, plus `.raw` and `.processed` for full access. The `to_dict()` and `to_timestamps_dict()` methods produce JSON-serializable dicts.

**HTTP (`api.py`)** wraps `TranscriptionService` in FastAPI routes. The model loads on first request and stays warm. Each endpoint maps to a specific response shape:

| Endpoint | Response |
|----------|----------|
| `POST /transcribe` | `{text, language, duration_seconds, word_count}` |
| `POST /transcribe/json` | Full result with segments, keywords, metadata |
| `POST /transcribe/timestamps` | Segments with formatted timestamps (format param) |

The timestamps endpoint returns `application/json` for format=json, `text/vtt` for format=vtt, `application/x-subrip` for format=srt, etc. This means a browser can directly load a VTT response as a video subtitle track.

### Error handling

Validation errors return HTTP 400/422 with descriptive messages. The pipeline catches `AudioValidationError` at every entry point (CLI, service, API) and maps it to the appropriate response. Unsupported formats get a list of what IS supported. Empty files get a clear message. Permission errors are caught during the read check, not deep in Whisper.

### Configuration

`PipelineConfig` is a frozen dataclass -- immutable after construction. This prevents bugs where one request accidentally modifies shared config. The config covers backend selection, model size, language, keyword extraction, output formats, and file writing. The service layer creates its own config at construction time, so multiple service instances can coexist with different settings.

## What I Would Add for Production

1. **Chunked processing** -- split long audio into 10-minute chunks with progress callbacks
2. **Timeout handling** -- wrap Whisper calls in threads with configurable timeouts
3. **Max duration guard** -- reject files exceeding a duration threshold
4. **Memory management** -- process and discard chunks incrementally instead of holding all segments in RAM
5. **Cancellation** -- async support so long transcriptions can be interrupted
6. **Rate limiting** -- for the HTTP API, prevent abuse
7. **Structured logging** -- JSON logs for production monitoring
8. **Health checks** -- model load status, memory usage, queue depth
