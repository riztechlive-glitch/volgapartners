"""
HTTP transcription service -- REST API for speech-to-text.

Run the server:
    python -m transcription_pipeline.api
    python -m transcription_pipeline.api --port 8080 --backend whisper --whisper-model base

Endpoints:
    POST /transcribe       Upload an audio file, get text back.
    POST /transcribe/json  Upload + get full JSON with segments/keywords.
    GET  /health           Health check + model status.
    GET  /                 This help page.

The API uses TranscriptionService under the hood, so the Whisper model
is loaded once on first request and reused for all subsequent requests.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from .audio_utils import SUPPORTED_EXTENSIONS, AudioValidationError
from .config import PipelineConfig
from .models import TranscriptionBackend
from .service import TranscriptionService

logger = logging.getLogger(__name__)

# FastAPI is optional -- if unavailable, we fall back to the stdlib server.
try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


def create_app(
    backend: str = "mock",
    whisper_model_size: str = "base",
    language: str | None = None,
) -> "FastAPI":
    """Build the FastAPI application with a shared TranscriptionService."""
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI not installed. Run: pip install fastapi uvicorn"
        )

    service = TranscriptionService(
        backend=backend,
        whisper_model_size=whisper_model_size,
        language=language,
        output_formats=["json"],
        write_files=False,
    )

    app = FastAPI(
        title="Transcription API",
        description="Speech-to-text service using OpenAI Whisper",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def index():
        return {
            "service": "Transcription API",
            "version": "1.0.0",
            "endpoints": {
                "POST /transcribe": "Upload audio file -> plain text transcript",
                "POST /transcribe/json": "Upload audio file -> full JSON (segments, keywords, metadata)",
                "POST /transcribe/timestamps": "Upload audio file -> segments with formatted timestamps",
                "GET /health": "Health check + model status",
            },
            "supported_formats": sorted(SUPPORTED_EXTENSIONS),
        }

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "backend": service.backend,
            "model_loaded": service.is_loaded,
            "supported_formats": sorted(SUPPORTED_EXTENSIONS),
        }

    @app.post("/transcribe")
    async def transcribe_text(
        file: UploadFile = File(...),
        language: str | None = Form(default=None),
    ):
        """Upload an audio file and return the plain text transcript."""
        return await _do_transcribe(file, service, language, plain_text=True)

    @app.post("/transcribe/json")
    async def transcribe_json(
        file: UploadFile = File(...),
        language: str | None = Form(default=None),
    ):
        """Upload an audio file and return the full structured result."""
        return await _do_transcribe(file, service, language, plain_text=False)

    @app.post("/transcribe/timestamps")
    async def transcribe_timestamps(
        file: UploadFile = File(...),
        language: str | None = Form(default=None),
        format: str = Form(default="json"),
    ):
        """
        Upload an audio file and return segments with timestamps.

        Query params:
            format: "json" (default), "srt", "vtt", "labeled", "tsv"

        Returns the appropriate format as the response body.
        """
        return await _do_transcribe_timestamps(file, service, language, format)

    return app


async def _do_transcribe(
    file: UploadFile,
    service: TranscriptionService,
    language: str | None,
    plain_text: bool,
) -> JSONResponse:
    """Shared handler for both /transcribe endpoints."""
    # Validate file was provided
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    # Check extension
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported format '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    # Transcribe
    try:
        result = service.transcribe_bytes(
            audio_bytes=content,
            filename=file.filename,
            language=language,
        )
    except AudioValidationError as e:
        raise HTTPException(status_code=422, detail=e.reason)
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    # Return appropriate response
    if plain_text:
        return JSONResponse(content={
            "text": result.text,
            "language": result.language,
            "duration_seconds": round(result.duration_seconds, 2),
            "word_count": result.word_count,
        })
    else:
        return JSONResponse(content=result.to_dict())


async def _do_transcribe_timestamps(
    file: UploadFile,
    service: TranscriptionService,
    language: str | None,
    fmt: str,
) -> JSONResponse:
    """Transcribe and return segments with formatted timestamps."""
    from fastapi.responses import PlainTextResponse

    from .timestamp_formats import SUPPORTED_TIMESTAMP_FORMATS, format_timestamps

    if fmt not in SUPPORTED_TIMESTAMP_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown format '{fmt}'. Supported: {SUPPORTED_TIMESTAMP_FORMATS}",
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported format '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    try:
        result = service.transcribe_bytes(
            audio_bytes=content,
            filename=file.filename,
            language=language,
        )
    except AudioValidationError as e:
        raise HTTPException(status_code=422, detail=e.reason)
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    formatted = format_timestamps(result.segments, fmt)

    if fmt == "json":
        return JSONResponse(content=result.to_timestamps_dict())
    else:
        media_types = {
            "srt": "application/x-subrip",
            "vtt": "text/vtt",
            "labeled": "text/plain",
            "tsv": "text/tab-separated-values",
        }
        return PlainTextResponse(content=formatted, media_type=media_types.get(fmt, "text/plain"))


# ── Stdlib fallback (no FastAPI required) ──────────────────────────────────────

def _run_stdlib_server(host: str, port: int, backend: str, model_size: str) -> None:
    """
    Minimal HTTP server using only the Python stdlib.
    Used when FastAPI is not installed.
    """
    import http.server
    import urllib.parse

    service = TranscriptionService(
        backend=backend,
        whisper_model_size=model_size,
        output_formats=["json"],
        write_files=False,
    )

    SUPPORTED_CONTENT_TYPES = {
        "audio/wav", "audio/wave", "audio/x-wav",
        "audio/mpeg", "audio/mp3",
        "audio/mp4", "audio/m4a", "audio/x-m4a",
        "audio/flac", "audio/x-flac",
        "audio/ogg", "audio/webm",
        "audio/x-ms-wma",
        "audio/aac",
    }

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._json_response(200, {
                    "status": "ok",
                    "backend": service.backend,
                    "model_loaded": service.is_loaded,
                })
            elif self.path == "/":
                self._json_response(200, {
                    "service": "Transcription API (stdlib)",
                    "note": "Install FastAPI for full multipart upload support.",
                    "endpoints": {
                        "POST /transcribe": "Send raw audio bytes in request body with Content-Type audio/*",
                        "GET /health": "Health check",
                    },
                })
            else:
                self._json_response(404, {"error": "Not found."})

        def do_POST(self):
            if self.path == "/transcribe":
                content_type = self.headers.get("Content-Type", "")
                content_length = int(self.headers.get("Content-Length", 0))

                if content_length == 0:
                    self._json_response(400, {"error": "Empty request body."})
                    return

                body = self.rfile.read(content_length)

                # Derive filename from content-type
                ext_map = {
                    "audio/wav": ".wav", "audio/mpeg": ".mp3",
                    "audio/mp4": ".m4a", "audio/flac": ".flac",
                    "audio/ogg": ".ogg", "audio/webm": ".webm",
                }
                ext = ".wav"
                for ct, e in ext_map.items():
                    if ct in content_type:
                        ext = e
                        break

                try:
                    result = service.transcribe_bytes(body, filename=f"upload{ext}")
                    self._json_response(200, result.to_dict())
                except AudioValidationError as e:
                    self._json_response(422, {"error": e.reason})
                except Exception as e:
                    self._json_response(500, {"error": str(e)})
            else:
                self._json_response(404, {"error": "Not found."})

        def _json_response(self, code: int, data: dict) -> None:
            payload = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            logger.info(format, *args)

    server = http.server.HTTPServer((host, port), Handler)
    print(f"Transcription API (stdlib) listening on http://{host}:{port}")
    print(f"  Backend: {backend}, Model: {model_size}")
    print(f"  POST /transcribe  -- send raw audio bytes")
    print(f"  GET  /health      -- health check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="transcription-api",
        description="Run the transcription HTTP service.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--backend", choices=["whisper", "mock"], default="mock")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--language", default=None, help="ISO 639-1 language code")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if HAS_FASTAPI:
        import uvicorn

        app = create_app(
            backend=args.backend,
            whisper_model_size=args.whisper_model,
            language=args.language,
        )
        print(f"Transcription API (FastAPI) starting on http://{args.host}:{args.port}")
        print(f"  Backend: {args.backend}, Model: {args.whisper_model}")
        print(f"  Interactive docs: http://{args.host}:{args.port}/docs")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    else:
        print("FastAPI not found -- using stdlib HTTP server (limited features).")
        print("For full support, run: pip install fastapi uvicorn")
        _run_stdlib_server(args.host, args.port, args.backend, args.whisper_model)


if __name__ == "__main__":
    main()
