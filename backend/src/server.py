"""EUNOCURA API entry point.

`uvicorn src.server:app --reload --port 8000`

Boots with zero configuration: no API keys, no MongoDB, no cloud storage. Each
missing piece is reported by `/healthz` as a degraded subsystem instead of
stopping the process.
"""

import importlib.util
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pymongo.errors import PyMongoError

from src import db
from src.config import PORT, PUBLIC_BASE_URL, STT_PROVIDER, UPLOAD_DIR
from src.llm import router
from src.llm.router import ModelUnavailable
from src.r2 import storage_mode
from src.routes.root import router as root_router
from src.services.local_speech import SpeechUnavailable

APP_NAME = "EUNOCURA"
VERSION = "2.0.0"

app = FastAPI(title=APP_NAME, version=VERSION)


@app.exception_handler(PyMongoError)
async def pymongo_exception_handler(request: Request, exc: PyMongoError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Database unavailable: MongoDB is not running or unreachable. Please start mongod or configure MONGODB_URI."},
    )


@app.exception_handler(ModelUnavailable)
async def model_unavailable_handler(request: Request, exc: ModelUnavailable):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


@app.exception_handler(SpeechUnavailable)
async def speech_unavailable_handler(request: Request, exc: SpeechUnavailable):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )

_origins = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Local storage fallback: uploaded documents are served from here when R2 is
# not configured, so the PDF viewer works on a clean clone.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(UPLOAD_DIR)), name="files")


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _stt_status() -> dict:
    """Local speech only: faster-whisper if installed, else text-only."""
    try:
        from src.services import local_speech
        stt = local_speech.status()["stt"]
        tts = local_speech.status()["tts"]
    except Exception as exc:  # never let a status probe break the server
        return {
            "provider": "none",
            "available": False,
            "note": f"speech module unavailable: {exc}",
        }

    if stt["available"]:
        return {
            "provider": "faster-whisper",
            "available": True,
            "model": stt.get("model"),
            "note": "local CPU transcription - no cloud, no key",
            "tts": tts,
        }
    return {
        "provider": "none",
        "available": False,
        "note": "text-only mode - run `pip install faster-whisper` for local speech",
        "tts": tts,
    }


@app.get("/")
async def root():
    return {
        "service": APP_NAME,
        "version": VERSION,
        "status": "/healthz",
        "docs": "/docs",
    }


@app.get("/health")
@app.get("/healthz")
async def healthz():
    """Preflight for the live demo: what is actually operational right now.

    Every probe is capped: a demo must answer in about a second even when the
    database is unreachable and the model host is firewalled.
    """
    import asyncio as _asyncio

    async def _bounded(coro, seconds, default):
        try:
            return await _asyncio.wait_for(coro, timeout=seconds)
        except Exception:
            return default

    model = await _bounded(_asyncio.to_thread(router.health), 2.5, {
        "provider": "unknown", "reachable": False, "detail": "timed out",
    })
    mongo_reachable = await _bounded(db.ping(), 2.0, False) if db.MONGO_CONFIGURED else False
    stt = _stt_status()

    # Report what actually initialised, not just what is installed: torch can be
    # present and still refuse to load (Windows Application Control, missing libs).
    try:
        from src.services import pdf_rag as _pdf_rag
        rag_reason = getattr(_pdf_rag, "RAG_UNAVAILABLE_REASON", "") or (
            "" if getattr(_pdf_rag, "pdf_rag_service", None) else "PDF RAG service did not initialise"
        )
    except Exception as exc:  # noqa: BLE001
        rag_reason = f"pdf_rag import failed: {exc}"
    embeddings = _has_module("faiss") and _has_module("sentence_transformers") and not rag_reason

    checks = {
        "model": model["reachable"] or model["provider"] == "none",
        "mongo": mongo_reachable,
        "storage": True,  # local disk is always writable enough to boot
        "embeddings": embeddings,
    }
    status = "ok" if all(checks.values()) else "degraded"

    return {
        "service": APP_NAME,
        "version": VERSION,
        "status": status,
        "checks": checks,
        "model": model,
        "mongo": {
            "configured": db.MONGO_CONFIGURED,
            "reachable": mongo_reachable,
            "hint": "" if mongo_reachable else "Set MONGODB_URI, or start a local mongod.",
        },
        "storage": {
            "mode": storage_mode(),
            "dir": str(UPLOAD_DIR),
            "public_base_url": PUBLIC_BASE_URL,
            "hint": "" if storage_mode() == "r2" else "R2 not configured — files are stored locally.",
        },
        "stt": stt,
        "rag": {
            "vector_store": "faiss" if _has_module("faiss") else "missing",
            "embeddings_model": "sentence-transformers/all-MiniLM-L6-v2",
            "available": embeddings,
            "hint": rag_reason if embeddings else (
                rag_reason or "pip install faiss-cpu sentence-transformers torch"
            ),
        },
    }


# All feature routers (auth, appointments, patients, doctors, ingest, chatbot).
app.include_router(root_router)
