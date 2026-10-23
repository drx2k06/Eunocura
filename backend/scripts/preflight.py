"""Preflight: can this machine run the EUNOCURA demo right now?

    cd backend
    python scripts/preflight.py

Checks, in the order that actually breaks demos:

  1. Python + the packages the pipeline imports
  2. The model server (Ollama) and whether the profile models are pulled
  3. MongoDB (optional for the AI paths, required for the portals)
  4. Storage mode (R2 or local disk) and the local storage directory
  5. Speech-to-text mode

Exit codes: 0 = demo-ready, 1 = something demo-blocking is missing,
2 = only optional pieces are missing. No keys are read or printed.
"""

from __future__ import annotations

import importlib.util
import shutil
import socket
import sys
from pathlib import Path
from typing import List, Tuple

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

OK, WARN, FAIL = "PASS", "WARN", "FAIL"
rows: List[Tuple[str, str, str]] = []


def record(status: str, name: str, note: str = "") -> None:
    rows.append((status, name, note))


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def check_python() -> None:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    record(OK if sys.version_info >= (3, 11) else FAIL, "python", version)


def check_packages() -> None:
    required = {
        "fastapi": "web framework",
        "uvicorn": "server",
        "pymongo": "MongoDB driver",
        "pdfplumber": "PDF text + word boxes",
        "langchain_core": "prompt chains",
        "langchain_openai": "model router transport",
    }
    optional = {
        "faiss": "document RAG index",
        "sentence_transformers": "MiniLM embeddings",
        "faster_whisper": "offline speech-to-text",
        "fitz": "evidence PNG rendering (pymupdf)",
    }
    missing_required = [name for name in required if not has_module(name)]
    if missing_required:
        record(FAIL, "packages", f"missing: {', '.join(missing_required)} — pip install -r requirements.txt")
    else:
        record(OK, "packages", "all required imports present")

    missing_optional = [name for name in optional if not has_module(name)]
    if missing_optional:
        record(WARN, "optional packages", f"missing: {', '.join(missing_optional)}")
    else:
        record(OK, "optional packages", "RAG, embeddings, STT and rendering installed")


def check_env_file() -> None:
    env_file = BACKEND / ".env"
    if env_file.exists():
        record(OK, ".env", "present (values never printed)")
    else:
        record(WARN, ".env", "absent — zero-key defaults still boot; copy .env.example if you need MongoDB")


def check_model() -> None:
    try:
        from src.llm.router import health
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "model router", f"could not import: {exc}")
        return

    status = health()
    if status["provider"] == "none":
        record(WARN, "model", "LLM_PROVIDER=none — model features disabled")
        return

    if status["provider"] == "gemini":
        record(OK if status["reachable"] else FAIL, "model (gemini)",
               "key present" if status["reachable"] else "GEMINI_API_KEY empty")
        return

    if not status["reachable"]:
        record(WARN, "model (ollama)", f"{status['hint']} (degraded mode: offline/deterministic paths still run)")
        return

    record(OK, "model (ollama)", status["endpoint"])
    if status["missing_models"]:
        record(WARN, "model weights", status["hint"])
    else:
        record(OK, "model weights", ", ".join(sorted(status["profile_models"].values())))


def check_mongo() -> None:
    try:
        from src.config import MONGODB_URI
    except Exception as exc:  # noqa: BLE001
        record(WARN, "mongo", f"config unavailable: {exc}")
        return

    if not MONGODB_URI:
        record(WARN, "mongo", "MONGODB_URI not set — portals need data; AI paths still run")
        return

    if not has_module("pymongo"):
        record(FAIL, "mongo", "pymongo missing")
        return

    try:
        from pymongo import MongoClient

        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=1500)
        client.admin.command("ping")
        record(OK, "mongo", "reachable")
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "mongo", f"not reachable: {exc.__class__.__name__}")


def check_storage() -> None:
    try:
        from src.config import UPLOAD_DIR
        from src.r2 import storage_mode
    except Exception as exc:  # noqa: BLE001
        record(WARN, "storage", f"config unavailable: {exc}")
        return

    mode = storage_mode()
    if mode == "r2":
        record(OK, "storage", "Cloudflare R2 configured")
        return
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        probe = UPLOAD_DIR / ".preflight"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        record(OK, "storage", f"local disk ({UPLOAD_DIR})")
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "storage", f"local disk not writable: {exc}")


def check_stt() -> None:
    try:
        from src.services.whisper_service import available
        status = available()
    except Exception:
        try:
            from src.services import local_speech
            status = local_speech.status().get("stt", {})
        except Exception as exc:  # noqa: BLE001
            record(WARN, "stt", f"unavailable: {exc.__class__.__name__}")
            return
    record(OK if status.get("available") else WARN, "stt", status.get("note", "text-only mode"))


def check_port(port: int = 8000) -> None:
    """Is something already holding the API port?"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        in_use = sock.connect_ex(("127.0.0.1", port)) == 0
    if in_use:
        record(WARN, "port 8000", "already in use — the API may already be running")
    else:
        record(OK, "port 8000", "free")


def check_ollama_binary() -> None:
    path = shutil.which("ollama")
    if path:
        record(OK, "ollama cli", path)
    else:
        record(WARN, "ollama cli", "not on PATH — you can also point OLLAMA_BASE_URL at a remote server")


def main() -> int:
    check_python()
    check_env_file()
    check_packages()
    check_ollama_binary()
    check_model()
    check_storage()
    check_stt()
    check_mongo()
    check_port()

    width = max(len(name) for _, name, _ in rows)
    print("\n  EUNOCURA preflight\n")
    for status, name, note in rows:
        print(f"  {status:<4}  {name:<{width}}  {note}")

    failures = sum(1 for status, _, _ in rows if status == FAIL)
    warnings = sum(1 for status, _, _ in rows if status == WARN)
    print(f"\n  {len(rows)} checks · {len(rows) - failures - warnings} pass · {warnings} warn · {failures} fail")

    if failures:
        print("\n  NOT DEMO-READY — fix the FAIL lines above.\n")
        return 1
    if warnings:
        print("\n  DEMO-READY with degraded features (see WARN).\n")
        return 2
    print("\n  DEMO-READY — every subsystem operational.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
