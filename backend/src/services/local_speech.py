"""
Local speech for EUNOCURA. No cloud, no keys.

    STT  faster-whisper  (CTranslate2, CPU int8 by default)
    TTS  pyttsx3         (SAPI5 on Windows, espeak-ng on Linux)

Both are optional. If a dependency or its model weights are missing the caller gets
`SpeechUnavailable` with the exact install command, and the app keeps working in text
mode — audio is a convenience, never a hard requirement.

This module replaces the previous Sarvam AI cloud calls. Nothing here opens a socket.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
from typing import Any, Dict, Optional

# Whisper language codes are ISO-639-1; callers speak BCP-47 ("hi-IN", "ta-IN").
_BCP47_TO_WHISPER = {
    "hi-IN": "hi", "ta-IN": "ta", "te-IN": "te", "bn-IN": "bn", "gu-IN": "gu",
    "kn-IN": "kn", "ml-IN": "ml", "mr-IN": "mr", "or-IN": "or", "pa-IN": "pa",
    "en-IN": "en", "en-US": "en", "en-GB": "en",
}

# ISO-639-1 -> the language name the SLM prompt uses.
_WHISPER_TO_NAME = {
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "bn": "Bengali", "gu": "Gujarati",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "or": "Odia", "pa": "Punjabi",
    "en": "English",
}


class SpeechUnavailable(RuntimeError):
    """Raised when a speech dependency is absent. Message always says how to fix it."""


_stt_lock = threading.Lock()
_stt_model = None
_stt_model_size: Optional[str] = None


def stt_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def tts_available() -> bool:
    try:
        import pyttsx3  # noqa: F401
        return True
    except Exception:
        return False


def status() -> Dict[str, Any]:
    return {
        "stt": {
            "available": stt_available(),
            "engine": "faster-whisper",
            "model": os.getenv("WHISPER_MODEL", "base"),
            "install": "pip install faster-whisper" if not stt_available() else None,
        },
        "tts": {
            "available": tts_available(),
            "engine": "pyttsx3",
            "install": "pip install pyttsx3" if not tts_available() else None,
        },
    }


def _get_stt_model(model_size: Optional[str] = None):
    """Load faster-whisper once and keep it. First call downloads weights (~75 MB for base)."""
    global _stt_model, _stt_model_size

    if not stt_available():
        raise SpeechUnavailable(
            "faster-whisper is not installed. Install it with:\n"
            "    pip install faster-whisper\n"
            "Text mode keeps working without it."
        )

    size = model_size or os.getenv("WHISPER_MODEL", "base")
    with _stt_lock:
        if _stt_model is None or _stt_model_size != size:
            from faster_whisper import WhisperModel

            # int8 on CPU: fast enough for a demo laptop and needs no GPU.
            _stt_model = WhisperModel(size, device="cpu", compute_type="int8")
            _stt_model_size = size
        return _stt_model


def _whisper_lang(language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    if language in _BCP47_TO_WHISPER:
        return _BCP47_TO_WHISPER[language]
    return language.split("-")[0][:2].lower()


def _suffix_for(audio: bytes) -> str:
    """faster-whisper decodes through PyAV, so the container matters."""
    head = audio[:12]
    if head.startswith(b"\x1aE\xdf\xa3"):
        return ".webm"
    if head[:4] == b"RIFF":
        return ".wav"
    if head[:4] == b"OggS":
        return ".ogg"
    if head[4:8] == b"ftyp":
        return ".m4a"
    return ".webm"


def transcribe_bytes(
    audio: bytes,
    language: Optional[str] = "hi-IN",
    model_size: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transcribe audio and translate it to English.

    faster-whisper's `transcribe` returns text in the spoken language; `task="translate"`
    makes it return English, which is what the clinical pipeline consumes.
    """
    if not audio:
        raise SpeechUnavailable("No audio received.")

    model = _get_stt_model(model_size)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=_suffix_for(audio)) as handle:
            handle.write(audio)
            tmp_path = handle.name

        try:
            segments, info = model.transcribe(
                tmp_path,
                language=_whisper_lang(language),
                task="translate",
                vad_filter=True,
                beam_size=1,
            )
            pieces = []
            for segment in segments:
                text = (segment.text or "").strip()
                if text:
                    pieces.append(text)

            transcript = " ".join(pieces).strip()
            detected = getattr(info, "language", None) or _whisper_lang(language) or "en"
            return {
                "translated_text": transcript,
                "detected_language": detected,
                "language_name": _WHISPER_TO_NAME.get(detected, detected),
                "duration": float(getattr(info, "duration", 0.0) or 0.0),
            }
        except Exception as exc:
            raise SpeechUnavailable(f"Audio transcription failed: {exc}") from exc
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


async def atranscribe_bytes(
    audio: bytes,
    language: Optional[str] = "hi-IN",
    model_size: Optional[str] = None,
) -> Dict[str, Any]:
    """Async wrapper — whisper is CPU-bound, so keep it off the event loop."""
    return await asyncio.to_thread(transcribe_bytes, audio, language, model_size)


def synthesize(text: str, language: str = "en-IN") -> bytes:
    """
    Speak `text` locally and return WAV bytes.

    Language is advisory: pyttsx3 uses whatever system voices are installed. If no
    voice matches the requested language we still speak it with the default voice
    rather than failing the request.
    """
    text = (text or "").strip()
    if not text:
        raise SpeechUnavailable("Nothing to say.")

    if not tts_available():
        raise SpeechUnavailable(
            "pyttsx3 is not installed. Install it with:\n"
            "    pip install pyttsx3\n"
            "Replies are still returned as text."
        )

    import pyttsx3

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as handle:
            tmp_path = handle.name

        engine = pyttsx3.init()
        wanted = _whisper_lang(language) or "en"
        for voice in engine.getProperty("voices") or []:
            haystack = f"{getattr(voice, 'id', '')} {getattr(voice, 'name', '')}".lower()
            if wanted == "hi" and ("hindi" in haystack or "hi-in" in haystack):
                engine.setProperty("voice", voice.id)
                break
            if wanted == "en" and ("english" in haystack or "en-us" in haystack):
                engine.setProperty("voice", voice.id)
                break
        engine.setProperty("rate", 165)
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        try:
            engine.stop()
        except Exception:
            pass

        with open(tmp_path, "rb") as handle:
            data = handle.read()
        if not data:
            raise SpeechUnavailable("The system speech engine produced no audio.")
        return data
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


async def asynthesize(text: str, language: str = "en-IN") -> bytes:
    return await asyncio.to_thread(synthesize, text, language)
