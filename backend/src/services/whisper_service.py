"""
Speech-to-text service.
Defaults to local faster-whisper. No cloud calls by default.
"""

import os
import tempfile
import asyncio
from typing import Dict, Any


class WhisperService:
    def __init__(self):
        self.model_name = os.getenv("WHISPER_MODEL", "base")
        self._model = None

    async def transcribe_audio(self, audio_data: bytes, language: str = "en") -> Dict[str, Any]:
        """Transcribe audio using faster-whisper. Degrade gracefully if missing."""
        try:
            from src.services import local_speech

            result = await local_speech.atranscribe_bytes(
                audio_data,
                language=language if "-" not in language else None,
            )
            return {
                "text": result.get("translated_text", ""),
                "language": result.get("detected_language", language),
                "segments": [],
                "engine": "faster-whisper",
            }
        except Exception as e:
            return {
                "text": "",
                "language": language,
                "segments": [],
                "error": str(e),
                "engine": "none",
            }

    async def transcribe_audio_file(self, file) -> str:
        """Used by routes - returns plain text."""
        try:
            content = await file.read()
            result = await self.transcribe_audio(content)
            return result.get("text", "")
        finally:
            try:
                await file.seek(0)
            except Exception:
                pass


whisper_service = WhisperService()


def available() -> Dict[str, Any]:
    """Check availability of speech-to-text service."""
    try:
        from src.services import local_speech
        status_info = local_speech.status()
        stt = status_info.get("stt", {})
        return {
            "available": stt.get("available", False),
            "model": stt.get("model", "base"),
            "note": "local CPU transcription - no cloud, no key" if stt.get("available") else "text-only mode - run `pip install faster-whisper` for local speech",
            "engine": stt.get("engine", "none"),
        }
    except Exception as exc:
        return {
            "available": False,
            "model": "base",
            "note": f"speech module unavailable: {exc}",
            "engine": "none",
        }


async def transcribe_audio_file(file) -> str:
    return await whisper_service.transcribe_audio_file(file)
