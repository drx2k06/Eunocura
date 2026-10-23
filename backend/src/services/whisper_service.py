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


async def transcribe_audio_file(file) -> str:
    return await whisper_service.transcribe_audio_file(file)
