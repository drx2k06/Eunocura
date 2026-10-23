"""
Streaming speech-to-text powered by local faster-whisper.

This replaces the previous Sarvam WebSocket implementation while keeping the same
public interface consumed by the routes:
  - StreamingTranscriber.connect() -> bool
  - StreamingTranscriber.process_audio_chunk(chunk: bytes) -> str
  - StreamingTranscriber.finalize() -> str
  - StreamingTranscriber.reset()
  - attributes: transcript_queue, current_partial, session_transcript, is_connected, chunk_count
"""

import asyncio
import os
import tempfile
import time
from typing import Optional
from collections import deque

from src.services import local_speech


class StreamingTranscriber:
    def __init__(self):
        self.is_connected = False
        self.ws = None
        self.transcript_queue = asyncio.Queue()
        self.current_partial = ""
        self.session_transcript = ""
        self.chunk_count = 0
        self._buffer: bytearray = bytearray()
        self._lock = asyncio.Lock()
        self._last_process = 0.0
        self._process_interval = 2.0
        self._min_bytes = 4096

    async def connect(self) -> bool:
        """Return true if local STT is available, false otherwise."""
        if local_speech.stt_available():
            self.is_connected = True
            self._last_process = time.time()
            return True
        self.is_connected = False
        return False

    def _suffix(self) -> str:
        head = bytes(self._buffer[:12])
        if head.startswith(b"\x1aE\xdf\xa3"):
            return ".webm"
        if head[:4] == b"RIFF":
            return ".wav"
        if head[4:8] == b"ftyp":
            return ".m4a"
        return ".webm"

    async def process_audio_chunk(self, chunk: bytes) -> str:
        """
        Append audio to the buffer. Every ~2 seconds transcribe what's accumulated
        so far and push a partial to the queue. This is local - no network calls.
        """
        if not self.is_connected:
            connected = await self.connect()
            if not connected:
                return self.current_partial

        self.chunk_count += 1
        self._buffer.extend(chunk)

        now = time.time()
        if len(self._buffer) < self._min_bytes:
            return self.current_partial
        if now - self._last_process < self._process_interval:
            return self.current_partial

        partial = await self._transcribe_buffer(accumulate=True)
        self._last_process = now
        if partial:
            self.current_partial = partial
            self.session_transcript = partial
            await self.transcript_queue.put(partial)
        return self.current_partial

    async def _transcribe_buffer(self, accumulate: bool = True) -> str:
        if not self._buffer:
            return self.current_partial

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=self._suffix()) as f:
                f.write(bytes(self._buffer))
                tmp_path = f.name

            def _run():
                model = local_speech._get_stt_model()
                from faster_whisper import WhisperModel  # noqa
                segments, info = model.transcribe(
                    tmp_path,
                    language=None,
                    task="translate",
                    vad_filter=True,
                    beam_size=1,
                )
                pieces = []
                for seg in segments:
                    t = (seg.text or "").strip()
                    if t:
                        pieces.append(t)
                return " ".join(pieces).strip()

            text = await asyncio.to_thread(_run)
            if accumulate and text:
                # keep the raw bytes we just decoded? faster-whisper decoded them fine.
                # trimming is tricky; easier to keep accumulating and rely on translate+vad.
                # to avoid re-sending identical prefixes the frontend resets on stop.
                pass
            return text or self.current_partial
        except Exception as e:
            print(f"Local streaming STT error: {e}")
            return self.current_partial
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    async def finalize(self) -> str:
        """Transcribe any remaining buffered audio and return the full transcript."""
        text = await self._transcribe_buffer(accumulate=False)
        if text:
            self.current_partial = text
            self.session_transcript = text
            try:
                await self.transcript_queue.put(text)
            except Exception:
                pass
        self.is_connected = False
        return self.session_transcript or self.current_partial

    def reset(self):
        self.is_connected = False
        self.ws = None
        self.current_partial = ""
        self.session_transcript = ""
        self.chunk_count = 0
        self._buffer = bytearray()
        self._last_process = 0.0
        while not self.transcript_queue.empty():
            try:
                self.transcript_queue.get_nowait()
            except Exception:
                break


async def transcribe_websocket_stream(websocket):
    """Helper kept for backwards compatibility."""
    transcriber = StreamingTranscriber()
    await transcriber.connect()
    while True:
        try:
            data = await websocket.receive_bytes()
            await transcriber.process_audio_chunk(data)
        except Exception:
            break
    return await transcriber.finalize()
