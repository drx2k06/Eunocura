"""One-shot: remove every Sarvam reference from the backend + docs."""
import io, os, re

def rw(path, fn):
    s = io.open(path, encoding="utf-8").read()
    out = fn(s)
    if out != s:
        io.open(path, "w", encoding="utf-8", newline="\n").write(out)
        print("patched", path)
    else:
        print("unchanged", path)

# ---------------------------------------------------------------- config.py
def config(s):
    s = s.replace(
        "# Audio transcription: Sarvam if a key is present, otherwise local Whisper,\n"
        "# otherwise a clean \"text only\" degrade. Never a crash.\n"
        "STT_PROVIDER = os.getenv('STT_PROVIDER', 'auto').strip().lower()\n",
        "# Audio transcription: local faster-whisper, or a clean \"text only\" degrade.\n"
        "# No cloud speech provider, no key. Never a crash.\n"
        "STT_PROVIDER = os.getenv('STT_PROVIDER', 'whisper').strip().lower()\n",
    )
    s = s.replace("SARVAM_AI_API_KEY = os.getenv('SARVAM_AI_API_KEY')\n", "")
    s = s.replace("FEATHERLESS_API_KEY = os.getenv('FEATHERLESS_API_KEY')\n", "")
    return s

rw("src/config.py", config)

# ---------------------------------------------------------------- server.py
def server(s):
    old_start = s.index("def _stt_status() -> dict:")
    old_end = s.index("@app.get(\"/\")", old_start)
    new = '''def _stt_status() -> dict:
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


'''
    return s[:old_start] + new + s[old_end:]

rw("src/server.py", server)

# ---------------------------------------------------------------- routes/chatbot.py
def chatbot(s):
    s = s.replace(
        '    Convert text to speech using Sarvam Bulbul TTS.\n',
        '    Convert text to speech using local system voices (pyttsx3).\n',
    )
    s = s.replace('    speaker: str = "anushka"\n',
                  '    speaker: str = "default"  # ignored, kept for compatibility\n')
    return s

rw("src/routes/chatbot.py", chatbot)

# ---------------------------------------------------------------- appointment.py comments
def appointment(s):
    s = s.replace("# Sarvam AI returns additive transcript, so we replace instead of appending",
                  "# The local transcriber returns the whole transcript, so we replace")
    s = s.replace("# process_audio_chunk sends to Sarvam, which triggers the sender_loop via the queue",
                  "# process_audio_chunk transcribes locally and pushes partials through the queue")
    return s

if os.path.exists("src/routes/appointment.py"):
    rw("src/routes/appointment.py", appointment)

# ---------------------------------------------------------------- .env.example
def envexample(s):
    s = s.replace("# auto | whisper | sarvam | off\nSTT_PROVIDER=auto\n",
                  "# whisper | off   (local faster-whisper only - no cloud speech)\nSTT_PROVIDER=whisper\n")
    s = s.replace("# Cloud STT for Indic languages. Optional - local whisper works without it.\n"
                  "SARVAM_AI_API_KEY=\n", "")
    return s

rw(".env.example", envexample)

# ---------------------------------------------------------------- preflight + docs
for doc in ("scripts/preflight.py", "../README.md", "../frontend/FEATURES.md"):
    if not os.path.exists(doc):
        continue
    def strip_doc(s):
        s = s.replace("| `SARVAM_AI_API_KEY` | Falls back to local `faster-whisper`, then to text-only notes |",
                      "| `faster-whisper` | Text-only mode; audio endpoints return a fix hint |")
        s = s.replace("| Speech-to-text | Sarvam cloud if keyed, else local faster-whisper, else text-only |",
                      "| Speech-to-text | local faster-whisper; text-only degrade if absent |")
        s = s.replace("| Text-to-speech | Sarvam Bulbul |", "| Text-to-speech | local pyttsx3 system voices |")
        s = s.replace("**faster-whisper / Sarvam AI**: local or cloud speech-to-text",
                      "**faster-whisper**: local speech-to-text; **pyttsx3**: local text-to-speech")
        s = s.replace("Sarvam AI cloud STT only if a key is set", "no cloud speech provider at all")
        s = s.replace("Sarvam → faster-whisper → text-only", "faster-whisper → text-only")
        s = s.replace("#   * speech  → local faster-whisper, or text-only mode",
                      "#   * speech  → local faster-whisper + pyttsx3, or text-only mode")
        s = s.replace("Sarvam cloud if keyed, else local faster-whisper, else text-only",
                      "local faster-whisper, text-only if absent")
        return s
    rw(doc, strip_doc)
