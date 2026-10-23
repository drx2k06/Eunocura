import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pwdlib import PasswordHash

load_dotenv()

PORT = os.getenv('PORT', '8000')
MONGODB_URI = os.getenv('MONGODB_URI')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# ---------------------------------------------------------------- model
# EUNOCURA runs open weights. Default: Ollama on this machine, no key, no cloud.
#   LLM_PROVIDER=ollama   local OpenAI-compatible server (default)
#   LLM_PROVIDER=gemini   optional cloud fallback (Gemma via Gemini API)
#   LLM_PROVIDER=none     refuse model calls; deterministic paths only
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'ollama').strip().lower()
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
# Small language model for extraction / intent / chat: Apache-2.0.
OLLAMA_SLM_MODEL = os.getenv('OLLAMA_SLM_MODEL', 'qwen2.5:7b-instruct')
# Medical profile: an open-weight medical model served the same way.
OLLAMA_MED_MODEL = os.getenv('OLLAMA_MED_MODEL', 'medgemma:4b')

# Optional cloud fallback. Never required to boot.
GEMMA_MODEL = os.getenv('GEMMA_MODEL', 'gemma-4-31b-it')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', GEMMA_MODEL)

# Every model call is cached on disk. The demo replays from cache when the
# network (or the rate limit) gives up.
LLM_CACHE_DIR = Path(os.getenv('LLM_CACHE_DIR', Path(__file__).resolve().parents[1] / '.cache' / 'llm'))
LLM_CACHE_ENABLED = os.getenv('LLM_CACHE_ENABLED', '1') not in ('0', 'false', 'False')

# ---------------------------------------------------------------- data
# Storage falls back to this directory when R2 credentials are absent.
DATA_DIR = Path(os.getenv('DATA_DIR', Path(__file__).resolve().parents[1] / 'data'))
UPLOAD_DIR = Path(os.getenv('UPLOAD_DIR', DATA_DIR / 'uploads'))
PUBLIC_BASE_URL = os.getenv('PUBLIC_BASE_URL', f'http://localhost:{PORT}')

# ---------------------------------------------------------------- voice
# Audio transcription: local faster-whisper, or a clean "text only" degrade.
# No cloud speech provider, no key. Never a crash.
STT_PROVIDER = os.getenv('STT_PROVIDER', 'whisper').strip().lower()
WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'base')
R2_ENDPOINT = os.getenv('R2_ENDPOINT')
R2_ACCESS_KEY = os.getenv('R2_ACCESS_KEY')
R2_SECRET_KEY = os.getenv('R2_SECRET_KEY')
R2_BUCKET=os.getenv('R2_BUCKET')
R2_PUBLIC_URL=os.getenv('R2_PUBLIC_URL')

# JWT Configuration - with defaults for development
SECRET_KEY = os.getenv('SECRET_KEY')
if not SECRET_KEY:
    # Default secret key for development only - CHANGE IN PRODUCTION!
    SECRET_KEY = "dev-secret-key-change-in-production-please-use-strong-random-key"
    print("WARNING: Using default SECRET_KEY. Set SECRET_KEY in .env for production!")

ALGORITHM = os.getenv('ALGORITHM', 'HS256')
ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', '10080')  # 7 days default

password_hash = PasswordHash.recommended()

# Use HTTPBearer for simpler Swagger UI (just Bearer token, no OAuth2 flow)
http_bearer = HTTPBearer(auto_error=False)