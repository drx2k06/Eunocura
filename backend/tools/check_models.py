"""List every Gemma model this key can actually call, then smoke-test the pinned one.

    python backend/tools/check_models.py

Uses the REST endpoint directly (httpx) so the answer never depends on an SDK
version. Reads GEMINI_API_KEY and GEMMA_MODEL from backend/.env.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

# load backend/.env whether we're run from the repo root or from backend/
HERE = Path(__file__).resolve()
for candidate in (HERE.parents[1] / ".env", HERE.parents[2] / ".env"):
    if candidate.exists():
        load_dotenv(candidate)
        break
else:
    load_dotenv()

API = "https://generativelanguage.googleapis.com/v1beta"
KEY = os.getenv("GEMINI_API_KEY", "").strip()
PINNED = os.getenv("GEMMA_MODEL", "").strip()


def die(msg: str) -> None:
    print(f"\n  FAIL  {msg}\n")
    sys.exit(1)


def main() -> None:
    if not KEY:
        die("GEMINI_API_KEY is empty. Put it in backend/.env (not in a chat).")

    print(f"\n  key  ...{KEY[-4:]}   pinned model: {PINNED or '(unset)'}\n")

    # 1. what is actually available on this key
    r = httpx.get(f"{API}/models", params={"key": KEY}, timeout=30)
    if r.status_code != 200:
        die(f"models.list returned {r.status_code}: {r.text[:400]}")

    gemmas = [m for m in r.json().get("models", []) if "gemma" in m["name"].lower()]
    if not gemmas:
        die("no Gemma models on this key — check the key's project / region")

    print("  available Gemma models")
    for m in gemmas:
        methods = ",".join(m.get("supportedGenerationMethods", []))
        print(f"    {m['name']:<45} {methods}")
        print(f"      {m.get('displayName', '')} · in {m.get('inputTokenLimit', '?')} tok")

    # 2. can we actually call the pinned one?
    target = PINNED or gemmas[0]["name"].removeprefix("models/")
    print(f"\n  smoke-testing {target} ...")
    t0 = time.perf_counter()
    r = httpx.post(
        f"{API}/models/{target}:generateContent",
        params={"key": KEY},
        json={
            "contents": [{"parts": [{"text": "Reply with exactly: OK"}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 16},
        },
        timeout=120,
    )
    ms = (time.perf_counter() - t0) * 1000

    if r.status_code != 200:
        die(f"{target} → {r.status_code}: {r.text[:600]}")

    try:
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        die(f"unexpected response shape: {r.text[:600]}")

    print(f'  first token back: "{text}"  in {ms:.0f} ms')
    print(f"\n  PASS  set GEMMA_MODEL={target} in backend/.env\n")


if __name__ == "__main__":
    main()
