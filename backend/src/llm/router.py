"""EUNOCURA model router — the only place this project talks to a model.

Open weights by default:

    Ollama (http://localhost:11434/v1)          no key, no cloud, runs offline
      ├── slm profile  → qwen2.5:7b-instruct    Apache-2.0
      └── med profile  → medgemma:4b            open weights

`LLM_PROVIDER=gemini` is an *optional* cloud fallback and `LLM_PROVIDER=none`
disables model calls entirely (deterministic paths still work). Nothing here
raises at import time: a missing Ollama server is a runtime error with a hint,
not a boot failure.

Three jobs, nothing else:

1. `chat_model(profile)` — a LangChain chat object pointed at the local server,
   so the existing `prompt | llm` chains keep working unchanged.
2. `ask()` / `ask_json()` — cached, retried text and JSON calls.
3. `health()` — what preflight and `/healthz` report.

Cache: every response is written to `.cache/llm/<sha256>.json`. If the model
server dies mid-demo the cached answer is replayed instantly instead.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Dict, Optional

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_CACHE_DIR,
    LLM_CACHE_ENABLED,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MED_MODEL,
    OLLAMA_SLM_MODEL,
)

# ------------------------------------------------------------------ profiles

PROFILE_MODELS = {
    "slm": OLLAMA_SLM_MODEL,   # extraction, intent, chat
    "med": OLLAMA_MED_MODEL,   # clinical reasoning / diagnosis
}

DEFAULT_PROFILE = "slm"


class ModelUnavailable(RuntimeError):
    """The model server is not reachable, or the provider is switched off."""


class ModelError(RuntimeError):
    """The model answered, but not with something we can use."""


# ------------------------------------------------------------------ plumbing

FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def provider() -> str:
    return LLM_PROVIDER


def model_for(profile: str = DEFAULT_PROFILE) -> str:
    return PROFILE_MODELS.get(profile, PROFILE_MODELS[DEFAULT_PROFILE])


def model_label(profile: str = DEFAULT_PROFILE) -> str:
    """`ollama:qwen2.5:7b-instruct` — what the UI can show next to an answer."""
    return f"{provider()}:{model_for(profile)}"


def strip_fences(text: str) -> str:
    return FENCE.sub("", (text or "").strip())


def parse_json(text: str) -> Any:
    """First JSON object/array in the text, tolerating fences and chatter."""
    cleaned = strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(opener), cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ModelError(f"model did not return JSON: {text[:300]}")


def chat_model(profile: str = DEFAULT_PROFILE, temperature: float = 0.0):
    """A LangChain chat object. Construction never touches the network.

    Raises ModelUnavailable only when a provider's own package is missing.
    """
    if provider() == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover - dependency hint
            raise ModelUnavailable(
                "LLM_PROVIDER=gemini needs `pip install langchain-google-genai`"
            ) from exc
        if not GEMINI_API_KEY:
            raise ModelUnavailable(
                "LLM_PROVIDER=gemini is set but GEMINI_API_KEY is empty. "
                "Unset LLM_PROVIDER to use local Ollama instead."
            )
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=temperature,
            google_api_key=GEMINI_API_KEY,
            max_retries=0,
        )

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise ModelUnavailable(
            "The model router needs `pip install langchain-openai openai`"
        ) from exc

    # Works for Ollama and any OpenAI-compatible server. The key is a literal:
    # a local server does not authenticate, and no env var is required to boot.
    return ChatOpenAI(
        model=model_for(profile),
        temperature=temperature,
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
        timeout=120,
        max_retries=0,
    )


def _cache_path(key: str):
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return LLM_CACHE_DIR / f"{key}.json"


def _key(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:32]


def _as_text(result: Any) -> str:
    """LangChain returns str or a list of content blocks depending on version."""
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


# ------------------------------------------------------------------ calls


def ask(
    prompt: str,
    *,
    profile: str = DEFAULT_PROFILE,
    temperature: Optional[float] = None,
    cache: bool = True,
) -> str:
    """One cached text call, retried, then a loud failure with a fix hint."""
    if provider() == "none":
        raise ModelUnavailable(
            "LLM_PROVIDER=none: model calls are disabled. Set LLM_PROVIDER=ollama."
        )

    label = model_label(profile)
    key = _key(label, str(temperature), prompt)
    path = _cache_path(key)

    if cache and LLM_CACHE_ENABLED and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["text"]

    model = chat_model(profile, temperature if temperature is not None else 0.0)

    last: Optional[Exception] = None
    for attempt in range(2):
        try:
            text = _as_text(model.invoke(prompt))
            if cache and LLM_CACHE_ENABLED:
                path.write_text(
                    json.dumps(
                        {"model": label, "prompt_sha": key, "text": text},
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            return text
        except Exception as exc:  # noqa: BLE001 - surface the last failure
            last = exc
            time.sleep(0.5 * (attempt + 1))

    raise ModelUnavailable(
        f"{label} failed: {last}. Is `ollama serve` running at {OLLAMA_BASE_URL}? "
        f"Start it, then `ollama pull {model_for(profile)}`."
    )


def _schema_prompt(prompt: str, schema: Optional[Dict[str, Any]]) -> str:
    if not schema:
        return prompt
    return (
        f"{prompt}\n\n"
        "Return ONLY a single JSON object that validates against this JSON Schema.\n"
        "No prose before or after it, no markdown fences.\n\n"
        f"{json.dumps(schema, indent=2)}\n"
    )


def ask_json(
    prompt: str,
    schema: Optional[Dict[str, Any]] = None,
    *,
    profile: str = DEFAULT_PROFILE,
    temperature: Optional[float] = None,
    cache: bool = True,
) -> Dict[str, Any]:
    """A text call that must return a JSON object. One repair attempt, then raise.

    Ollama's OpenAI-compatible endpoint accepts a JSON Schema response format;
    we try it first and fall back to prompt-constrained JSON + a repair retry,
    which is what keeps this working against any compatible server.
    """
    if provider() == "gemini":
        # Gemma on the Gemini API has no JSON mode — prompt-constrained only.
        raw = ask(_schema_prompt(prompt, schema), profile=profile, temperature=temperature, cache=cache)
        return _repair_parse(raw, prompt, schema, profile, temperature, cache)

    model = None
    if schema:
        try:
            model = chat_model(profile, temperature if temperature is not None else 0.0)
            bound = model.bind(
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "eunocura_response", "schema": schema},
                }
            )
            raw = _as_text(bound.invoke(f"{prompt}\n\nReturn only the JSON object."))
            return parse_json(raw)  # type: ignore[return-value]
        except ModelUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - fall through to prompt mode
            print(f"  [router] structured JSON mode unavailable ({exc}); using prompt mode")

    raw = ask(_schema_prompt(prompt, schema), profile=profile, temperature=temperature, cache=cache)
    return _repair_parse(raw, prompt, schema, profile, temperature, cache)


def _repair_parse(
    raw: str,
    prompt: str,
    schema: Optional[Dict[str, Any]],
    profile: str,
    temperature: Optional[float],
    cache: bool,
) -> Dict[str, Any]:
    try:
        parsed = parse_json(raw)
        if isinstance(parsed, dict):
            return parsed
        raise ModelError("model returned JSON that is not an object")
    except ModelError as first:
        repair = (
            f"{_schema_prompt(prompt, schema)}\n\n"
            f"Your previous reply was not a valid JSON object.\n"
            f"--- reply ---\n{raw[:1500]}\n--- end ---\n"
            f"Return ONLY the corrected JSON object."
        )
        raw2 = ask(repair, profile=profile, temperature=temperature, cache=cache)
        try:
            parsed = parse_json(raw2)
            if isinstance(parsed, dict):
                return parsed
            raise ModelError("model returned JSON that is not an object")
        except ModelError:
            raise ModelError(
                f"unparseable JSON twice. first={first}. second={raw2[:300]}"
            )


# ------------------------------------------------------------------ health


def health(timeout: float = 2.0) -> Dict[str, Any]:
    """What `/healthz` and `scripts/preflight.py` report. Never raises."""
    out: Dict[str, Any] = {
        "provider": provider(),
        "profile_models": dict(PROFILE_MODELS),
        "reachable": False,
        "installed": [],
        "missing_models": [],
        "hint": "",
    }

    if provider() == "none":
        out["hint"] = "LLM_PROVIDER=none — model features are switched off by config."
        return out

    if provider() == "gemini":
        out["reachable"] = bool(GEMINI_API_KEY)
        out["endpoint"] = GEMINI_MODEL
        if not GEMINI_API_KEY:
            out["hint"] = "GEMINI_API_KEY is empty; unset LLM_PROVIDER to use Ollama."
        return out

    out["endpoint"] = OLLAMA_BASE_URL
    try:
        import httpx

        response = httpx.get(f"{OLLAMA_BASE_URL.rstrip('/')}/models", timeout=timeout)
        response.raise_for_status()
        names = [m.get("id", "") for m in response.json().get("data", [])]
        out["installed"] = names
        out["reachable"] = True
        wanted = set(PROFILE_MODELS.values())
        out["missing_models"] = sorted(m for m in wanted if m not in names)
        if out["missing_models"]:
            out["hint"] = "ollama pull " + " && ollama pull ".join(out["missing_models"])
    except Exception as exc:  # noqa: BLE001 - health must never raise
        out["hint"] = (
            f"Ollama not reachable at {OLLAMA_BASE_URL} ({exc.__class__.__name__}). "
            "Run `ollama serve`, then `ollama pull qwen2.5:7b-instruct`."
        )
    return out
