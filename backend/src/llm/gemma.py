"""Deprecated alias for `src.llm.router`.

Kept only so older imports keep working. New code should import from
`src.llm.router` directly. There is exactly one model client in this project
and it lives there.
"""

from src.llm.router import (  # noqa: F401
    ModelError,
    ModelUnavailable,
    ask,
    ask_json,
    chat_model,
    health,
    model_label,
    parse_json,
    strip_fences,
)

__all__ = [
    "ModelError",
    "ModelUnavailable",
    "ask",
    "ask_json",
    "chat_model",
    "health",
    "model_label",
    "parse_json",
    "strip_fences",
]
