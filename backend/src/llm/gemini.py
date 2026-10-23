"""Open-weight chat model used by every text feature.

Historically this module pinned Google's proprietary `gemini-2.0-flash-exp`.
EUNOCURA now routes through `src.llm.router`, which defaults to a local
Ollama server (Qwen2.5-7B-Instruct, Apache-2.0). The module name is kept so
existing `prompt | llm` chains and imports do not move.

The cloud path still exists if someone sets `LLM_PROVIDER=gemini`, but it is
never required to boot or to run the demo.
"""

from src.llm.router import chat_model

# `slm` profile: extraction, intent classification, patient chat.
llm = chat_model("slm")
