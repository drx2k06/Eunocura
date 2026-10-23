"""Medical-profile chat model.

Historically this pinned `m42-health/Llama3-Med42-70B` on Featherless AI and
raised `ValueError` at import when `FEATHERLESS_API_KEY` was absent — which
meant a clean clone could not even start the server. Neither is true now:

* no key, no import-time failure;
* default is the open-weight `med` profile served locally by Ollama
  (`OLLAMA_MED_MODEL`, default `medgemma:4b`);
* point `OLLAMA_MED_MODEL` at any medical model you have pulled instead.
"""

from src.llm.router import chat_model

# `med` profile: clinical reasoning and diagnosis generation.
llm = chat_model("med")
