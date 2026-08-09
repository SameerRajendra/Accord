"""LangChain `ChatOpenAI` factory pointed at self-hosted SGLang.

Every LLM call in the repo (sentiment, behaviors, recommendation) goes
through `chat_model()` — one place to swap models, adjust temperature, or
retarget the base URL (e.g. to a managed OSS-inference fallback per §1) without
touching call sites.

`base_url` defaults to Modal's colocated SGLang endpoint (`http://127.0.0.1:30000/v1`);
override with env `SGLANG_BASE_URL` to point at a different SGLang instance
(e.g. the cluster's port 30001 during local development). The `api_key` is a
dummy — SGLang doesn't check it — but LangChain requires a non-empty string.

DESIGN.md §5: "OpenAI" names the wire protocol these clients speak
(OpenAI-compatible REST), **not** the provider. No hosted OpenAI API is
involved anywhere in this project.
"""

from __future__ import annotations

import os
from functools import lru_cache


DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@lru_cache(maxsize=4)
def chat_model(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
):
    """Return a `ChatOpenAI` instance pointed at the self-hosted SGLang endpoint.

    Cached per (model, temperature, max_tokens) so repeated calls in the same
    process share one client (and its HTTP connection pool). Not thread-safe
    across process forks; Modal containers are single-process so this is fine.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=os.environ.get("SGLANG_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.environ.get("SGLANG_API_KEY", "sglang-no-auth"),
        model=os.environ.get("SGLANG_MODEL", model),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=120,
    )
