"""Langfuse observability wiring — env-gated so local runs without keys work.

Every LangChain call and LangGraph node in the pipeline attaches the callback
handler returned by `langfuse_callbacks()`. When `LANGFUSE_PUBLIC_KEY` is
unset (local dev, CI), this returns an empty list and LangChain runs
un-instrumented — no crash, no warning spam.

DESIGN.md §5 committed to Langfuse managed cloud (free tier) rather than a
self-hosted deployment; the callback wiring is identical either way — only
`LANGFUSE_HOST` differs.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _handler():
    """Construct the Langfuse CallbackHandler once. `None` if unconfigured."""
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return None
    try:
        # langfuse >= 2.x exposes the LangChain handler at this path; the
        # location has moved across versions, so a failed import is treated
        # as "tracing unavailable" rather than a hard error.
        from langfuse.callback import CallbackHandler  # type: ignore
    except ImportError:
        try:
            from langfuse.langchain import CallbackHandler  # type: ignore
        except ImportError:
            logger.warning(
                "langfuse installed but no CallbackHandler found — tracing disabled"
            )
            return None
    return CallbackHandler()


def langfuse_callbacks() -> List:
    """Return `[handler]` when Langfuse is configured, otherwise `[]`.

    Pass into any LangChain `.invoke(input, config={"callbacks": ...})` call
    or `RunnableConfig` — LangChain accepts an empty list without complaint.
    """
    h = _handler()
    return [h] if h is not None else []
