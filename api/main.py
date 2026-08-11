"""FastAPI app exposing /analyze and /health.

Runs as a Modal ASGI app colocated with SGLang in the same GPU container
(see [infra/modal/app.py](../infra/modal/app.py)). The FastAPI process talks
to SGLang over localhost, so no network hop leaves the container for
inference.

`build_app()` is called by Modal's ASGI adapter; running this file directly
also works for a local dev loop pointed at any SGLang endpoint (set
`SGLANG_BASE_URL`).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException

from agent.graph import run as run_graph
from api.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalyzeThreadRequest,
    HealthResponse,
    ParsedTurnPayload,
    RecommendationPayload,
)
from data.schema import Transcript

logger = logging.getLogger(__name__)


def _check_sglang_ready(base_url: Optional[str] = None) -> bool:
    url = base_url or os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000/v1")
    try:
        r = httpx.get(url.rstrip("/") + "/models", timeout=2.0)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _check_outcome_model_loaded() -> bool:
    from analysis.outcome_service import MissingModelError, _load

    try:
        _load()
        return True
    except MissingModelError:
        return False
    except Exception:  # noqa: BLE001
        return False


def build_app() -> FastAPI:
    app = FastAPI(
        title="Accord — Negotiation Intelligence API",
        version="0.1.0",
        description="Analyze negotiation transcripts: sentiment, per-party stance, where the "
        "discussion is heading, extreme-behavior flags, breakdown risk, precedent retrieval, "
        "and a de-escalation recommendation.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            sglang_ready=_check_sglang_ready(),
            outcome_model_loaded=_check_outcome_model_loaded(),
            rag_configured=bool(os.environ.get("DATABASE_URL")),
        )

    def _run(
        transcript: Transcript,
        use_rag: bool,
        retrieval_query: Optional[str],
        parsed: Optional[list] = None,
    ) -> AnalyzeResponse:
        """Shared analysis path for both entry points."""
        try:
            final = run_graph(transcript, use_rag=use_rag, retrieval_query=retrieval_query)
        except Exception as exc:  # noqa: BLE001
            logger.exception("graph invocation failed")
            raise HTTPException(status_code=500, detail=f"analysis failed: {exc}") from exc

        rec = final.get("recommendation")
        rec_payload = RecommendationPayload(
            next_move=rec.next_move if rec else "",
            tactic=rec.tactic if rec else "other",
            rationale=rec.rationale if rec else "recommendation missing",
            grounded_case_ids=rec.grounded_case_ids if rec else [],
        )
        return AnalyzeResponse(
            sentiment=final.get("sentiment", []),
            party_stances=final.get("party_stances", []),
            # None (not a synthesized "unknown") when the node never ran — the
            # stance stage builds its own explicit unknown when it merely failed.
            trajectory=final.get("trajectory"),
            behaviors=final.get("behaviors", []),
            outcome_prob=final.get("outcome_prob"),
            retrieved=final.get("retrieved", []),
            recommendation=rec_payload,
            parsed=parsed or [],
        )

    @app.post("/analyze", response_model=AnalyzeResponse)
    def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
        """Typed entry point — caller supplies an already-normalized Transcript."""
        return _run(req.transcript, req.use_rag, req.retrieval_query)

    @app.post("/analyze/thread", response_model=AnalyzeResponse)
    def analyze_thread(req: AnalyzeThreadRequest) -> AnalyzeResponse:
        """Raw-text entry point — paste an email thread, the LLM structures it.

        Parse failures return 422 (the caller's input is unusable) rather than
        500, so a malformed paste is distinguishable from a broken pipeline.
        """
        from analysis.parse_thread import ThreadParseError, parse_thread

        try:
            transcript = parse_thread(req.thread_text)
        except ThreadParseError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("unexpected thread-parsing failure")
            raise HTTPException(status_code=500, detail=f"thread parsing failed: {exc}") from exc

        parsed = [
            ParsedTurnPayload(index=t.index, speaker=t.speaker, text=t.text)
            for t in transcript.turns
        ]
        return _run(transcript, req.use_rag, req.retrieval_query, parsed=parsed)

    return app


app = build_app()
