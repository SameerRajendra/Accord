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
    HealthResponse,
    RecommendationPayload,
)

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
        description="Analyze negotiation transcripts: sentiment, extreme-behavior flags, "
        "breakdown risk, precedent retrieval, and a de-escalation recommendation.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            sglang_ready=_check_sglang_ready(),
            outcome_model_loaded=_check_outcome_model_loaded(),
            rag_configured=bool(os.environ.get("DATABASE_URL")),
        )

    @app.post("/analyze", response_model=AnalyzeResponse)
    def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
        try:
            final = run_graph(
                req.transcript,
                use_rag=req.use_rag,
                retrieval_query=req.retrieval_query,
            )
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
            behaviors=final.get("behaviors", []),
            outcome_prob=final.get("outcome_prob"),
            retrieved=final.get("retrieved", []),
            recommendation=rec_payload,
        )

    return app


app = build_app()
