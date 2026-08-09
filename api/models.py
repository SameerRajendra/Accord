"""Pydantic v2 request/response contracts for the API.

Public API contract: `AnalyzeRequest` (a `Transcript` + `use_rag` toggle) →
`AnalyzeResponse` (per-turn sentiment, extreme-behavior flags, outcome
probability, retrieved precedents, recommendation). Kept intentionally close
to the domain models — the API is a thin surface over `agent.graph.run`, not
a translation layer.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from analysis.behaviors import BehaviorFlag
from analysis.sentiment import PerTurnSentiment
from data.schema import Transcript
from rag.retriever import RetrievedCase


class AnalyzeRequest(BaseModel):
    transcript: Transcript
    use_rag: bool = Field(True, description="Toggle RAG for the ablation. Defaults to on.")
    retrieval_query: Optional[str] = Field(
        default=None,
        description="Override the query used for precedent retrieval. Defaults to the last few turns.",
    )


class RecommendationPayload(BaseModel):
    next_move: str
    tactic: str
    rationale: str
    grounded_case_ids: List[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    sentiment: List[PerTurnSentiment]
    behaviors: List[BehaviorFlag]
    outcome_prob: Optional[float] = Field(
        None,
        description="Calibrated P(agreement_reached); null if the outcome model artifact is missing.",
    )
    retrieved: List[RetrievedCase]
    recommendation: RecommendationPayload


class HealthResponse(BaseModel):
    status: str
    sglang_ready: bool
    outcome_model_loaded: bool
    rag_configured: bool
