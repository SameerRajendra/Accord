"""Pydantic v2 request/response contracts for the API.

Public API contract: `AnalyzeRequest` (a `Transcript` + `use_rag` toggle) →
`AnalyzeResponse` (per-turn sentiment, per-party stance, discussion
trajectory, extreme-behavior flags, outcome probability, retrieved precedents,
recommendation). Kept intentionally close to the domain models — the API is a
thin surface over `agent.graph.run`, not a translation layer.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from analysis.behaviors import BehaviorFlag
from analysis.sentiment import PerTurnSentiment
from analysis.stance import PartyStance, Trajectory
from data.schema import Transcript
from rag.retriever import RetrievedCase


class AnalyzeRequest(BaseModel):
    transcript: Transcript
    use_rag: bool = Field(True, description="Toggle RAG for the ablation. Defaults to on.")
    retrieval_query: Optional[str] = Field(
        default=None,
        description="Override the query used for precedent retrieval. Defaults to the last few turns.",
    )


class AnalyzeThreadRequest(BaseModel):
    """Raw-text entry point: paste an email thread instead of authoring JSON."""

    thread_text: str = Field(
        ...,
        description="A raw email thread, or a plain `Speaker: message` transcript.",
    )
    use_rag: bool = Field(True, description="Toggle RAG for the ablation. Defaults to on.")
    retrieval_query: Optional[str] = Field(
        default=None,
        description="Override the query used for precedent retrieval. Defaults to the last few turns.",
    )


class ParsedTurnPayload(BaseModel):
    """What the parser extracted — echoed back so a caller can verify it."""

    index: int
    speaker: str
    text: str


class RecommendationPayload(BaseModel):
    next_move: str
    tactic: str
    rationale: str
    grounded_case_ids: List[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    sentiment: List[PerTurnSentiment]
    party_stances: List[PartyStance] = Field(
        default_factory=list,
        description="Whole-thread stance per participant, least flexible first. A party the "
        "model skipped comes back with mood/flexibility 'unknown' rather than a guess.",
    )
    trajectory: Optional[Trajectory] = Field(
        None,
        description="Where the discussion is heading. `direction='unknown'` (confidence 0.0) "
        "means the stance stage returned no reading; null means the stage never ran.",
    )
    behaviors: List[BehaviorFlag]
    outcome_prob: Optional[float] = Field(
        None,
        description="Calibrated P(agreement_reached); null if the outcome model artifact is missing.",
    )
    retrieved: List[RetrievedCase]
    recommendation: RecommendationPayload
    parsed: List[ParsedTurnPayload] = Field(
        default_factory=list,
        description="Turns extracted from a raw thread. Empty when a Transcript was supplied "
        "directly — populated only by /analyze/thread, so the caller can check what the "
        "parser understood before trusting the analysis built on it.",
    )


class HealthResponse(BaseModel):
    status: str
    sglang_ready: bool
    outcome_model_loaded: bool
    rag_configured: bool
