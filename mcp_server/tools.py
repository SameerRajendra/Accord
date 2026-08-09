"""FastMCP tool layer exposing Accord's negotiation-intelligence pipeline.

Three MCP tools — `retrieve_precedent`, `predict_outcome`, `analyze_sentiment`
— each a thin wrapper around the same `analysis.*`/`rag.*` implementation
`agent/tools.py` uses (DESIGN.md §5). Nothing in this module duplicates
pipeline logic.

Run as a stdio MCP server:
    python -m mcp_server.tools
"""

from __future__ import annotations

import logging
from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from analysis.behaviors import BehaviorFlag, detect
from analysis.outcome_service import predict_from_transcript
from analysis.sentiment import PerTurnSentiment, analyze
from data.schema import Transcript
from rag.retriever import RetrievedCase, retrieve

logger = logging.getLogger(__name__)

mcp = FastMCP("accord")


@mcp.tool()
def retrieve_precedent(query: str, k: int = 5) -> List[RetrievedCase]:
    """Return the top-k most similar precedents from the case corpus."""
    return retrieve(query, k=k)


@mcp.tool()
def predict_outcome(transcript: Transcript) -> Optional[float]:
    """Return calibrated P(agreement_reached) in [0,1], or None if the model isn't loaded."""
    return predict_from_transcript(transcript)


@mcp.tool()
def analyze_sentiment(transcript: Transcript) -> List[PerTurnSentiment]:
    """Score every text turn's emotion + escalation in one batched LLM call."""
    return analyze(transcript)


@mcp.tool()
def detect_behaviors(transcript: Transcript) -> List[BehaviorFlag]:
    """Flag extreme-behavior categories (threats, ultimatums, stonewalling, ...)."""
    return detect(transcript)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()
