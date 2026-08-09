"""LangGraph-facing tool wrappers.

Thin adapters over the same `analysis.*`/`rag.*` implementations that
`mcp_server/tools.py` exposes to MCP clients (DESIGN.md §5, "two thin
adapters over one implementation, not two implementations"). Nothing in this
module talks to the MCP server as a client — the agent calls the underlying
functions directly to avoid a pointless self-network-hop.
"""

from __future__ import annotations

from typing import List, Optional

from analysis.behaviors import BehaviorFlag, detect
from analysis.outcome_service import predict_from_transcript
from analysis.sentiment import PerTurnSentiment, analyze
from data.schema import Transcript
from rag.retriever import RetrievedCase, retrieve


def analyze_sentiment_tool(transcript: Transcript) -> List[PerTurnSentiment]:
    return analyze(transcript)


def detect_behaviors_tool(transcript: Transcript) -> List[BehaviorFlag]:
    return detect(transcript)


def predict_outcome_tool(transcript: Transcript) -> Optional[float]:
    return predict_from_transcript(transcript)


def retrieve_precedent_tool(query: str, k: int = 5) -> List[RetrievedCase]:
    return retrieve(query, k=k)
