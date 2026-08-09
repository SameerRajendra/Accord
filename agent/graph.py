"""LangGraph agent: analyze → retrieve → recommend.

Four analysis nodes (sentiment, behaviors, outcome, retrieve) fan out from
START in parallel — each writes a distinct key into `AgentState`, so
LangGraph runs them concurrently without state collisions. `recommend`
barriers on all four before generating the final recommendation.

The `use_rag` flag on `AgentState` gates the retrieve node — set `False` to
run the RAG-vs-no-RAG ablation (DESIGN.md §7 signature experiment) without
duplicating the graph.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agent.callbacks import langfuse_callbacks
from agent.llm import chat_model
from agent.tools import (
    analyze_sentiment_tool,
    detect_behaviors_tool,
    predict_outcome_tool,
    retrieve_precedent_tool,
)
from analysis.behaviors import BehaviorFlag
from analysis.sentiment import PerTurnSentiment
from data.schema import Transcript
from rag.retriever import RetrievedCase

logger = logging.getLogger(__name__)


class Recommendation(BaseModel):
    """Structured output for the recommendation node."""

    next_move: str = Field(..., description="One concrete next action for the negotiator.")
    tactic: str = Field(
        ...,
        description="Named tactic used: one of mirror / label / calibrated-question / accusation-audit / value-swap / walk-away / other.",
    )
    rationale: str = Field(..., description="Why this move, grounded in the analysis + retrieved cases.")
    grounded_case_ids: List[str] = Field(
        default_factory=list,
        description="Which retrieved case_ids this recommendation cites. Empty when no-RAG.",
    )


class AgentState(TypedDict, total=False):
    transcript: Transcript
    use_rag: bool
    retrieval_query: Optional[str]
    sentiment: List[PerTurnSentiment]
    behaviors: List[BehaviorFlag]
    outcome_prob: Optional[float]
    retrieved: List[RetrievedCase]
    recommendation: Recommendation


# --- nodes ----------------------------------------------------------------


def _node_sentiment(state: AgentState) -> AgentState:
    return {"sentiment": analyze_sentiment_tool(state["transcript"])}


def _node_behaviors(state: AgentState) -> AgentState:
    return {"behaviors": detect_behaviors_tool(state["transcript"])}


def _node_outcome(state: AgentState) -> AgentState:
    return {"outcome_prob": predict_outcome_tool(state["transcript"])}


def _default_query(transcript: Transcript) -> str:
    """Concatenate the last few text turns as the retrieval query."""
    text_turns = [t for t in transcript.turns if t.action is None]
    tail = text_turns[-4:] if len(text_turns) > 4 else text_turns
    return " ".join(f"{t.speaker}: {t.text}" for t in tail)


def _node_retrieve(state: AgentState) -> AgentState:
    if not state.get("use_rag", True):
        return {"retrieved": []}
    query = state.get("retrieval_query") or _default_query(state["transcript"])
    return {"retrieved": retrieve_precedent_tool(query, k=5)}


_RECOMMEND_SYSTEM = (
    "You are a negotiation coach. Given the analysis of a transcript and (optionally) "
    "relevant precedent cases, recommend one concrete next move for the negotiator "
    "on the buyer side. Use a named tactic from {mirror, label, calibrated-question, "
    "accusation-audit, value-swap, walk-away, other}. When precedent cases are "
    "provided, cite the case_ids you actually used in `grounded_case_ids`; leave it "
    "empty otherwise. Rationale must reference specific signals from the analysis, "
    "not restate the transcript."
)


def _format_analysis(state: AgentState) -> str:
    parts: List[str] = []
    sentiment = state.get("sentiment") or []
    if sentiment:
        parts.append("SENTIMENT (per-turn, latest 5):")
        for s in sentiment[-5:]:
            parts.append(
                f"  turn {s.turn_index}: {s.emotion.value} (escalation={s.escalation:.2f}) — {s.rationale}"
            )

    behaviors = state.get("behaviors") or []
    present = [b for b in behaviors if b.present]
    if present:
        parts.append("EXTREME BEHAVIORS FLAGGED:")
        for b in present:
            parts.append(f"  - {b.name} (conf={b.confidence:.2f}): {b.evidence}")
    else:
        parts.append("EXTREME BEHAVIORS FLAGGED: none")

    outcome = state.get("outcome_prob")
    if outcome is not None:
        parts.append(f"OUTCOME (P(agreement_reached)): {outcome:.2f}")

    retrieved = state.get("retrieved") or []
    if retrieved:
        parts.append("RETRIEVED PRECEDENTS:")
        for r in retrieved:
            snippet = r.text if len(r.text) <= 400 else r.text[:400] + "…"
            parts.append(f"  [{r.case_id}] (score={r.score:.3f}) {snippet}")
    else:
        parts.append("RETRIEVED PRECEDENTS: (none — RAG disabled or empty result)")

    return "\n".join(parts)


def _node_recommend(state: AgentState) -> AgentState:
    model = chat_model(temperature=0.2, max_tokens=512).with_structured_output(Recommendation)
    analysis = _format_analysis(state)
    prompt = [
        ("system", _RECOMMEND_SYSTEM),
        ("user", analysis + "\n\nWhat is the buyer's next move?"),
    ]
    try:
        rec: Recommendation = model.invoke(prompt, config={"callbacks": langfuse_callbacks()})
    except Exception as exc:  # noqa: BLE001
        logger.exception("recommendation LLM call failed: %s", exc)
        rec = Recommendation(
            next_move="Pause and ask a calibrated question to buy time.",
            tactic="calibrated-question",
            rationale=f"recommendation model failed ({type(exc).__name__}); safe default.",
            grounded_case_ids=[],
        )
    return {"recommendation": rec}


# --- graph construction ---------------------------------------------------


def build_graph():
    """Compile the LangGraph state machine. Callers hold the compiled graph."""
    from langgraph.graph import END, START, StateGraph

    g: StateGraph = StateGraph(AgentState)
    g.add_node("sentiment", _node_sentiment)
    g.add_node("behaviors", _node_behaviors)
    g.add_node("outcome", _node_outcome)
    g.add_node("retrieve", _node_retrieve)
    g.add_node("recommend", _node_recommend)

    # Fan out from START to all four analysis nodes (LangGraph runs them
    # concurrently; each writes a distinct state key).
    for node in ("sentiment", "behaviors", "outcome", "retrieve"):
        g.add_edge(START, node)
        g.add_edge(node, "recommend")

    g.add_edge("recommend", END)
    return g.compile()


def run(
    transcript: Transcript,
    use_rag: bool = True,
    retrieval_query: Optional[str] = None,
) -> AgentState:
    """Convenience: build (or reuse) the graph and run one transcript through.

    `retrieval_query` overrides the default (last-few-turns-concatenated) query
    used by `_node_retrieve`. Ignored when `use_rag=False`.
    """
    graph = _cached_graph()
    initial: AgentState = {"transcript": transcript, "use_rag": use_rag}
    if retrieval_query is not None:
        initial["retrieval_query"] = retrieval_query
    final = graph.invoke(initial, config={"callbacks": langfuse_callbacks()})
    return final  # type: ignore[return-value]


_GRAPH = None


def _cached_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH
