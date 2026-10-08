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
from rag.graph_retriever import (
    GraphQueryPlan,
    GraphRetrievedCase,
    graph_retrieve,
    plan_for_transcript,
)
from rag.retriever import RetrievedCase, retrieve


def analyze_sentiment_tool(transcript: Transcript) -> List[PerTurnSentiment]:
    return analyze(transcript)


def detect_behaviors_tool(transcript: Transcript) -> List[BehaviorFlag]:
    return detect(transcript)


def predict_outcome_tool(transcript: Transcript) -> Optional[float]:
    return predict_from_transcript(transcript)


def retrieve_precedent_tool(
    query: str, k: int = 5, namespace: Optional[str] = None
) -> List[RetrievedCase]:
    """Vector precedent. `namespace=None` reads the frozen CaSiNo benchmark.

    A live namespace is the product path — the user's own ingested contracts,
    threads and playbooks. The benchmark is what the evals score against and
    what a demo falls back to when nothing has been ingested yet.
    """
    return retrieve(query, k=k, namespace=namespace)


def plan_precedent_graph_tool(transcript: Transcript, query: str) -> GraphQueryPlan:
    """Resolve a transcript (+ query) to graph anchors, without retrieving.

    Pure and deterministic — regex over a hand-written lexicon plus the
    parties' priority rankings, no I/O and no LLM. The agent builds the plan
    first so it can report *why* a graph arm returned nothing: an empty plan
    means the traversal never ran, which is an ordinary outcome for an input
    outside the corpus domain, not a broken deployment.
    """
    return plan_for_transcript(transcript, query=query)


def retrieve_precedent_graph_tool(
    transcript: Transcript,
    query: str,
    k: int = 5,
    use_vector: bool = True,
    plan: Optional[GraphQueryPlan] = None,
    namespace: Optional[str] = None,
) -> List[GraphRetrievedCase]:
    """Graph-anchored precedent, optionally fused with vector similarity.

    Anchors on the *transcript* rather than only on the query text because the
    agent holds the real `Transcript`: the parties' priority rankings make the
    conflict structure and the contested/traded issues known facts rather than
    words to be guessed from the last four turns (infra/graph/README.md §8).
    `query` is merged in on top, so anything the caller typed still anchors.

    `use_vector=False` is the graph-only ablation arm; `True` is the hybrid.
    Returns `GraphRetrievedCase`, a subclass of `RetrievedCase`, so the agent's
    state schema needs no change.

    Pass `plan` to reuse one built by `plan_precedent_graph_tool` — same
    result, one less planning pass, and it guarantees the plan the caller
    reports on is the plan that ran.

    Degradation is inherited, not added: if the graph tables are missing or
    unreachable `graph_retrieve` logs and returns the vector result instead of
    raising. That is deliberate but silent — `agent.graph._node_retrieve`
    inspects the hits afterwards and records whether the graph actually
    contributed, so a misconfiguration doesn't just look like a weak graph.
    """
    plan = plan if plan is not None else plan_precedent_graph_tool(transcript, query)
    return graph_retrieve(
        query or "", k=k, plan=plan, use_vector=use_vector, namespace=namespace
    )
