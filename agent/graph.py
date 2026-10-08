"""LangGraph agent: analyze → retrieve → recommend.

Five analysis nodes (sentiment, behaviors, stance, outcome, retrieve) fan out
from START in parallel — each writes distinct keys into `AgentState`, so
LangGraph runs them concurrently without state collisions. `recommend`
barriers on all five before generating the final recommendation.

Two orthogonal knobs drive the retrieval ablation, both on `AgentState` and
neither duplicating the graph:

- `use_rag` — off runs the no-RAG control (DESIGN.md §7 signature experiment).
- `retrieval_mode` — which retriever answers when RAG is on: `vector`
  (pgvector baseline), `graph` (graph traversal only), or `hybrid` (both,
  fused). Ignored when `use_rag=False`, so there is no undefined combination.

That gives four arms — none / vector / graph / hybrid — off two booleans-worth
of state (infra/graph/README.md §8).
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import List, Optional, Sequence, Union

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agent.callbacks import langfuse_callbacks
from agent.llm import chat_model
from agent.tools import (
    analyze_sentiment_tool,
    detect_behaviors_tool,
    plan_precedent_graph_tool,
    predict_outcome_tool,
    retrieve_precedent_graph_tool,
    retrieve_precedent_tool,
)
from analysis.behaviors import BehaviorFlag
from analysis.sentiment import PerTurnSentiment
from analysis.stance import Direction, PartyStance, Trajectory
from analysis.stance import analyze as analyze_stance
from data.schema import Transcript
from rag.retriever import RetrievedCase

logger = logging.getLogger(__name__)


class RetrievalMode(str, Enum):
    """Which retriever answers when RAG is on. Orthogonal to `use_rag`."""

    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


#: What `_node_retrieve` reports when RAG is off. Not a member of
#: `RetrievalMode` on purpose: "no retrieval" is the `use_rag` axis, and
#: making it a mode would recreate the undefined `use_rag=False, mode=graph`
#: combination the two-knob split exists to avoid.
RETRIEVAL_MODE_NONE = "none"


class RetrievalInfo(BaseModel):
    """What the retrieve node actually did — not what it was asked to do.

    Exists because `rag.graph_retriever.graph_retrieve` degrades to
    vector-only when the graph tables are missing or unreachable, by design
    and *without raising* (infra/graph/README.md §6). Without this, that
    misconfiguration is indistinguishable from "the graph didn't help" — which
    is exactly the way to accidentally publish a benchmark of a graph that was
    never queried.
    """

    mode: str = Field(..., description="Arm requested: 'none' | 'vector' | 'graph' | 'hybrid'.")
    n_retrieved: int = 0
    n_graph_grounded: int = Field(
        default=0, description="Hits carrying at least one graph-evidence line."
    )
    graph_effective: bool = Field(
        default=False,
        description="A graph arm was requested AND at least one hit came back with graph evidence.",
    )
    plan: Optional[str] = Field(
        default=None,
        description="One-line summary of the anchors the graph planner resolved. Null on the "
        "vector and no-RAG arms, where no plan is built.",
    )
    namespace: Optional[str] = Field(
        default=None,
        description="Live knowledge-base namespace that was searched. Null means the frozen "
        "CaSiNo benchmark corpus was searched instead.",
    )
    corpus: str = Field(
        default="benchmark",
        description="'live' (the user's ingested documents) or 'benchmark' (CaSiNo). Which one "
        "answered decides how much a precedent is worth: benchmark hits are campsite "
        "bartering and are a demo, not domain-relevant advice.",
    )
    note: str = Field(default="", description="Populated only when something needs explaining.")


class Recommendation(BaseModel):
    """Structured output for the recommendation node."""

    next_move: str = Field(..., description="One concrete next action for the negotiator.")
    tactic: str = Field(
        ...,
        description=(
            "Named tactic used: one of mirror / label / calibrated-question / "
            "accusation-audit / value-swap / walk-away / other."
        ),
    )
    rationale: str = Field(
        ..., description="Why this move, grounded in the analysis + retrieved cases."
    )
    grounded_case_ids: List[str] = Field(
        default_factory=list,
        description="Which retrieved case_ids this recommendation cites. Empty when no-RAG.",
    )


class AgentState(TypedDict, total=False):
    transcript: Transcript
    use_rag: bool
    retrieval_mode: RetrievalMode
    retrieval_query: Optional[str]
    #: Live knowledge-base namespace to retrieve from. `None` reads the frozen
    #: CaSiNo benchmark corpus instead — see rag/documents.py for why those are
    #: two collections and not one.
    namespace: Optional[str]
    sentiment: List[PerTurnSentiment]
    behaviors: List[BehaviorFlag]
    party_stances: List[PartyStance]
    trajectory: Optional[Trajectory]
    outcome_prob: Optional[float]
    # `GraphRetrievedCase` subclasses `RetrievedCase`, so the graph arms need
    # no widening here — they just carry extra fields a reader may ignore.
    retrieved: List[RetrievedCase]
    retrieval_info: RetrievalInfo
    recommendation: Recommendation


# --- nodes ----------------------------------------------------------------


def _node_sentiment(state: AgentState) -> AgentState:
    return {"sentiment": analyze_sentiment_tool(state["transcript"])}


def _node_behaviors(state: AgentState) -> AgentState:
    return {"behaviors": detect_behaviors_tool(state["transcript"])}


def _node_stance(state: AgentState) -> AgentState:
    # Called directly rather than through `agent/tools.py`: that module exists to
    # keep the MCP and agent surfaces over one implementation, and stance isn't
    # exposed over MCP, so a wrapper would have nothing on the other side.
    # One call returns both keys — see analysis/stance.py for why they share it.
    report = analyze_stance(state["transcript"])
    return {"party_stances": report.parties, "trajectory": report.trajectory}


def _node_outcome(state: AgentState) -> AgentState:
    return {"outcome_prob": predict_outcome_tool(state["transcript"])}


def _default_query(transcript: Transcript) -> str:
    """Concatenate the last few text turns as the retrieval query."""
    text_turns = [t for t in transcript.turns if t.action is None]
    tail = text_turns[-4:] if len(text_turns) > 4 else text_turns
    return " ".join(f"{t.speaker}: {t.text}" for t in tail)


def _graph_evidence_count(cases: Sequence[RetrievedCase]) -> int:
    """How many hits carry actual graph provenance.

    Counts `evidence`, **not** `matched_by`. In a hybrid result
    `rag.graph_retriever.fuse` gives a vector-only candidate the placeholder
    line "vector similarity only — no graph anchor matched this case", so
    `matched_by` is non-empty for hits the graph never touched; counting it
    would report a graph as effective on a run where the traversal returned
    nothing. `evidence` is empty for exactly those hits.

    `getattr` rather than `isinstance` because the vector arm returns plain
    `RetrievedCase`: this measures whether the *graph* contributed, not which
    code path was taken.
    """
    return sum(1 for case in cases if getattr(case, "evidence", None))


def _retrieval_info(
    mode: str,
    cases: Sequence[RetrievedCase],
    plan: Optional[object] = None,
    namespace: Optional[str] = None,
) -> RetrievalInfo:
    """Describe what retrieval did. `plan` is a `GraphQueryPlan` on graph arms.

    Three distinguishable ways to come back with nothing useful, because they
    have three different fixes and only one of them is a fault:

    * **Empty live namespace** — nothing has been ingested yet. Expected on a
      fresh deployment; the fix is to add documents, not to debug retrieval.
    * **Empty plan** (graph arms) — see below.
    * **Plan built, no evidence** (graph arms) — see below.

    The two ways a graph arm comes back empty-handed are separated on purpose:

    * **Empty plan** — nothing in the transcript or query resolved to a graph
      anchor, so the traversal never ran. Ordinary and expected for input
      outside the corpus domain (a business email thread has no campsite
      priority rankings and rarely trips the CaSiNo lexicon). Not a fault.
    * **Plan built, no evidence returned** — the traversal ran and produced
      nothing, or the graph tables are missing and `graph_retrieve` degraded
      to vector-only without raising. That one needs investigating.

    Collapsing these into one "graph didn't help" is how a graph that was
    never consulted ends up recorded as a graph that underperformed.
    """
    grounded = _graph_evidence_count(cases)
    wants_graph = mode in (RetrievalMode.GRAPH.value, RetrievalMode.HYBRID.value)
    plan_empty = bool(plan is not None and plan.is_empty())  # type: ignore[union-attr]

    note = ""
    if namespace is not None and not cases:
        # Checked before the graph diagnostics: an empty knowledge base
        # explains an empty result completely, and blaming the graph tables
        # for it would send someone to debug the wrong subsystem.
        note = (
            f"namespace '{namespace}' has no documents matching this query — if you have "
            "not ingested anything yet, that is the expected result and the recommendation "
            "is running ungrounded (same as the no-RAG arm). Add precedent with POST "
            "/corpus/documents."
        )
    elif wants_graph and not grounded:
        if plan_empty:
            note = (
                "no graph anchors resolved from this transcript or query, so the traversal "
                "never ran — these results are vector-only. Expected for input outside the "
                "corpus domain: the planner anchors on campsite issues, CaSiNo persuasion "
                "tactics and priority rankings, none of which a business email thread carries."
            )
        else:
            note = (
                f"the '{mode}' arm was requested and anchors resolved, but no hit came back with "
                "graph evidence — graph_retrieve degrades to vector-only when the graph tables "
                "are missing or unreachable, so verify before reading anything into this: "
                'python -c "from rag.graph_db import graph_is_populated; '
                'print(graph_is_populated())"'
            )
    elif mode == RetrievalMode.VECTOR.value and grounded:
        # Would mean a graph-scored hit reached the vector arm — a wiring bug,
        # not a data condition, so say so rather than silently reporting it.
        note = "vector arm returned graph-grounded hits — unexpected; check _node_retrieve dispatch"

    return RetrievalInfo(
        mode=mode,
        n_retrieved=len(cases),
        n_graph_grounded=grounded,
        graph_effective=bool(wants_graph and grounded),
        plan=plan.describe() if plan is not None else None,  # type: ignore[union-attr]
        namespace=namespace,
        corpus="live" if namespace is not None else "benchmark",
        note=note,
    )


def _node_retrieve(state: AgentState) -> AgentState:
    if not state.get("use_rag", True):
        return {
            "retrieved": [],
            "retrieval_info": RetrievalInfo(
                mode=RETRIEVAL_MODE_NONE,
                note="no-RAG control arm: retrieval was skipped, not attempted and empty",
            ),
        }

    mode = RetrievalMode(state.get("retrieval_mode") or RetrievalMode.VECTOR)
    query = state.get("retrieval_query") or _default_query(state["transcript"])
    namespace = state.get("namespace")

    # This is the ONLY analysis node that reaches an external service (Neon), and
    # scale-to-zero makes that connection the pipeline's flakiest point. Every
    # other node degrades to a safe default on failure; retrieval must too, or a
    # sleepy database turns a full, useful analysis (sentiment, stance,
    # trajectory, behaviors, recommendation) into a 500 for whoever is watching.
    # The recommendation node already handles empty precedent — it becomes the
    # ungrounded (no-RAG) path — so degrading here is a real answer, not a stub.
    #
    # The graph arms catch their own I/O errors and fall back to vector inside
    # `graph_retrieve`; the vector arm calls `retrieve` directly, which raises on
    # a genuine connection failure. Both are covered by this one guard.
    try:
        if mode is RetrievalMode.VECTOR:
            cases: List[RetrievedCase] = list(
                retrieve_precedent_tool(query, k=5, namespace=namespace)
            )
            return {
                "retrieved": cases,
                "retrieval_info": _retrieval_info(mode.value, cases, namespace=namespace),
            }

        # Plan first, then retrieve with that same plan: the diagnosis in
        # `_retrieval_info` has to describe the anchors that actually ran, not a
        # second planning pass that might disagree.
        plan = plan_precedent_graph_tool(state["transcript"], query)
        cases = list(
            retrieve_precedent_graph_tool(
                state["transcript"],
                query=query,
                k=5,
                use_vector=(mode is RetrievalMode.HYBRID),
                plan=plan,
                namespace=namespace,
            )
        )
        return {
            "retrieved": cases,
            "retrieval_info": _retrieval_info(mode.value, cases, plan, namespace=namespace),
        }
    except Exception as exc:  # noqa: BLE001 — a retrieval outage must not sink the analysis
        logger.exception("retrieval node failed; degrading to no precedent: %s", exc)
        return {
            "retrieved": [],
            "retrieval_info": RetrievalInfo(
                mode=mode.value,
                n_retrieved=0,
                namespace=namespace,
                corpus="live" if namespace is not None else "benchmark",
                note=(
                    f"retrieval was unavailable this run ({type(exc).__name__}: {exc}) — "
                    "the analysis continued without precedent, so the recommendation is ungrounded "
                    "(same as the "
                    "no-RAG arm). This is a degraded response, not an error."
                ),
            ),
        }


_RECOMMEND_SYSTEM = (
    "You are a negotiation coach. Given the analysis of a transcript and (optionally) "
    "relevant precedent cases, recommend one concrete next move for the first party "
    "(the one who should act next to steer toward a better, non-broken outcome). Use a "
    "named tactic from {mirror, label, calibrated-question, accusation-audit, "
    "value-swap, walk-away, other}. When precedent cases are provided, cite the "
    "case_ids you actually used in `grounded_case_ids`, copied character-for-character "
    "from the bracketed id above the case — do not shorten, reformat, or deduplicate "
    "them, and never cite an id that is not printed in the context. Any claim you make "
    "about a precedent must be supported by that case's own text or by its 'why this "
    "case' evidence lines. Leave `grounded_case_ids` empty when no cases are provided. "
    "Rationale must reference specific signals from the analysis, not restate the "
    "transcript."
)


def _format_analysis(state: AgentState) -> str:
    parts: List[str] = []
    sentiment = state.get("sentiment") or []
    if sentiment:
        parts.append("SENTIMENT (per-turn, latest 5):")
        for s in sentiment[-5:]:
            parts.append(
                f"  turn {s.turn_index}: {s.emotion.value} "
                f"(escalation={s.escalation:.2f}) — {s.rationale}"
            )

    stances = state.get("party_stances") or []
    if stances:
        parts.append("PARTY STANCE (whole thread, per participant):")
        for ps in stances:
            turns = ", ".join(str(i) for i in ps.evidence_turns) or "none cited"
            parts.append(
                f"  - {ps.party}: mood={ps.mood.value}, flexibility={ps.flexibility.value}; "
                f"holding: {ps.position or 'unstated'} (turns {turns}) — {ps.rationale}"
            )

    trajectory = state.get("trajectory")
    if trajectory is not None and trajectory.direction != Direction.UNKNOWN:
        turned = (
            f", tone turned at turn {trajectory.turning_point_turn}"
            if trajectory.turning_point_turn is not None
            else ""
        )
        parts.append(
            f"TRAJECTORY: {trajectory.direction.value} "
            f"(confidence={trajectory.confidence:.2f}{turned}) — {trajectory.reasoning}"
        )
    else:
        # Say so rather than omit it — a missing section reads as "calm" to the model.
        parts.append("TRAJECTORY: (not available — the stance stage returned no reading)")

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
        info = state.get("retrieval_info")
        mode = info.mode if info is not None else RetrievalMode.VECTOR.value
        corpus = info.corpus if info is not None else "benchmark"
        header = f"RETRIEVED PRECEDENTS (retrieval={mode}, corpus={corpus}):"
        if corpus == "benchmark":
            # Without this the model will cheerfully advise a contract dispute
            # from campsite-bartering precedent and present it as domain
            # experience. The corpus mismatch is a known limitation; hiding it
            # from the model makes the output overclaim.
            header += (
                "\n  NOTE: these come from a campsite resource-bartering research corpus, "
                "not this organisation's own history. Use them for tactical structure only, "
                "and do not present them as comparable deals."
            )
        parts.append(header)
        for r in retrieved:
            snippet = r.text if len(r.text) <= 400 else r.text[:400] + "…"
            # `citation_label` names a live chunk by document title + clause,
            # which is checkable; the bare id is not.
            label = r.citation_label()
            title = f' "{label}"' if label != r.case_id else ""
            parts.append(f"  [{r.case_id}]{title} (score={r.score:.3f}) {snippet}")
            # The relation that justified the hit, not just the text. This is
            # the direct countermeasure to the citation fabrication logged in
            # evals/agent_eval.py: a model that is told *why* a case was
            # retrieved has less room to invent a reason, and a reader can
            # check the claim against the stated relation afterwards.
            for line in getattr(r, "matched_by", None) or []:
                parts.append(f"      · why this case: {line}")
    else:
        parts.append("RETRIEVED PRECEDENTS: (none — RAG disabled or empty result)")

    return "\n".join(parts)


def _node_recommend(state: AgentState) -> AgentState:
    model = chat_model(temperature=0.2, max_tokens=512).with_structured_output(Recommendation)
    analysis = _format_analysis(state)
    prompt = [
        ("system", _RECOMMEND_SYSTEM),
        ("user", analysis + "\n\nWhat is the recommended next move?"),
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
    g.add_node("stance", _node_stance)
    g.add_node("outcome", _node_outcome)
    g.add_node("retrieve", _node_retrieve)
    g.add_node("recommend", _node_recommend)

    # Fan out from START to all five analysis nodes (LangGraph runs them
    # concurrently; each writes state keys no other node writes).
    for node in ("sentiment", "behaviors", "stance", "outcome", "retrieve"):
        g.add_edge(START, node)
        g.add_edge(node, "recommend")

    g.add_edge("recommend", END)
    return g.compile()


def run(
    transcript: Transcript,
    use_rag: bool = True,
    retrieval_query: Optional[str] = None,
    retrieval_mode: Union[RetrievalMode, str] = RetrievalMode.VECTOR,
    namespace: Optional[str] = None,
) -> AgentState:
    """Convenience: build (or reuse) the graph and run one transcript through.

    `retrieval_query` overrides the default (last-few-turns-concatenated) query
    used by `_node_retrieve`. Ignored when `use_rag=False`.

    `retrieval_mode` picks the arm — `"vector"` (default, the deployed
    baseline), `"graph"`, or `"hybrid"`. Also ignored when `use_rag=False`.
    Accepts the enum or its string value so a caller reading a mode off a
    request body or a CLI flag doesn't have to convert first; an unknown
    string raises `ValueError` here rather than silently falling back to
    vector, because a benchmark arm that quietly ran a different retriever
    than it reported is worse than a crash.

    `namespace` selects the live knowledge base to retrieve from. Left `None`
    it reads the frozen CaSiNo benchmark corpus, which keeps every existing
    eval and the RAG ablation pointed at labeled data — the live path is the
    product, the benchmark path is the measurement, and they are deliberately
    different collections (`rag/documents.py`).
    """
    graph = _cached_graph()
    initial: AgentState = {
        "transcript": transcript,
        "use_rag": use_rag,
        "retrieval_mode": RetrievalMode(retrieval_mode),
        "namespace": namespace,
    }
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
