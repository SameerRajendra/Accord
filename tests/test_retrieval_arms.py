"""Tests for the four-arm retrieval ablation in `agent/graph.py`.

No database, no LLM: every retriever is patched. What's under test is the
dispatch and the *reporting*, both of which can be wrong silently:

- dispatch — a benchmark arm that quietly ran a different retriever than it
  reported is worse than one that crashes;
- reporting — `rag.graph_retriever.graph_retrieve` degrades to vector-only
  without raising when the graph tables are missing, so "the graph didn't
  help" and "the graph was never queried" look identical from the outside
  unless something says which happened.
"""

from unittest.mock import patch

import pytest

from agent.graph import (
    RETRIEVAL_MODE_NONE,
    RetrievalMode,
    _format_analysis,
    _node_retrieve,
    _retrieval_info,
)
from data.schema import Outcome, Party, Transcript, Turn
from rag.graph_retriever import GraphEvidence, GraphQueryPlan, GraphRetrievedCase
from rag.retriever import RetrievedCase


def _transcript() -> Transcript:
    return Transcript(
        dialogue_id="casino-1",
        source="casino",
        domain="campsite_resources",
        parties=[
            Party(
                party_id="agent_1",
                priorities={"Firewood": "High", "Food": "Medium", "Water": "Low"},
            ),
            Party(
                party_id="agent_2",
                priorities={"Firewood": "High", "Water": "Medium", "Food": "Low"},
            ),
        ],
        turns=[
            Turn(index=0, speaker="agent_1", text="I need the firewood for my group."),
            Turn(index=1, speaker="agent_2", text="So do I, and I am not giving it up."),
        ],
        outcome=Outcome(agreement_reached=False),
    )


def _vector_case(case_id: str = "casino-casino-1") -> RetrievedCase:
    return RetrievedCase(
        case_id=case_id, source="casino", kind="case", text="a case", score=0.43
    )


def _graph_case(case_id: str = "casino-casino-2") -> GraphRetrievedCase:
    evidence = GraphEvidence(
        kind="outcome",
        anchor_id="outcome:no_agreement",
        anchor_label="no_agreement",
        contribution=0.8,
    )
    return GraphRetrievedCase(
        case_id=case_id,
        source="casino",
        kind="case",
        text="a graph-found case",
        score=0.71,
        graph_score=0.8,
        vector_score=0.42,
        evidence=[evidence],
        matched_by=[evidence.summary()],
    )


# --- dispatch -------------------------------------------------------------


def test_no_rag_arm_skips_retrieval_entirely():
    """`use_rag=False` must not call any retriever, and must say so."""
    with patch("agent.graph.retrieve_precedent_tool") as vector, patch(
        "agent.graph.retrieve_precedent_graph_tool"
    ) as graph:
        out = _node_retrieve({"transcript": _transcript(), "use_rag": False})

    vector.assert_not_called()
    graph.assert_not_called()
    assert out["retrieved"] == []
    info = out["retrieval_info"]
    assert info.mode == RETRIEVAL_MODE_NONE
    # "skipped" and "attempted, returned nothing" are different claims.
    assert "skipped" in info.note


def test_vector_arm_uses_the_vector_retriever_only():
    with (
        patch("agent.graph.retrieve_precedent_tool", return_value=[_vector_case()]) as vector,
        patch("agent.graph.retrieve_precedent_graph_tool") as graph,
    ):
        out = _node_retrieve(
            {"transcript": _transcript(), "use_rag": True, "retrieval_mode": RetrievalMode.VECTOR}
        )

    vector.assert_called_once()
    graph.assert_not_called()
    info = out["retrieval_info"]
    assert info.mode == "vector"
    assert info.graph_effective is False
    assert info.plan is None       # no plan is built on this arm
    assert info.note == ""


@pytest.mark.parametrize(
    "mode,expect_use_vector",
    [(RetrievalMode.GRAPH, False), (RetrievalMode.HYBRID, True)],
)
def test_graph_arms_dispatch_with_the_right_fusion_flag(mode, expect_use_vector):
    """graph = traversal only; hybrid = traversal fused with vector similarity."""
    with patch("agent.graph.retrieve_precedent_tool") as vector, patch(
        "agent.graph.retrieve_precedent_graph_tool", return_value=[_graph_case()]
    ) as graph, patch(
        "agent.graph.plan_precedent_graph_tool", return_value=GraphQueryPlan(issues=["Firewood"])
    ):
        out = _node_retrieve(
            {"transcript": _transcript(), "use_rag": True, "retrieval_mode": mode}
        )

    vector.assert_not_called()
    graph.assert_called_once()
    assert graph.call_args.kwargs["use_vector"] is expect_use_vector
    assert out["retrieval_info"].graph_effective is True


def test_graph_arm_retrieves_with_the_plan_it_reports():
    """The reported plan must be the plan that ran, not a second planning pass."""
    plan = GraphQueryPlan(issues=["Firewood"], outcomes=["no_agreement"])
    with (
        patch("agent.graph.retrieve_precedent_graph_tool", return_value=[_graph_case()]) as graph,
        patch("agent.graph.plan_precedent_graph_tool", return_value=plan),
    ):
        out = _node_retrieve(
            {"transcript": _transcript(), "use_rag": True, "retrieval_mode": RetrievalMode.GRAPH}
        )

    assert graph.call_args.kwargs["plan"] is plan
    assert out["retrieval_info"].plan == plan.describe()


def test_retrieval_outage_degrades_instead_of_crashing_the_pipeline():
    """A Neon failure must not 500 a recruiter's analysis — the one hardening that matters."""
    with patch(
        "agent.graph.retrieve_precedent_tool",
        side_effect=RuntimeError("neon connection reset"),
    ):
        out = _node_retrieve(
            {"transcript": _transcript(), "use_rag": True, "retrieval_mode": RetrievalMode.VECTOR}
        )

    assert out["retrieved"] == []
    info = out["retrieval_info"]
    assert info.n_retrieved == 0
    # Distinguishable as a degraded response, not a silent empty result.
    assert "unavailable" in info.note and "neon connection reset" in info.note


def test_unknown_mode_raises_rather_than_falling_back():
    """A typo'd arm must fail loudly — a silent fallback misattributes results."""
    with pytest.raises(ValueError):
        _node_retrieve(
            {"transcript": _transcript(), "use_rag": True, "retrieval_mode": "graphh"}
        )


# --- reporting: the two ways a graph arm comes back empty -----------------


def test_empty_plan_is_reported_as_never_queried_not_as_a_weak_graph():
    info = _retrieval_info("graph", [_vector_case()], GraphQueryPlan())

    assert info.graph_effective is False
    assert "never ran" in info.note
    # Must NOT accuse the deployment of a missing graph — nothing was queried.
    assert "graph_is_populated" not in info.note


def test_built_plan_with_no_evidence_points_at_the_graph_tables():
    info = _retrieval_info("hybrid", [_vector_case()], GraphQueryPlan(issues=["Food"]))

    assert info.graph_effective is False
    assert "graph_is_populated" in info.note


def test_vector_only_hits_in_a_hybrid_result_do_not_count_as_graph_evidence():
    """`fuse` gives vector-only candidates a placeholder `matched_by` line.

    Counting `matched_by` instead of `evidence` would report a graph as
    effective on a run where the traversal returned nothing at all.
    """
    placeholder = GraphRetrievedCase(
        case_id="casino-casino-9",
        source="casino",
        kind="case",
        text="found by cosine alone",
        score=0.4,
        evidence=[],
        matched_by=["vector similarity only — no graph anchor matched this case"],
    )
    info = _retrieval_info("hybrid", [placeholder], GraphQueryPlan(issues=["Food"]))

    assert info.n_graph_grounded == 0
    assert info.graph_effective is False


def test_graph_grounded_count_is_per_hit():
    info = _retrieval_info(
        "hybrid", [_graph_case("a"), _vector_case("b")], GraphQueryPlan(issues=["Food"])
    )

    assert info.n_retrieved == 2
    assert info.n_graph_grounded == 1
    assert info.graph_effective is True
    assert info.note == ""


# --- prompt assembly ------------------------------------------------------


def test_format_analysis_surfaces_the_evidence_behind_each_precedent():
    """Provenance in the prompt is the countermeasure to citation fabrication."""
    case = _graph_case()
    prompt = _format_analysis(
        {
            "retrieved": [case],
            "retrieval_info": _retrieval_info("graph", [case], GraphQueryPlan(issues=["Firewood"])),
        }
    )

    assert "retrieval=graph" in prompt
    assert f"[{case.case_id}]" in prompt
    assert "why this case" in prompt
    assert "ended as no_agreement" in prompt


def test_format_analysis_still_works_on_the_vector_arm():
    """Plain `RetrievedCase` has no `matched_by`; that must not raise."""
    prompt = _format_analysis({"retrieved": [_vector_case()]})

    # Header now tags which corpus answered — a benchmark hit must be marked so
    # the model doesn't present campsite precedent as domain experience.
    assert "RETRIEVED PRECEDENTS (retrieval=vector, corpus=benchmark):" in prompt
    assert "why this case" not in prompt
