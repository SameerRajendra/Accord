"""Tests for graph query planning and rerank fusion (pure — no database).

`graph_retrieve` itself needs Postgres and is not covered here; what *is*
covered is everything that decides what it looks up and how it orders what
comes back. Those are the parts that can be wrong silently.
"""

import pytest

from data.schema import Outcome, Party, Transcript, Turn
from rag.graph_retriever import (
    GraphEvidence,
    GraphQueryPlan,
    GraphRetrievedCase,
    fuse,
    merge_plans,
    plan_from_transcript,
    plan_query,
)
from rag.graph_schema import (
    OUTCOME_BALANCED,
    OUTCOME_LOPSIDED,
    OUTCOME_NO_AGREEMENT,
    STRUCT_COMPLEMENTARY,
    STRUCT_HIGH_CLASH,
    STRUCT_IDENTICAL,
)
from rag.retriever import RetrievedCase

HIGH_A = {"Firewood": "High", "Food": "Medium", "Water": "Low"}
HIGH_B_CLASH = {"Firewood": "High", "Water": "Medium", "Food": "Low"}
HIGH_B_COMPLEMENT = {"Water": "High", "Food": "Medium", "Firewood": "Low"}


# --- query planning: the hostile-query failure this layer answers to -------


def test_hostile_query_anchors_on_adversarial_structure_not_words():
    """The corpus never says "hostile"; the planner maps it onto facts that are stored."""
    plan = plan_query("find precedents where the other side was hostile and aggressive")

    assert "adversarial" in plan.polarities
    assert "uv-part" in plan.strategies              # the adversarial tactic
    assert OUTCOME_NO_AGREEMENT in plan.outcomes
    assert OUTCOME_LOPSIDED in plan.outcomes
    assert STRUCT_HIGH_CLASH in plan.structures
    assert STRUCT_IDENTICAL in plan.structures       # subsumed by high_clash
    assert not plan.is_empty()
    assert plan.notes                                # the expansion is explained


def test_breakdown_and_issue_query():
    plan = plan_query("both wanted the firewood, but the negotiation broke down")
    assert plan.issues == ["Firewood"]
    assert OUTCOME_NO_AGREEMENT in plan.outcomes
    assert STRUCT_HIGH_CLASH in plan.structures


def test_integrative_query_anchors_the_other_way():
    plan = plan_query("a collaborative win-win deal where they traded complementary priorities")
    assert "integrative" in plan.polarities
    assert "promote-coordination" in plan.strategies
    assert OUTCOME_BALANCED in plan.outcomes
    assert STRUCT_COMPLEMENTARY in plan.structures


def test_plan_is_empty_when_nothing_anchors():
    plan = plan_query("what is the weather like on tuesday")
    assert plan.is_empty()
    assert plan.strategies == [] and plan.outcomes == []
    assert any("vector-only" in note for note in plan.notes)


def test_matching_is_word_bounded():
    """`\\bfair\\b` must not fire on "affair" — a lexicon planner's classic failure."""
    assert "vouch-fair" not in plan_query("they discussed the affair").strategies
    assert "vouch-fair" in plan_query("they wanted a fair split").strategies


def test_prefix_stems_match_inflections():
    assert "showing-empathy" in plan_query("she was empathising with him").strategies
    assert "showing-empathy" in plan_query("he showed empathy").strategies


def test_issue_synonyms_resolve_to_canonical_names():
    assert plan_query("we argued about the wood").issues == ["Firewood"]
    assert plan_query("they were thirsty").issues == ["Water"]
    assert plan_query("nobody was hungry").issues == ["Food"]


def test_matched_terms_records_what_triggered_each_anchor():
    plan = plan_query("a hostile fight over firewood")
    assert "Firewood" in plan.matched_terms
    assert "firewood" in plan.matched_terms["Firewood"]
    assert "adversarial" in plan.matched_terms


def test_anchor_ids_are_well_formed_node_ids():
    plan = plan_query("hostile fight over firewood")
    ids = plan.anchor_ids()
    assert "issue:Firewood" in ids["issue_ids"]
    assert "strategy:uv-part" in ids["strategy_ids"]
    assert "outcome:no_agreement" in ids["outcome_ids"]
    assert "conflict:high_clash" in ids["structure_ids"]


def test_describe_is_loggable():
    text = plan_query("hostile fight over firewood").describe()
    assert text.startswith("plan[text](")
    assert "Firewood" in text


# --- planning from a transcript (the exact path) ---------------------------


def _transcript(prio_b, strategies=None, dialogue_id="casino-9") -> Transcript:
    strategies = strategies or []
    return Transcript(
        dialogue_id=dialogue_id,
        source="casino",
        domain="campsite_resources",
        parties=[
            Party(party_id="agent_1", priorities=dict(HIGH_A)),
            Party(party_id="agent_2", priorities=dict(prio_b)),
        ],
        turns=[
            Turn(index=0, speaker="agent_1", text="hello", strategies=list(strategies)),
            Turn(index=1, speaker="agent_2", text="hi", strategies=list(strategies)),
        ],
        outcome=Outcome(agreement_reached=False, final_deal=None, points={}),
        has_strategy_annotations=bool(strategies),
        metadata={"split": "test"},
    )


def test_plan_from_transcript_uses_priority_rankings_as_facts():
    plan = plan_from_transcript(_transcript(HIGH_B_CLASH))
    assert plan.origin == "transcript"
    assert plan.structures == [STRUCT_HIGH_CLASH, STRUCT_IDENTICAL]
    assert plan.issues == ["Firewood"]          # contested, computed not guessed
    assert plan.outcomes == []                  # the outcome is what we're predicting


def test_plan_from_transcript_complementary_case_anchors_traded_issues():
    plan = plan_from_transcript(_transcript(HIGH_B_COMPLEMENT))
    assert plan.structures == [STRUCT_COMPLEMENTARY]
    assert plan.issues == ["Firewood", "Water"]


def test_plan_from_transcript_states_missing_annotations_rather_than_hiding_them():
    plan = plan_from_transcript(_transcript(HIGH_B_CLASH))
    assert plan.strategies == []
    assert any("no strategy annotations" in note for note in plan.notes)


def test_plan_from_transcript_ranks_annotated_strategies():
    plan = plan_from_transcript(_transcript(HIGH_B_CLASH, strategies=["uv-part", "self-need"]))
    assert set(plan.strategies) == {"uv-part", "self-need"}


def test_plan_from_transcript_drops_out_of_vocabulary_labels():
    plan = plan_from_transcript(_transcript(HIGH_B_CLASH, strategies=["not-a-real-strategy"]))
    assert plan.strategies == []


def test_plan_from_transcript_target_outcome_is_opt_in():
    plan = plan_from_transcript(_transcript(HIGH_B_CLASH), target_outcome=OUTCOME_NO_AGREEMENT)
    assert plan.outcomes == [OUTCOME_NO_AGREEMENT]


def test_merge_plans_unions_without_duplicates():
    merged = merge_plans(
        plan_from_transcript(_transcript(HIGH_B_CLASH)),
        plan_query("a hostile fight over firewood"),
    )
    assert merged.origin == "merged"
    assert merged.issues == ["Firewood"]                 # present in both, kept once
    assert "uv-part" in merged.strategies
    assert merged.structures.count(STRUCT_HIGH_CLASH) == 1


# --- rerank / fusion -------------------------------------------------------


def _graph_case(case_id, graph_score, rank) -> GraphRetrievedCase:
    return GraphRetrievedCase(
        case_id=case_id,
        source="casino",
        kind="case",
        text="graph text for {}".format(case_id),
        score=0.0,
        metadata={"dialogue_id": case_id},
        graph_score=graph_score,
        graph_rank=rank,
        evidence=[
            GraphEvidence(
                kind="strategy",
                anchor_id="strategy:uv-part",
                anchor_label="uv-part",
                contribution=graph_score,
                detail={"party_id": "party:{}:agent_1".format(case_id), "count": 2},
            )
        ],
        matched_by=["agent_1 used strategy 'uv-part' x2"],
    )


def _vector_case(case_id, score) -> RetrievedCase:
    return RetrievedCase(
        case_id=case_id,
        source="casino",
        kind="case",
        text="vector text for {}".format(case_id),
        score=score,
        metadata={"dialogue_id": case_id},
    )


def _fixture_lists():
    graph = [
        _graph_case("casino-a", 3.0, 1),
        _graph_case("casino-b", 2.0, 2),
        _graph_case("casino-d", 1.0, 3),
    ]
    vector = [
        _vector_case("casino-a", 0.44),
        _vector_case("casino-c", 0.43),
        _vector_case("casino-b", 0.42),
    ]
    return graph, vector


def test_weighted_fusion_blends_both_signals():
    graph, vector = _fixture_lists()
    fused = fuse(graph, vector, k=4, fusion="weighted", alpha=0.6)

    assert [c.case_id for c in fused] == ["casino-a", "casino-b", "casino-c", "casino-d"]
    scores = {c.case_id: round(c.score, 6) for c in fused}
    assert scores["casino-a"] == 1.0        # top of both lists
    assert scores["casino-b"] == 0.3        # 0.6*0.5 + 0.4*0.0
    assert scores["casino-c"] == 0.2        # 0.6*0.0 + 0.4*0.5
    assert scores["casino-d"] == 0.0


def test_rrf_fusion_uses_ranks_not_magnitudes():
    graph, vector = _fixture_lists()
    fused = fuse(graph, vector, k=4, fusion="rrf", alpha=0.6, rrf_k=60)
    # casino-d (graph rank 3, absent from vector) outranks casino-c
    # (vector rank 2 only) because the graph side carries more weight.
    assert [c.case_id for c in fused] == ["casino-a", "casino-b", "casino-d", "casino-c"]


def test_alpha_zero_reduces_to_the_vector_baseline_ordering():
    graph, vector = _fixture_lists()
    fused = fuse(graph, vector, k=3, fusion="weighted", alpha=0.0)
    assert [c.case_id for c in fused][:2] == ["casino-a", "casino-c"]


def test_alpha_one_ignores_the_vector_signal():
    graph, vector = _fixture_lists()
    fused = fuse(graph, vector, k=4, fusion="weighted", alpha=1.0)
    assert [c.case_id for c in fused][:2] == ["casino-a", "casino-b"]
    # The vector-only candidate contributes nothing and sinks into a zero tie.
    # A *true* graph-only ablation is `graph_retrieve(..., use_vector=False)`,
    # which never fetches the vector list at all — see the empty-vector test.
    assert fused[-1].score == 0.0


def test_vector_only_hits_are_marked_as_carrying_no_graph_evidence():
    graph, vector = _fixture_lists()
    fused = {c.case_id: c for c in fuse(graph, vector, k=4)}
    only_vector = fused["casino-c"]
    assert only_vector.graph_score == 0.0
    assert only_vector.graph_rank is None
    assert only_vector.evidence == []
    assert "vector similarity only" in only_vector.matched_by[0]
    assert only_vector.vector_score == 0.43
    assert only_vector.vector_rank == 2


def test_graph_only_hits_report_a_null_vector_score():
    """Stated as unknown rather than faked — see `fuse`'s documented bias."""
    graph, vector = _fixture_lists()
    fused = {c.case_id: c for c in fuse(graph, vector, k=4)}
    assert fused["casino-d"].vector_score is None
    assert fused["casino-d"].vector_rank is None
    assert fused["casino-d"].graph_score == 1.0


def test_overlapping_hits_prefer_the_embedded_text():
    """Both retrievers must hand the prompt identical bytes for the same case."""
    graph, vector = _fixture_lists()
    fused = {c.case_id: c for c in fuse(graph, vector, k=4)}
    assert fused["casino-a"].text == "vector text for casino-a"
    assert fused["casino-d"].text == "graph text for casino-d"  # graph-only: no vector text


def test_evidence_survives_fusion():
    graph, vector = _fixture_lists()
    fused = {c.case_id: c for c in fuse(graph, vector, k=4)}
    evidence = fused["casino-a"].evidence
    assert len(evidence) == 1
    assert evidence[0].anchor_id == "strategy:uv-part"
    assert fused["casino-a"].fusion == "weighted"


def test_fusion_rejects_an_unknown_method():
    graph, vector = _fixture_lists()
    with pytest.raises(ValueError, match="unknown fusion"):
        fuse(graph, vector, k=3, fusion="magic")


def test_fusion_handles_an_empty_graph_side():
    _, vector = _fixture_lists()
    fused = fuse([], vector, k=3, fusion="weighted", alpha=0.6)
    assert [c.case_id for c in fused] == ["casino-a", "casino-c", "casino-b"]
    assert all(c.graph_score == 0.0 for c in fused)


def test_fusion_handles_an_empty_vector_side():
    graph, _ = _fixture_lists()
    fused = fuse(graph, [], k=3, fusion="weighted", alpha=0.6)
    assert [c.case_id for c in fused] == ["casino-a", "casino-b", "casino-d"]
    assert all(c.vector_score is None for c in fused)


def test_fusion_is_deterministic_on_ties():
    tied = [_graph_case("casino-z", 1.0, 1), _graph_case("casino-a", 1.0, 2)]
    fused = fuse(tied, [], k=2)
    assert [c.case_id for c in fused] == ["casino-a", "casino-z"]  # tie broken by id


# --- contract with the existing retrieval path -----------------------------


def test_graph_result_is_a_drop_in_retrieved_case():
    """`agent/graph.py` types its state as List[RetrievedCase]; these must fit."""
    case = _graph_case("casino-a", 1.0, 1)
    assert isinstance(case, RetrievedCase)
    assert set(RetrievedCase.model_fields).issubset(set(GraphRetrievedCase.model_fields))


def test_evidence_summaries_are_human_readable():
    evidence = [
        GraphEvidence(
            kind="strategy",
            anchor_id="strategy:uv-part",
            anchor_label="uv-part",
            contribution=1.0,
            hops=1,
            detail={"party_id": "party:casino-1:agent_2", "count": 3},
        ),
        GraphEvidence(
            kind="outcome",
            anchor_id="outcome:no_agreement",
            anchor_label="no_agreement",
            contribution=0.8,
            detail={"rel": "RESULTED_IN"},
        ),
        GraphEvidence(
            kind="issue",
            anchor_id="issue:Firewood",
            anchor_label="Firewood",
            contribution=0.6,
            detail={"rel": "CONTESTED"},
        ),
    ]
    summaries = [e.summary() for e in evidence]
    assert summaries[0] == "agent_2 used strategy 'uv-part' x3 (reached by 1 co-occurrence hop)"
    assert summaries[1] == "ended as no_agreement"
    assert summaries[2] == "contested issue Firewood"


def test_empty_plan_round_trips_through_pydantic():
    plan = GraphQueryPlan()
    assert plan.is_empty()
    assert GraphQueryPlan.model_validate_json(plan.model_dump_json()) == plan
