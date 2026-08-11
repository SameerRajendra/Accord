"""Tests for the knowledge-graph builder (pure, deterministic — no LLM, no database).

Only `rag.graph_ingest`'s *pure* half is exercised here: `build_graph` and the
derivations feeding it. `load_graph` talks to Postgres and is deliberately not
covered — a test that needs a live Neon instance is not a unit test.

Fixtures build `Transcript` objects directly in CaSiNo's shape, matching the
style of `tests/test_build_case_corpus.py`, so these stay independent of the
raw-parquet format.
"""

from data.schema import Action, CaseDocument, Outcome, Party, Transcript, Turn
from rag.graph_ingest import (
    build_global_nodes,
    build_graph,
    build_negotiation,
    derive_cooccurrence_edges,
    derive_preceded_edges,
)
from rag.graph_schema import (
    NODE_CONFLICT,
    NODE_ISSUE,
    NODE_NEGOTIATION,
    NODE_OUTCOME,
    NODE_PARTY,
    NODE_STRATEGY,
    NODE_TYPES,
    OUTCOME_BALANCED,
    OUTCOME_LOPSIDED,
    OUTCOME_NO_AGREEMENT,
    OUTCOME_UNSCORED,
    REL_ALLOCATED,
    REL_CO_OCCURS_WITH,
    REL_CONTESTED,
    REL_HAS_PARTY,
    REL_HAS_STRUCTURE,
    REL_NEGOTIATED_WITH,
    REL_PRECEDED,
    REL_PRIORITIZES,
    REL_RESULTED_IN,
    REL_TRADED,
    REL_TYPES,
    REL_USED_STRATEGY,
    STRUCT_COMPLEMENTARY,
    STRUCT_HIGH_CLASH,
    STRUCT_IDENTICAL,
    STRUCT_PARTIAL,
    STRUCT_UNKNOWN,
    conflict_structure,
    contested_issues,
    outcome_class,
    traded_issues,
)

HIGH_A = {"Firewood": "High", "Food": "Medium", "Water": "Low"}
HIGH_B_CLASH = {"Firewood": "High", "Water": "Medium", "Food": "Low"}
HIGH_B_COMPLEMENT = {"Water": "High", "Food": "Medium", "Firewood": "Low"}


def _clash_transcript() -> Transcript:
    """Both campers want Firewood most — the contested-resource shape."""
    return Transcript(
        dialogue_id="casino-1",
        source="casino",
        domain="campsite_resources",
        parties=[
            Party(
                party_id="agent_1",
                priorities=dict(HIGH_A),
                outcome_points=22,
                satisfaction="Extremely satisfied",
                metadata={"personality": {"svo": "proself"}},
            ),
            Party(
                party_id="agent_2",
                priorities=dict(HIGH_B_CLASH),
                outcome_points=12,
                metadata={"personality": {"svo": "prosocial"}},
            ),
        ],
        turns=[
            Turn(index=0, speaker="agent_1", text="Hi", strategies=["small-talk"]),
            Turn(index=1, speaker="agent_2", text="I need firewood", strategies=["self-need"]),
            Turn(index=2, speaker="agent_1", text="You don't really need it", strategies=["uv-part"]),
            Turn(index=3, speaker="agent_1", text="Seriously, you don't", strategies=["uv-part"]),
            Turn(
                index=4,
                speaker="agent_1",
                text="Submit-Deal",
                action=Action.SUBMIT_DEAL,
                action_data={"issue2youget": {"Firewood": "3"}, "issue2theyget": {"Water": "3"}},
            ),
            Turn(index=5, speaker="agent_2", text="Accept-Deal", action=Action.ACCEPT_DEAL),
        ],
        outcome=Outcome(
            agreement_reached=True,
            final_deal={
                "agent_1": {"Firewood": 3, "Water": 0, "Food": 0},
                "agent_2": {"Firewood": 0, "Water": 3, "Food": 3},
            },
            points={"agent_1": 22, "agent_2": 12},
        ),
        has_strategy_annotations=True,
        metadata={"split": "train"},
    )


def _breakdown_transcript() -> Transcript:
    return Transcript(
        dialogue_id="casino-2",
        source="casino",
        domain="campsite_resources",
        parties=[
            Party(party_id="agent_1", priorities=dict(HIGH_A)),
            Party(party_id="agent_2", priorities=dict(HIGH_B_CLASH)),
        ],
        turns=[
            Turn(index=0, speaker="agent_1", text="All the wood is mine", strategies=["self-need"]),
            Turn(index=1, speaker="agent_2", text="Your claim is nonsense", strategies=["uv-part"]),
            Turn(index=2, speaker="agent_1", text="Walk-Away", action=Action.WALK_AWAY),
        ],
        outcome=Outcome(agreement_reached=False, final_deal=None, points={}),
        has_strategy_annotations=True,
        metadata={"split": "test"},
    )


def _case_doc(dialogue_id: str) -> CaseDocument:
    """Mirrors the real corpus, including its `casino-casino-N` case_id quirk."""
    return CaseDocument(
        case_id="casino-{}".format(dialogue_id),
        source="casino",
        kind="case",
        text="Case casino-{} — rendered precedent text.".format(dialogue_id),
        metadata={"dialogue_id": dialogue_id, "split": "train"},
    )


def _edges_of(edges, rel):
    return [e for e in edges if e.rel == rel]


# --- derivations ----------------------------------------------------------


def test_conflict_structure_classes():
    assert conflict_structure(HIGH_A, HIGH_A) == STRUCT_IDENTICAL
    assert conflict_structure(HIGH_A, HIGH_B_CLASH) == STRUCT_HIGH_CLASH
    assert conflict_structure(HIGH_A, HIGH_B_COMPLEMENT) == STRUCT_COMPLEMENTARY
    # Different tops, and neither top is the other's bottom.
    partial_b = {"Water": "High", "Firewood": "Medium", "Food": "Low"}
    assert conflict_structure(HIGH_A, partial_b) == STRUCT_PARTIAL
    assert conflict_structure(HIGH_A, None) == STRUCT_UNKNOWN
    assert conflict_structure(HIGH_A, {"Firewood": "High"}) == STRUCT_UNKNOWN


def test_contested_and_traded_issues():
    assert contested_issues(HIGH_A, HIGH_B_CLASH) == ["Firewood"]
    assert traded_issues(HIGH_A, HIGH_B_CLASH) == []
    assert contested_issues(HIGH_A, HIGH_B_COMPLEMENT) == []
    assert traded_issues(HIGH_A, HIGH_B_COMPLEMENT) == ["Firewood", "Water"]


def test_outcome_class_buckets_on_the_documented_threshold():
    assert outcome_class(False, {}) == OUTCOME_NO_AGREEMENT
    assert outcome_class(True, {"a": 20, "b": 15}) == OUTCOME_BALANCED   # gap 5
    assert outcome_class(True, {"a": 21, "b": 15}) == OUTCOME_LOPSIDED   # gap 6
    assert outcome_class(True, {}) == OUTCOME_UNSCORED
    assert outcome_class(True, {"a": 20}) == OUTCOME_UNSCORED


# --- global nodes ---------------------------------------------------------


def test_global_nodes_cover_the_closed_vocabulary():
    nodes = build_global_nodes()
    by_type = {}
    for node in nodes:
        by_type.setdefault(node.node_type, []).append(node)
    assert len(by_type[NODE_ISSUE]) == 3
    assert len(by_type[NODE_STRATEGY]) == 10       # CaSiNo's 10-strategy taxonomy
    assert len(by_type[NODE_OUTCOME]) == 4
    assert len(by_type[NODE_CONFLICT]) == 5
    ids = {n.node_id for n in nodes}
    assert "strategy:uv-part" in ids
    assert "issue:Firewood" in ids
    assert "outcome:no_agreement" in ids
    assert "conflict:high_clash" in ids


def test_strategy_nodes_carry_polarity_and_playbook_text():
    nodes = {n.node_id: n for n in build_global_nodes()}
    assert nodes["strategy:uv-part"].props["polarity"] == "adversarial"
    assert nodes["strategy:promote-coordination"].props["polarity"] == "integrative"
    assert nodes["strategy:uv-part"].props["text"].startswith("Strategy: uv-part.")


def test_strategy_nodes_prefer_verbatim_corpus_text():
    """Byte-identical text to the vector store is a benchmarking requirement."""
    doc = CaseDocument(
        case_id="strategy-uv-part",
        source="playbook",
        kind="strategy",
        text="Strategy: uv-part. Verbatim corpus rendering.",
        metadata={"strategy": "uv-part"},
    )
    build = build_graph([], [doc])
    node = {n.node_id: n for n in build.nodes}["strategy:uv-part"]
    assert node.props["text"] == doc.text
    assert node.props["case_id"] == "strategy-uv-part"


# --- per-negotiation ------------------------------------------------------


def test_negotiation_node_props():
    nodes, _ = build_negotiation(_clash_transcript(), _case_doc("casino-1"))
    neg = nodes[0]
    assert neg.node_type == NODE_NEGOTIATION
    assert neg.node_id == "negotiation:casino-1"
    props = neg.props
    assert props["conflict_structure"] == STRUCT_HIGH_CLASH
    assert props["contested_issues"] == ["Firewood"]
    assert props["traded_issues"] == []
    assert props["outcome_class"] == OUTCOME_LOPSIDED   # 22 vs 12 → gap 10
    assert props["point_gap"] == 10
    assert props["joint_points"] == 34
    assert props["split"] == "train"
    assert props["n_text_turns"] == 4                   # protocol turns excluded
    assert props["dominant_strategies"][0] == "uv-part"  # used twice


def test_negotiation_carries_verbatim_case_text_matched_on_dialogue_id():
    """The corpus's case_id is `casino-casino-1`; matching keys on dialogue_id."""
    case = _case_doc("casino-1")
    assert case.case_id == "casino-casino-1"
    nodes, _ = build_negotiation(_clash_transcript(), case)
    assert nodes[0].props["text"] == case.text
    assert nodes[0].props["case_id"] == "casino-casino-1"


def test_negotiation_edges():
    _, edges = build_negotiation(_clash_transcript(), None)

    assert len(_edges_of(edges, REL_HAS_PARTY)) == 2
    assert len(_edges_of(edges, REL_NEGOTIATED_WITH)) == 2      # stored both ways
    assert len(_edges_of(edges, REL_PRIORITIZES)) == 6          # 2 parties x 3 issues
    assert len(_edges_of(edges, REL_ALLOCATED)) == 6            # final deal, both parties
    assert len(_edges_of(edges, REL_RESULTED_IN)) == 1
    assert len(_edges_of(edges, REL_HAS_STRUCTURE)) == 1

    contested = _edges_of(edges, REL_CONTESTED)
    assert [e.dst_id for e in contested] == ["issue:Firewood"]
    assert _edges_of(edges, REL_TRADED) == []

    structure = _edges_of(edges, REL_HAS_STRUCTURE)[0]
    assert structure.dst_id == "conflict:high_clash"
    assert _edges_of(edges, REL_RESULTED_IN)[0].dst_id == "outcome:agreement_lopsided"


def test_used_strategy_edges_aggregate_counts_and_turn_span():
    _, edges = build_negotiation(_clash_transcript(), None)
    used = {(e.src_id, e.dst_id): e for e in _edges_of(edges, REL_USED_STRATEGY)}

    uv = used[("party:casino-1:agent_1", "strategy:uv-part")]
    assert uv.props["count"] == 2          # one edge with a count, not two edges
    assert uv.props["first_turn"] == 2
    assert uv.props["last_turn"] == 3
    assert uv.weight == 2.0

    assert ("party:casino-1:agent_2", "strategy:self-need") in used
    assert ("party:casino-1:agent_1", "strategy:small-talk") in used


def test_allocated_edges_denormalize_own_priority():
    _, edges = build_negotiation(_clash_transcript(), None)
    allocated = {(e.src_id, e.dst_id): e for e in _edges_of(edges, REL_ALLOCATED)}
    firewood = allocated[("party:casino-1:agent_1", "issue:Firewood")]
    assert firewood.props["quantity"] == 3
    assert firewood.props["own_priority"] == "High"
    # agent_2 surrendered the issue it also ranked High — visible without a hop.
    conceded = allocated[("party:casino-1:agent_2", "issue:Firewood")]
    assert conceded.props["quantity"] == 0
    assert conceded.props["own_priority"] == "High"


def test_party_nodes_record_role_and_satisfaction():
    nodes, _ = build_negotiation(_clash_transcript(), None)
    by_id = {n.node_id: n for n in nodes if n.node_type == NODE_PARTY}
    assert by_id["party:casino-1:agent_1"].props["role"] == "winner"
    assert by_id["party:casino-1:agent_2"].props["role"] == "loser"
    assert by_id["party:casino-1:agent_1"].props["satisfaction"] == "Extremely satisfied"
    assert by_id["party:casino-1:agent_2"].props["svo"] == "prosocial"


def test_breakdown_negotiation_has_no_allocations():
    nodes, edges = build_negotiation(_breakdown_transcript(), None)
    assert nodes[0].props["outcome_class"] == OUTCOME_NO_AGREEMENT
    assert nodes[0].props["point_gap"] is None
    assert _edges_of(edges, REL_ALLOCATED) == []
    assert _edges_of(edges, REL_RESULTED_IN)[0].dst_id == "outcome:no_agreement"


# --- corpus-level aggregates ----------------------------------------------


def test_cooccurrence_weights_are_directional_conditionals():
    a, b, c = "vouch-fair", "promote-coordination", "uv-part"
    sets = [{a, b}] * 5 + [{a}] * 5 + [{c}] * 2      # 12 dialogues
    edges = derive_cooccurrence_edges(sets, min_count=5, min_pmi=0.1)

    by_pair = {(e.src_id, e.dst_id): e for e in edges}
    assert len(edges) == 2                            # one pair, stored both ways
    ab = by_pair[("strategy:vouch-fair", "strategy:promote-coordination")]
    ba = by_pair[("strategy:promote-coordination", "strategy:vouch-fair")]
    assert ab.props["count"] == 5
    assert ab.weight == 0.5                           # P(b|a) = 5/10
    assert ba.weight == 1.0                           # P(a|b) = 5/5
    assert 0 < ab.props["pmi"] < 1
    assert all(e.rel == REL_CO_OCCURS_WITH for e in edges)


def test_cooccurrence_respects_the_support_floor():
    a, b = "vouch-fair", "uv-part"
    edges = derive_cooccurrence_edges([{a, b}] * 4, min_count=5)
    assert edges == []


def test_preceded_lift_uses_the_annotated_subset_as_its_base_rate():
    rows = [({"vouch-fair"}, OUTCOME_BALANCED)] * 8 + [({"uv-part"}, OUTCOME_NO_AGREEMENT)] * 2
    edges = derive_preceded_edges(rows)

    by_pair = {(e.src_id, e.dst_id): e for e in edges}
    # 2 strategies x the 2 outcome classes that actually occurred.
    assert len(edges) == 4
    assert all(e.rel == REL_PRECEDED for e in edges)

    hostile = by_pair[("strategy:uv-part", "outcome:no_agreement")]
    assert hostile.props["support"] == 2
    assert hostile.props["confidence"] == 1.0
    assert hostile.props["base_rate"] == 0.2          # 2/10 annotated, not 2/1030
    assert hostile.props["lift"] == 5.0
    assert hostile.props["n_annotated_dialogues"] == 10

    protective = by_pair[("strategy:vouch-fair", "outcome:no_agreement")]
    assert protective.props["support"] == 0
    assert protective.props["lift"] == 0.0            # kept, so the zero is visible


def test_preceded_lift_is_clamped():
    rows = [({"uv-part"}, OUTCOME_NO_AGREEMENT)] + [({"vouch-fair"}, OUTCOME_BALANCED)] * 99
    edges = {(e.src_id, e.dst_id): e for e in derive_preceded_edges(rows)}
    hostile = edges[("strategy:uv-part", "outcome:no_agreement")]
    assert hostile.props["lift"] == 100.0             # raw value preserved...
    assert hostile.weight == 5.0                      # ...but the score weight is capped


# --- whole-corpus build ---------------------------------------------------


def test_build_graph_stats_and_uniqueness():
    transcripts = [_clash_transcript(), _breakdown_transcript()]
    cases = [_case_doc("casino-1"), _case_doc("casino-2")]
    build = build_graph(transcripts, cases)

    stats = build.stats
    assert stats["transcripts"] == 2
    assert stats["cases_matched"] == 2
    assert stats["cases_unmatched"] == 0
    assert stats["annotated_dialogues"] == 2
    assert stats["outcome_distribution"] == {OUTCOME_LOPSIDED: 1, OUTCOME_NO_AGREEMENT: 1}
    assert stats["structure_distribution"] == {STRUCT_HIGH_CLASH: 2}
    assert stats["contested_issue_counts"] == {"Firewood": 2}

    # Node ids unique — the table's primary key.
    node_ids = [n.node_id for n in build.nodes]
    assert len(node_ids) == len(set(node_ids))

    # (src, rel, dst) unique — the table's UNIQUE constraint. A duplicate here
    # would be silently swallowed by the loader's ON CONFLICT DO UPDATE and
    # quietly drop data, so the build must guarantee it.
    keys = [(e.src_id, e.rel, e.dst_id) for e in build.edges]
    assert len(keys) == len(set(keys))

    # Every edge endpoint must resolve — the table has FK constraints.
    known = set(node_ids)
    assert all(e.src_id in known and e.dst_id in known for e in build.edges)


def test_build_graph_emits_only_declared_types():
    """The vocabulary in graph_schema.py is the contract; ingestion must respect it."""
    build = build_graph([_clash_transcript(), _breakdown_transcript()], [])
    assert {n.node_type for n in build.nodes} <= set(NODE_TYPES)
    assert {e.rel for e in build.edges} <= set(REL_TYPES)


def test_build_graph_backfills_corpus_counts_onto_global_nodes():
    build = build_graph([_clash_transcript(), _breakdown_transcript()], [])
    by_id = {n.node_id: n for n in build.nodes}
    assert by_id["strategy:uv-part"].props["n_annotated_dialogues"] == 2
    assert by_id["strategy:vouch-fair"].props["n_annotated_dialogues"] == 0
    assert by_id["outcome:no_agreement"].props["n_negotiations"] == 1
    assert by_id["outcome:no_agreement"].props["share"] == 0.5
    assert by_id["conflict:high_clash"].props["n_negotiations"] == 2
    assert by_id["issue:Firewood"].props["n_contested"] == 2
    assert by_id["issue:Water"].props["n_contested"] == 0


def test_build_graph_is_deterministic():
    transcripts = [_clash_transcript(), _breakdown_transcript()]
    first = build_graph(transcripts, [_case_doc("casino-1")])
    second = build_graph(transcripts, [_case_doc("casino-1")])
    assert [n.model_dump() for n in first.nodes] == [n.model_dump() for n in second.nodes]
    assert [e.model_dump() for e in first.edges] == [e.model_dump() for e in second.edges]


def test_build_graph_reports_unmatched_cases():
    build = build_graph([_clash_transcript()], [])
    assert build.stats["cases_matched"] == 0
    assert build.stats["cases_unmatched"] == 1
    neg = {n.node_id: n for n in build.nodes}["negotiation:casino-1"]
    assert neg.props["text"] is None
