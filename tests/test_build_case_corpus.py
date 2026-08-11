"""Tests for the RAG case-corpus builder (pure, deterministic — no LLM).

Fixtures build `Transcript` objects directly in CaSiNo's shape (agent_1/agent_2
party_ids, priority rankings, strategy-annotated turns, item-split outcome with
points) rather than importing from ingestion, keeping these tests independent
of the raw-parquet format.
"""

from pathlib import Path

from data.build_case_corpus import build_case, build_cases, build_corpus, build_playbook
from data.schema import Action, CaseDocument, Outcome, Party, Transcript, Turn


def _agreement_transcript() -> Transcript:
    return Transcript(
        dialogue_id="1",
        source="casino",
        domain="campsite_resources",
        parties=[
            Party(
                party_id="agent_1",
                priorities={"Firewood": "High", "Food": "Medium", "Water": "Low"},
                outcome_points=25,
                satisfaction="Extremely satisfied",
                opponent_likeness="Extremely like",
                metadata={"personality": {"svo": "prosocial"}},
            ),
            Party(
                party_id="agent_2",
                priorities={"Water": "High", "Food": "Medium", "Firewood": "Low"},
                outcome_points=13,
                metadata={"personality": {"svo": "proself"}},
            ),
        ],
        turns=[
            Turn(index=0, speaker="agent_1", text="Hi there!", strategies=["small-talk", "elicit-pref"]),
            Turn(index=1, speaker="agent_2", text="I need water", strategies=["self-need"]),
            Turn(index=2, speaker="agent_1", text="Let's find a fair split", strategies=["vouch-fair", "promote-coordination"]),
            Turn(
                index=3,
                speaker="agent_1",
                text="Submit-Deal",
                action=Action.SUBMIT_DEAL,
                action_data={"issue2youget": {"Firewood": "3", "Food": "2"}, "issue2theyget": {"Water": "3", "Food": "1"}},
            ),
            Turn(index=4, speaker="agent_2", text="Accept-Deal", action=Action.ACCEPT_DEAL),
        ],
        outcome=Outcome(
            agreement_reached=True,
            final_deal={"agent_1": {"Firewood": 3, "Food": 2}, "agent_2": {"Water": 3, "Food": 1}},
            points={"agent_1": 25, "agent_2": 13},
        ),
        has_strategy_annotations=True,
        metadata={"split": "train"},
    )


def _no_agreement_transcript() -> Transcript:
    return Transcript(
        dialogue_id="2",
        source="casino",
        domain="campsite_resources",
        parties=[
            Party(party_id="agent_1", priorities={"Firewood": "High", "Food": "Medium", "Water": "Low"}, metadata={}),
            Party(party_id="agent_2", priorities={"Firewood": "High", "Water": "Medium", "Food": "Low"}, metadata={}),
        ],
        turns=[
            Turn(index=0, speaker="agent_1", text="I really need all the firewood", strategies=["self-need"]),
            Turn(index=1, speaker="agent_2", text="No, I need it more", strategies=["uv-part"]),
            Turn(index=2, speaker="agent_1", text="Walk-Away", action=Action.WALK_AWAY),
        ],
        outcome=Outcome(agreement_reached=False, final_deal=None, points={}),
        has_strategy_annotations=True,
        metadata={"split": "validation"},
    )


def test_agreement_case_renders_priorities_strategies_outcome_lesson():
    case = build_case(_agreement_transcript())
    assert case.kind == "case"
    assert case.case_id == "casino-1"
    text = case.text
    assert "agent_1 priorities: High=Firewood" in text
    assert "agent_2 priorities: High=Water" in text
    assert "vouch-fair (1)" in text or "promote-coordination (1)" in text
    assert "Agreement reached" in text
    assert "agent_1=25" in text and "agent_2=13" in text  # points rendered
    assert "favored agent_1" in text  # 25 > 13


def test_agreement_case_metadata():
    case = build_case(_agreement_transcript())
    m = case.metadata
    assert m["outcome_label"] == "agreement"
    assert m["agreement_reached"] is True
    assert m["points"] == {"agent_1": 25, "agent_2": 13}
    assert m["split"] == "train"
    assert m["has_strategy_annotations"] is True
    assert len(m["dominant_strategies"]) > 0


def test_protocol_turns_excluded_from_strategies():
    """Submit-Deal / Accept-Deal turns carry no strategies and shouldn't surface."""
    case = build_case(_agreement_transcript())
    assert "Submit-Deal" not in case.text
    assert "Accept-Deal" not in case.text


def test_no_agreement_case_flags_breakdown():
    case = build_case(_no_agreement_transcript())
    assert "No agreement" in case.text
    assert "ended via walk_away" in case.text
    assert "broke down" in case.text
    assert case.metadata["outcome_label"] == "no_agreement"
    assert case.metadata["points"] == {}


def test_playbook_covers_all_strategies():
    docs = build_playbook()
    assert len(docs) == 10  # the CaSiNo 10-strategy taxonomy
    assert all(d.kind == "strategy" and d.source == "playbook" for d in docs)
    ids = {d.case_id for d in docs}
    assert "strategy-vouch-fair" in ids and "strategy-uv-part" in ids


def test_case_document_roundtrips():
    case = build_case(_agreement_transcript())
    assert CaseDocument.model_validate_json(case.model_dump_json()) == case


def test_build_corpus_writes_cases_plus_playbook(tmp_path: Path):
    import json

    in_path = tmp_path / "casino.jsonl"
    in_path.write_text(
        _agreement_transcript().model_dump_json() + "\n" + _no_agreement_transcript().model_dump_json() + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "case_corpus.jsonl"

    summary = build_corpus(in_path, out_path)

    assert summary["cases"] == 2
    assert summary["strategy_docs"] == 10
    assert summary["documents_total"] == 12
    assert summary["agreements"] == 1
    assert summary["agreement_rate"] == 0.5

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 12
    docs = [json.loads(line) for line in lines]
    kinds = {d["kind"] for d in docs}
    assert kinds == {"case", "strategy"}


def test_build_corpus_no_playbook_flag(tmp_path: Path):
    in_path = tmp_path / "casino.jsonl"
    in_path.write_text(_agreement_transcript().model_dump_json() + "\n", encoding="utf-8")
    out_path = tmp_path / "case_corpus.jsonl"
    summary = build_corpus(in_path, out_path, include_playbook=False)
    assert summary["strategy_docs"] == 0
    assert summary["documents_total"] == 1
