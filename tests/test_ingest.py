"""Ingestion tests: pure-function transform on a fixture, plus an optional
smoke test against the real corpus if it has been downloaded."""

from pathlib import Path

import pytest

from data.ingest_casino import DEFAULT_INPUT, ingest, normalize_dialogue
from data.schema import Action

# A hand-built raw CaSiNo dialogue exercising: utterances, sparse text-keyed
# annotations, a Submit-Deal (submitter-perspective quantities), and Accept-Deal.
RAW_AGREEMENT = {
    "dialogue_id": 42,
    "chat_logs": [
        {"text": "Hello there!", "task_data": {}, "id": "mturk_agent_1"},
        {"text": "I need firewood.", "task_data": {}, "id": "mturk_agent_2"},
        {
            "text": "Submit-Deal",
            "task_data": {
                "issue2youget": {"Firewood": "1", "Water": "3", "Food": "2"},
                "issue2theyget": {"Firewood": "2", "Water": "0", "Food": "1"},
            },
            "id": "mturk_agent_1",
        },
        {"text": "Accept-Deal", "task_data": {"data": "accept_deal"}, "id": "mturk_agent_2"},
    ],
    "participant_info": {
        "mturk_agent_1": {
            "value2issue": {"High": "Water", "Medium": "Food", "Low": "Firewood"},
            "outcomes": {"points_scored": 24, "satisfaction": "Extremely satisfied"},
        },
        "mturk_agent_2": {
            "value2issue": {"High": "Firewood", "Medium": "Food", "Low": "Water"},
            "outcomes": {"points_scored": 18, "satisfaction": "Slightly satisfied"},
        },
    },
    "annotations": [
        ["Hello there!", "small-talk"],
        # multi-label utterance with one bogus label that must be filtered out.
        ["I need firewood.", "self-need,other-need,not-a-real-label"],
    ],
}

RAW_WALKAWAY = {
    "dialogue_id": 7,
    "chat_logs": [
        {"text": "Hi", "task_data": {}, "id": "mturk_agent_1"},
        {"text": "No deal.", "task_data": {}, "id": "mturk_agent_2"},
        {"text": "Walk-Away", "task_data": {}, "id": "mturk_agent_2"},
    ],
    "participant_info": {
        "mturk_agent_1": {"outcomes": {"points_scored": 5}},
        "mturk_agent_2": {"outcomes": {"points_scored": 5}},
    },
    "annotations": [],
}


def test_agreement_dialogue_shape():
    t = normalize_dialogue(RAW_AGREEMENT)
    assert t.dialogue_id == "42"
    assert [p.party_id for p in t.parties] == ["agent_1", "agent_2"]
    # priorities inverted from CaSiNo value2issue (priority->issue) to issue->priority.
    assert t.parties[0].priorities == {"Water": "High", "Food": "Medium", "Firewood": "Low"}
    assert t.outcome.points == {"agent_1": 24, "agent_2": 18}


def test_submit_deal_perspective_resolves_to_both_parties():
    t = normalize_dialogue(RAW_AGREEMENT)
    assert t.outcome.agreement_reached is True
    # Submitter is agent_1, so issue2youget belongs to agent_1.
    assert t.outcome.final_deal["agent_1"] == {"Firewood": 1, "Water": 3, "Food": 2}
    assert t.outcome.final_deal["agent_2"] == {"Firewood": 2, "Water": 0, "Food": 1}


def test_actions_tagged_and_quantities_are_ints():
    t = normalize_dialogue(RAW_AGREEMENT)
    submit = t.turns[2]
    assert submit.action is Action.SUBMIT_DEAL
    assert t.turns[3].action is Action.ACCEPT_DEAL
    assert all(isinstance(v, int) for v in t.outcome.final_deal["agent_1"].values())


def test_annotations_are_text_keyed_and_filtered():
    t = normalize_dialogue(RAW_AGREEMENT)
    assert t.turns[0].strategies == ["small-talk"]
    # bogus label dropped, valid ones kept
    assert t.turns[1].strategies == ["self-need", "other-need"]
    assert t.has_strategy_annotations is True


def test_walkaway_has_no_agreement_or_deal():
    t = normalize_dialogue(RAW_WALKAWAY)
    assert t.outcome.agreement_reached is False
    assert t.outcome.final_deal is None
    assert t.turns[-1].action is Action.WALK_AWAY
    assert t.has_strategy_annotations is False


def test_ingest_writes_jsonl(tmp_path: Path):
    import json

    raw_path = tmp_path / "casino.json"
    raw_path.write_text(json.dumps([RAW_AGREEMENT, RAW_WALKAWAY]), encoding="utf-8")
    out_path = tmp_path / "out.jsonl"

    summary = ingest(raw_path, out_path)

    assert summary["dialogues_written"] == 2
    assert summary["with_annotations"] == 1
    assert summary["agreements"] == 1
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["source"] == "casino" for line in lines)


@pytest.mark.skipif(
    not DEFAULT_INPUT.exists(),
    reason="real casino.json not downloaded; run `python -m data.ingest_casino --download`",
)
def test_real_corpus_smoke():
    """If the real corpus is present, every dialogue must normalize + validate."""
    import json

    raw = json.loads(DEFAULT_INPUT.read_text(encoding="utf-8"))
    assert len(raw) == 1030
    annotated = 0
    for d in raw:
        t = normalize_dialogue(d)  # raises on any schema violation
        annotated += int(t.has_strategy_annotations)
    assert annotated == 396
