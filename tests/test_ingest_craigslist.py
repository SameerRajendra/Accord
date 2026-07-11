"""Tests for CraigslistBargain ingestion.

Fixtures are trimmed but verbatim-shaped copies of real dialogues fetched from
the CodaLab source (see data/ingest_craigslist.py's module docstring for the
verification notes) — not invented structures.
"""

from pathlib import Path

from data.ingest_craigslist import ingest, normalize_dialogue
from data.schema import Action

# Real agreement dialogue (GoPro Hero4), trimmed to the events that matter.
RAW_AGREEMENT = {
    "uuid": "C_91d39147df0946bfa0278f0286421796",
    "scenario_uuid": "S_To118PXuNicOd8SO",
    "scenario": {
        "category": "electronics",
        "post_id": "6122134540",
        "kbs": [
            {
                "personal": {"Role": "buyer", "Bottomline": None, "Target": 243},
                "item": {
                    "Category": "electronics",
                    "Title": "GoPro Hero4 Black + Battery BacPac",
                    "Price": 265,
                    "Description": ["- HERO4 Black Camera", "- Standard Housing"],
                    "Images": ["electronics/6122134540_0.jpg"],
                },
            },
            {
                "personal": {"Role": "seller", "Bottomline": None, "Target": 265},
                "item": {
                    "Category": "electronics",
                    "Title": "GoPro Hero4 Black + Battery BacPac",
                    "Price": 265,
                    "Description": ["- HERO4 Black Camera", "- USB Cable", "- Battery BacPac"],
                    "Images": ["electronics/6122134540_0.jpg"],
                },
            },
        ],
    },
    "events": [
        {"agent": 0, "action": "message", "data": "hi there", "metadata": {"intent": "intro", "price": None}},
        {"agent": 1, "action": "message", "data": "Good Day!", "metadata": {"intent": "unknown", "price": None}},
        {
            "agent": 0,
            "action": "message",
            "data": "how about $243 for it?",
            "metadata": {"intent": "init-price", "price": 243.0},
        },
        {
            "agent": 1,
            "action": "offer",
            "data": {"price": 243.0, "sides": ""},
            "metadata": {"intent": "offer", "price": 243.0},
        },
        {"agent": 0, "action": "accept", "data": None, "metadata": {"intent": "accept"}},
    ],
    "outcome": {"reward": 1, "offer": {"price": 243.0, "sides": ""}},
    "agents": {"0": "human", "1": "human"},
}

# Real no-deal dialogue (housing rental) ending in quit after an offer — the
# offer that was made is never accepted, so reward stays 0.
RAW_QUIT_AFTER_OFFER = {
    "uuid": "C_quit_example_0001",
    "scenario_uuid": "S_L7F3EuuCeGp3YW2v",
    "scenario": {
        "category": "housing",
        "post_id": "6132705525",
        "kbs": [
            {
                "personal": {"Role": "seller", "Bottomline": None, "Target": 2350},
                "item": {"Category": "housing", "Title": "Berkeley apartment", "Price": 2350,
                          "Description": ["Nice place"], "Images": []},
            },
            {
                "personal": {"Role": "buyer", "Bottomline": None, "Target": 1200},
                "item": {"Category": "housing", "Title": "Berkeley apartment", "Price": 2350,
                          "Description": ["Nice place"], "Images": []},
            },
        ],
    },
    "events": [
        {"agent": 0, "action": "message", "data": "Are you interested?", "metadata": {"intent": "intro", "price": None}},
        {
            "agent": 0,
            "action": "message",
            "data": "I can offer you a price of $1800 a month.",
            "metadata": {"intent": "init-price", "price": 1800.0},
        },
        {
            "agent": 0,
            "action": "offer",
            "data": {"price": 1800.0, "sides": ""},
            "metadata": {"intent": "offer", "price": 1800.0},
        },
        {"agent": 0, "action": "quit", "data": None, "metadata": {"intent": "quit"}},
    ],
    "outcome": {"reward": 0, "offer": None},
    "agents": {"0": "human", "1": "human"},
}

# Real "rejected offer" shape: outcome.offer is populated even though
# reward == 0 — must NOT be treated as agreement.
RAW_REJECTED_OFFER = {
    "uuid": "C_reject_example_0001",
    "scenario_uuid": "S_reject_0001",
    "scenario": {
        "category": "bike",
        "post_id": "1",
        "kbs": [
            {"personal": {"Role": "buyer", "Bottomline": None, "Target": 50},
             "item": {"Category": "bike", "Title": "Old bike", "Price": 80,
                       "Description": ["Used"], "Images": []}},
            {"personal": {"Role": "seller", "Bottomline": None, "Target": 80},
             "item": {"Category": "bike", "Title": "Old bike", "Price": 80,
                       "Description": ["Used"], "Images": []}},
        ],
    },
    "events": [
        {"agent": 1, "action": "offer", "data": {"price": 75.0, "sides": ""}, "metadata": {"intent": "offer", "price": 75.0}},
        {"agent": 0, "action": "reject", "data": None, "metadata": {"intent": "unknown"}},
    ],
    "outcome": {"reward": 0, "offer": {"price": 75.0, "sides": ""}},
    "agents": {"0": "human", "1": "human"},
}

# Test-split shape: events carry metadata: null throughout (labels withheld).
RAW_TEST_NO_METADATA = {
    "uuid": "C_test_example_0001",
    "scenario_uuid": "S_test_0001",
    "scenario": {
        "category": "phone",
        "post_id": "2",
        "kbs": [
            {"personal": {"Role": "buyer", "Bottomline": None, "Target": 100},
             "item": {"Category": "phone", "Title": "Phone", "Price": 150,
                       "Description": ["Good condition"], "Images": []}},
            {"personal": {"Role": "seller", "Bottomline": None, "Target": 150},
             "item": {"Category": "phone", "Title": "Phone", "Price": 150,
                       "Description": ["Good condition"], "Images": []}},
        ],
    },
    "events": [
        {"agent": 0, "action": "message", "data": "Interested, could do $100.", "metadata": None},
    ],
    "outcome": {"reward": 0, "offer": None},
    "agents": {"0": "human", "1": "human"},
}


def test_agreement_dialogue_shape():
    t = normalize_dialogue(RAW_AGREEMENT, split="validation")
    assert t.dialogue_id == "C_91d39147df0946bfa0278f0286421796"
    assert t.source == "craigslist_bargain"
    assert {p.party_id for p in t.parties} == {"buyer", "seller"}
    assert t.outcome.agreement_reached is True
    assert t.outcome.final_deal == {"buyer": {"price_usd": 243}, "seller": {"price_usd": 243}}
    assert t.metadata["split"] == "validation"
    assert t.metadata["category"] == "electronics"


def test_speakers_resolved_to_role_not_raw_int():
    t = normalize_dialogue(RAW_AGREEMENT, split="train")
    speakers = {turn.speaker for turn in t.turns}
    assert speakers == {"buyer", "seller"}
    assert all(isinstance(turn.speaker, str) for turn in t.turns)


def test_actions_map_onto_existing_action_enum():
    t = normalize_dialogue(RAW_AGREEMENT, split="train")
    action_turns = [turn for turn in t.turns if turn.action is not None]
    assert [turn.action for turn in action_turns] == [Action.SUBMIT_DEAL, Action.ACCEPT_DEAL]
    offer_turn = action_turns[0]
    assert offer_turn.action_data == {"price": 243.0, "sides": ""}
    accept_turn = action_turns[1]
    assert accept_turn.action_data is None  # accept's raw data is null


def test_party_metadata_captures_target_and_item_info():
    t = normalize_dialogue(RAW_AGREEMENT, split="train")
    buyer = next(p for p in t.parties if p.party_id == "buyer")
    assert buyer.priorities is None
    assert buyer.outcome_points is None  # no native score in this source
    assert buyer.metadata["target"] == 243
    assert buyer.metadata["bottomline"] is None
    assert buyer.metadata["item_title"] == "GoPro Hero4 Black + Battery BacPac"
    assert "HERO4 Black Camera" in buyer.metadata["item_description"]


def test_quit_after_offer_is_not_an_agreement():
    t = normalize_dialogue(RAW_QUIT_AFTER_OFFER, split="train")
    assert t.outcome.agreement_reached is False
    assert t.outcome.final_deal is None
    assert t.turns[-1].action is Action.WALK_AWAY


def test_rejected_offer_with_outcome_offer_populated_is_not_agreement():
    """outcome.offer can be non-null even when reward==0 (the rejected offer) —
    reward must be the sole authority, not offer presence."""
    t = normalize_dialogue(RAW_REJECTED_OFFER, split="train")
    assert t.outcome.agreement_reached is False
    assert t.outcome.final_deal is None
    assert t.turns[-1].action is Action.REJECT_DEAL


def test_null_event_metadata_becomes_empty_dict():
    """Test split carries metadata: null throughout — must not crash, must not
    surface as a Python None on Turn.metadata (which is dict, not Optional)."""
    t = normalize_dialogue(RAW_TEST_NO_METADATA, split="test")
    assert t.turns[0].metadata == {}
    assert t.outcome.agreement_reached is False


def test_ingest_combines_splits_into_one_jsonl(tmp_path: Path):
    import json

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "craigslist_train.json").write_text(
        json.dumps([RAW_AGREEMENT, RAW_REJECTED_OFFER]), encoding="utf-8"
    )
    (raw_dir / "craigslist_validation.json").write_text(
        json.dumps([RAW_QUIT_AFTER_OFFER]), encoding="utf-8"
    )
    out_path = tmp_path / "craigslist_bargain.jsonl"

    summary = ingest(raw_dir, out_path, splits=["train", "validation"])

    assert summary["splits"] == {"train": 2, "validation": 1}
    assert summary["dialogues_written"] == 3
    assert summary["agreements"] == 1  # only RAW_AGREEMENT

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    docs = [json.loads(line) for line in lines]
    assert all(d["source"] == "craigslist_bargain" for d in docs)
    assert {d["metadata"]["split"] for d in docs} == {"train", "validation"}
