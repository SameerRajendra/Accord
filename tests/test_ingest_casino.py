"""Tests for CaSiNo ingestion (normalize_dialogue is pure — testable offline).

The key adaptation vs. the old raw-JSON ingestion is that the HF parquet
delivers `annotations` as numpy arrays, not Python lists; these tests assert
the annotation index survives both shapes, plus the outcome/priority/strategy
mapping and the deterministic split.
"""

import numpy as np

from data.ingest_casino import (
    _as_pairs,
    _build_annotation_index,
    _split_for,
    normalize_dialogue,
)
from data.schema import Action


def _raw_dialogue():
    return {
        "chat_logs": [
            {"text": "Hi! Let's work out a deal.", "task_data": {"data": "", "issue2youget": {}, "issue2theyget": {}}, "id": "mturk_agent_1"},
            {"text": "I need firewood for my dog.", "task_data": {"data": ""}, "id": "mturk_agent_2"},
            {
                "text": "Submit-Deal",
                "task_data": {"data": "", "issue2youget": {"Firewood": "3", "Food": "1", "Water": "0"}, "issue2theyget": {"Firewood": "0", "Food": "2", "Water": "3"}},
                "id": "mturk_agent_1",
            },
            {"text": "Accept-Deal", "task_data": {"data": "accept_deal"}, "id": "mturk_agent_2"},
        ],
        "participant_info": {
            "mturk_agent_1": {
                "value2issue": {"Low": "Water", "Medium": "Food", "High": "Firewood"},
                "value2reason": {"High": "cold nights"},
                "outcomes": {"points_scored": 19, "satisfaction": "Slightly satisfied", "opponent_likeness": "Slightly like"},
                "demographics": {"age": 43, "gender": "male"},
                "personality": {"svo": "proself", "big-five": {"agreeableness": 6.0}},
            },
            "mturk_agent_2": {
                "value2issue": {"Low": "Food", "Medium": "Water", "High": "Firewood"},
                "value2reason": {"High": "fleas"},
                "outcomes": {"points_scored": 18, "satisfaction": "Extremely satisfied", "opponent_likeness": "Extremely like"},
                "demographics": {"age": 22, "gender": "female"},
                "personality": {"svo": "proself", "big-five": {"agreeableness": 6.0}},
            },
        },
        # numpy arrays, exactly as the HF parquet delivers them
        "annotations": np.array(
            [
                np.array(["Hi! Let's work out a deal.", "small-talk,elicit-pref"], dtype=object),
                np.array(["I need firewood for my dog.", "self-need,other-need"], dtype=object),
            ],
            dtype=object,
        ),
    }


def test_as_pairs_handles_numpy_arrays():
    raw = _raw_dialogue()
    pairs = _as_pairs(raw["annotations"])
    assert len(pairs) == 2
    assert pairs[0] == ("Hi! Let's work out a deal.", "small-talk,elicit-pref")


def test_as_pairs_handles_plain_lists():
    pairs = _as_pairs([["hello", "small-talk"], ["bye", "non-strategic"]])
    assert pairs == [("hello", "small-talk"), ("bye", "non-strategic")]


def test_annotation_index_filters_to_known_vocab():
    idx = _build_annotation_index([["x", "small-talk,not-a-real-strategy"]])
    assert idx == {"x": ["small-talk"]}  # unknown label dropped


def test_normalize_parties_and_priorities():
    t = normalize_dialogue(_raw_dialogue(), "casino-0", "train")
    ids = {p.party_id for p in t.parties}
    assert ids == {"agent_1", "agent_2"}  # mturk_ prefix stripped
    a1 = next(p for p in t.parties if p.party_id == "agent_1")
    # priorities inverted to issue->priority
    assert a1.priorities == {"Water": "Low", "Food": "Medium", "Firewood": "High"}
    assert a1.outcome_points == 19
    assert a1.satisfaction == "Slightly satisfied"
    assert a1.opponent_likeness == "Slightly like"


def test_normalize_turns_and_strategies():
    t = normalize_dialogue(_raw_dialogue(), "casino-0", "train")
    assert t.has_strategy_annotations is True
    # message turn carries strategies joined by text
    assert t.turns[0].strategies == ["small-talk", "elicit-pref"]
    assert t.turns[1].strategies == ["self-need", "other-need"]
    # protocol turns detected
    assert t.turns[2].action is Action.SUBMIT_DEAL
    assert t.turns[3].action is Action.ACCEPT_DEAL
    # accept detected via task_data.data fallback would also work; here text matches
    assert t.turns[0].action is None


def test_normalize_outcome_agreement_and_deal():
    t = normalize_dialogue(_raw_dialogue(), "casino-0", "train")
    assert t.outcome.agreement_reached is True
    # submitter (agent_1) gets issue2youget; the other gets issue2theyget.
    # "0" coerces to 0 (getting none of an item is a real term); only ""/non-numeric drop.
    assert t.outcome.final_deal["agent_1"] == {"Firewood": 3, "Food": 1, "Water": 0}
    assert t.outcome.points == {"agent_1": 19, "agent_2": 18}


def test_no_accept_means_no_agreement():
    raw = _raw_dialogue()
    raw["chat_logs"] = raw["chat_logs"][:3]  # drop the Accept-Deal
    raw["chat_logs"].append({"text": "Walk-Away", "task_data": {"data": ""}, "id": "mturk_agent_2"})
    t = normalize_dialogue(raw, "casino-0", "train")
    assert t.outcome.agreement_reached is False
    assert t.outcome.final_deal is None


def test_split_is_deterministic_and_valid():
    s1 = _split_for("casino-0")
    s2 = _split_for("casino-0")
    assert s1 == s2
    assert all(_split_for(f"casino-{i}") in {"train", "validation", "test"} for i in range(50))


def test_transcript_validates_end_to_end():
    """normalize_dialogue must return a schema-valid Transcript (contiguous
    indices, known speakers, agreement<->final_deal integrity)."""
    t = normalize_dialogue(_raw_dialogue(), "casino-0", "train")
    # round-trips through pydantic validation without raising
    from data.schema import Transcript

    assert Transcript.model_validate_json(t.model_dump_json()) == t
