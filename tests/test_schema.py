"""Sanity checks on the normalized transcript schema."""

import pytest
from pydantic import ValidationError

from data.schema import Action, Outcome, Party, Transcript, Turn


def _minimal_transcript(**overrides) -> Transcript:
    base = dict(
        dialogue_id="t1",
        source="casino",
        domain="campsite_resources",
        parties=[Party(party_id="agent_1"), Party(party_id="agent_2")],
        turns=[
            Turn(index=0, speaker="agent_1", text="hi"),
            Turn(index=1, speaker="agent_2", text="hello"),
        ],
        outcome=Outcome(agreement_reached=False, final_deal=None, points={}),
    )
    base.update(overrides)
    return Transcript(**base)


def test_roundtrip_json():
    t = _minimal_transcript()
    restored = Transcript.model_validate_json(t.model_dump_json())
    assert restored == t


def test_non_contiguous_turn_indices_rejected():
    with pytest.raises(ValidationError):
        _minimal_transcript(
            turns=[
                Turn(index=0, speaker="agent_1", text="hi"),
                Turn(index=5, speaker="agent_2", text="hello"),
            ]
        )


def test_unknown_speaker_rejected():
    with pytest.raises(ValidationError):
        _minimal_transcript(
            turns=[Turn(index=0, speaker="ghost", text="boo")],
        )


def test_agreement_requires_final_deal():
    with pytest.raises(ValidationError):
        _minimal_transcript(
            outcome=Outcome(agreement_reached=True, final_deal=None, points={}),
        )


def test_no_agreement_forbids_final_deal():
    with pytest.raises(ValidationError):
        _minimal_transcript(
            outcome=Outcome(
                agreement_reached=False,
                final_deal={"agent_1": {"Food": 1}},
                points={},
            ),
        )


def test_action_enum_serialization():
    turn = Turn(index=0, speaker="agent_1", text="Submit-Deal", action=Action.SUBMIT_DEAL)
    assert '"submit_deal"' in turn.model_dump_json()
