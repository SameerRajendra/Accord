"""Tests for the outcome-model feature engineering (no-leakage, order-invariant).

CaSiNo's agent_1/agent_2 labeling is arbitrary, so features must be symmetric
across the two parties — these tests assert that swapping the party order
leaves the feature row unchanged, plus the no-leakage guarantee.
"""

import math

from analysis.outcome_model import build_feature_matrix, extract_features
from data.schema import Action, Outcome, Party, Transcript, Turn


_ISSUES = ("Firewood", "Water", "Food")


def _party(pid, high, svo="proself", agree=6.0, emo=5.0, openness=5.5):
    """Build a party whose priority ranking is always internally consistent.

    Medium/Low are *derived* from `high` rather than passed in: a dict literal
    like `{x: "High", y: "Medium", x: "Low"}` silently keeps only the last
    value, leaving the party with no High issue at all — which is exactly the
    fixture bug this signature prevents.
    """
    rest = [i for i in _ISSUES if i != high]
    return Party(
        party_id=pid,
        priorities={high: "High", rest[0]: "Medium", rest[1]: "Low"},
        outcome_points=20,
        satisfaction="Slightly satisfied",
        opponent_likeness="Slightly like",
        metadata={
            "personality": {
                "svo": svo,
                "big-five": {
                    "agreeableness": agree,
                    "emotional-stability": emo,
                    "openness-to-experiences": openness,
                },
            }
        },
    )


def _agreement_transcript(high_a="Firewood", high_b="Water") -> Transcript:
    return Transcript(
        dialogue_id="casino-1",
        source="casino",
        domain="campsite_resources",
        parties=[
            _party("agent_1", high_a),
            _party("agent_2", high_b, svo="prosocial", agree=4.0, emo=3.0),
        ],
        turns=[
            Turn(index=0, speaker="agent_1", text="Hi!", strategies=["small-talk"]),
            Turn(index=1, speaker="agent_2", text="I need firewood", strategies=["self-need"]),
            Turn(index=2, speaker="agent_1", text="Let's trade", strategies=["promote-coordination"]),
            Turn(
                index=3,
                speaker="agent_1",
                text="Submit-Deal",
                action=Action.SUBMIT_DEAL,
                action_data={"issue2youget": {"Firewood": "3"}, "issue2theyget": {"Water": "3"}},
            ),
            Turn(index=4, speaker="agent_2", text="Accept-Deal", action=Action.ACCEPT_DEAL),
        ],
        outcome=Outcome(
            agreement_reached=True,
            final_deal={"agent_1": {"Firewood": 3}, "agent_2": {"Water": 3}},
            points={"agent_1": 20, "agent_2": 18},
        ),
        has_strategy_annotations=True,
        metadata={"split": "train"},
    )


def _no_personality_transcript() -> Transcript:
    return Transcript(
        dialogue_id="casino-2",
        source="casino",
        domain="campsite_resources",
        parties=[
            Party(party_id="agent_1", priorities={"Water": "High"}, metadata={}),
            Party(party_id="agent_2", priorities={"Water": "High"}, metadata={}),
        ],
        turns=[Turn(index=0, speaker="agent_1", text="hi")],
        outcome=Outcome(agreement_reached=False, final_deal=None, points={}),
        metadata={"split": "test"},
    )


def test_extract_features_basic_values():
    f = extract_features(_agreement_transcript())
    assert f["num_message_turns"] == 3.0  # 3 text turns, not the 2 protocol turns
    assert f["num_offers"] == 1.0
    assert f["high_conflict"] == 0.0  # Firewood vs Water — different High issues
    assert f["num_proself"] == 1.0  # agent_1 proself, agent_2 prosocial
    assert f["both_proself"] == 0.0
    assert f["mean_agreeableness"] == 5.0  # (6 + 4) / 2
    assert f["min_agreeableness"] == 4.0


def test_high_conflict_detected_when_same_high_issue():
    f = extract_features(_agreement_transcript(high_a="Firewood", high_b="Firewood"))
    assert f["high_conflict"] == 1.0


def test_features_are_order_invariant():
    """Swapping which party is agent_1 must not change the feature row —
    the labeling is arbitrary."""
    t = _agreement_transcript()
    swapped = t.model_copy(deep=True)
    swapped.parties = list(reversed(swapped.parties))
    assert extract_features(t) == extract_features(swapped)


def test_no_leakage_no_outcome_features():
    """The feature set must never encode the resolution or any outcome field."""
    f = extract_features(_agreement_transcript())
    forbidden = ("accept", "reject", "quit", "agreement", "final_deal", "points", "satisfaction", "likeness")
    for key in f:
        for bad in forbidden:
            assert bad not in key.lower(), f"feature '{key}' looks like it leaks an outcome"


def test_strategies_not_used_as_features():
    """Strategy annotations exist for only ~38% of dialogues, so their presence
    is a selection artifact — they must not appear as model features."""
    f = extract_features(_agreement_transcript())
    assert not any("strateg" in k.lower() for k in f)
    assert not any("small_talk" in k.lower() or "self_need" in k.lower() for k in f)


def test_missing_personality_yields_nan_not_crash():
    f = extract_features(_no_personality_transcript())
    assert math.isnan(f["mean_agreeableness"])
    assert math.isnan(f["min_emotional_stability"])
    assert f["num_proself"] == 0.0
    assert f["high_conflict"] == 1.0  # both rank Water High


def test_build_feature_matrix_fixed_numeric_schema():
    X, y = build_feature_matrix([_agreement_transcript(), _no_personality_transcript()])
    assert list(y) == [1, 0]
    # Every column numeric, no categorical leftovers.
    assert all(str(dt).startswith(("float", "int")) for dt in X.dtypes)
    assert "high_conflict" in X.columns
    assert "mean_openness_to_experiences" in X.columns
