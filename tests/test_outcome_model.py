"""Tests for the outcome-model feature engineering (the no-leakage design)."""

import math

from analysis.outcome_model import CATEGORY_VALUES, build_feature_matrix, extract_features
from data.schema import Action, Outcome, Party, Transcript, Turn


def _agreement_transcript() -> Transcript:
    return Transcript(
        dialogue_id="C_1",
        source="craigslist_bargain",
        domain="craigslist_price_negotiation",
        parties=[
            Party(party_id="buyer", metadata={"target": 243, "item_listed_price": 265}),
            Party(party_id="seller", metadata={"target": 265, "item_listed_price": 265}),
        ],
        turns=[
            Turn(index=0, speaker="buyer", text="hi", metadata={"intent": "intro"}),
            Turn(index=1, speaker="seller", text="hey", metadata={"intent": "unknown"}),
            Turn(index=2, speaker="buyer", text="$243?", metadata={"intent": "init-price"}),
            Turn(index=3, speaker="seller", text="ok, that's fine", metadata={"intent": "agree"}),
            Turn(
                index=4,
                speaker="seller",
                text="Offer",
                action=Action.SUBMIT_DEAL,
                action_data={"price": 243.0, "sides": ""},
                metadata={"intent": "offer"},
            ),
            Turn(index=5, speaker="buyer", text="Accept", action=Action.ACCEPT_DEAL, metadata={"intent": "accept"}),
        ],
        outcome=Outcome(
            agreement_reached=True,
            final_deal={"buyer": {"price_usd": 243}, "seller": {"price_usd": 243}},
            points={},
        ),
        metadata={"split": "train", "category": "electronics"},
    )


def _no_target_transcript() -> Transcript:
    """Defensive case: targets missing (shouldn't happen in real data, but the
    real data's Bottomline is always null, so missing-value handling matters)."""
    return Transcript(
        dialogue_id="C_2",
        source="craigslist_bargain",
        domain="craigslist_price_negotiation",
        parties=[Party(party_id="buyer", metadata={}), Party(party_id="seller", metadata={})],
        turns=[Turn(index=0, speaker="buyer", text="hi")],
        outcome=Outcome(agreement_reached=False, final_deal=None, points={}),
        metadata={"split": "test", "category": "bike"},
    )


def test_extract_features_basic_values():
    f = extract_features(_agreement_transcript())
    assert f["buyer_target"] == 243.0
    assert f["seller_target"] == 265.0
    assert f["listed_price"] == 265.0
    assert f["target_gap"] == 22.0  # seller - buyer
    assert f["num_message_turns"] == 4.0  # intro, unknown, init-price, agree (not offer/accept)
    assert f["num_offers"] == 1.0
    assert f["first_mover_is_buyer"] == 1.0
    assert f["category"] == "electronics"


def test_extract_features_counts_meaningful_intents_only():
    f = extract_features(_agreement_transcript())
    assert f["intent_intro"] == 1.0
    assert f["intent_init_price"] == 1.0
    assert f["intent_agree"] == 1.0
    # "unknown", "offer", "accept" are not meaningful intents -> no such keys
    assert not any(k.startswith("intent_") and "unknown" in k for k in f)
    assert not any(k.startswith("intent_") and "offer" in k for k in f)
    assert not any(k.startswith("intent_") and "accept" in k for k in f)


def test_no_leakage_no_resolution_action_features():
    """The feature set must never directly encode the resolution action
    (accept/reject/quit) — that would just re-encode the label."""
    f = extract_features(_agreement_transcript())
    forbidden_substrings = ("accept", "reject", "quit", "agreement_reached", "final_deal")
    for key in f:
        for bad in forbidden_substrings:
            assert bad not in key.lower(), f"feature '{key}' looks like it leaks the resolution"


def test_extract_features_handles_missing_targets():
    f = extract_features(_no_target_transcript())
    assert math.isnan(f["buyer_target"])
    assert math.isnan(f["seller_target"])
    assert math.isnan(f["listed_price"])
    assert math.isnan(f["target_gap"])
    assert f["num_message_turns"] == 1.0
    assert f["num_offers"] == 0.0
    assert f["category"] == "bike"


def test_build_feature_matrix_has_all_category_columns_regardless_of_split_content():
    """Even though these two transcripts only cover 'electronics' and 'bike',
    every CATEGORY_VALUES column must exist (fixed schema across splits)."""
    X, y = build_feature_matrix([_agreement_transcript(), _no_target_transcript()])
    for cat in CATEGORY_VALUES:
        assert f"category_{cat}" in X.columns
    assert "category" not in X.columns  # dropped after one-hot encoding
    assert list(y) == [1, 0]
    assert X.loc[0, "category_electronics"] == 1.0
    assert X.loc[0, "category_bike"] == 0.0
    assert X.loc[1, "category_bike"] == 1.0
