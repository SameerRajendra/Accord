"""Tests for the RAG case-corpus builder (pure, deterministic — no LLM).

Fixtures build `Transcript` objects directly in CraigslistBargain's shape
(buyer/seller party_ids, target-based metadata, single shared-price outcome)
rather than importing from ingest_craigslist, keeping this module's tests
independent of ingestion's raw-dict format.
"""

from pathlib import Path

from data.build_case_corpus import build_case, build_cases, build_corpus
from data.schema import Action, CaseDocument, Outcome, Party, Transcript, Turn


def _agreement_transcript() -> Transcript:
    return Transcript(
        dialogue_id="C_agreement_0001",
        source="craigslist_bargain",
        domain="craigslist_price_negotiation",
        parties=[
            Party(
                party_id="buyer",
                priorities=None,
                outcome_points=None,
                metadata={
                    "role": "buyer",
                    "target": 243,
                    "bottomline": None,
                    "item_category": "electronics",
                    "item_title": "GoPro Hero4 Black",
                    "item_listed_price": 265,
                },
            ),
            Party(
                party_id="seller",
                priorities=None,
                outcome_points=None,
                metadata={
                    "role": "seller",
                    "target": 265,
                    "bottomline": None,
                    "item_category": "electronics",
                    "item_title": "GoPro Hero4 Black",
                    "item_listed_price": 265,
                },
            ),
        ],
        turns=[
            Turn(index=0, speaker="buyer", text="hi there", metadata={"intent": "intro"}),
            Turn(index=1, speaker="seller", text="Good Day!", metadata={"intent": "unknown"}),
            Turn(
                index=2,
                speaker="buyer",
                text="how about $243?",
                metadata={"intent": "init-price", "price": 243.0},
            ),
            Turn(
                index=3,
                speaker="seller",
                text="Offer",
                action=Action.SUBMIT_DEAL,
                action_data={"price": 243.0, "sides": ""},
                metadata={"intent": "offer", "price": 243.0},
            ),
            Turn(index=4, speaker="buyer", text="Accept", action=Action.ACCEPT_DEAL, metadata={"intent": "accept"}),
        ],
        outcome=Outcome(
            agreement_reached=True,
            final_deal={"buyer": {"price_usd": 243}, "seller": {"price_usd": 243}},
            points={},
        ),
        has_strategy_annotations=False,
        metadata={"split": "train", "category": "electronics"},
    )


def _no_agreement_transcript() -> Transcript:
    return Transcript(
        dialogue_id="C_quit_0001",
        source="craigslist_bargain",
        domain="craigslist_price_negotiation",
        parties=[
            Party(
                party_id="buyer",
                metadata={"role": "buyer", "target": 1200, "bottomline": None,
                          "item_category": "housing", "item_title": "Apartment", "item_listed_price": 2350},
            ),
            Party(
                party_id="seller",
                metadata={"role": "seller", "target": 2350, "bottomline": None,
                          "item_category": "housing", "item_title": "Apartment", "item_listed_price": 2350},
            ),
        ],
        turns=[
            Turn(index=0, speaker="seller", text="Interested?", metadata={"intent": "intro"}),
            Turn(
                index=1,
                speaker="seller",
                text="Offer",
                action=Action.SUBMIT_DEAL,
                action_data={"price": 1800.0, "sides": ""},
                metadata={"intent": "offer", "price": 1800.0},
            ),
            Turn(index=2, speaker="seller", text="Quit", action=Action.WALK_AWAY, metadata={"intent": "quit"}),
        ],
        outcome=Outcome(agreement_reached=False, final_deal=None, points={}),
        has_strategy_annotations=False,
        metadata={"split": "validation", "category": "housing"},
    )


def test_agreement_case_renders_setup_acts_outcome_lesson():
    case = build_case(_agreement_transcript())
    assert case.kind == "case"
    assert case.case_id == "craigslist_bargain-C_agreement_0001"
    text = case.text
    assert "Listed at $265 (electronics)" in text
    assert "buyer target: $243" in text and "seller target: $265" in text
    assert "init-price (1)" in text  # meaningful intent counted
    assert "Outcome: Agreement reached at $243." in text
    # price ($243) sits exactly at buyer's target -> favors the buyer
    assert "favored the buyer" in text


def test_agreement_case_metadata():
    case = build_case(_agreement_transcript())
    m = case.metadata
    assert m["outcome_label"] == "agreement"
    assert m["agreement_reached"] is True
    assert m["final_price_usd"] == 243
    assert m["buyer_target"] == 243
    assert m["seller_target"] == 265
    assert m["category"] == "electronics"
    assert m["split"] == "train"
    assert "init-price" in m["dominant_dialogue_acts"]


def test_protocol_intents_excluded_from_dialogue_acts():
    """offer/accept/unknown intents shouldn't appear as 'dialogue acts observed'
    (they're protocol echoes, already captured in the outcome line)."""
    case = build_case(_agreement_transcript())
    assert "offer (1)" not in case.text
    assert "accept (1)" not in case.text
    assert "unknown" not in case.text


def test_no_agreement_case_flags_breakdown():
    case = build_case(_no_agreement_transcript())
    assert "No agreement" in case.text
    assert "ended via walk_away" in case.text
    assert "broke down" in case.text
    assert case.metadata["outcome_label"] == "no_agreement"
    assert case.metadata["final_price_usd"] is None


def test_case_document_roundtrips():
    case = build_case(_agreement_transcript())
    assert CaseDocument.model_validate_json(case.model_dump_json()) == case


def test_build_cases_covers_every_transcript():
    """Unlike CaSiNo, there's no annotated/unannotated split — every dialogue
    becomes a case."""
    cases = build_cases([_agreement_transcript(), _no_agreement_transcript()])
    assert len(cases) == 2


def test_build_corpus_writes_jsonl(tmp_path: Path):
    import json

    in_path = tmp_path / "craigslist_bargain.jsonl"
    in_path.write_text(
        _agreement_transcript().model_dump_json() + "\n" + _no_agreement_transcript().model_dump_json() + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "case_corpus.jsonl"

    summary = build_corpus(in_path, out_path)

    assert summary["cases"] == 2
    assert summary["agreements"] == 1
    assert summary["agreement_rate"] == 0.5

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    docs = [json.loads(line) for line in lines]
    assert all(d["kind"] == "case" and d["source"] == "craigslist_bargain" for d in docs)
