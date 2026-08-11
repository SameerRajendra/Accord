"""Tests for email-thread parsing.

The plain-transcript fast path and the Transcript construction are pure and
deterministic, so they're tested directly. The LLM path is mocked — what
matters there is that a bad or unusable result becomes a clear
`ThreadParseError` rather than a malformed Transcript reaching the pipeline.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from analysis.parse_thread import (
    MAX_CHARS,
    ParsedThread,
    ParsedTurn,
    ThreadParseError,
    _parse_plain_transcript,
    _to_transcript,
    parse_thread,
)
from data.schema import Transcript


# --- plain-transcript fast path -------------------------------------------


def test_plain_transcript_parsed_without_llm():
    text = "Priya: Can you explain the 40% uplift?\nDaniel: It reflects market rates."
    with patch("analysis.parse_thread.chat_model") as mock_model:
        t = parse_thread(text)
    mock_model.assert_not_called()  # fast path must not hit the LLM
    assert [turn.speaker for turn in t.turns] == ["Priya", "Daniel"]
    assert t.turns[0].text == "Can you explain the 40% uplift?"


def test_plain_transcript_joins_continuation_lines():
    turns = _parse_plain_transcript(
        "Priya: We budgeted for a modest increase\nnot forty percent.\nDaniel: Understood."
    )
    assert len(turns) == 2
    assert turns[0].text == "We budgeted for a modest increase not forty percent."


def test_leading_prose_is_not_the_plain_format():
    """A thread starting with narrative text must fall through to the LLM."""
    assert _parse_plain_transcript("Here is the thread below.\nPriya: hello") == []


# --- Transcript construction ----------------------------------------------


def test_parties_are_derived_from_turns():
    """Deriving parties from turns is what guarantees Transcript's validator
    (every speaker must be a known party) can never fail."""
    turns = [
        ParsedTurn(speaker="Priya Raman", text="a"),
        ParsedTurn(speaker="Daniel Okafor", text="b"),
        ParsedTurn(speaker="Priya Raman", text="c"),
    ]
    t = _to_transcript(turns, "thread-1", subject="MSA renewal")
    assert [p.party_id for p in t.parties] == ["Priya Raman", "Daniel Okafor"]  # order preserved, deduped
    assert [turn.index for turn in t.turns] == [0, 1, 2]
    assert t.metadata["subject"] == "MSA renewal"
    # Round-trips through pydantic validation without raising.
    assert Transcript.model_validate_json(t.model_dump_json()) == t


def test_outcome_is_a_neutral_placeholder():
    """A live thread's outcome is unknown; the schema requires one anyway."""
    t = _to_transcript([ParsedTurn(speaker="A", text="x"), ParsedTurn(speaker="B", text="y")], "t", None)
    assert t.outcome.agreement_reached is False
    assert t.outcome.final_deal is None


def test_email_metadata_absent_so_outcome_features_are_empty():
    """Email threads carry no priorities/personality — the outcome model
    should degrade, not receive fabricated values."""
    t = _to_transcript([ParsedTurn(speaker="A", text="x"), ParsedTurn(speaker="B", text="y")], "t", None)
    assert all(p.priorities is None and p.metadata == {} for p in t.parties)


# --- error handling --------------------------------------------------------


def test_empty_thread_rejected():
    with pytest.raises(ThreadParseError, match="empty"):
        parse_thread("   ")


def test_oversized_thread_rejected():
    with pytest.raises(ThreadParseError, match="limit"):
        parse_thread("x" * (MAX_CHARS + 1))


def test_single_speaker_rejected():
    """One-sided input isn't a negotiation; fail loudly rather than analyze it."""
    with patch("analysis.parse_thread.chat_model") as mock_model:
        mock_model.return_value.with_structured_output.return_value.invoke.return_value = (
            ParsedThread(subject=None, turns=[
                ParsedTurn(speaker="Priya", text="one"),
                ParsedTurn(speaker="Priya", text="two"),
            ])
        )
        with pytest.raises(ThreadParseError, match="same sender"):
            parse_thread("A long unstructured email body that isn't Speaker: form.")


def test_too_few_turns_rejected():
    with patch("analysis.parse_thread.chat_model") as mock_model:
        mock_model.return_value.with_structured_output.return_value.invoke.return_value = (
            ParsedThread(subject=None, turns=[ParsedTurn(speaker="Priya", text="only one")])
        )
        with pytest.raises(ThreadParseError, match="fewer than two"):
            parse_thread("Some unstructured prose that needs the model.")


def test_llm_failure_becomes_parse_error():
    """A model/serving failure must surface as ThreadParseError so the API
    returns 422 rather than an opaque 500."""
    with patch("analysis.parse_thread.chat_model") as mock_model:
        mock_model.return_value.with_structured_output.return_value.invoke.side_effect = RuntimeError("boom")
        with pytest.raises(ThreadParseError, match="Could not parse"):
            parse_thread("Some unstructured prose that needs the model.")


def test_blank_turns_filtered_before_count_check():
    with patch("analysis.parse_thread.chat_model") as mock_model:
        mock_model.return_value.with_structured_output.return_value.invoke.return_value = (
            ParsedThread(subject=None, turns=[
                ParsedTurn(speaker="Priya", text="real content"),
                ParsedTurn(speaker="Daniel", text="   "),
            ])
        )
        with pytest.raises(ThreadParseError, match="fewer than two"):
            parse_thread("Some unstructured prose that needs the model.")
