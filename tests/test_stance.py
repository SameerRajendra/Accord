"""Tests for per-party stance + discussion trajectory.

The LLM call is mocked throughout — what's under test is the reconciliation
around it, which is where this stage can quietly lie: attributing a stance to
someone who never spoke, citing a turn that doesn't exist, or dressing up a
failed call as a calm reading. Each of those is asserted against explicitly.
"""

from __future__ import annotations

from unittest.mock import patch

from analysis.stance import (
    Direction,
    Flexibility,
    PartyMood,
    PartyStance,
    StanceReport,
    Trajectory,
    analyze,
)
from data.schema import Action, Outcome, Party, Transcript, Turn


def _transcript(specs) -> Transcript:
    """Build a schema-valid Transcript from (speaker, text[, action]) tuples."""
    turns = [
        Turn(
            index=i,
            speaker=spec[0],
            text=spec[1],
            action=(spec[2] if len(spec) > 2 else None),
        )
        for i, spec in enumerate(specs)
    ]
    party_ids: list = []
    for t in turns:
        if t.speaker not in party_ids:
            party_ids.append(t.speaker)
    return Transcript(
        dialogue_id="stance-test",
        source="test",
        domain="unit-test",
        parties=[Party(party_id=p, metadata={}) for p in party_ids],
        turns=turns,
        outcome=Outcome(agreement_reached=False, final_deal=None, points={}),
    )


TWO_PARTY = _transcript(
    [
        ("Priya", "Can you walk me through the 40% uplift?"),
        ("Daniel", "It reflects market rates. I won't itemise our costs."),
        ("Priya", "This is starting to feel like a hostage situation."),
        ("Daniel", "38% and the auto-renewal stands. Sign by Friday or we lapse."),
    ]
)


def _stance(party, flexibility=Flexibility.MODERATE, mood=PartyMood.GUARDED, turns=None):
    return PartyStance(
        party=party,
        mood=mood,
        flexibility=flexibility,
        position="unchanged terms",
        evidence_turns=turns if turns is not None else [],
        rationale="test fixture",
    )


def _trajectory(direction=Direction.ESCALATING, turning_point=None, confidence=0.8):
    return Trajectory(
        direction=direction,
        confidence=confidence,
        reasoning="test fixture",
        turning_point_turn=turning_point,
    )


def _run(transcript, report):
    """Run `analyze` with the LLM mocked to return `report`. Yields (result, mock)."""
    with patch("analysis.stance.chat_model") as mock_model:
        mock_model.return_value.with_structured_output.return_value.invoke.return_value = report
        result = analyze(transcript)
    return result, mock_model


# --- ordering --------------------------------------------------------------


def test_parties_ordered_least_flexible_first():
    """The party with no room to move is the one blocking the deal — surface
    them first regardless of the order the model happened to emit."""
    report = StanceReport(
        parties=[
            _stance("Priya", Flexibility.OPEN),
            _stance("Daniel", Flexibility.RIGID),
        ],
        trajectory=_trajectory(),
    )
    result, _ = _run(TWO_PARTY, report)
    assert [p.party for p in result.parties] == ["Daniel", "Priya"]


def test_equal_flexibility_keeps_first_appearance_order():
    """The sort is stable, so ties fall back to who spoke first rather than to
    whatever order the model returned."""
    transcript = _transcript([("A", "one"), ("B", "two"), ("C", "three")])
    report = StanceReport(
        parties=[
            _stance("C", Flexibility.MODERATE),
            _stance("B", Flexibility.MODERATE),
            _stance("A", Flexibility.MODERATE),
        ],
        trajectory=_trajectory(),
    )
    result, _ = _run(transcript, report)
    assert [p.party for p in result.parties] == ["A", "B", "C"]


def test_unknown_flexibility_sorts_last():
    """A filled-in default carries no information, so it must not outrank a
    real reading."""
    report = StanceReport(
        parties=[_stance("Daniel", Flexibility.OPEN)],  # Priya omitted -> unknown
        trajectory=_trajectory(),
    )
    result, _ = _run(TWO_PARTY, report)
    assert [p.party for p in result.parties] == ["Daniel", "Priya"]
    assert result.parties[-1].flexibility is Flexibility.UNKNOWN


# --- defaults and reconciliation -------------------------------------------


def test_omitted_party_defaults_to_unknown_not_a_guess():
    """A party the model skipped must read as 'no reading', never as neutral —
    'we don't know' and 'they're calm' are different claims."""
    report = StanceReport(
        parties=[_stance("Daniel", Flexibility.RIGID, PartyMood.HARDENED)],
        trajectory=_trajectory(),
    )
    result, _ = _run(TWO_PARTY, report)
    priya = next(p for p in result.parties if p.party == "Priya")
    assert priya.mood is PartyMood.UNKNOWN
    assert priya.flexibility is Flexibility.UNKNOWN
    assert priya.evidence_turns == []


def test_stance_for_a_non_participant_is_dropped():
    """A stance attributed to someone who never spoke is a hallucination the UI
    would otherwise render as fact."""
    report = StanceReport(
        parties=[
            _stance("Priya", Flexibility.MODERATE),
            _stance("Daniel", Flexibility.RIGID),
            _stance("Legal Team", Flexibility.LOW),
        ],
        trajectory=_trajectory(),
    )
    result, _ = _run(TWO_PARTY, report)
    assert [p.party for p in result.parties] == ["Daniel", "Priya"]


def test_party_matching_tolerates_case_and_spacing():
    """The model re-types the speaker name; matching on the exact string would
    silently turn every entry into an 'omitted party' default."""
    report = StanceReport(
        parties=[_stance("  daniel ", Flexibility.RIGID), _stance("PRIYA", Flexibility.OPEN)],
        trajectory=_trajectory(),
    )
    result, _ = _run(TWO_PARTY, report)
    # Matched, and echoed back with the transcript's spelling so the UI can key on it.
    assert [p.party for p in result.parties] == ["Daniel", "Priya"]
    assert result.parties[0].flexibility is Flexibility.RIGID


def test_duplicate_party_entries_collapse_to_the_first():
    """Two cards for one person would double-count them in the UI."""
    report = StanceReport(
        parties=[
            _stance("Daniel", Flexibility.RIGID),
            _stance("Daniel", Flexibility.OPEN),
        ],
        trajectory=_trajectory(),
    )
    result, _ = _run(TWO_PARTY, report)
    assert [p.party for p in result.parties] == ["Daniel", "Priya"]
    assert result.parties[0].flexibility is Flexibility.RIGID


def test_evidence_turns_are_filtered_and_deduped():
    """Evidence links back to a turn in the UI, so an index the transcript
    doesn't have would point at nothing."""
    report = StanceReport(
        parties=[_stance("Daniel", Flexibility.RIGID, turns=[1, 3, 3, 99, -1])],
        trajectory=_trajectory(),
    )
    result, _ = _run(TWO_PARTY, report)
    assert result.parties[0].evidence_turns == [1, 3]


def test_turning_point_outside_the_transcript_is_dropped():
    report = StanceReport(parties=[], trajectory=_trajectory(turning_point=42))
    result, _ = _run(TWO_PARTY, report)
    assert result.trajectory.turning_point_turn is None
    assert result.trajectory.direction is Direction.ESCALATING  # the rest survives


def test_valid_turning_point_is_kept():
    report = StanceReport(parties=[], trajectory=_trajectory(turning_point=3))
    result, _ = _run(TWO_PARTY, report)
    assert result.trajectory.turning_point_turn == 3


# --- degradation -----------------------------------------------------------


def test_llm_failure_degrades_to_an_explicit_unknown():
    """A serving failure must never reach the API as an exception — and must not
    be reported as a reading either."""
    with patch("analysis.stance.chat_model") as mock_model:
        mock_model.return_value.with_structured_output.return_value.invoke.side_effect = (
            RuntimeError("sglang down")
        )
        result = analyze(TWO_PARTY)

    assert result.trajectory.direction is Direction.UNKNOWN
    assert result.trajectory.confidence == 0.0
    assert "RuntimeError" in result.trajectory.reasoning
    assert [p.party for p in result.parties] == ["Priya", "Daniel"]  # transcript order
    assert all(p.mood is PartyMood.UNKNOWN for p in result.parties)


def test_transcript_without_text_turns_skips_the_llm():
    """Protocol events carry no stance signal — there is nothing to send."""
    transcript = _transcript([("a", "", Action.WALK_AWAY)])
    with patch("analysis.stance.chat_model") as mock_model:
        result = analyze(transcript)

    mock_model.assert_not_called()
    assert result.parties == []
    assert result.trajectory.direction is Direction.UNKNOWN


# --- prompt wiring ---------------------------------------------------------


def test_prompt_renders_speaker_and_text_only():
    """Protocol events are excluded and each turn is rendered with its real
    index, which is what makes `evidence_turns` checkable."""
    transcript = _transcript(
        [
            ("Priya", "opening ask"),
            ("Daniel", "", Action.SUBMIT_DEAL),
            ("Daniel", "final position"),
        ]
    )
    report = StanceReport(parties=[], trajectory=_trajectory())
    _, mock_model = _run(transcript, report)

    call = mock_model.return_value.with_structured_output.return_value.invoke.call_args
    user_message = call.args[0][1][1]
    assert "[turn_index=0] Priya: opening ask" in user_message
    assert "[turn_index=2] Daniel: final position" in user_message
    assert "turn_index=1" not in user_message  # the submit_deal event
    assert "Participants: Priya, Daniel" in user_message


def test_structured_output_and_langfuse_callbacks_are_wired():
    """Same contract as every other LLM stage: schema-constrained decode plus a
    Langfuse trace (a no-op list when Langfuse is unconfigured)."""
    report = StanceReport(parties=[], trajectory=_trajectory())
    _, mock_model = _run(TWO_PARTY, report)

    mock_model.return_value.with_structured_output.assert_called_once_with(StanceReport)
    call = mock_model.return_value.with_structured_output.return_value.invoke.call_args
    assert "callbacks" in call.kwargs["config"]
