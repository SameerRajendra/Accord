"""Per-party stance and discussion trajectory — one batched structured call.

`analysis.sentiment` scores individual turns; nothing in the pipeline turned
those readings into a per-*person* answer, and averaging a party's escalation
scores is not one. Tone and movement are different axes: a negotiator can write
one sharp message while still conceding ground, or stay perfectly courteous
while refusing to move at all. So this stage asks the model for mood and
flexibility directly over the whole thread rather than deriving them from the
per-turn scores.

Stance and trajectory come back from **one** call because they are a single
judgement over the same evidence, and because the recommendation node is
already the latency bottleneck (DESIGN.md §6) — a second fan-out call would add
to the critical path for a reading the first call has to form anyway.

Degradation is deliberately visible rather than silent: a party the model
skipped comes back `mood=unknown, flexibility=unknown`, and a failed call
yields `direction=unknown` with confidence 0.0. A caller can then tell "no
reading" apart from "a calm reading" — the two are not the same claim.

Not validated: CaSiNo carries no stance or trajectory labels, so there is no
ground truth to score these against. The taxonomies below are a design choice,
and `confidence` is a model self-report on an uncalibrated scale — the same
caveat that applies to `behaviors.confidence`.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from agent.callbacks import langfuse_callbacks
from agent.llm import chat_model
from data.schema import Transcript

logger = logging.getLogger(__name__)


class PartyMood(str, Enum):
    """A party's settled disposition across the whole thread.

    Deliberately a different vocabulary from `sentiment.Emotion`, which labels
    one utterance — "this message reads frustrated" and "this person has
    hardened" are different claims, and collapsing them would let a single
    heated line rewrite a party's whole posture.
    """

    COOPERATIVE = "cooperative"
    GUARDED = "guarded"
    FRUSTRATED = "frustrated"
    HARDENED = "hardened"
    ANXIOUS = "anxious"
    DISENGAGED = "disengaged"
    UNKNOWN = "unknown"  # pipeline fill-in only — never a model choice


class Flexibility(str, Enum):
    """How much room to move a party is signalling. Orthogonal to mood."""

    RIGID = "rigid"
    LOW = "low"
    MODERATE = "moderate"
    OPEN = "open"
    UNKNOWN = "unknown"  # pipeline fill-in only


class Direction(str, Enum):
    """Where the exchange is heading, judged over the turn sequence."""

    CONVERGING = "converging"
    HOLDING = "holding"
    STALLING = "stalling"
    ESCALATING = "escalating"
    BREAKING_DOWN = "breaking_down"
    UNKNOWN = "unknown"  # pipeline fill-in only


# Least flexible first: the party with no room to move is the one blocking the
# deal, so that's the card a reader should hit first. `UNKNOWN` carries no
# information and sorts last.
_FLEXIBILITY_RANK: Dict[Flexibility, int] = {
    Flexibility.RIGID: 0,
    Flexibility.LOW: 1,
    Flexibility.MODERATE: 2,
    Flexibility.OPEN: 3,
    Flexibility.UNKNOWN: 4,
}


class PartyStance(BaseModel):
    """One participant's whole-thread posture."""

    party: str = Field(..., description="Speaker name, exactly as it appears in the transcript.")
    mood: PartyMood = Field(..., description="Settled disposition across the thread.")
    flexibility: Flexibility = Field(
        ..., description="How much movement this party is signalling on their position."
    )
    position: str = Field(
        ..., description="The concrete thing this party is holding out for, in one clause."
    )
    evidence_turns: List[int] = Field(
        default_factory=list,
        description="Turn indices that evidence the reading. Real indices from the input only.",
    )
    rationale: str = Field(
        "", description="One short phrase explaining mood + flexibility — trace-friendly."
    )


class Trajectory(BaseModel):
    """Where the discussion as a whole is heading."""

    direction: Direction = Field(..., description="Direction of travel for the exchange.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model self-reported confidence in [0,1]. Uncalibrated."
    )
    reasoning: str = Field(
        ..., description="Two sentences at most, citing what changed across turns."
    )
    turning_point_turn: Optional[int] = Field(
        default=None,
        description="Turn index where the tone changed. Null when no single turn did.",
    )


class StanceReport(BaseModel):
    """Return contract for the batched stance call."""

    parties: List[PartyStance]
    trajectory: Trajectory


_SYSTEM = (
    "You read a whole negotiation thread and report two things: where each "
    "participant now stands, and where the exchange is heading.\n\n"
    "For every participant, return one PartyStance:\n"
    "- party: the speaker's name exactly as it appears in the input.\n"
    "- mood: their disposition across the WHOLE thread, one of {cooperative, "
    "guarded, frustrated, hardened, anxious, disengaged}. This is not the tone "
    "of their last message — someone can write one sharp reply and still be "
    "working toward a deal.\n"
    "- flexibility: how much room to move they are signalling, one of {rigid, "
    "low, moderate, open}. Judge what they say about their own room to move, "
    "not how politely they say it.\n"
    "- position: the concrete thing they are holding out for, in one clause.\n"
    "- evidence_turns: the turn indices that show it. Use real indices from the input.\n"
    "- rationale: one short phrase, not a paragraph.\n\n"
    "Then return exactly one Trajectory for the exchange:\n"
    "- direction: converging (positions closing), holding (steady, neither side "
    "moving), stalling (repetition without progress), escalating (tension rising "
    "turn over turn), breaking_down (heading toward no deal or walkout).\n"
    "- confidence in [0,1].\n"
    "- reasoning: at most two sentences, citing what changed across turns.\n"
    "- turning_point_turn: the turn index where the tone changed, or null if no "
    "single turn did.\n\n"
    "Never output 'unknown' for mood, flexibility or direction — that value is "
    "reserved for entries the pipeline fills in when a reading is missing."
)


def _render_turns(transcript: Transcript) -> tuple[str, List[int]]:
    """Format text-only turns as a numbered block. Returns (rendered, original_indices)."""
    lines: List[str] = []
    original_indices: List[int] = []
    for turn in transcript.turns:
        if turn.action is not None:
            continue  # protocol event — carries no stance signal
        original_indices.append(turn.index)
        lines.append(f"[turn_index={turn.index}] {turn.speaker}: {turn.text}")
    return "\n".join(lines), original_indices


def _speakers(transcript: Transcript) -> List[str]:
    """Participants who actually spoke, in first-appearance order.

    Taken from the turns rather than `Transcript.parties` because a stance has
    to be evidenced by something the person said; a listed-but-silent party
    would otherwise get a reading with nothing behind it.
    """
    seen: List[str] = []
    for turn in transcript.turns:
        if turn.action is not None:
            continue
        if turn.speaker not in seen:
            seen.append(turn.speaker)
    return seen


def _norm(name: str) -> str:
    """Normalize a party name for matching — the model re-types casing and spacing."""
    return " ".join(name.split()).casefold()


def _clean_indices(indices: List[int], valid: Set[int]) -> List[int]:
    """Keep only real text-turn indices, de-duplicated, in the model's order.

    The UI links these back to turns, so an index the transcript doesn't have
    would render as evidence that doesn't exist.
    """
    cleaned: List[int] = []
    for idx in indices or []:
        if idx in valid and idx not in cleaned:
            cleaned.append(idx)
    dropped = len(indices or []) - len(cleaned)
    if dropped:
        logger.warning(
            "stance: dropped %d evidence turn index(es) — unknown or duplicated", dropped
        )
    return cleaned


def _flexibility_rank(stance: PartyStance) -> int:
    return _FLEXIBILITY_RANK.get(stance.flexibility, len(_FLEXIBILITY_RANK))


def _default_stance(party: str) -> PartyStance:
    return PartyStance(
        party=party,
        mood=PartyMood.UNKNOWN,
        flexibility=Flexibility.UNKNOWN,
        position="",
        evidence_turns=[],
        rationale="no reading returned for this party",
    )


def _unknown_trajectory(reason: str) -> Trajectory:
    return Trajectory(
        direction=Direction.UNKNOWN,
        confidence=0.0,
        reasoning=reason,
        turning_point_turn=None,
    )


def _default_report(speakers: List[str], reason: str) -> StanceReport:
    return StanceReport(
        parties=[_default_stance(s) for s in speakers],
        trajectory=_unknown_trajectory(reason),
    )


def _reconcile_parties(
    returned: List[PartyStance], speakers: List[str], valid_indices: Set[int]
) -> List[PartyStance]:
    """Match the model's entries back to real speakers, fill gaps, then order.

    Entries matching no speaker are dropped: a stance attributed to someone who
    never spoke is a hallucination the UI would otherwise present as fact.
    """
    by_norm: Dict[str, PartyStance] = {}
    for entry in returned or []:
        key = _norm(entry.party)
        if key not in by_norm:
            by_norm[key] = entry  # first wins — a repeated party is model drift

    resolved: List[PartyStance] = []
    for speaker in speakers:
        entry = by_norm.get(_norm(speaker))
        if entry is None:
            logger.warning("stance: model omitted party=%r, defaulting to unknown", speaker)
            resolved.append(_default_stance(speaker))
            continue
        resolved.append(
            entry.model_copy(
                update={
                    # Echo the transcript's spelling so the UI can key on it.
                    "party": speaker,
                    "evidence_turns": _clean_indices(entry.evidence_turns, valid_indices),
                }
            )
        )

    unmatched = sorted(set(by_norm) - {_norm(s) for s in speakers})
    if unmatched:
        logger.warning("stance: dropped entries for non-participants: %s", unmatched)

    # Stable sort — parties with equal flexibility stay in first-appearance order.
    return sorted(resolved, key=_flexibility_rank)


def _reconcile_trajectory(trajectory: Trajectory, valid_indices: Set[int]) -> Trajectory:
    """Drop a turning point that doesn't name a real turn."""
    turn = trajectory.turning_point_turn
    if turn is not None and turn not in valid_indices:
        logger.warning("stance: turning_point_turn=%s is not a text turn, dropping", turn)
        return trajectory.model_copy(update={"turning_point_turn": None})
    return trajectory


def analyze(transcript: Transcript) -> StanceReport:
    """Read the whole thread in one batched LLM call.

    Returns one `PartyStance` per speaking participant (least flexible first)
    and one `Trajectory`. Never raises: a failed call returns an explicit
    unknown reading so the API degrades to "we don't know" rather than a 500.
    """
    rendered, indices = _render_turns(transcript)
    speakers = _speakers(transcript)
    if not speakers:
        return _default_report([], "no free-text turns to read")

    model = chat_model(temperature=0.0, max_tokens=1024).with_structured_output(StanceReport)
    prompt = [
        ("system", _SYSTEM),
        (
            "user",
            "Participants: "
            + ", ".join(speakers)
            + "\n\nReturn one PartyStance for each participant above, plus one "
            "Trajectory for the exchange as a whole.\n\n"
            + rendered,
        ),
    ]
    try:
        result: StanceReport = model.invoke(prompt, config={"callbacks": langfuse_callbacks()})
    except Exception as exc:  # noqa: BLE001 — degrade to "unknown" rather than 500
        logger.exception("stance LLM call failed: %s", exc)
        return _default_report(speakers, f"stance model failed ({type(exc).__name__})")

    valid = set(indices)
    return StanceReport(
        parties=_reconcile_parties(result.parties, speakers, valid),
        trajectory=_reconcile_trajectory(result.trajectory, valid),
    )
