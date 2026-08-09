"""Per-turn sentiment/escalation scoring via LangChain + SGLang structured output.

One batched LLM call scores every text turn in the transcript at once (§4
"Sentiment/escalation — one batched call, not N round-trips"). The model
returns a JSON array of `PerTurnSentiment`, one entry per input turn, in the
same order. If SGLang's structured-output guarantee drops a turn (rare) the
missing entries are filled with a neutral default so downstream code can
assume list-length equals turn-count.

Emotion taxonomy: a 6-class demo subset of the planned 10-class scheme
(Never Split the Difference + the thesis). The planned full taxonomy — anger,
fear, distrust, tactical-empathy, stonewalling, urgency, loss-aversion,
anchoring-aggression, fairness-indignation, collaborative-optimism — is
Phase-5 fine-tuning work; the 6-class version is what a zero-shot 7B can hit
reliably.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from agent.callbacks import langfuse_callbacks
from agent.llm import chat_model
from data.schema import Transcript

logger = logging.getLogger(__name__)


class Emotion(str, Enum):
    NEUTRAL = "neutral"
    COLLABORATIVE = "collaborative"
    FRUSTRATED = "frustrated"
    ANXIOUS = "anxious"
    DISTRUSTFUL = "distrustful"
    URGENT = "urgent"


class PerTurnSentiment(BaseModel):
    """One turn's sentiment reading. Ordered by `turn_index`."""

    turn_index: int = Field(..., ge=0, description="0-based index into `Transcript.turns`.")
    emotion: Emotion = Field(..., description="Dominant emotion label from the 6-class taxonomy.")
    escalation: float = Field(
        ..., ge=0.0, le=1.0,
        description="How much this turn escalates tension vs. the running conversation (0=de-escalating, 1=maximally escalating).",
    )
    rationale: str = Field(..., description="One short phrase explaining the labels — trace-friendly.")


class SentimentBatch(BaseModel):
    """Return contract for the batched LLM call."""

    turns: List[PerTurnSentiment]


_SYSTEM = (
    "You analyze negotiation transcripts turn-by-turn. For every free-text turn "
    "in the input, output one PerTurnSentiment entry, in the same order as the "
    "input. Assign one emotion from {neutral, collaborative, frustrated, anxious, "
    "distrustful, urgent} and an escalation score in [0,1] where 0=this turn "
    "cools the conversation and 1=this turn maximally raises tension. The "
    "rationale must be one short phrase, not a paragraph."
)


def _render_turns(transcript: Transcript) -> tuple[str, List[int]]:
    """Format text-only turns as a numbered block. Returns (rendered, original_indices)."""
    lines: List[str] = []
    original_indices: List[int] = []
    for turn in transcript.turns:
        if turn.action is not None:
            continue  # protocol event — not scored
        original_indices.append(turn.index)
        lines.append(f"[turn_index={turn.index}] {turn.speaker}: {turn.text}")
    return "\n".join(lines), original_indices


def _neutral_default(turn_index: int) -> PerTurnSentiment:
    return PerTurnSentiment(
        turn_index=turn_index,
        emotion=Emotion.NEUTRAL,
        escalation=0.0,
        rationale="model omitted this turn — neutral default",
    )


def analyze(transcript: Transcript) -> List[PerTurnSentiment]:
    """Score every text turn in one batched LLM call. Returns a list aligned
    to text-turn order (protocol-event turns are skipped, not stubbed)."""
    rendered, indices = _render_turns(transcript)
    if not indices:
        return []

    model = chat_model(temperature=0.0, max_tokens=1024).with_structured_output(SentimentBatch)
    prompt = [
        ("system", _SYSTEM),
        (
            "user",
            "Score the following turns. Return exactly one entry per turn, in input order.\n\n"
            + rendered,
        ),
    ]
    try:
        result: SentimentBatch = model.invoke(prompt, config={"callbacks": langfuse_callbacks()})
    except Exception as exc:  # noqa: BLE001 — degrade to neutral rather than 500
        logger.exception("sentiment LLM call failed: %s", exc)
        return [_neutral_default(i) for i in indices]

    # Index the model's output by its self-reported turn_index, then walk our
    # expected indices to preserve order + fill gaps with a neutral default.
    by_index = {entry.turn_index: entry for entry in result.turns}
    ordered: List[PerTurnSentiment] = []
    for idx in indices:
        entry = by_index.get(idx)
        if entry is None:
            logger.warning("sentiment: model omitted turn_index=%d, defaulting", idx)
            ordered.append(_neutral_default(idx))
        else:
            ordered.append(entry)
    return ordered
