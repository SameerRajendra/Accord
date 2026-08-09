"""Extreme-behavior detection via structured LLM output.

One LLM call over the full transcript flags whether any turn exhibits each of
six extreme behaviors (thesis taxonomy §5 + Voss's tactical categories),
along with the turn indices that triggered the flag. Flag = boolean + a
confidence in [0,1] (self-reported by the model; the eval harness in
`evals/behavior_eval.py` calibrates this against labels — Phase 5).

Returns one `BehaviorFlags` per transcript, not per turn — extreme-behavior
detection is a dialogue-level signal.
"""

from __future__ import annotations

import logging
from typing import List

from pydantic import BaseModel, Field

from agent.callbacks import langfuse_callbacks
from agent.llm import chat_model
from data.schema import Transcript

logger = logging.getLogger(__name__)


class BehaviorFlag(BaseModel):
    """One extreme-behavior category and whether the transcript exhibits it."""

    name: str = Field(..., description="Behavior category (see BEHAVIOR_CATEGORIES).")
    present: bool = Field(..., description="True if any turn in the transcript exhibits this behavior.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model self-reported confidence in [0,1].")
    turn_indices: List[int] = Field(
        default_factory=list,
        description="Turn indices that triggered the flag. Empty if `present=False`.",
    )
    evidence: str = Field(
        "",
        description="One-sentence quote or paraphrase of the strongest evidence. Empty if not present.",
    )


class BehaviorFlags(BaseModel):
    """All behavior categories for a transcript, ordered as BEHAVIOR_CATEGORIES."""

    flags: List[BehaviorFlag]


BEHAVIOR_CATEGORIES = [
    "threats",
    "ultimatums",
    "stonewalling",
    "personal_attacks",
    "deception_signals",
    "extreme_anchoring",
]


_SYSTEM = (
    "You detect extreme negotiation behaviors in transcripts. For each of the "
    "six categories below, return one BehaviorFlag entry. `present=True` only "
    "when a turn clearly exhibits the behavior — err toward `False` when in doubt. "
    "`turn_indices` must reference real turn indices from the input. `evidence` is "
    "one short sentence; leave empty when `present=False`.\n\n"
    "Categories:\n"
    "- threats: explicit or implicit threats of harm, walkout, retaliation.\n"
    "- ultimatums: 'take it or leave it' framings, single non-negotiable demand.\n"
    "- stonewalling: refusal to engage, one-word dismissals, silence-as-tactic.\n"
    "- personal_attacks: attacks on the person, not the position; insults, ad hominem.\n"
    "- deception_signals: contradictions, evasions, obvious misrepresentation of facts.\n"
    "- extreme_anchoring: opening offer wildly outside the reasonable zone (>2x from listed/target)."
)


def _render_turns(transcript: Transcript) -> str:
    lines: List[str] = []
    for turn in transcript.turns:
        if turn.action is not None:
            continue
        lines.append(f"[turn_index={turn.index}] {turn.speaker}: {turn.text}")
    return "\n".join(lines)


def _default_flags() -> List[BehaviorFlag]:
    return [
        BehaviorFlag(name=name, present=False, confidence=0.0, turn_indices=[], evidence="")
        for name in BEHAVIOR_CATEGORIES
    ]


def detect(transcript: Transcript) -> List[BehaviorFlag]:
    """Return flags in `BEHAVIOR_CATEGORIES` order, with unknown categories dropped."""
    rendered = _render_turns(transcript)
    if not rendered:
        return _default_flags()

    model = chat_model(temperature=0.0, max_tokens=1024).with_structured_output(BehaviorFlags)
    prompt = [
        ("system", _SYSTEM),
        ("user", "Analyze the following transcript.\n\n" + rendered),
    ]
    try:
        result: BehaviorFlags = model.invoke(prompt, config={"callbacks": langfuse_callbacks()})
    except Exception as exc:  # noqa: BLE001 — degrade to all-false
        logger.exception("behaviors LLM call failed: %s", exc)
        return _default_flags()

    # Reorder to canonical order, add any missing category as a false flag,
    # drop unrecognized category names (defensive against structured-output
    # drift).
    by_name = {f.name: f for f in result.flags}
    ordered: List[BehaviorFlag] = []
    for name in BEHAVIOR_CATEGORIES:
        entry = by_name.get(name)
        if entry is None:
            ordered.append(
                BehaviorFlag(name=name, present=False, confidence=0.0, turn_indices=[], evidence="")
            )
        else:
            ordered.append(entry)
    return ordered
