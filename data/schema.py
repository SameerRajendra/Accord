"""Normalized transcript schema — the data spine of Accord.

Every downstream component (analysis, RAG, agent, evals) consumes this schema,
never a raw dataset format. Keeping one source-agnostic representation means a
new corpus (DealOrNoDeal, CraigslistBargain, ...) only needs a new ingestion
script that emits `Transcript` objects — the rest of the pipeline is unchanged.

Design notes
------------
- Pydantic v2 models: they double as the validation layer and the on-disk
  contract (each transcript is one line of JSONL via `model_dump_json()`).
- `action` on a turn distinguishes real utterances from protocol events
  (deal submission / accept / reject / walk-away). Analysis models look at
  `text` turns; outcome logic looks at `action` turns.
- Strategy annotations are optional and sparse (only ~38% of CaSiNo dialogues
  are annotated), so `Turn.strategies` defaults to empty and
  `Transcript.has_strategy_annotations` records whether labels were available.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Action(str, Enum):
    """Protocol events that are not free-text negotiation turns."""

    SUBMIT_DEAL = "submit_deal"
    ACCEPT_DEAL = "accept_deal"
    REJECT_DEAL = "reject_deal"
    WALK_AWAY = "walk_away"


class Party(BaseModel):
    """One negotiator and everything known about them and their result."""

    party_id: str = Field(..., description="Stable id within the dialogue, e.g. 'agent_1'.")
    priorities: Optional[dict[str, str]] = Field(
        default=None,
        description="issue name -> priority label ('High'/'Medium'/'Low'). Source-specific.",
    )
    outcome_points: Optional[int] = Field(
        default=None, description="Points this party scored (dataset-provided, not recomputed)."
    )
    satisfaction: Optional[str] = None
    opponent_likeness: Optional[str] = None
    metadata: dict = Field(default_factory=dict, description="Demographics, personality, reasons.")


class Turn(BaseModel):
    """A single message. Either a free-text utterance or a protocol action."""

    index: int = Field(..., ge=0, description="0-based position in the dialogue.")
    speaker: str = Field(..., description="party_id of the speaker.")
    text: str
    strategies: list[str] = Field(
        default_factory=list, description="Utterance-level strategy labels, if annotated."
    )
    action: Optional[Action] = Field(
        default=None, description="Set for protocol events; None for ordinary utterances."
    )
    action_data: Optional[dict] = Field(
        default=None, description="Structured payload for an action (e.g. deal terms)."
    )


class Outcome(BaseModel):
    """How the negotiation ended."""

    agreement_reached: bool
    final_deal: Optional[dict[str, dict[str, int]]] = Field(
        default=None,
        description="party_id -> {issue -> quantity} for the accepted deal; None if no agreement.",
    )
    points: dict[str, int] = Field(
        default_factory=dict, description="party_id -> points scored."
    )


class Transcript(BaseModel):
    """A fully normalized negotiation dialogue."""

    dialogue_id: str
    source: str = Field(..., description="Originating dataset, e.g. 'casino'.")
    domain: str = Field(..., description="Negotiation domain, e.g. 'campsite_resources'.")
    parties: list[Party]
    turns: list[Turn]
    outcome: Outcome
    has_strategy_annotations: bool = False
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_integrity(self) -> "Transcript":
        # Turn indices must be contiguous and 0-based.
        expected = list(range(len(self.turns)))
        actual = [t.index for t in self.turns]
        if actual != expected:
            raise ValueError(f"turn indices must be 0..N-1 contiguous, got {actual}")

        # Every speaker must be a known party.
        party_ids = {p.party_id for p in self.parties}
        unknown = {t.speaker for t in self.turns} - party_ids
        if unknown:
            raise ValueError(f"turns reference unknown speakers: {sorted(unknown)}")

        # An agreement must name a concrete deal; no agreement must not.
        if self.outcome.agreement_reached and self.outcome.final_deal is None:
            raise ValueError("agreement_reached=True but final_deal is None")
        if not self.outcome.agreement_reached and self.outcome.final_deal is not None:
            raise ValueError("agreement_reached=False but final_deal is set")
        return self
