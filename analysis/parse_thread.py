"""Parse a raw email thread into Accord's normalized `Transcript`.

The pipeline's analysis stages only ever need *who said what, in order* — so
rather than asking a user to hand-author JSON, this module hands the raw
pasted thread to the same self-hosted model that does the rest of the work and
asks it for structure.

Email threads are messy in ways a regex handles badly and an LLM handles well:

- **Quoted reply chains.** `> original text` and `On Tue, X wrote:` blocks
  repeat earlier messages verbatim. Counting them again would inflate turn
  counts and make escalation scoring nonsense.
- **Newest-first ordering.** Most clients stack the latest reply on top.
  Escalation analysis is order-dependent, so the parser must return
  chronological order regardless of how the thread was pasted.
- **Signatures, disclaimers, headers.** Boilerplate that isn't negotiation
  content.

Design notes
------------
- `Transcript.outcome` is **required by the schema but unknown for a live
  thread** — the whole point is to predict where it's heading. We fill a
  neutral placeholder (`agreement_reached=False`) and every analysis stage
  ignores it. That mismatch is a known wart: `Transcript` was designed for
  *stored, completed* dialogues and is being reused as a request body. Fixing
  it properly means splitting the schema; noted, not done.
- Parties are derived **from the parsed turns**, never from a separate list
  the model returns. `Transcript` validates that every `turn.speaker` is a
  known `party_id`, and a model that lists participants separately will
  eventually disagree with itself. Deriving guarantees consistency.
- `priorities` and `personality` come back `None`/empty: an email thread
  carries no CaSiNo-style priority ranking or Big-Five profile. Sentiment and
  behaviour analysis don't use them; the XGBoost outcome model does, so it
  will return `None` on this path.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from pydantic import BaseModel, Field

from agent.callbacks import langfuse_callbacks
from agent.llm import chat_model
from data.schema import Outcome, Party, Transcript, Turn

logger = logging.getLogger(__name__)

SOURCE = "email_thread"
DOMAIN = "business_negotiation"

# Guardrail: a pasted thread far beyond this is likely a whole mailbox dump.
MAX_CHARS = 24_000


class ParsedTurn(BaseModel):
    """One message in the thread, after de-duplication and cleanup."""

    speaker: str = Field(
        ...,
        description="Sender's display name, normalized consistently across the thread "
        "(e.g. always 'Sarah Chen', never sometimes 'sarah.chen@acme.com').",
    )
    text: str = Field(
        ...,
        description="The message the sender newly wrote: quoted history, signatures, "
        "legal disclaimers and header lines removed.",
    )


class ParsedThread(BaseModel):
    """Return contract for the parsing call."""

    subject: Optional[str] = Field(None, description="Thread subject line, if present.")
    turns: List[ParsedTurn] = Field(
        ..., description="Messages in CHRONOLOGICAL order — oldest first."
    )


class ThreadParseError(RuntimeError):
    """Raised when a thread can't be turned into at least two usable turns."""


_SYSTEM = (
    "You convert raw email threads into a structured list of negotiation turns.\n\n"
    "Rules:\n"
    "1. Return turns in CHRONOLOGICAL order, oldest message first. Email clients "
    "usually stack the newest reply on top — reverse it if so.\n"
    "2. Include each message EXACTLY ONCE. Quoted history ('> ...', 'On Mon, "
    "... wrote:', 'From: ... Sent: ...') repeats messages you have already "
    "captured — never emit those as separate turns.\n"
    "3. For each turn, `text` is only what that sender newly wrote. Strip "
    "headers, quoted blocks, signature blocks, and legal disclaimers.\n"
    "4. Normalize each person to one consistent display name for the whole "
    "thread. If only an email address is available, use the local part in "
    "title case (j.smith@acme.com -> 'J Smith').\n"
    "5. Preserve the sender's wording verbatim. Do not summarize, soften, "
    "translate or rephrase — downstream stages score tone, and paraphrasing "
    "destroys the signal being measured.\n"
    "6. Skip automated messages (out-of-office, delivery failures, calendar "
    "notifications) — they are not negotiation turns."
)


def _looks_like_transcript_line(line: str) -> bool:
    """`Name: message` — the plain-transcript form people often paste instead."""
    return bool(re.match(r"^\s*[\w][\w .'-]{0,40}\s*:\s+\S", line))


def _parse_plain_transcript(text: str) -> List[ParsedTurn]:
    """Fast path for `Speaker: message` transcripts — no LLM call needed.

    Worth special-casing because it's the most common way people paste a
    conversation, it's unambiguous, and skipping the model makes it instant
    and free.
    """
    turns: List[ParsedTurn] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not _looks_like_transcript_line(line):
            # Continuation of the previous speaker's message.
            if turns:
                turns[-1].text = (turns[-1].text + " " + line).strip()
                continue
            return []  # leading prose — not this format
        speaker, _, body = line.partition(":")
        turns.append(ParsedTurn(speaker=speaker.strip(), text=body.strip()))
    return turns


def _to_transcript(turns: List[ParsedTurn], dialogue_id: str, subject: Optional[str]) -> Transcript:
    """Build a schema-valid Transcript, deriving parties from the turns themselves."""
    speakers: List[str] = []
    for t in turns:
        if t.speaker not in speakers:
            speakers.append(t.speaker)

    parties = [
        Party(party_id=s, priorities=None, outcome_points=None, metadata={})
        for s in speakers
    ]
    schema_turns = [
        Turn(index=i, speaker=t.speaker, text=t.text, strategies=[], action=None)
        for i, t in enumerate(turns)
    ]

    return Transcript(
        dialogue_id=dialogue_id,
        source=SOURCE,
        domain=DOMAIN,
        parties=parties,
        turns=schema_turns,
        # Placeholder: the outcome of a live thread is unknown — that's what the
        # pipeline is asked to reason about. Every analysis stage ignores this.
        outcome=Outcome(agreement_reached=False, final_deal=None, points={}),
        has_strategy_annotations=False,
        metadata={"split": "live", "subject": subject, "parsed_by": "llm"},
    )


def parse_thread(text: str, dialogue_id: str = "thread-1") -> Transcript:
    """Turn a pasted email thread (or plain transcript) into a `Transcript`.

    Tries the deterministic `Speaker: message` path first, then falls back to
    the LLM for real email threads. Raises `ThreadParseError` when the result
    isn't a usable two-party-or-more conversation.
    """
    text = (text or "").strip()
    if not text:
        raise ThreadParseError("The thread is empty — paste an email thread or conversation.")
    if len(text) > MAX_CHARS:
        raise ThreadParseError(
            f"Thread is {len(text):,} characters; the limit is {MAX_CHARS:,}. "
            "Paste the relevant portion of the exchange."
        )

    plain = _parse_plain_transcript(text)
    if len(plain) >= 2 and len({t.speaker for t in plain}) >= 2:
        logger.info("Parsed %d turns via the plain-transcript fast path", len(plain))
        return _to_transcript(plain, dialogue_id, subject=None)

    model = chat_model(temperature=0.0, max_tokens=2048).with_structured_output(ParsedThread)
    try:
        parsed: ParsedThread = model.invoke(
            [("system", _SYSTEM), ("user", "Parse this thread:\n\n" + text)],
            config={"callbacks": langfuse_callbacks()},
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("thread parsing failed")
        raise ThreadParseError(f"Could not parse the thread ({type(exc).__name__}).") from exc

    turns = [t for t in (parsed.turns or []) if t.text.strip()]
    if len(turns) < 2:
        raise ThreadParseError(
            "Found fewer than two messages. Make sure the thread contains a back-and-forth "
            "exchange between at least two people."
        )
    if len({t.speaker for t in turns}) < 2:
        raise ThreadParseError(
            "Every message resolved to the same sender. Check that the thread includes "
            "replies from more than one person."
        )

    logger.info("Parsed %d turns from %d participants via LLM",
                len(turns), len({t.speaker for t in turns}))
    return _to_transcript(turns, dialogue_id, parsed.subject)
