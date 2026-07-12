"""Build the RAG case corpus from normalized CraigslistBargain transcripts.

The corpus is what the agent retrieves precedent from (Phase 2 embeds it into
pgvector; Phase 3's recommendation node conditions on it). Each retrievable
unit is a [`CaseDocument`](data/schema.py) of kind `"case"`: one past
negotiation rendered as precedent — setup (listed price, buyer/seller
targets), dialogue acts observed, outcome (agreed price or no deal), and a
short distilled lesson about who the deal favored.

This is a **pure, deterministic transform** — no LLM. The `text` field is a
templated natural-language rendering; retrieval quality is measured in
Phase 2, not asserted here. (LLM-generated hard cases are a separate concern
— `synthetic_gen.py`.)

Note on scope: CaSiNo had its own 10-strategy persuasion taxonomy, which the
first version of this module rendered as a "playbook" of retrievable
strategy definitions. CraigslistBargain carries no equivalent taxonomy — its
per-turn signal is coarser dialogue-act *intents* (`init-price`,
`counter-price`, `agree`, ...), a different kind of label living in
`Turn.metadata`, not `Turn.strategies`. There is no playbook to build here;
the planned 10-class Voss/Cranfield-thesis sentiment taxonomy is a Phase 1
analysis concern (fine-tuning a classifier), not a RAG corpus concern, so it
is intentionally out of scope for this module.

Usage::

    # ingest first: python -m data.ingest_craigslist --download
    python -m data.build_case_corpus
    python -m data.build_case_corpus --input data/processed/craigslist_bargain.jsonl \
        --output data/processed/case_corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data.schema import CaseDocument, Transcript

DEFAULT_INPUT = Path("data/processed/craigslist_bargain.jsonl")
DEFAULT_OUTPUT = Path("data/processed/case_corpus.jsonl")

DOMAIN_BLURB = "a real Craigslist price negotiation between a buyer and a seller"

# Dialogue-act intents that carry real signal — excludes "unknown"
# (uninformative) and the four action-echoing intents (offer/accept/reject/
# quit), which are protocol events, not tone. Shared with analysis/
# outcome_model.py for feature engineering, not just this module's case text.
MEANINGFUL_INTENTS = frozenset(
    {"intro", "disagree", "agree", "inquiry", "inform", "vague-price", "init-price", "counter-price"}
)


def _setup_line(transcript: Transcript) -> str:
    """Listed at $265 (electronics). buyer target: $243. seller target: $265."""
    by_role = {p.party_id: p for p in transcript.parties}
    listed = next((p.metadata.get("item_listed_price") for p in transcript.parties), None)
    category = next((p.metadata.get("item_category") for p in transcript.parties), None)
    title = next((p.metadata.get("item_title") for p in transcript.parties), None)

    header = f"Listed at ${listed}" if listed is not None else "Listing price unknown"
    if category:
        header += f" ({category})"
    if title:
        header += f': "{title}".'
    else:
        header += "."

    target_parts = []
    for role in ("buyer", "seller"):
        party = by_role.get(role)
        target = party.metadata.get("target") if party else None
        target_parts.append(f"{role} target: ${target}" if target is not None else f"{role} target: unknown")
    return header + " " + " ".join(target_parts)


def _dialogue_act_counts(transcript: Transcript) -> dict[str, dict[str, int]]:
    """party_id -> {intent -> count}, restricted to meaningful (non-protocol) intents."""
    counts: dict[str, dict[str, int]] = {p.party_id: {} for p in transcript.parties}
    for turn in transcript.turns:
        intent = turn.metadata.get("intent")
        if intent not in MEANINGFUL_INTENTS:
            continue
        party = counts.setdefault(turn.speaker, {})
        party[intent] = party.get(intent, 0) + 1
    return counts


def _dialogue_act_lines(counts: dict[str, dict[str, int]]) -> list[str]:
    lines = []
    for party_id, intent_counts in counts.items():
        if not intent_counts:
            continue
        ordered = sorted(intent_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        rendered = ", ".join(f"{name} ({n})" for name, n in ordered)
        lines.append(f"- {party_id}: {rendered}")
    return lines


def _dominant_dialogue_acts(counts: dict[str, dict[str, int]], top: int = 2) -> list[str]:
    totals: dict[str, int] = {}
    for intent_counts in counts.values():
        for name, n in intent_counts.items():
            totals[name] = totals.get(name, 0) + n
    ordered = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    return [name for name, _ in ordered[:top]]


def _final_price(transcript: Transcript) -> float | None:
    """The agreed price, read off Outcome.final_deal (same value for both parties)."""
    final_deal = transcript.outcome.final_deal
    if not final_deal:
        return None
    return next(iter(final_deal.values()))["price_usd"]


def _target(transcript: Transcript, role: str) -> float | None:
    party = next((p for p in transcript.parties if p.party_id == role), None)
    return party.metadata.get("target") if party else None


def _outcome_line(transcript: Transcript) -> str:
    if not transcript.outcome.agreement_reached:
        last_action = transcript.turns[-1].action if transcript.turns else None
        how = f" (ended via {last_action.value})" if last_action else ""
        return f"Outcome: No agreement — the negotiation ended without a deal{how}."
    price = _final_price(transcript)
    return f"Outcome: Agreement reached at ${price}."


def _lesson(transcript: Transcript) -> str:
    if not transcript.outcome.agreement_reached:
        return (
            "Lesson: The negotiation broke down without agreement — a cautionary "
            "example of a bargaining gap that couldn't be closed."
        )

    price = _final_price(transcript)
    buyer_target = _target(transcript, "buyer")
    seller_target = _target(transcript, "seller")

    if price is None or buyer_target is None or seller_target is None:
        return "Lesson: Reached an agreed deal."

    dist_to_buyer = abs(price - buyer_target)
    dist_to_seller = abs(price - seller_target)
    total_gap = abs(seller_target - buyer_target)
    if total_gap == 0:
        balance = "landed exactly at both parties' target"
    elif dist_to_buyer < dist_to_seller * 0.6:
        balance = "closed much nearer the buyer's target — favored the buyer"
    elif dist_to_seller < dist_to_buyer * 0.6:
        balance = "closed much nearer the seller's target — favored the seller"
    else:
        balance = "split roughly evenly between both targets"

    return f"Lesson: The final price {balance}."


def build_case(transcript: Transcript) -> CaseDocument:
    """Render one normalized transcript into a retrievable precedent (pure)."""
    counts = _dialogue_act_counts(transcript)

    sections = [
        f"Case {transcript.source}-{transcript.dialogue_id} — {DOMAIN_BLURB}.",
        _setup_line(transcript),
    ]
    act_lines = _dialogue_act_lines(counts)
    if act_lines:
        sections.append("Dialogue acts observed:\n" + "\n".join(act_lines))
    sections.append(_outcome_line(transcript))
    sections.append(_lesson(transcript))
    text = "\n\n".join(sections)

    buyer_target = _target(transcript, "buyer")
    seller_target = _target(transcript, "seller")

    return CaseDocument(
        case_id=f"{transcript.source}-{transcript.dialogue_id}",
        source=transcript.source,
        kind="case",
        text=text,
        metadata={
            "dialogue_id": transcript.dialogue_id,
            "domain": transcript.domain,
            "category": transcript.metadata.get("category"),
            "split": transcript.metadata.get("split"),
            "agreement_reached": transcript.outcome.agreement_reached,
            "outcome_label": "agreement" if transcript.outcome.agreement_reached else "no_agreement",
            "final_price_usd": _final_price(transcript),
            "buyer_target": buyer_target,
            "seller_target": seller_target,
            "dominant_dialogue_acts": _dominant_dialogue_acts(counts),
        },
    )


def build_cases(transcripts: list[Transcript]) -> list[CaseDocument]:
    return [build_case(t) for t in transcripts]


def _load_transcripts(path: Path) -> list[Transcript]:
    return [
        Transcript.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_corpus(input_path: Path, output_path: Path) -> dict:
    """Build the case corpus and write JSONL. Returns a summary."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run the ingestion first:\n"
            f"  python -m data.ingest_craigslist --download"
        )

    transcripts = _load_transcripts(input_path)
    cases = build_cases(transcripts)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for doc in cases:
            fh.write(doc.model_dump_json())
            fh.write("\n")

    agreements = sum(1 for c in cases if c.metadata["agreement_reached"])
    return {
        "input": str(input_path),
        "output": str(output_path),
        "transcripts_read": len(transcripts),
        "cases": len(cases),
        "agreements": agreements,
        "agreement_rate": round(agreements / len(cases), 4) if cases else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the RAG case corpus from transcripts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Normalized JSONL.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Corpus JSONL out.")
    args = parser.parse_args(argv)

    summary = build_corpus(args.input, args.output)
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
