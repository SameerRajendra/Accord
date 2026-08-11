"""Build the RAG case corpus from normalized CaSiNo transcripts.

The corpus is what the agent retrieves precedent from (Phase 2 embeds it into
pgvector; Phase 3's recommendation node conditions on it). Two kinds of
retrievable [`CaseDocument`](data/schema.py):

- **kind="case"** — one past negotiation rendered as precedent: each party's
  priority ranking, the persuasion strategies observed, the outcome (agreed
  item split + points scored, or breakdown), and a distilled lesson.
- **kind="strategy"** — one entry from CaSiNo's 10-strategy persuasion
  playbook, so the agent can retrieve a tactic *definition* and not only
  similar past dialogues. CraigslistBargain carried no equivalent taxonomy,
  so this playbook returned when Accord re-based on CaSiNo (DESIGN.md §1).

This is a **pure, deterministic transform** — no LLM. The `text` field is a
templated natural-language rendering; retrieval quality is measured in
Phase 2, not asserted here.

Usage::

    # ingest first: python -m data.ingest_casino --download
    python -m data.build_case_corpus
    python -m data.build_case_corpus --input data/processed/casino.jsonl \
        --output data/processed/case_corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data.ingest_casino import STRATEGY_VOCAB
from data.schema import CaseDocument, Transcript

DEFAULT_INPUT = Path("data/processed/casino.jsonl")
DEFAULT_OUTPUT = Path("data/processed/case_corpus.jsonl")

DOMAIN_BLURB = "a campsite negotiation between two campers bartering over Food, Water, and Firewood packages"

# Human-readable definitions for CaSiNo's persuasion-strategy taxonomy. Kept in
# this module (not ingestion) because it's a RAG-corpus artifact, not a
# property of any single dialogue. Ordered high-level → filler.
STRATEGY_PLAYBOOK: dict = {
    "small-talk": "Build rapport with off-topic friendliness (asking about the trip, pets, "
    "the weather) to create a cooperative tone before bargaining.",
    "elicit-pref": "Ask about the partner's priorities and preferences to find trades that "
    "create joint value rather than assuming a zero-sum split.",
    "showing-empathy": "Acknowledge and validate the partner's situation or needs, lowering "
    "defensiveness and signaling good faith.",
    "promote-coordination": "Propose mutually beneficial trades and frame the deal as a shared "
    "problem to solve together — the integrative move.",
    "no-need": "Signal a low need for an item so the partner can take it, unlocking a trade on "
    "the items you actually want.",
    "self-need": "Justify a claim by pointing to your own requirements ('I need extra water, "
    "I'm dehydrated') — a distributive, position-based appeal.",
    "other-need": "Justify a claim by appealing to the needs of others you're responsible for "
    "(family, a pet, elderly group members).",
    "vouch-fair": "Invoke fairness or a fair-division principle ('let's split it evenly') to "
    "anchor the deal on a norm.",
    "uv-part": "Undervalue the partner — downplay or dismiss their stated needs to weaken their "
    "claim (an adversarial tactic).",
    "non-strategic": "Logistics and filler with no persuasive intent (confirming numbers, "
    "closing pleasantries).",
}


def _priority_line(transcript: Transcript) -> str:
    """agent_1 priorities: High=Firewood, Medium=Food, Low=Water. agent_2 ..."""
    parts = []
    for party in transcript.parties:
        prio = party.priorities or {}
        # priorities is issue->priority; render as priority=issue for readability.
        by_level = {level: issue for issue, level in prio.items()}
        rendered = ", ".join(
            f"{level}={by_level[level]}" for level in ("High", "Medium", "Low") if level in by_level
        )
        parts.append(f"{party.party_id} priorities: {rendered}" if rendered else f"{party.party_id} priorities: unknown")
    return " ".join(parts)


def _strategy_counts(transcript: Transcript) -> dict:
    """party_id -> {strategy -> count}, from utterance-level strategy annotations."""
    counts: dict = {p.party_id: {} for p in transcript.parties}
    for turn in transcript.turns:
        for strat in turn.strategies:
            if strat not in STRATEGY_VOCAB:
                continue
            party = counts.setdefault(turn.speaker, {})
            party[strat] = party.get(strat, 0) + 1
    return counts


def _strategy_lines(counts: dict) -> list:
    lines = []
    for party_id, strat_counts in counts.items():
        if not strat_counts:
            continue
        ordered = sorted(strat_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        rendered = ", ".join(f"{name} ({n})" for name, n in ordered)
        lines.append(f"- {party_id}: {rendered}")
    return lines


def _dominant_strategies(counts: dict, top: int = 3) -> list:
    totals: dict = {}
    for strat_counts in counts.values():
        for name, n in strat_counts.items():
            totals[name] = totals.get(name, 0) + n
    ordered = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    return [name for name, _ in ordered[:top]]


def _points(transcript: Transcript) -> dict:
    return dict(transcript.outcome.points or {})


def _outcome_line(transcript: Transcript) -> str:
    if not transcript.outcome.agreement_reached:
        last_action = transcript.turns[-1].action if transcript.turns else None
        how = f" (ended via {last_action.value})" if last_action else ""
        return f"Outcome: No agreement — the negotiation ended without a deal{how}."

    deal = transcript.outcome.final_deal or {}
    deal_parts = []
    for party_id, items in deal.items():
        items_str = ", ".join(f"{qty} {issue}" for issue, qty in items.items()) or "nothing"
        deal_parts.append(f"{party_id} gets {items_str}")
    points = _points(transcript)
    points_str = ""
    if points:
        points_str = " Points: " + ", ".join(f"{pid}={pts}" for pid, pts in points.items()) + "."
    return f"Outcome: Agreement reached — {'; '.join(deal_parts)}.{points_str}"


def _lesson(transcript: Transcript) -> str:
    if not transcript.outcome.agreement_reached:
        return (
            "Lesson: The negotiation broke down without agreement — a cautionary example of "
            "a value gap the parties couldn't bridge."
        )
    points = _points(transcript)
    if len(points) == 2:
        (a, pa), (b, pb) = list(points.items())
        if pa == pb:
            return f"Lesson: A balanced deal — both parties scored {pa} points."
        winner, wpts = (a, pa) if pa > pb else (b, pb)
        loser, lpts = (b, pb) if pa > pb else (a, pa)
        return (
            f"Lesson: The deal favored {winner} ({wpts} vs {lpts} points) — study which "
            f"priorities {winner} protected and which trades {loser} conceded."
        )
    return "Lesson: Reached an agreed division of the packages."


def build_case(transcript: Transcript) -> CaseDocument:
    """Render one normalized transcript into a retrievable precedent (pure)."""
    counts = _strategy_counts(transcript)

    sections = [
        f"Case {transcript.source}-{transcript.dialogue_id} — {DOMAIN_BLURB}.",
        _priority_line(transcript),
    ]
    strat_lines = _strategy_lines(counts)
    if strat_lines:
        sections.append("Persuasion strategies observed:\n" + "\n".join(strat_lines))
    sections.append(_outcome_line(transcript))
    sections.append(_lesson(transcript))
    text = "\n\n".join(sections)

    return CaseDocument(
        case_id=f"{transcript.source}-{transcript.dialogue_id}",
        source=transcript.source,
        kind="case",
        text=text,
        metadata={
            "dialogue_id": transcript.dialogue_id,
            "domain": transcript.domain,
            "split": transcript.metadata.get("split"),
            "agreement_reached": transcript.outcome.agreement_reached,
            "outcome_label": "agreement" if transcript.outcome.agreement_reached else "no_agreement",
            "points": _points(transcript),
            "has_strategy_annotations": transcript.has_strategy_annotations,
            "dominant_strategies": _dominant_strategies(counts),
        },
    )


def build_playbook() -> list:
    """One CaseDocument per CaSiNo persuasion strategy (kind='strategy')."""
    docs = []
    for name, definition in STRATEGY_PLAYBOOK.items():
        docs.append(
            CaseDocument(
                case_id=f"strategy-{name}",
                source="playbook",
                kind="strategy",
                text=f"Strategy: {name}. {definition}",
                metadata={"strategy": name},
            )
        )
    return docs


def build_cases(transcripts: list) -> list:
    return [build_case(t) for t in transcripts]


def _load_transcripts(path: Path) -> list:
    return [
        Transcript.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_corpus(input_path: Path, output_path: Path, include_playbook: bool = True) -> dict:
    """Build the case corpus (+ strategy playbook) and write JSONL. Returns a summary."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run the ingestion first:\n"
            f"  python -m data.ingest_casino --download"
        )

    transcripts = _load_transcripts(input_path)
    cases = build_cases(transcripts)
    playbook = build_playbook() if include_playbook else []
    all_docs = cases + playbook

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for doc in all_docs:
            fh.write(doc.model_dump_json())
            fh.write("\n")

    agreements = sum(1 for c in cases if c.metadata["agreement_reached"])
    return {
        "input": str(input_path),
        "output": str(output_path),
        "transcripts_read": len(transcripts),
        "cases": len(cases),
        "strategy_docs": len(playbook),
        "documents_total": len(all_docs),
        "agreements": agreements,
        "agreement_rate": round(agreements / len(cases), 4) if cases else 0.0,
    }


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(description="Build the RAG case corpus from transcripts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Normalized JSONL.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Corpus JSONL out.")
    parser.add_argument(
        "--no-playbook", action="store_true", help="Skip the strategy-definition documents."
    )
    args = parser.parse_args(argv)

    summary = build_corpus(args.input, args.output, include_playbook=not args.no_playbook)
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
