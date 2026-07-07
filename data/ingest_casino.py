"""Ingest the CaSiNo corpus into Accord's normalized transcript schema.

CaSiNo (Chawla et al., NAACL 2021) is 1030 campsite resource-negotiation
dialogues between two MTurk workers bartering over Food / Water / Firewood.
396 dialogues additionally carry utterance-level strategy annotations.

Raw shape (per dialogue)::

    {
      "dialogue_id": 0,
      "chat_logs": [
        {"text": "Hello!...", "task_data": {}, "id": "mturk_agent_1"},
        ...
        {"text": "Submit-Deal",
         "task_data": {"issue2youget": {...}, "issue2theyget": {...}},
         "id": "mturk_agent_2"},
        {"text": "Accept-Deal", "task_data": {"data": "accept_deal"}, "id": "mturk_agent_1"}
      ],
      "participant_info": {"mturk_agent_1": {...}, "mturk_agent_2": {...}},
      "annotations": [["<utterance text>", "strat-a,strat-b"], ...]   # sparse
    }

Key quirks handled here:
- `annotations` is text-keyed, NOT index-aligned to chat_logs (protocol lines
  like Submit-Deal are never annotated), so we join on utterance text.
- Deal quantities in task_data are strings; we coerce to int.
- `issue2youget` is from the *submitter's* perspective ("you" = submitter).

Usage::

    python -m data.ingest_casino --download                 # fetch raw + convert
    python -m data.ingest_casino --input data/raw/casino.json \
        --output data/processed/casino.jsonl
    python -m data.ingest_casino --download --annotated-only # only the 396 labeled
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlopen

from data.schema import Action, Outcome, Party, Transcript, Turn

CASINO_URL = "https://raw.githubusercontent.com/kushalchawla/CaSiNo/master/data/casino.json"

DEFAULT_INPUT = Path("data/raw/casino.json")
DEFAULT_OUTPUT = Path("data/processed/casino.jsonl")

SOURCE = "casino"
DOMAIN = "campsite_resources"

# Raw "text" values that denote a protocol action rather than an utterance.
_ACTION_BY_TEXT = {
    "Submit-Deal": Action.SUBMIT_DEAL,
    "Accept-Deal": Action.ACCEPT_DEAL,
    "Reject-Deal": Action.REJECT_DEAL,
    "Walk-Away": Action.WALK_AWAY,
}

# The 10-strategy CaSiNo annotation vocabulary (empty labels are dropped).
STRATEGY_VOCAB = frozenset(
    {
        "small-talk",
        "self-need",
        "other-need",
        "no-need",
        "elicit-pref",
        "uv-part",
        "vouch-fair",
        "showing-empathy",
        "promote-coordination",
        "non-strategic",
    }
)


def _norm_party_id(raw_id: str) -> str:
    """'mturk_agent_1' -> 'agent_1'."""
    return raw_id.replace("mturk_", "")


def _coerce_quantities(mapping: dict) -> dict[str, int]:
    """{'Firewood': '3', ...} -> {'Firewood': 3, ...}, skipping non-numeric."""
    out: dict[str, int] = {}
    for issue, qty in (mapping or {}).items():
        try:
            out[issue] = int(qty)
        except (TypeError, ValueError):
            continue
    return out


def _build_annotation_index(annotations: list) -> dict[str, list[str]]:
    """Map utterance text -> list of valid strategy labels.

    CaSiNo annotations are ``[[text, "a,b"], ...]``. We key on text rather than
    position: 3/396 annotated dialogues have fewer annotations than utterances
    (some turns are simply unlabeled), which breaks positional alignment. Text
    keying is unambiguous here because no annotated dialogue contains two
    utterances with identical text (verified against the full corpus).
    """
    index: dict[str, list[str]] = {}
    for entry in annotations or []:
        if not (isinstance(entry, list) and len(entry) == 2):
            continue
        text, label_str = entry
        labels = [lab.strip() for lab in str(label_str).split(",") if lab.strip()]
        labels = [lab for lab in labels if lab in STRATEGY_VOCAB]
        if labels:
            index[text] = labels
    return index


def _parse_parties(participant_info: dict) -> list[Party]:
    parties: list[Party] = []
    for raw_id, info in participant_info.items():
        outcomes = info.get("outcomes") or {}
        parties.append(
            Party(
                party_id=_norm_party_id(raw_id),
                priorities=_priorities_from_value2issue(info.get("value2issue")),
                outcome_points=outcomes.get("points_scored"),
                satisfaction=outcomes.get("satisfaction"),
                opponent_likeness=outcomes.get("opponent_likeness"),
                metadata={
                    "demographics": info.get("demographics"),
                    "personality": info.get("personality"),
                    "value2reason": info.get("value2reason"),
                },
            )
        )
    return parties


def _priorities_from_value2issue(value2issue: dict | None) -> dict[str, str] | None:
    """CaSiNo stores priority->issue ('High': 'Firewood'); invert to issue->priority."""
    if not value2issue:
        return None
    return {issue: priority for priority, issue in value2issue.items()}


def _parse_turns(
    chat_logs: list, annotation_index: dict[str, list[str]]
) -> list[Turn]:
    turns: list[Turn] = []
    for i, entry in enumerate(chat_logs):
        text = entry.get("text", "")
        action = _ACTION_BY_TEXT.get(text)
        action_data = entry.get("task_data") or None
        turns.append(
            Turn(
                index=i,
                speaker=_norm_party_id(entry.get("id", "")),
                text=text,
                strategies=annotation_index.get(text, []),
                action=action,
                action_data=action_data if action else None,
            )
        )
    return turns


def _parse_outcome(turns: list[Turn], parties: list[Party]) -> Outcome:
    """Resolve agreement + final deal from the action stream.

    A dialogue ends in agreement iff an Accept-Deal is present. The accepted
    terms are the Submit-Deal immediately preceding that acceptance.
    """
    points = {p.party_id: p.outcome_points for p in parties if p.outcome_points is not None}

    accept_idx = next(
        (i for i in range(len(turns) - 1, -1, -1) if turns[i].action is Action.ACCEPT_DEAL),
        None,
    )
    if accept_idx is None:
        return Outcome(agreement_reached=False, final_deal=None, points=points)

    submit = next(
        (turns[i] for i in range(accept_idx - 1, -1, -1) if turns[i].action is Action.SUBMIT_DEAL),
        None,
    )
    if submit is None or not submit.action_data:
        # Accept without a resolvable Submit — treat as no usable deal.
        return Outcome(agreement_reached=False, final_deal=None, points=points)

    submitter = submit.speaker
    other = next(p.party_id for p in parties if p.party_id != submitter)
    final_deal = {
        submitter: _coerce_quantities(submit.action_data.get("issue2youget", {})),
        other: _coerce_quantities(submit.action_data.get("issue2theyget", {})),
    }
    return Outcome(agreement_reached=True, final_deal=final_deal, points=points)


def normalize_dialogue(raw: dict) -> Transcript:
    """Convert one raw CaSiNo dialogue into a validated `Transcript`.

    Pure function (no I/O) so it can be unit-tested against small fixtures.
    """
    annotation_index = _build_annotation_index(raw.get("annotations", []))
    parties = _parse_parties(raw.get("participant_info", {}))
    turns = _parse_turns(raw.get("chat_logs", []), annotation_index)
    outcome = _parse_outcome(turns, parties)

    return Transcript(
        dialogue_id=str(raw.get("dialogue_id")),
        source=SOURCE,
        domain=DOMAIN,
        parties=parties,
        turns=turns,
        outcome=outcome,
        has_strategy_annotations=bool(annotation_index),
        metadata={"raw_annotation_count": len(annotation_index)},
    )


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CaSiNo from {url} ...", file=sys.stderr)
    with urlopen(url) as resp:  # noqa: S310 — fixed, trusted GitHub raw URL
        dest.write_bytes(resp.read())
    print(f"Saved raw corpus to {dest}", file=sys.stderr)


def ingest(
    input_path: Path,
    output_path: Path,
    annotated_only: bool = False,
    download: bool = False,
) -> dict:
    """Run the full ingestion and write JSONL. Returns a summary dict."""
    if download and not input_path.exists():
        _download(CASINO_URL, input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Re-run with --download to fetch it from:\n  {CASINO_URL}"
        )

    raw_dialogues = json.loads(input_path.read_text(encoding="utf-8"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    annotated = 0
    agreements = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for raw in raw_dialogues:
            transcript = normalize_dialogue(raw)
            if annotated_only and not transcript.has_strategy_annotations:
                continue
            fh.write(transcript.model_dump_json())
            fh.write("\n")
            written += 1
            annotated += int(transcript.has_strategy_annotations)
            agreements += int(transcript.outcome.agreement_reached)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "dialogues_written": written,
        "with_annotations": annotated,
        "agreements": agreements,
        "agreement_rate": round(agreements / written, 4) if written else 0.0,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest CaSiNo -> normalized transcripts (JSONL).")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Raw casino.json path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument(
        "--annotated-only",
        action="store_true",
        help="Keep only the 396 strategy-annotated dialogues.",
    )
    parser.add_argument(
        "--download", action="store_true", help="Download raw casino.json if missing."
    )
    args = parser.parse_args(argv)

    summary = ingest(args.input, args.output, args.annotated_only, args.download)
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
