"""Ingest the CaSiNo corpus into Accord's normalized transcript schema.

CaSiNo (Chawla et al., NAACL 2021, *CaSiNo: A Corpus of Campsite Negotiation
Dialogues for Automatic Negotiation Systems*) is 1,030 campsite
resource-negotiation dialogues between two MTurk workers bartering over
Food / Water / Firewood packages. 396 dialogues additionally carry
utterance-level persuasion-strategy annotations.

**Source (2026-07 pivot):** read directly from the HuggingFace parquet mirror
`kchawla123/casino` (auto-converted, no loading script, no CodaLab dependency).
This replaces the earlier CraigslistBargain ingestion after CodaLab — the only
host for CraigslistBargain's raw parsed.json — went permanently 500. CaSiNo is
also the dataset Accord's `schema.py` was originally designed for
(`Party.satisfaction`, `opponent_likeness`, `priorities`, `outcome_points`,
`Turn.strategies` are all CaSiNo-native), so this re-base fits the schema
better than CraigslistBargain ever did. See DESIGN.md §1.

Raw shape (per dialogue, as stored in the HF parquet)::

    {
      "chat_logs": [
        {"text": "Hello!...", "task_data": {"data": "", "issue2youget": {...},
                                             "issue2theyget": {...}}, "id": "mturk_agent_1"},
        ...
        {"text": "Submit-Deal",
         "task_data": {"data": "", "issue2youget": {"Firewood": "3", ...},
                       "issue2theyget": {...}}, "id": "mturk_agent_2"},
        {"text": "Accept-Deal", "task_data": {"data": "accept_deal", ...}, "id": "mturk_agent_1"}
      ],
      "participant_info": {"mturk_agent_1": {"value2issue": {"High": "Firewood", ...},
                                             "value2reason": {...},
                                             "outcomes": {"points_scored": 19,
                                                          "satisfaction": "...",
                                                          "opponent_likeness": "..."},
                                             "demographics": {...},
                                             "personality": {"svo": "...", "big-five": {...}}},
                           "mturk_agent_2": {...}},
      "annotations": [["<utterance text>", "strat-a,strat-b"], ...]   # numpy arrays; sparse
    }

Key quirks handled here:
- The HF parquet stores `annotations` as an array of 2-element numpy arrays,
  not Python lists — the annotation index must not gate on `isinstance(list)`
  or every strategy label is silently dropped.
- `annotations` is text-keyed, NOT index-aligned to chat_logs (protocol lines
  like Submit-Deal are never annotated), so we join on utterance text.
- Deal quantities in task_data are strings (or "" on non-deal turns); coerced
  to int, empties skipped.
- `issue2youget` is from the *submitter's* perspective ("you" = submitter).
- The HF mirror carries no `dialogue_id` and no train/val/test split; we
  synthesize a stable `casino-<i>` id and a deterministic hash-based 80/10/10
  split so the outcome model's splits are reproducible from this repo alone.

Usage::

    python -m data.ingest_casino --download                  # fetch HF parquet + convert
    python -m data.ingest_casino --input data/raw/casino.parquet \
        --output data/processed/casino.jsonl
    python -m data.ingest_casino --download --annotated-only # only the 396 labeled
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

from data.schema import Action, Outcome, Party, Transcript, Turn

# HuggingFace parquet mirror (auto-converted branch — script-free, reliable).
HF_REPO = "kchawla123/casino"
HF_PARQUET_PATH = "hf://datasets/kchawla123/casino@refs/convert/parquet/default/train/0000.parquet"

DEFAULT_INPUT = Path("data/raw/casino.parquet")
DEFAULT_OUTPUT = Path("data/processed/casino.jsonl")

SOURCE = "casino"
DOMAIN = "campsite_resources"

# Deterministic split proportions (train, validation, test). Must sum to 1.0.
_SPLIT_TRAIN = 0.8
_SPLIT_VAL = 0.1  # test = remainder

# Raw "text" values that denote a protocol action rather than an utterance.
_ACTION_BY_TEXT = {
    "Submit-Deal": Action.SUBMIT_DEAL,
    "Accept-Deal": Action.ACCEPT_DEAL,
    "Reject-Deal": Action.REJECT_DEAL,
    "Walk-Away": Action.WALK_AWAY,
}
# Fallback: some turns carry the action in task_data.data instead of text.
_ACTION_BY_DATA = {
    "accept_deal": Action.ACCEPT_DEAL,
    "reject_deal": Action.REJECT_DEAL,
}

# The 10-strategy CaSiNo annotation vocabulary (empty/unknown labels dropped).
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
    return str(raw_id).replace("mturk_", "")


def _split_for(dialogue_id: str) -> str:
    """Deterministic hash-based split so the same dialogue always lands the same way."""
    h = int(hashlib.md5(dialogue_id.encode("utf-8")).hexdigest(), 16)  # noqa: S324 — non-crypto bucketing
    frac = (h % 10_000) / 10_000
    if frac < _SPLIT_TRAIN:
        return "train"
    if frac < _SPLIT_TRAIN + _SPLIT_VAL:
        return "validation"
    return "test"


def _coerce_quantities(mapping: dict) -> dict:
    """{'Firewood': '3', 'Water': ''} -> {'Firewood': 3}, skipping non-numeric/empty."""
    out: dict = {}
    for issue, qty in (mapping or {}).items():
        try:
            out[issue] = int(qty)
        except (TypeError, ValueError):
            continue
    return out


def _as_pairs(annotations) -> list:
    """Normalize the annotations field into a list of (text, label_str) pairs.

    The HF parquet delivers each annotation as a 2-element numpy array, so we
    index positionally rather than assuming Python lists (the previous
    raw-JSON ingestion gated on `isinstance(list)`, which drops everything
    here)."""
    pairs = []
    for entry in annotations if annotations is not None else []:
        try:
            if len(entry) != 2:
                continue
        except TypeError:
            continue
        pairs.append((str(entry[0]), str(entry[1])))
    return pairs


def _build_annotation_index(annotations) -> dict:
    """Map utterance text -> list of valid strategy labels.

    Keyed on text, not position: some annotated dialogues have fewer
    annotation rows than utterances, which breaks positional alignment. Text
    keying is unambiguous because no annotated dialogue repeats an utterance
    verbatim (verified against the full corpus in the original ingestion)."""
    index: dict = {}
    for text, label_str in _as_pairs(annotations):
        labels = [lab.strip() for lab in label_str.split(",") if lab.strip()]
        labels = [lab for lab in labels if lab in STRATEGY_VOCAB]
        if labels:
            index[text] = labels
    return index


def _priorities_from_value2issue(value2issue: Optional[dict]) -> Optional[dict]:
    """CaSiNo stores priority->issue ('High': 'Firewood'); invert to issue->priority."""
    if not value2issue:
        return None
    return {issue: priority for priority, issue in value2issue.items()}


def _parse_parties(participant_info: dict) -> list:
    parties = []
    for raw_id, info in (participant_info or {}).items():
        info = info or {}
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


def _detect_action(entry: dict) -> Optional[Action]:
    text = entry.get("text", "")
    if text in _ACTION_BY_TEXT:
        return _ACTION_BY_TEXT[text]
    data = (entry.get("task_data") or {}).get("data")
    return _ACTION_BY_DATA.get(data)


def _parse_turns(chat_logs, annotation_index: dict) -> list:
    turns = []
    for i, raw_entry in enumerate(chat_logs if chat_logs is not None else []):
        entry = dict(raw_entry)  # numpy record / dict -> plain dict
        text = entry.get("text", "") or ""
        action = _detect_action(entry)
        task_data = entry.get("task_data") or None
        turns.append(
            Turn(
                index=i,
                speaker=_norm_party_id(entry.get("id", "")),
                text=text,
                strategies=annotation_index.get(text, []),
                action=action,
                action_data=dict(task_data) if (action and task_data) else None,
            )
        )
    return turns


def _parse_outcome(turns: list, parties: list) -> Outcome:
    """Resolve agreement + final deal from the action stream.

    Agreement iff an Accept-Deal is present; the accepted terms are the
    Submit-Deal immediately preceding that acceptance."""
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
        return Outcome(agreement_reached=False, final_deal=None, points=points)

    submitter = submit.speaker
    other = next((p.party_id for p in parties if p.party_id != submitter), None)
    final_deal = {
        submitter: _coerce_quantities(submit.action_data.get("issue2youget", {})),
    }
    if other is not None:
        final_deal[other] = _coerce_quantities(submit.action_data.get("issue2theyget", {}))
    return Outcome(agreement_reached=True, final_deal=final_deal, points=points)


def normalize_dialogue(raw: dict, dialogue_id: str, split: str) -> Transcript:
    """Convert one raw CaSiNo dialogue into a validated `Transcript`.

    Pure function (no I/O) so it can be unit-tested against small fixtures.
    """
    annotation_index = _build_annotation_index(raw.get("annotations"))
    parties = _parse_parties(raw.get("participant_info") or {})
    turns = _parse_turns(raw.get("chat_logs"), annotation_index)
    outcome = _parse_outcome(turns, parties)

    return Transcript(
        dialogue_id=dialogue_id,
        source=SOURCE,
        domain=DOMAIN,
        parties=parties,
        turns=turns,
        outcome=outcome,
        has_strategy_annotations=bool(annotation_index),
        metadata={"split": split},
    )


def _load_raw_rows(input_path: Path, download: bool) -> list:
    """Return a list of raw dialogue dicts (chat_logs/participant_info/annotations)."""
    import pandas as pd

    if download:
        print(f"Reading CaSiNo from HF parquet: {HF_PARQUET_PATH}", file=sys.stderr)
        df = pd.read_parquet(HF_PARQUET_PATH)
    else:
        if not input_path.exists():
            raise FileNotFoundError(
                f"{input_path} not found. Re-run with --download to fetch from HuggingFace:\n"
                f"  {HF_REPO}"
            )
        df = pd.read_parquet(input_path)
    return df.to_dict(orient="records")


def ingest(
    input_path: Path,
    output_path: Path,
    download: bool = False,
    annotated_only: bool = False,
) -> dict:
    """Convert CaSiNo into one combined JSONL with a deterministic split column."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_raw_rows(input_path, download)

    written = 0
    agreements = 0
    annotated = 0
    per_split: dict = {"train": 0, "validation": 0, "test": 0}

    with output_path.open("w", encoding="utf-8") as out_fh:
        for i, raw in enumerate(rows):
            dialogue_id = f"casino-{i}"
            split = _split_for(dialogue_id)
            transcript = normalize_dialogue(raw, dialogue_id, split)

            if annotated_only and not transcript.has_strategy_annotations:
                continue

            out_fh.write(transcript.model_dump_json())
            out_fh.write("\n")
            written += 1
            per_split[split] += 1
            agreements += int(transcript.outcome.agreement_reached)
            annotated += int(transcript.has_strategy_annotations)

    return {
        "input": HF_PARQUET_PATH if download else str(input_path),
        "output": str(output_path),
        "splits": per_split,
        "dialogues_written": written,
        "annotated": annotated,
        "agreements": agreements,
        "agreement_rate": round(agreements / written, 4) if written else 0.0,
    }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest CaSiNo -> normalized transcripts (JSONL)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Local parquet path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument(
        "--download", action="store_true", help="Read the CaSiNo parquet from HuggingFace."
    )
    parser.add_argument(
        "--annotated-only",
        action="store_true",
        help="Keep only the ~396 strategy-annotated dialogues.",
    )
    args = parser.parse_args(argv)

    summary = ingest(args.input, args.output, args.download, args.annotated_only)
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
