"""Ingest CraigslistBargain into Accord's normalized transcript schema.

CraigslistBargain (He et al., EMNLP 2018, *Decoupling Strategy and Generation in
Negotiation Dialogues*) is 6,682 real buyer/seller price-haggling dialogues
scraped from Craigslist across six categories (housing, furniture, electronics,
bike, car, phone). Hosted on CodaLab; the HF dataset card
(`stanfordnlp/craigslist_bargains`) points at these three JSON blobs.

Raw shape (per dialogue), verified against the live data (not the HF loader's
flattened summary, which paraphrases field names loosely)::

    {
      "uuid": "C_91d39147df0946bfa0278f0286421796",
      "scenario_uuid": "S_To118PXuNicOd8SO",
      "scenario": {
        "category": "electronics",
        "post_id": "...",
        "kbs": [
          {"personal": {"Role": "buyer", "Bottomline": null, "Target": 243},
           "item": {"Category": "electronics", "Title": "...", "Price": 265,
                     "Description": ["- line one", "- line two", ...], "Images": [...]}},
          {"personal": {"Role": "seller", ...}, "item": {...}}
        ]
      },
      "events": [
        {"agent": 0, "action": "message", "data": "hi there",
         "metadata": {"intent": "intro", "price": null}},
        {"agent": 1, "action": "offer", "data": {"price": 243.0, "sides": ""},
         "metadata": {"intent": "offer", "price": 243.0}},
        {"agent": 0, "action": "accept", "data": null, "metadata": {"intent": "accept"}}
      ],
      "outcome": {"reward": 1, "offer": {"price": 243.0, "sides": ""}},
      "agents": {"0": "human", "1": "human"}
    }

Key facts verified against the full validation split (597 dialogues) and
spot-checked against test (838 dialogues) before writing this parser:

- `outcome.reward` (0/1) is the **authoritative** agreement signal — it agreed
  with "does any event have action == accept" on all 597 validation dialogues,
  0 mismatches. `outcome.offer` is populated even on some reward==0 dialogues
  (the last, *rejected* offer) — so offer.price is only a real final deal when
  reward == 1, never trust it standalone.
- `Bottomline` (each party's private walk-away price) is **always null** in
  this public release — there is no per-party outcome score analogous to
  CaSiNo's `points_scored`. We deliberately do NOT invent one (e.g. from
  Target vs. final price) during ingestion; `Party.outcome_points` stays
  `None` and `Outcome.points` stays `{}`. Any such derived metric belongs in
  an explicit, documented analysis step later, not silently baked into the
  data spine.
- Every dialogue has exactly one `buyer` and one `seller` (verified across
  all 597) — `Role` is a stable, meaningful `party_id`, unlike CaSiNo's
  arbitrary `mturk_agent_N`.
- The test split's events carry `metadata: null` throughout (labels
  withheld — a shared-task holdout characteristic) — handled as `{}`, not a
  parse error.
- Actions map cleanly onto Accord's existing `Action` enum: `offer` ->
  SUBMIT_DEAL, `accept` -> ACCEPT_DEAL, `reject` -> REJECT_DEAL, `quit` ->
  WALK_AWAY. No schema change needed for actions (unlike Turn.metadata, added
  for the dialogue-act intents, which are a different taxonomy than CaSiNo's
  persuasion strategies).
- The final price is a single shared scalar (not a per-party resource split
  like CaSiNo's Firewood/Water/Food). We represent it in `Outcome.final_deal`
  as `{party_id: {"price_usd": <int>}}` for *both* parties symmetrically —
  reusing the existing per-party-dict shape to mean "the agreed transaction
  price," not an allocation split. Documented here so a future reader isn't
  confused by the reused shape meaning something different per source.

Usage::

    python -m data.ingest_craigslist --download                  # all 3 splits -> one JSONL
    python -m data.ingest_craigslist --download --split train    # one split only
    python -m data.ingest_craigslist --input data/raw/craigslist_train.json \
        --output data/processed/craigslist_bargain.jsonl --split train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlopen

from data.schema import Action, Outcome, Party, Transcript, Turn

# CodaLab bundle URLs (from stanfordnlp/craigslist_bargains' loading script).
# NOTE: the test URL has no filename suffix by design — the bundle serves a
# single file at that path with Content-Disposition: filename="test.json";
# appending "/parsed.json" resolves to a different (wrong) resource. Verified
# via `curl -I` against both variants before hardcoding this.
CODALAB_URLS: dict[str, str] = {
    "train": "https://worksheets.codalab.org/rest/bundles/0xd34bbbc5fb3b4fccbd19e10756ca8dd7/contents/blob/parsed.json",
    "validation": "https://worksheets.codalab.org/rest/bundles/0x15c4160b43d44ee3a8386cca98da138c/contents/blob/parsed.json",
    "test": "https://worksheets.codalab.org/rest/bundles/0x54d325bbcfb2463583995725ed8ca42b/contents/blob/",
}

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUTPUT = Path("data/processed/craigslist_bargain.jsonl")

SOURCE = "craigslist_bargain"
DOMAIN = "craigslist_price_negotiation"

_ACTION_BY_NAME = {
    "offer": Action.SUBMIT_DEAL,
    "accept": Action.ACCEPT_DEAL,
    "reject": Action.REJECT_DEAL,
    "quit": Action.WALK_AWAY,
}
_ACTION_TEXT = {"offer": "Offer", "accept": "Accept", "reject": "Reject", "quit": "Quit"}

# Dialogue-act intents observed in event metadata (reference only — passed
# through as-is, not filtered, since this is a real annotation layer from the
# original paper rather than a vocabulary we validate against).
KNOWN_DIALOGUE_ACTS = frozenset(
    {
        "intro",
        "unknown",
        "disagree",
        "agree",
        "inquiry",
        "inform",
        "vague-price",
        "init-price",
        "counter-price",
        "offer",
        "accept",
        "reject",
        "quit",
    }
)


def _item_metadata(kb: dict) -> dict:
    item = kb.get("item") or {}
    personal = kb.get("personal") or {}
    description = item.get("Description")
    return {
        "role": personal.get("Role"),
        "target": personal.get("Target"),
        "bottomline": personal.get("Bottomline"),  # always None in this release; kept for transparency
        "item_category": item.get("Category"),
        "item_title": item.get("Title"),
        "item_listed_price": item.get("Price"),
        "item_description": "\n".join(description) if isinstance(description, list) else description,
    }


def _parse_parties(scenario: dict) -> list[Party]:
    kbs = scenario.get("kbs") or []
    parties = []
    for kb in kbs:
        role = (kb.get("personal") or {}).get("Role")
        parties.append(
            Party(
                party_id=role,
                priorities=None,  # no discrete issue/priority ranking in this source
                outcome_points=None,  # no native per-party score (Bottomline always null)
                metadata=_item_metadata(kb),
            )
        )
    return parties


def _parse_turns(events: list, parties: list[Party]) -> list[Turn]:
    """Build turns with speakers resolved to party_id (role) up front.

    `event["agent"]` is a raw int index (0/1) into `scenario.kbs` — it must be
    mapped to the party's `Role` *before* constructing the Turn, since
    `Turn.speaker` is typed `str` and pydantic v2 does not coerce int -> str.
    """
    role_by_index = {i: p.party_id for i, p in enumerate(parties)}
    turns = []
    for i, event in enumerate(events):
        raw_action = event.get("action")
        action = _ACTION_BY_NAME.get(raw_action)
        if action is None:
            text = event.get("data") or ""
            action_data = None
        else:
            text = _ACTION_TEXT[raw_action]
            # Only "offer" carries informative data ({price, sides}); accept/
            # reject/quit all have data: null in the source.
            action_data = event.get("data") if action is Action.SUBMIT_DEAL else None
        turns.append(
            Turn(
                index=i,
                speaker=role_by_index[event.get("agent")],
                text=text,
                strategies=[],
                action=action,
                action_data=action_data,
                metadata=event.get("metadata") or {},  # None on the test split
            )
        )
    return turns


def _parse_outcome(raw_outcome: dict, parties: list[Party]) -> Outcome:
    reward = (raw_outcome or {}).get("reward")
    if reward != 1:
        return Outcome(agreement_reached=False, final_deal=None, points={})

    offer = (raw_outcome or {}).get("offer") or {}
    price = offer.get("price")
    if price is None:
        # reward==1 with no resolvable price is not expected (0/597 in the
        # verified sample) but degrade safely rather than raise.
        return Outcome(agreement_reached=False, final_deal=None, points={})

    price_int = round(price)
    final_deal = {p.party_id: {"price_usd": price_int} for p in parties}
    return Outcome(agreement_reached=True, final_deal=final_deal, points={})


def normalize_dialogue(raw: dict, split: str) -> Transcript:
    """Convert one raw CraigslistBargain dialogue into a validated `Transcript`.

    Pure function (no I/O) so it can be unit-tested against small fixtures.
    """
    scenario = raw.get("scenario") or {}
    parties = _parse_parties(scenario)
    turns = _parse_turns(raw.get("events", []), parties)
    outcome = _parse_outcome(raw.get("outcome") or {}, parties)

    return Transcript(
        dialogue_id=raw["uuid"],
        source=SOURCE,
        domain=DOMAIN,
        parties=parties,
        turns=turns,
        outcome=outcome,
        has_strategy_annotations=False,
        metadata={
            "split": split,
            "category": scenario.get("category"),
            "post_id": scenario.get("post_id"),
            "scenario_uuid": raw.get("scenario_uuid"),
            "agents": raw.get("agents"),
        },
    )


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} from {url} ...", file=sys.stderr)
    with urlopen(url) as resp:  # noqa: S310 — fixed, trusted CodaLab URL
        dest.write_bytes(resp.read())
    print(f"Saved to {dest}", file=sys.stderr)


def ingest(
    raw_dir: Path,
    output_path: Path,
    splits: list[str],
    download: bool = False,
) -> dict:
    """Run ingestion across the requested splits and write one combined JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    agreements = 0
    per_split: dict[str, int] = {}

    with output_path.open("w", encoding="utf-8") as out_fh:
        for split in splits:
            raw_path = raw_dir / f"craigslist_{split}.json"
            if download and not raw_path.exists():
                _download(CODALAB_URLS[split], raw_path)
            if not raw_path.exists():
                raise FileNotFoundError(
                    f"{raw_path} not found. Re-run with --download to fetch it from:\n"
                    f"  {CODALAB_URLS[split]}"
                )

            raw_dialogues = json.loads(raw_path.read_text(encoding="utf-8"))
            split_count = 0
            for raw in raw_dialogues:
                transcript = normalize_dialogue(raw, split)
                out_fh.write(transcript.model_dump_json())
                out_fh.write("\n")
                written += 1
                split_count += 1
                agreements += int(transcript.outcome.agreement_reached)
            per_split[split] = split_count

    return {
        "raw_dir": str(raw_dir),
        "output": str(output_path),
        "splits": per_split,
        "dialogues_written": written,
        "agreements": agreements,
        "agreement_rate": round(agreements / written, 4) if written else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest CraigslistBargain -> normalized transcripts (JSONL)."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Raw JSON directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test", "all"],
        default="all",
        help="Which split(s) to ingest (default: all three, combined into one file).",
    )
    parser.add_argument(
        "--download", action="store_true", help="Download raw split JSON(s) if missing."
    )
    args = parser.parse_args(argv)

    splits = list(CODALAB_URLS.keys()) if args.split == "all" else [args.split]
    summary = ingest(args.raw_dir, args.output, splits, args.download)
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
