"""Seed a demo knowledge base with in-domain business-negotiation precedent.

Purpose: the public demo's default thread is a business contract renewal (an MSA
price-uplift + auto-renewal standoff). Retrieving that against the CaSiNo
research corpus returns campsite firewood bartering — honest (the UI warns), but
a weak first impression for a recruiter, who sees the product grounding a
contract dispute in precedent about tents.

This module ingests a small, **clearly synthetic** set of business-negotiation
precedents and playbook rules into a namespace (default ``demo``), chosen to
overlap the default thread's themes — uplift, auto-renewal, liability caps,
indemnity, ultimatums — so the demo shows relevant precedent *with provenance*.

Honesty guardrails:

- This is **demo data, not evaluation ground truth.** It lives in a live
  namespace (``accord_ns_demo``), never in the frozen benchmark corpus the evals
  score. No metric is computed against it; it only makes retrieval visible.
- The documents are generic and invented — no real company, person, or contract.
  They illustrate the mechanism; they are not a claim about real deals.

Run::

    python -m data.demo_seed                       # -> namespace "demo" (needs DATABASE_URL)
    python -m data.demo_seed --namespace acme-demo
    modal run infra/modal/app.py::seed_demo        # the cloud path
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

from rag.documents import SourceType

DEMO_NAMESPACE = "demo"

# (title, source_type, text). Kept short and vocabulary-rich so retrieval can
# actually discriminate between them rather than seeing one shared template —
# the exact failure the CaSiNo case corpus has.
_DOCS: List[Tuple[str, SourceType, str]] = [
    (
        "MSA renewal — 35% uplift resolved via multi-year term",
        SourceType.ANALYZED_THREAD,
        "A vendor opened a master services agreement renewal at a 35% annual uplift with a "
        "twelve-month auto-renewal. The customer refused the headline number but signalled "
        "flexibility on term length. Resolution: the uplift was brought to 18% in exchange for "
        "a three-year commitment and a capped 6% annual escalator thereafter. Lesson: when a "
        "price anchor is aggressive, trade on term length and escalator caps rather than "
        "fighting the headline percentage head-on.",
    ),
    (
        "Auto-renewal clause — standard fallback position",
        SourceType.PLAYBOOK,
        "Rule: never accept an auto-renewal longer than twelve months without a "
        "termination-for-convenience window of at least sixty days. Preferred fallback: convert "
        "an evergreen auto-renewal into a fixed term with an explicit renewal notice period. If "
        "the counterparty insists on auto-renewal, require a price-protection cap so the renewal "
        "cannot reprice above a stated percentage.",
    ),
    (
        "Limitation of liability — cap negotiated to 12 months' fees",
        SourceType.CONTRACT,
        "The counterparty sought unlimited liability. Position held: total aggregate liability "
        "capped at the fees paid in the preceding twelve months, with a carve-out only for "
        "breach of confidentiality and IP indemnity. Consequential and indirect losses were "
        "mutually excluded. Outcome: agreed. This is the standard defensible cap for a "
        "recurring-revenue contract.",
    ),
    (
        "Indemnity deadlock — unlimited demand led to no deal",
        SourceType.ANALYZED_THREAD,
        "The counterparty demanded uncapped indemnity for any and all third-party claims without "
        "limitation and regardless of fault, and refused to itemise the underlying risk basis. "
        "Repeated attempts to offer a tiered cap were rejected. The negotiation broke down and "
        "no agreement was reached. Lesson: an unlimited, open-ended indemnity demand paired with "
        "a refusal to explain the cost basis is a strong predictor of breakdown; escalate to a "
        "principal early rather than trading concessions into a vacuum.",
    ),
    (
        "Price uplift — anchoring and concession sequencing",
        SourceType.PLAYBOOK,
        "Rule: when facing an aggressive uplift, do not counter with a single number. Ask for the "
        "cost basis first (a calibrated question), then concede on non-price levers — term "
        "length, payment timing, volume commitment — before moving on headline price. Concede "
        "slowly and in decreasing increments so the pattern signals you are near your limit.",
    ),
    (
        "Payment terms — Net-30 conceded to Net-60 for volume",
        SourceType.CONTRACT,
        "The customer requested Net-60 payment terms against a standard Net-30. Rather than "
        "refuse, the vendor tied the extension to a committed annual volume and a small early-"
        "payment discount to preserve cash-flow optionality. Outcome: agreed at Net-45 with a "
        "1.5% early-pay discount. Lesson: payment timing is a cheap concession to trade for a "
        "commitment that de-risks revenue.",
    ),
    (
        "Ultimatum handling — de-escalation without capitulation",
        SourceType.PLAYBOOK,
        "Rule: when the counterparty issues a 'sign by Friday or we walk' ultimatum, do not "
        "counter-ultimatum and do not capitulate. Label the tactic ('it sounds like there's a "
        "hard deadline on your side') and ask a calibrated question about what is driving it. "
        "This converts a threat into information and buys time without conceding the point. "
        "Walk away only when the walk-away is genuinely better than the deal on the table.",
    ),
    (
        "Vendor lock-in — termination and data-portability rights",
        SourceType.CONTRACT,
        "A renewal was used to introduce stronger exit rights: a documented data-export format on "
        "termination, a ninety-day transition-assistance period at agreed rates, and removal of a "
        "clause that voided support during any dispute. Outcome: agreed. Lesson: renewals are the "
        "moment to fix lock-in terms, because leverage is highest before signature and lowest once "
        "the auto-renewal has lapsed.",
    ),
]


def seed(namespace: str = DEMO_NAMESPACE, database_url: str = "") -> dict:
    """Ingest the demo documents into `namespace`. Idempotent (content-addressed)."""
    from rag.documents import build_document
    from rag.embed import upsert_document

    results = []
    for title, source_type, text in _DOCS:
        doc = build_document(
            title=title,
            text=text,
            namespace=namespace,
            source_type=source_type,
            metadata={"demo": True, "synthetic": True},
        )
        results.append(upsert_document(doc, database_url=database_url))
    total_chunks = sum(int(r["chunks"]) for r in results)
    return {"namespace": namespace, "documents": len(results), "chunks": total_chunks}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Seed the demo knowledge base.")
    parser.add_argument("--namespace", default=DEMO_NAMESPACE)
    args = parser.parse_args(argv)
    summary = seed(args.namespace)
    print(f"Seeded {summary['documents']} documents ({summary['chunks']} chunks) "
          f"into namespace '{summary['namespace']}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
