"""Build the Accord knowledge graph from normalized CaSiNo transcripts.

Same discipline as [`data/build_case_corpus.py`](../data/build_case_corpus.py):
a **pure, deterministic transform** — no LLM, no heuristics that depend on
model output, no randomness. `build_graph()` takes parsed
[`Transcript`](../data/schema.py) / [`CaseDocument`](../data/schema.py) objects
and returns nodes and edges as plain Pydantic models. `load_graph()` is the
only function that touches Postgres, and it is a separate step so the whole
build is unit-testable with no database.

What gets built
---------------
Per negotiation (facts read directly off one dialogue):

    (negotiation)-[:HAS_PARTY]->(party)
    (party)-[:PRIORITIZES {level}]->(issue)
    (party)-[:ALLOCATED {quantity, own_priority}]->(issue)      # from the final deal
    (party)-[:USED_STRATEGY {count, first_turn, last_turn}]->(strategy)
    (party)-[:NEGOTIATED_WITH]->(party)
    (negotiation)-[:CONTESTED]->(issue)        # both parties ranked it High
    (negotiation)-[:TRADED]->(issue)           # one party's High is the other's Low
    (negotiation)-[:HAS_STRUCTURE]->(conflict)
    (negotiation)-[:RESULTED_IN]->(outcome)

Corpus-level aggregates (statistics *across* dialogues, not facts about any
one of them — the relations flat document retrieval structurally cannot hold):

    (strategy)-[:CO_OCCURS_WITH {count, pmi, p_given_src}]->(strategy)
    (strategy)-[:PRECEDED {support, confidence, base_rate, lift}]->(outcome)

`PRECEDED` is correlational and is **deliberately excluded from default
retrieval scoring** (see `graph_retriever.plan_query(expand_by_lift=...)`):
CaSiNo reaches agreement in ~97.6% of dialogues and only 396 carry strategy
annotations, so the `no_agreement` denominator is a couple of dozen dialogues.
Ranking precedent by an association estimated from that would be exactly the
kind of unearned claim this repo exists to avoid. The edges are still stored,
with their raw support counts, because *inspecting* them is honest analysis —
using them to rank without a benchmark is not.

Usage::

    # prerequisites: python -m data.ingest_casino --download
    #                python -m data.build_case_corpus
    python -m rag.graph_ingest --dry-run       # build + print stats, no database
    python -m rag.graph_ingest                 # create tables, wipe, load
    python -m rag.graph_ingest --dump /tmp/graph.jsonl --dry-run

Requires env `DATABASE_URL` (or `ACCORD_GRAPH_DATABASE_URL`) for anything
other than `--dry-run` — same Neon string the vector path uses.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field

from data.build_case_corpus import STRATEGY_PLAYBOOK
from data.ingest_casino import STRATEGY_VOCAB
from data.schema import CaseDocument, Transcript
from rag.graph_schema import (
    CONFLICT_DESCRIPTIONS,
    CONFLICT_STRUCTURES,
    EDGE_TABLE,
    ISSUES,
    NODE_CONFLICT,
    NODE_ISSUE,
    NODE_NEGOTIATION,
    NODE_OUTCOME,
    NODE_PARTY,
    NODE_STRATEGY,
    NODE_TABLE,
    OUTCOME_CLASSES,
    OUTCOME_DESCRIPTIONS,
    REL_ALLOCATED,
    REL_CO_OCCURS_WITH,
    REL_CONTESTED,
    REL_HAS_PARTY,
    REL_HAS_STRUCTURE,
    REL_NEGOTIATED_WITH,
    REL_PRECEDED,
    REL_PRIORITIZES,
    REL_RESULTED_IN,
    REL_TRADED,
    REL_USED_STRATEGY,
    STRATEGY_POLARITY,
    STRUCT_UNKNOWN,
    GraphEdge,
    GraphNode,
    canonical_issue,
    conflict_node_id,
    conflict_structure,
    contested_issues,
    issue_node_id,
    joint_points,
    negotiation_node_id,
    outcome_class,
    outcome_node_id,
    party_node_id,
    point_gap,
    strategy_node_id,
    traded_issues,
)

logger = logging.getLogger(__name__)

DEFAULT_TRANSCRIPTS = Path("data/processed/casino.jsonl")
DEFAULT_CORPUS = Path("data/processed/case_corpus.jsonl")

#: A strategy pair needs this many co-occurring dialogues before it earns a
#: `CO_OCCURS_WITH` edge. Below it, the association is noise from a 396-dialogue
#: annotated subset. Threshold chosen, not fitted.
MIN_COOCCURRENCE = 5

#: ...and this much positive pointwise mutual information. `> 0` alone would
#: keep pairs that merely reflect both strategies being common; a small floor
#: keeps only genuinely associated tactics.
MIN_COOCCURRENCE_PMI = 0.10


# --------------------------------------------------------------------------
# Build result
# --------------------------------------------------------------------------


class GraphBuild(BaseModel):
    """Everything `build_graph` produced — inspectable before it touches a DB."""

    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)

    def counts_by_node_type(self) -> Dict[str, int]:
        out: Dict[str, int] = defaultdict(int)
        for node in self.nodes:
            out[node.node_type] += 1
        return dict(out)

    def counts_by_rel(self) -> Dict[str, int]:
        out: Dict[str, int] = defaultdict(int)
        for edge in self.edges:
            out[edge.rel] += 1
        return dict(out)


# --------------------------------------------------------------------------
# Small pure helpers
# --------------------------------------------------------------------------


def _strategy_counts(transcript: Transcript) -> Dict[str, Dict[str, Dict[str, int]]]:
    """party_id -> strategy -> {count, first_turn, last_turn}.

    Only labels inside `STRATEGY_VOCAB` count; protocol turns carry none.
    """
    out: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(dict)
    for turn in transcript.turns:
        for strat in turn.strategies:
            if strat not in STRATEGY_VOCAB:
                continue
            bucket = out[turn.speaker].get(strat)
            if bucket is None:
                out[turn.speaker][strat] = {
                    "count": 1,
                    "first_turn": turn.index,
                    "last_turn": turn.index,
                }
            else:
                bucket["count"] += 1
                bucket["last_turn"] = turn.index
    return dict(out)


def _dominant_strategies(counts: Dict[str, Dict[str, Dict[str, int]]], top: int = 3) -> List[str]:
    totals: Dict[str, int] = defaultdict(int)
    for per_party in counts.values():
        for name, meta in per_party.items():
            totals[name] += int(meta["count"])
    ordered = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    return [name for name, _ in ordered[:top]]


def _party_role(points: Dict[str, int], party_id: str) -> str:
    """'winner' / 'loser' / 'tie' / 'unscored' for a two-party scored deal."""
    values = [v for v in points.values() if v is not None]
    if len(values) != 2 or party_id not in points or points[party_id] is None:
        return "unscored"
    mine = int(points[party_id])
    theirs = int([v for pid, v in points.items() if pid != party_id][0])
    if mine == theirs:
        return "tie"
    return "winner" if mine > theirs else "loser"


def _case_index(cases: Optional[Iterable[CaseDocument]]) -> Tuple[Dict[str, CaseDocument], Dict[str, CaseDocument]]:
    """Split the corpus into (dialogue_id -> case doc, strategy name -> playbook doc).

    Keyed on `metadata['dialogue_id']`, **not** `case_id`: the corpus builder
    composes `case_id` as `f"{source}-{dialogue_id}"` while `dialogue_id`
    already carries the source, so real ids look like `casino-casino-0`.
    Matching on the metadata field sidesteps that quirk instead of unpicking
    it (unpicking it would mean editing `data/`, which this layer does not own).
    """
    by_dialogue: Dict[str, CaseDocument] = {}
    by_strategy: Dict[str, CaseDocument] = {}
    for case in cases or []:
        if case.kind == "case":
            dialogue_id = case.metadata.get("dialogue_id")
            if dialogue_id:
                by_dialogue[str(dialogue_id)] = case
        elif case.kind == "strategy":
            name = case.metadata.get("strategy")
            if name:
                by_strategy[str(name)] = case
    return by_dialogue, by_strategy


# --------------------------------------------------------------------------
# Global (corpus-wide) nodes
# --------------------------------------------------------------------------


def build_global_nodes(strategy_docs: Optional[Dict[str, CaseDocument]] = None) -> List[GraphNode]:
    """The ~22 nodes shared by every negotiation: issues, strategies, outcomes, structures.

    Strategy nodes carry the playbook document text **verbatim from
    `case_corpus.jsonl`** when it is available, so a strategy hit returned by
    graph retrieval is byte-identical to the same document returned by vector
    retrieval. That is a benchmarking requirement, not a nicety: a head-to-head
    comparison where the two retrievers return differently-worded text for the
    same document measures the rendering, not the retrieval.
    """
    strategy_docs = strategy_docs or {}
    nodes: List[GraphNode] = []

    for issue in ISSUES:
        nodes.append(
            GraphNode(
                node_id=issue_node_id(issue),
                node_type=NODE_ISSUE,
                label=issue,
                props={"issue": issue},
            )
        )

    for name in sorted(STRATEGY_VOCAB):
        doc = strategy_docs.get(name)
        definition = STRATEGY_PLAYBOOK.get(name, "")
        nodes.append(
            GraphNode(
                node_id=strategy_node_id(name),
                node_type=NODE_STRATEGY,
                label=name,
                props={
                    "strategy": name,
                    "polarity": STRATEGY_POLARITY.get(name, "neutral"),
                    "definition": definition,
                    "case_id": doc.case_id if doc else "strategy-{}".format(name),
                    "source": doc.source if doc else "playbook",
                    "text": doc.text if doc else "Strategy: {}. {}".format(name, definition),
                },
            )
        )

    for cls in OUTCOME_CLASSES:
        nodes.append(
            GraphNode(
                node_id=outcome_node_id(cls),
                node_type=NODE_OUTCOME,
                label=cls,
                props={"outcome_class": cls, "description": OUTCOME_DESCRIPTIONS[cls]},
            )
        )

    for structure in CONFLICT_STRUCTURES:
        nodes.append(
            GraphNode(
                node_id=conflict_node_id(structure),
                node_type=NODE_CONFLICT,
                label=structure,
                props={"structure": structure, "description": CONFLICT_DESCRIPTIONS[structure]},
            )
        )

    return nodes


# --------------------------------------------------------------------------
# Per-negotiation nodes + edges
# --------------------------------------------------------------------------


def build_negotiation(
    transcript: Transcript,
    case: Optional[CaseDocument] = None,
) -> Tuple[List[GraphNode], List[GraphEdge]]:
    """Nodes + edges for one dialogue. Pure; references global nodes by id."""
    dialogue_id = transcript.dialogue_id
    neg_id = negotiation_node_id(dialogue_id)
    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []

    parties = list(transcript.parties)
    priorities = [p.priorities for p in parties]
    prio_a = priorities[0] if len(priorities) > 0 else None
    prio_b = priorities[1] if len(priorities) > 1 else None

    structure = conflict_structure(prio_a, prio_b) if len(parties) == 2 else STRUCT_UNKNOWN
    contested = contested_issues(prio_a, prio_b) if len(parties) == 2 else []
    traded = traded_issues(prio_a, prio_b) if len(parties) == 2 else []

    points = {pid: pts for pid, pts in (transcript.outcome.points or {}).items() if pts is not None}
    out_class = outcome_class(transcript.outcome.agreement_reached, points)
    counts = _strategy_counts(transcript)
    text_turns = [t for t in transcript.turns if t.action is None]

    nodes.append(
        GraphNode(
            node_id=neg_id,
            node_type=NODE_NEGOTIATION,
            label=dialogue_id,
            props={
                "dialogue_id": dialogue_id,
                "source": transcript.source,
                "domain": transcript.domain,
                "split": transcript.metadata.get("split"),
                "case_id": case.case_id if case else None,
                "kind": "case",
                "agreement_reached": transcript.outcome.agreement_reached,
                "outcome_class": out_class,
                "point_gap": point_gap(points),
                "joint_points": joint_points(points),
                "points": points,
                "conflict_structure": structure,
                "contested_issues": contested,
                "traded_issues": traded,
                "dominant_strategies": _dominant_strategies(counts),
                "has_strategy_annotations": transcript.has_strategy_annotations,
                "n_turns": len(transcript.turns),
                "n_text_turns": len(text_turns),
                # Verbatim from case_corpus.jsonl so graph and vector retrieval
                # return the same document text for the same case (see
                # build_global_nodes for why that matters).
                "text": case.text if case else None,
            },
        )
    )

    edges.append(GraphEdge(src_id=neg_id, dst_id=conflict_node_id(structure), rel=REL_HAS_STRUCTURE))
    edges.append(GraphEdge(src_id=neg_id, dst_id=outcome_node_id(out_class), rel=REL_RESULTED_IN))
    for issue in contested:
        edges.append(GraphEdge(src_id=neg_id, dst_id=issue_node_id(issue), rel=REL_CONTESTED))
    for issue in traded:
        edges.append(GraphEdge(src_id=neg_id, dst_id=issue_node_id(issue), rel=REL_TRADED))

    final_deal = transcript.outcome.final_deal or {}

    for party in parties:
        pid = party_node_id(dialogue_id, party.party_id)
        nodes.append(
            GraphNode(
                node_id=pid,
                node_type=NODE_PARTY,
                label="{} ({})".format(party.party_id, dialogue_id),
                props={
                    "party_id": party.party_id,
                    "dialogue_id": dialogue_id,
                    "points": party.outcome_points,
                    "satisfaction": party.satisfaction,
                    "opponent_likeness": party.opponent_likeness,
                    "svo": ((party.metadata or {}).get("personality") or {}).get("svo"),
                    "role": _party_role(points, party.party_id),
                    "priorities": party.priorities or {},
                },
            )
        )
        edges.append(GraphEdge(src_id=neg_id, dst_id=pid, rel=REL_HAS_PARTY))

        for issue, level in (party.priorities or {}).items():
            canon = canonical_issue(issue)
            if canon is None:
                continue
            edges.append(
                GraphEdge(
                    src_id=pid,
                    dst_id=issue_node_id(canon),
                    rel=REL_PRIORITIZES,
                    props={"level": level},
                )
            )

        for issue, qty in (final_deal.get(party.party_id) or {}).items():
            canon = canonical_issue(issue)
            if canon is None:
                continue
            edges.append(
                GraphEdge(
                    src_id=pid,
                    dst_id=issue_node_id(canon),
                    rel=REL_ALLOCATED,
                    weight=float(qty),
                    props={
                        "quantity": int(qty),
                        # Denormalized so "did this party surrender their top
                        # priority?" is answerable without a second hop.
                        "own_priority": (party.priorities or {}).get(canon),
                    },
                )
            )

        for strat, meta in (counts.get(party.party_id) or {}).items():
            edges.append(
                GraphEdge(
                    src_id=pid,
                    dst_id=strategy_node_id(strat),
                    rel=REL_USED_STRATEGY,
                    weight=float(meta["count"]),
                    props=dict(meta),
                )
            )

    # Symmetric opponent link, stored both ways so a traversal can start at
    # either party without an OR in the join.
    if len(parties) == 2:
        a = party_node_id(dialogue_id, parties[0].party_id)
        b = party_node_id(dialogue_id, parties[1].party_id)
        edges.append(GraphEdge(src_id=a, dst_id=b, rel=REL_NEGOTIATED_WITH))
        edges.append(GraphEdge(src_id=b, dst_id=a, rel=REL_NEGOTIATED_WITH))

    return nodes, edges


# --------------------------------------------------------------------------
# Corpus-level derived edges
# --------------------------------------------------------------------------


def derive_cooccurrence_edges(
    strategy_sets: Sequence[Set[str]],
    min_count: int = MIN_COOCCURRENCE,
    min_pmi: float = MIN_COOCCURRENCE_PMI,
) -> List[GraphEdge]:
    """`(strategy)-[:CO_OCCURS_WITH]->(strategy)` from dialogue-level co-occurrence.

    One `strategy_sets` entry per **annotated** dialogue: the set of distinct
    strategies any party used in it. Emits a directed edge per ordered pair, so
    the weight can be the asymmetric conditional `P(dst | src)` — which is what
    the retriever's recursive expansion multiplies through, and which stays in
    (0, 1] so the walk strictly decays.

    Pairs are kept only above `min_count` dialogues **and** `min_pmi`; both are
    chosen thresholds on a 396-dialogue subset, not fitted values.
    """
    n_total = len(strategy_sets)
    if n_total == 0:
        return []

    single: Dict[str, int] = defaultdict(int)
    pair: Dict[Tuple[str, str], int] = defaultdict(int)
    for strategies in strategy_sets:
        present = sorted(s for s in strategies if s in STRATEGY_VOCAB)
        for i, a in enumerate(present):
            single[a] += 1
            for b in present[i + 1:]:
                pair[(a, b)] += 1

    edges: List[GraphEdge] = []
    for (a, b), n_ab in sorted(pair.items()):
        if n_ab < min_count:
            continue
        n_a, n_b = single[a], single[b]
        if n_a == 0 or n_b == 0:
            continue
        pmi = math.log((n_ab * n_total) / float(n_a * n_b))
        if pmi < min_pmi:
            continue
        for src, dst, n_src in ((a, b, n_a), (b, a, n_b)):
            edges.append(
                GraphEdge(
                    src_id=strategy_node_id(src),
                    dst_id=strategy_node_id(dst),
                    rel=REL_CO_OCCURS_WITH,
                    weight=round(n_ab / float(n_src), 6),
                    props={
                        "count": n_ab,
                        "pmi": round(pmi, 6),
                        "p_given_src": round(n_ab / float(n_src), 6),
                        "n_dialogues": n_total,
                    },
                )
            )
    return edges


def derive_preceded_edges(
    rows: Sequence[Tuple[Set[str], str]],
) -> List[GraphEdge]:
    """`(strategy)-[:PRECEDED]->(outcome)` — corpus-level association, not causation.

    `rows` is one `(strategies_used, outcome_class)` pair per **annotated**
    dialogue. Both the numerator and the base rate are computed over that same
    annotated subset; mixing a 396-dialogue numerator with a 1,030-dialogue
    base rate would inflate every lift.

    Every edge carries `support` (the raw co-occurrence count) precisely so a
    reader can see how thin the evidence is. On CaSiNo the `no_agreement`
    denominator is a couple of dozen dialogues, so lifts against it are noisy
    and are **not** used for ranking by default. "PRECEDED" names temporal
    order in the corpus, nothing stronger — no causal claim is made or implied.
    """
    n_total = len(rows)
    if n_total == 0:
        return []

    n_strategy: Dict[str, int] = defaultdict(int)
    n_outcome: Dict[str, int] = defaultdict(int)
    n_joint: Dict[Tuple[str, str], int] = defaultdict(int)
    for strategies, outcome in rows:
        n_outcome[outcome] += 1
        for strat in sorted(s for s in strategies if s in STRATEGY_VOCAB):
            n_strategy[strat] += 1
            n_joint[(strat, outcome)] += 1

    edges: List[GraphEdge] = []
    for strat in sorted(n_strategy):
        for outcome in OUTCOME_CLASSES:
            n_o = n_outcome.get(outcome, 0)
            if n_o == 0:
                continue
            n_s = n_strategy[strat]
            n_so = n_joint.get((strat, outcome), 0)
            confidence = n_so / float(n_s)
            base_rate = n_o / float(n_total)
            lift = confidence / base_rate if base_rate > 0 else 0.0
            edges.append(
                GraphEdge(
                    src_id=strategy_node_id(strat),
                    dst_id=outcome_node_id(outcome),
                    rel=REL_PRECEDED,
                    # Clamped: a lift computed off a handful of dialogues can be
                    # arbitrarily large and would dominate any score it entered.
                    weight=round(min(lift, 5.0), 6),
                    props={
                        "support": n_so,
                        "confidence": round(confidence, 6),
                        "base_rate": round(base_rate, 6),
                        "lift": round(lift, 6),
                        "n_strategy_dialogues": n_s,
                        "n_outcome_dialogues": n_o,
                        "n_annotated_dialogues": n_total,
                    },
                )
            )
    return edges


# --------------------------------------------------------------------------
# Whole-corpus build
# --------------------------------------------------------------------------


def build_graph(
    transcripts: Sequence[Transcript],
    cases: Optional[Sequence[CaseDocument]] = None,
    min_cooccurrence: int = MIN_COOCCURRENCE,
) -> GraphBuild:
    """Turn the whole corpus into nodes + edges. Pure — no I/O, no database."""
    by_dialogue, by_strategy = _case_index(cases)

    nodes: List[GraphNode] = build_global_nodes(by_strategy)
    edges: List[GraphEdge] = []

    annotated_strategy_sets: List[Set[str]] = []
    annotated_rows: List[Tuple[Set[str], str]] = []
    matched_cases = 0
    outcome_hist: Dict[str, int] = defaultdict(int)
    structure_hist: Dict[str, int] = defaultdict(int)
    contested_hist: Dict[str, int] = defaultdict(int)
    traded_hist: Dict[str, int] = defaultdict(int)

    for transcript in transcripts:
        case = by_dialogue.get(transcript.dialogue_id)
        matched_cases += int(case is not None)
        n_nodes, n_edges = build_negotiation(transcript, case)
        nodes.extend(n_nodes)
        edges.extend(n_edges)

        neg_props = n_nodes[0].props
        outcome_hist[neg_props["outcome_class"]] += 1
        structure_hist[neg_props["conflict_structure"]] += 1
        for issue in neg_props["contested_issues"]:
            contested_hist[issue] += 1
        for issue in neg_props["traded_issues"]:
            traded_hist[issue] += 1

        used = {
            strat
            for per_party in _strategy_counts(transcript).values()
            for strat in per_party
        }
        if used:
            annotated_strategy_sets.append(used)
            annotated_rows.append((used, neg_props["outcome_class"]))

    edges.extend(derive_cooccurrence_edges(annotated_strategy_sets, min_count=min_cooccurrence))
    edges.extend(derive_preceded_edges(annotated_rows))

    # Backfill corpus-level counts onto the global nodes so a reader of the
    # graph can see support without running an aggregate.
    strategy_dialogue_counts: Dict[str, int] = defaultdict(int)
    for used in annotated_strategy_sets:
        for strat in used:
            strategy_dialogue_counts[strat] += 1

    n_negotiations = len(transcripts)
    for node in nodes:
        if node.node_type == NODE_STRATEGY:
            name = node.props["strategy"]
            node.props["n_annotated_dialogues"] = strategy_dialogue_counts.get(name, 0)
        elif node.node_type == NODE_OUTCOME:
            cls = node.props["outcome_class"]
            node.props["n_negotiations"] = outcome_hist.get(cls, 0)
            node.props["share"] = (
                round(outcome_hist.get(cls, 0) / float(n_negotiations), 6) if n_negotiations else 0.0
            )
        elif node.node_type == NODE_CONFLICT:
            structure = node.props["structure"]
            node.props["n_negotiations"] = structure_hist.get(structure, 0)
            node.props["share"] = (
                round(structure_hist.get(structure, 0) / float(n_negotiations), 6)
                if n_negotiations
                else 0.0
            )
        elif node.node_type == NODE_ISSUE:
            issue = node.props["issue"]
            node.props["n_contested"] = contested_hist.get(issue, 0)
            node.props["n_traded"] = traded_hist.get(issue, 0)

    build = GraphBuild(nodes=nodes, edges=edges)
    build.stats = {
        "transcripts": n_negotiations,
        "cases_matched": matched_cases,
        "cases_unmatched": n_negotiations - matched_cases,
        "annotated_dialogues": len(annotated_strategy_sets),
        "nodes_total": len(nodes),
        "edges_total": len(edges),
        "nodes_by_type": build.counts_by_node_type(),
        "edges_by_rel": build.counts_by_rel(),
        "outcome_distribution": dict(outcome_hist),
        "structure_distribution": dict(structure_hist),
        "contested_issue_counts": dict(contested_hist),
        "traded_issue_counts": dict(traded_hist),
    }
    return build


# --------------------------------------------------------------------------
# Loading (the only part that talks to Postgres)
# --------------------------------------------------------------------------


_UPSERT_NODE = """
INSERT INTO {table} (node_id, node_type, label, props)
VALUES (%s, %s, %s, %s)
ON CONFLICT (node_id) DO UPDATE
   SET node_type = EXCLUDED.node_type,
       label     = EXCLUDED.label,
       props     = EXCLUDED.props
""".format(table=NODE_TABLE)

_UPSERT_EDGE = """
INSERT INTO {table} (src_id, dst_id, rel, weight, props)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (src_id, rel, dst_id) DO UPDATE
   SET weight = EXCLUDED.weight,
       props  = EXCLUDED.props
""".format(table=EDGE_TABLE)


def ddl_path() -> Path:
    return Path(__file__).resolve().parent / "graph_schema.sql"


def apply_schema(database_url: str = "") -> Path:
    """Apply `rag/graph_schema.sql` without needing `psql` on PATH.

    Equivalent to `psql "$DATABASE_URL" -f rag/graph_schema.sql`; provided
    because psql is not installed by default on Windows, where this repo is
    developed. Idempotent — the DDL is all `CREATE ... IF NOT EXISTS`.
    """
    from rag.graph_db import execute_script

    path = ddl_path()
    execute_script(path.read_text(encoding="utf-8"), url=database_url)
    return path


def load_graph(
    build: GraphBuild,
    database_url: str = "",
    apply_ddl: bool = True,
    wipe: bool = True,
    batch_size: int = 1000,
) -> Dict[str, int]:
    """Write `build` to Postgres in **one transaction**. Returns row counts.

    All-or-nothing on purpose: a partially loaded graph would score candidates
    against missing edges and hand back provenance paths that do not exist —
    the precise failure this layer was added to prevent.

    `wipe=True` truncates first, mirroring `rag/embed.py`'s wipe-and-re-embed
    default. For a ~3k-node / ~25k-edge demo corpus that is simpler and safer
    than a diff-based upsert.
    """
    from psycopg.types.json import Jsonb

    from rag.graph_db import connect

    node_rows = [
        (n.node_id, n.node_type, n.label, Jsonb(n.props)) for n in build.nodes
    ]
    edge_rows = [
        (e.src_id, e.dst_id, e.rel, float(e.weight), Jsonb(e.props)) for e in build.edges
    ]

    ddl_sql = ddl_path().read_text(encoding="utf-8") if apply_ddl else None

    with connect(database_url) as conn:
        with conn.cursor() as cur:
            if ddl_sql:
                logger.info("applying %s", ddl_path())
                cur.execute(ddl_sql)
            if wipe:
                logger.info("truncating %s / %s", EDGE_TABLE, NODE_TABLE)
                cur.execute(
                    "TRUNCATE {edges}, {nodes} RESTART IDENTITY CASCADE".format(
                        edges=EDGE_TABLE, nodes=NODE_TABLE
                    )
                )
            logger.info("inserting %d nodes", len(node_rows))
            for start in range(0, len(node_rows), batch_size):
                cur.executemany(_UPSERT_NODE, node_rows[start:start + batch_size])
            logger.info("inserting %d edges", len(edge_rows))
            for start in range(0, len(edge_rows), batch_size):
                cur.executemany(_UPSERT_EDGE, edge_rows[start:start + batch_size])
        conn.commit()

    return {"nodes": len(node_rows), "edges": len(edge_rows)}


# --------------------------------------------------------------------------
# File loading + CLI
# --------------------------------------------------------------------------


def load_transcripts(path: Path) -> List[Transcript]:
    if not path.exists():
        raise FileNotFoundError(
            "{} not found. Run the ingestion first:\n"
            "  python -m data.ingest_casino --download".format(path)
        )
    return [
        Transcript.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_cases(path: Path) -> List[CaseDocument]:
    if not path.exists():
        raise FileNotFoundError(
            "{} not found. Build the case corpus first:\n"
            "  python -m data.build_case_corpus".format(path)
        )
    return [
        CaseDocument.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dump_jsonl(build: GraphBuild, path: Path) -> None:
    """Write nodes then edges as JSONL — lets `--dry-run` output be diffed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for node in build.nodes:
            fh.write(json.dumps({"record": "node", **node.model_dump()}))
            fh.write("\n")
        for edge in build.edges:
            fh.write(json.dumps({"record": "edge", **edge.model_dump()}))
            fh.write("\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the Accord knowledge graph and load it into Postgres."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_TRANSCRIPTS,
                        help="Normalized transcripts JSONL.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                        help="Case corpus JSONL (supplies verbatim document text).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build and report only — never opens a database connection.")
    parser.add_argument("--dump", type=Path, default=None,
                        help="Also write the built nodes/edges to this JSONL path.")
    parser.add_argument("--no-ddl", action="store_true",
                        help="Skip applying rag/graph_schema.sql before loading.")
    parser.add_argument("--no-wipe", action="store_true",
                        help="Upsert into the existing graph instead of truncating first.")
    parser.add_argument("--min-cooccurrence", type=int, default=MIN_COOCCURRENCE,
                        help="Dialogue count floor for a CO_OCCURS_WITH edge.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    transcripts = load_transcripts(args.input)
    cases = load_cases(args.corpus) if args.corpus.exists() else []
    if not cases:
        logger.warning(
            "no case corpus at %s — negotiation nodes will carry no document text, "
            "so graph-only hits cannot return the same text as the vector path",
            args.corpus,
        )

    build = build_graph(transcripts, cases, min_cooccurrence=args.min_cooccurrence)

    if args.dump:
        dump_jsonl(build, args.dump)

    summary: Dict[str, Any] = dict(build.stats)
    summary["input"] = str(args.input)
    summary["corpus"] = str(args.corpus)
    summary["dump"] = str(args.dump) if args.dump else None

    if args.dry_run:
        summary["loaded"] = False
    else:
        loaded = load_graph(
            build,
            apply_ddl=not args.no_ddl,
            wipe=not args.no_wipe,
        )
        summary["loaded"] = True
        summary["rows_written"] = loaded

    json.dump(summary, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
