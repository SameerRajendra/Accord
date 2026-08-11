"""Knowledge-graph schema for Accord's negotiation corpus — nodes, edges, vocab.

This module is the **declarative** half of the graph layer: it defines what a
node and an edge *are*, what types exist, and the deterministic derivations
(conflict structure, outcome class, strategy polarity) that turn a
[`Transcript`](../data/schema.py) into graph facts. It holds no I/O and no
database dependency, so it imports cleanly in a test that never touches
Postgres.

Why a graph at all
------------------
Retrieval today is pure dense-vector similarity over
`data/processed/case_corpus.jsonl`. Every case document is rendered from the
same template (`data/build_case_corpus.py`), so a large fraction of each
document's tokens are boilerplate shared by all 1,030 cases — the same domain
blurb, the same section headers, the same "Lesson:" framing. Cosine similarity
between two such documents is dominated by that shared template rather than by
the facts that distinguish them, which is the mechanism behind the observed
0.42–0.44 score band and the failure to separate a hostile-conflict query from
amicable balanced-deal cases.

Worse, the discriminating facts are frequently *absent from the text as
words*. The corpus contains no hostility vocabulary at all: a negotiation where
one party repeatedly used `uv-part` (undervalue-partner) and the deal collapsed
is rendered in the same neutral register as one that ended in a handshake. No
embedding of that text can recover a distinction the text does not make.

The graph puts those facts back as first-class, exactly-matchable structure:
which strategies a *party* actually used, which issue both parties ranked High,
how the negotiation actually ended, and how those relate across the corpus.

**Honest status:** whether graph-anchored retrieval beats the vector baseline on
this corpus is **unmeasured**. Nothing here is validated; `evals/` owns that
verdict. See `infra/graph/README.md` for the benchmark that would settle it.

Backend note
------------
The vocabulary below is deliberately backend-neutral: node/edge type names are
uppercase relation names in the Cypher idiom, so the same schema loads into
Neo4j unchanged if the project ever ports (the Cypher DDL is committed in
`infra/graph/README.md`). The shipped implementation stores it in the Neon
Postgres instance the project already runs — see that README's ADR section for
the decision and its cost.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Table names (mirrors rag/schema.sql's role for the vector side)
# --------------------------------------------------------------------------

NODE_TABLE = "accord_graph_nodes"
EDGE_TABLE = "accord_graph_edges"


# --------------------------------------------------------------------------
# Node types
# --------------------------------------------------------------------------

NODE_NEGOTIATION = "negotiation"
NODE_PARTY = "party"
NODE_ISSUE = "issue"
NODE_STRATEGY = "strategy"
NODE_OUTCOME = "outcome"
NODE_CONFLICT = "conflict"

NODE_TYPES: Tuple[str, ...] = (
    NODE_NEGOTIATION,
    NODE_PARTY,
    NODE_ISSUE,
    NODE_STRATEGY,
    NODE_OUTCOME,
    NODE_CONFLICT,
)

#: Node types there are only a handful of, shared by every negotiation. Useful
#: because the retriever anchors on these and never needs to page through them.
GLOBAL_NODE_TYPES: Tuple[str, ...] = (NODE_ISSUE, NODE_STRATEGY, NODE_OUTCOME, NODE_CONFLICT)


# --------------------------------------------------------------------------
# Edge (relation) types
# --------------------------------------------------------------------------

REL_HAS_PARTY = "HAS_PARTY"              # negotiation -> party
REL_NEGOTIATED_WITH = "NEGOTIATED_WITH"  # party -> party (both directions)
REL_PRIORITIZES = "PRIORITIZES"          # party -> issue   {level: High|Medium|Low}
REL_ALLOCATED = "ALLOCATED"              # party -> issue   {quantity} (final deal)
REL_USED_STRATEGY = "USED_STRATEGY"      # party -> strategy {count, first_turn, last_turn}
REL_CONTESTED = "CONTESTED"              # negotiation -> issue (both parties ranked it High)
REL_TRADED = "TRADED"                    # negotiation -> issue (one party's High is the other's Low)
REL_HAS_STRUCTURE = "HAS_STRUCTURE"      # negotiation -> conflict
REL_RESULTED_IN = "RESULTED_IN"          # negotiation -> outcome
REL_PRECEDED = "PRECEDED"                # strategy -> outcome  (derived, corpus-level aggregate)
REL_CO_OCCURS_WITH = "CO_OCCURS_WITH"    # strategy -> strategy (derived, corpus-level aggregate)

REL_TYPES: Tuple[str, ...] = (
    REL_HAS_PARTY,
    REL_NEGOTIATED_WITH,
    REL_PRIORITIZES,
    REL_ALLOCATED,
    REL_USED_STRATEGY,
    REL_CONTESTED,
    REL_TRADED,
    REL_HAS_STRUCTURE,
    REL_RESULTED_IN,
    REL_PRECEDED,
    REL_CO_OCCURS_WITH,
)

#: Relations whose weights are *statistics computed over the corpus*, not facts
#: read off a single dialogue. They are correlational and are excluded from
#: default retrieval scoring — see `graph_ingest.derive_preceded_edges`.
DERIVED_RELS: Tuple[str, ...] = (REL_PRECEDED, REL_CO_OCCURS_WITH)


# --------------------------------------------------------------------------
# Domain vocabulary
# --------------------------------------------------------------------------

#: CaSiNo's three tradeable resources. Every dialogue involves all three, which
#: is why there is no `INVOLVES` edge: it would match every negotiation and
#: carry exactly zero retrieval signal. Only `CONTESTED` / `TRADED` are stored.
ISSUES: Tuple[str, ...] = ("Food", "Water", "Firewood")

_ISSUE_BY_LOWER = {name.lower(): name for name in ISSUES}

PRIORITY_LEVELS: Tuple[str, ...] = ("High", "Medium", "Low")


#: Strategy -> coarse negotiation-theory polarity. This is an *interpretive*
#: grouping of CaSiNo's 10-class taxonomy (the dataset does not ship it); it
#: exists so a query like "hostile" can anchor on `uv-part` without the word
#: "hostile" ever appearing in the corpus. Documented as a judgement call, not
#: a dataset fact.
STRATEGY_POLARITY: Dict[str, str] = {
    "small-talk": "rapport",
    "showing-empathy": "rapport",
    "elicit-pref": "integrative",
    "promote-coordination": "integrative",
    "no-need": "integrative",
    "self-need": "distributive",
    "other-need": "distributive",
    "vouch-fair": "distributive",
    "uv-part": "adversarial",
    "non-strategic": "neutral",
}

POLARITIES: Tuple[str, ...] = ("rapport", "integrative", "distributive", "adversarial", "neutral")


# --------------------------------------------------------------------------
# Derived classes: outcome
# --------------------------------------------------------------------------

OUTCOME_NO_AGREEMENT = "no_agreement"
OUTCOME_BALANCED = "agreement_balanced"
OUTCOME_LOPSIDED = "agreement_lopsided"
OUTCOME_UNSCORED = "agreement_unscored"

OUTCOME_CLASSES: Tuple[str, ...] = (
    OUTCOME_NO_AGREEMENT,
    OUTCOME_BALANCED,
    OUTCOME_LOPSIDED,
    OUTCOME_UNSCORED,
)

#: Point gap at or above which an agreement is bucketed `agreement_lopsided`.
#: CaSiNo scores 3 units of each issue at 5/4/3 points for High/Medium/Low, so
#: a party's total lives in roughly [0, 36]. 6 points is chosen as "more than
#: one full High-priority unit of advantage" — a threshold, not a finding. The
#: raw gap is stored on the node (`point_gap`) so any consumer can re-bucket
#: without re-ingesting.
POINT_GAP_LOPSIDED = 6

OUTCOME_DESCRIPTIONS: Dict[str, str] = {
    OUTCOME_NO_AGREEMENT: "The negotiation ended with no accepted deal.",
    OUTCOME_BALANCED: f"Agreement reached with a point gap under {POINT_GAP_LOPSIDED}.",
    OUTCOME_LOPSIDED: f"Agreement reached, but one party led by {POINT_GAP_LOPSIDED}+ points.",
    OUTCOME_UNSCORED: "Agreement reached but per-party points were not recorded.",
}


# --------------------------------------------------------------------------
# Derived classes: conflict structure
# --------------------------------------------------------------------------

STRUCT_IDENTICAL = "identical_rankings"
STRUCT_HIGH_CLASH = "high_clash"
STRUCT_COMPLEMENTARY = "complementary"
STRUCT_PARTIAL = "partial_overlap"
STRUCT_UNKNOWN = "unknown_structure"

CONFLICT_STRUCTURES: Tuple[str, ...] = (
    STRUCT_IDENTICAL,
    STRUCT_HIGH_CLASH,
    STRUCT_COMPLEMENTARY,
    STRUCT_PARTIAL,
    STRUCT_UNKNOWN,
)

CONFLICT_DESCRIPTIONS: Dict[str, str] = {
    STRUCT_IDENTICAL: "Both parties ranked all three issues the same way — maximally zero-sum.",
    STRUCT_HIGH_CLASH: "Both parties named the same issue their top priority.",
    STRUCT_COMPLEMENTARY: "Each party's top priority is the other's lowest — maximal trade potential.",
    STRUCT_PARTIAL: "Priorities overlap partially; some trades create value, some do not.",
    STRUCT_UNKNOWN: "Priority rankings were missing or incomplete for at least one party.",
}

#: A structure query for `high_clash` should also surface `identical_rankings`
#: cases, because identical rankings are a strictly stronger clash. Expansion
#: happens in the query planner, not in the stored edges — one negotiation gets
#: exactly one `HAS_STRUCTURE` edge, always the most specific class.
STRUCTURE_SUBSUMES: Dict[str, Tuple[str, ...]] = {
    STRUCT_HIGH_CLASH: (STRUCT_IDENTICAL,),
}


# --------------------------------------------------------------------------
# Node-id conventions
# --------------------------------------------------------------------------
#
# `<type>:<local id>` — one flat, human-readable, globally unique key space.
# Party ids nest the dialogue id (`party:casino-0:agent_1`) because `agent_1`
# is only unique *within* a dialogue. The retriever derives display labels by
# splitting on the first colon, so no extra lookup is needed for provenance.


def negotiation_node_id(dialogue_id: str) -> str:
    return "{}:{}".format(NODE_NEGOTIATION, dialogue_id)


def party_node_id(dialogue_id: str, party_id: str) -> str:
    return "{}:{}:{}".format(NODE_PARTY, dialogue_id, party_id)


def issue_node_id(issue: str) -> str:
    return "{}:{}".format(NODE_ISSUE, canonical_issue(issue) or issue)


def strategy_node_id(strategy: str) -> str:
    return "{}:{}".format(NODE_STRATEGY, strategy)


def outcome_node_id(outcome_class: str) -> str:
    return "{}:{}".format(NODE_OUTCOME, outcome_class)


def conflict_node_id(structure: str) -> str:
    return "{}:{}".format(NODE_CONFLICT, structure)


def node_local_id(node_id: str) -> str:
    """`strategy:uv-part` -> `uv-part`; `party:casino-0:agent_1` -> `casino-0:agent_1`."""
    _, sep, rest = node_id.partition(":")
    return rest if sep else node_id


def node_type_of(node_id: str) -> str:
    head, sep, _ = node_id.partition(":")
    return head if sep else ""


def canonical_issue(name: Optional[str]) -> Optional[str]:
    """Case-insensitively map a raw issue string onto one of `ISSUES`."""
    if not name:
        return None
    return _ISSUE_BY_LOWER.get(str(name).strip().lower())


# --------------------------------------------------------------------------
# Deterministic derivations
# --------------------------------------------------------------------------


def outcome_class(agreement_reached: bool, points: Optional[Dict[str, int]]) -> str:
    """Bucket a negotiation's ending into one of `OUTCOME_CLASSES`. Pure.

    `points` is `party_id -> points` exactly as `Outcome.points` carries it.
    Only the two-party case can be scored for balance; anything else falls back
    to `agreement_unscored` rather than inventing a gap.
    """
    if not agreement_reached:
        return OUTCOME_NO_AGREEMENT
    values = list((points or {}).values())
    if len(values) != 2 or any(v is None for v in values):
        return OUTCOME_UNSCORED
    gap = abs(int(values[0]) - int(values[1]))
    return OUTCOME_LOPSIDED if gap >= POINT_GAP_LOPSIDED else OUTCOME_BALANCED


def point_gap(points: Optional[Dict[str, int]]) -> Optional[int]:
    values = [v for v in (points or {}).values() if v is not None]
    if len(values) != 2:
        return None
    return abs(int(values[0]) - int(values[1]))


def joint_points(points: Optional[Dict[str, int]]) -> Optional[int]:
    values = [v for v in (points or {}).values() if v is not None]
    if not values:
        return None
    return int(sum(values))


def _level_to_issue(priorities: Optional[Dict[str, str]]) -> Dict[str, str]:
    """`{'Firewood': 'High', ...}` -> `{'High': 'Firewood', ...}` (canonicalized)."""
    out: Dict[str, str] = {}
    for issue, level in (priorities or {}).items():
        canon = canonical_issue(issue)
        if canon is None or level not in PRIORITY_LEVELS:
            continue
        out[level] = canon
    return out


def conflict_structure(
    priorities_a: Optional[Dict[str, str]],
    priorities_b: Optional[Dict[str, str]],
) -> str:
    """Classify how two parties' priority rankings relate. Pure, deterministic.

    Returns the *most specific* applicable class: `identical_rankings` implies a
    high clash, and only the former is emitted. `STRUCTURE_SUBSUMES` records
    that relationship for the query planner to expand at retrieval time.
    """
    a = _level_to_issue(priorities_a)
    b = _level_to_issue(priorities_b)
    if len(a) != len(PRIORITY_LEVELS) or len(b) != len(PRIORITY_LEVELS):
        return STRUCT_UNKNOWN
    if a == b:
        return STRUCT_IDENTICAL
    if a["High"] == b["High"]:
        return STRUCT_HIGH_CLASH
    if a["High"] == b["Low"] and b["High"] == a["Low"]:
        return STRUCT_COMPLEMENTARY
    return STRUCT_PARTIAL


def contested_issues(
    priorities_a: Optional[Dict[str, str]],
    priorities_b: Optional[Dict[str, str]],
) -> List[str]:
    """Issues both parties ranked High — the genuinely scarce resource."""
    a = _level_to_issue(priorities_a)
    b = _level_to_issue(priorities_b)
    if "High" not in a or "High" not in b:
        return []
    return [a["High"]] if a["High"] == b["High"] else []


def traded_issues(
    priorities_a: Optional[Dict[str, str]],
    priorities_b: Optional[Dict[str, str]],
) -> List[str]:
    """Issues one party ranked High and the other Low — free value on the table."""
    a = _level_to_issue(priorities_a)
    b = _level_to_issue(priorities_b)
    out: List[str] = []
    for x, y in ((a, b), (b, a)):
        if "High" in x and "Low" in y and x["High"] == y["Low"]:
            out.append(x["High"])
    # Deterministic order, no duplicates.
    return sorted(set(out))


# --------------------------------------------------------------------------
# Wire models
# --------------------------------------------------------------------------


class GraphNode(BaseModel):
    """One vertex. `props` is stored as JSONB and is queryable via containment."""

    node_id: str = Field(..., description="Globally unique, e.g. 'strategy:uv-part'.")
    node_type: str = Field(..., description="One of NODE_TYPES.")
    label: str = Field(..., description="Human-readable display name.")
    props: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """One directed, typed, weighted relation.

    `(src_id, rel, dst_id)` is unique — a party that used `vouch-fair` four
    times gets ONE edge with `props['count'] == 4`, not four edges. That keeps
    the load idempotent and makes counts available to scoring without an
    aggregate.
    """

    src_id: str
    dst_id: str
    rel: str = Field(..., description="One of REL_TYPES.")
    weight: float = Field(default=1.0, description="Relation strength; 1.0 for plain facts.")
    props: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "NODE_TABLE",
    "EDGE_TABLE",
    "NODE_TYPES",
    "GLOBAL_NODE_TYPES",
    "REL_TYPES",
    "DERIVED_RELS",
    "ISSUES",
    "PRIORITY_LEVELS",
    "STRATEGY_POLARITY",
    "POLARITIES",
    "OUTCOME_CLASSES",
    "OUTCOME_DESCRIPTIONS",
    "POINT_GAP_LOPSIDED",
    "CONFLICT_STRUCTURES",
    "CONFLICT_DESCRIPTIONS",
    "STRUCTURE_SUBSUMES",
    "GraphNode",
    "GraphEdge",
    "canonical_issue",
    "conflict_structure",
    "contested_issues",
    "traded_issues",
    "outcome_class",
    "point_gap",
    "joint_points",
    "negotiation_node_id",
    "party_node_id",
    "issue_node_id",
    "strategy_node_id",
    "outcome_node_id",
    "conflict_node_id",
    "node_local_id",
    "node_type_of",
]
