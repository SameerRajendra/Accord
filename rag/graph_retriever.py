"""Graph-aware precedent retrieval — plan, traverse, rerank, with provenance.

Sits **alongside** `rag/retriever.py`, never replacing it. The public entry
point mirrors that module's signature so the two can be benchmarked head to
head::

    from rag.retriever       import retrieve         # vector baseline
    from rag.graph_retriever import graph_retrieve   # graph + vector hybrid

    a = retrieve(query, k=5)
    b = graph_retrieve(query, k=5)

Three stages
------------
1. **Query planning** (`plan_query`) — deterministic, lexicon-driven, no LLM.
   Maps free text onto graph anchors: issues, persuasion strategies, outcome
   classes, conflict structures. `plan_from_transcript` does the same from a
   live [`Transcript`](../data/schema.py), where the priority rankings are
   *known facts* rather than words to be guessed at — a strictly stronger
   anchor set, and the reason the agent should prefer it.
2. **Traversal** (`_traverse`) — one SQL round trip. A recursive CTE expands
   the seed strategies one or more hops along `CO_OCCURS_WITH`, walks back
   `strategy <- party <- negotiation`, unions in issue / outcome / structure
   matches, and sums a weighted score per candidate negotiation.
3. **Rerank** (`fuse`) — combines the graph score with the existing vector
   similarity from `rag.retriever.retrieve`, by normalized weighted sum
   (default) or reciprocal rank fusion.

Why this exists
---------------
The vector baseline scored 0.42–0.44 on a hostile-conflict query and returned
five amicable balanced-deal cases, after which the recommendation node
fabricated a citation about one of them. Two mechanisms are at work and the
graph addresses both:

* Every case document is rendered from one template, so cosine similarity is
  dominated by shared boilerplate rather than by what distinguishes the cases.
  Graph anchors match on *structure* (`RESULTED_IN no_agreement`), which the
  template flattens but does not remove.
* The words the query used ("hostile", "aggressive") never appear in the
  corpus at all — CaSiNo renders an adversarial negotiation in the same neutral
  register as a friendly one. No embedding can recover a distinction the text
  does not make. The planner instead maps "hostile" onto the *adversarial*
  strategy polarity (`uv-part`), the `no_agreement` / `agreement_lopsided`
  outcome classes, and the `high_clash` conflict structure — facts the graph
  stores explicitly.

Every returned hit carries `evidence`: which anchor matched, along which path,
contributing how much. Given that the failure this layer answers to was a
*fabricated citation*, traceability is a requirement rather than a nicety —
a caller can check that a cited case actually stands in the claimed relation.

Honest status
-------------
**Unproven.** No benchmark has yet compared this against the vector baseline on
this corpus. The scoring weights below are hand-set priors, not fitted values;
the choice between weighted-sum and RRF fusion is a guess; and the planner's
lexicon is hand-written, so its recall against real user phrasing is unknown.
`evals/retrieval_eval.py` (recall@k / MRR on held-out cases) is the experiment
that would settle it, and until it runs, "graph retrieval improves precedent
quality" is a hypothesis this module implements — not a result it demonstrates.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from data.schema import Transcript
from rag.graph_schema import (
    EDGE_TABLE,
    NODE_TABLE,
    OUTCOME_BALANCED,
    OUTCOME_LOPSIDED,
    OUTCOME_NO_AGREEMENT,
    STRATEGY_POLARITY,
    STRUCT_COMPLEMENTARY,
    STRUCT_HIGH_CLASH,
    STRUCT_IDENTICAL,
    STRUCTURE_SUBSUMES,
    conflict_node_id,
    conflict_structure,
    contested_issues,
    issue_node_id,
    node_local_id,
    outcome_node_id,
    strategy_node_id,
    traded_issues,
)
from rag.retriever import RetrievedCase

logger = logging.getLogger(__name__)


# ==========================================================================
# 1. Query planning
# ==========================================================================
#
# Lexicons are hand-written and deliberately so: the same "pure, deterministic,
# no LLM" discipline as data/build_case_corpus.py. An LLM planner would be more
# flexible and completely unreproducible, and every retrieval benchmark run
# against it would measure the planner's sampling as much as the retrieval.
#
# Convention: a trailing `*` means prefix match ("empathi*" matches "empathise",
# "empathy", "empathetic"). Everything else is whole-word.

_ISSUE_TERMS: Dict[str, Tuple[str, ...]] = {
    "Firewood": ("firewood", "fire wood", "wood", "fire", "campfire", "warmth", "warm", "heat", "cold"),
    "Food": ("food", "meal*", "eat*", "hungry", "hunger", "snack*", "provision*", "ration*"),
    "Water": ("water", "drink*", "thirst*", "hydrat*", "dehydrat*"),
}

_STRATEGY_TERMS: Dict[str, Tuple[str, ...]] = {
    "small-talk": ("small talk", "smalltalk", "chit chat", "chitchat", "rapport", "pleasantr*", "greeting*"),
    "elicit-pref": (
        "elicit*", "ask what they", "ask them what", "probe*", "preference*", "their priorit*",
        "what do you need", "discover*", "find out what",
    ),
    "showing-empathy": ("empath*", "sympath*", "acknowledg*", "validat*", "understanding"),
    "promote-coordination": (
        "coordinat*", "collaborat*", "cooperat*", "work together", "mutual*", "joint",
        "integrative", "logroll*", "trade off", "trade-off", "value creat*",
    ),
    "no-need": ("no need", "don't need", "do not need", "dont need", "give it up", "give up", "concede*", "concession*"),
    "self-need": ("self need", "my need*", "i need", "personal need", "justif*"),
    "other-need": ("other need", "my family", "my group", "my kids", "my children", "my dog", "my pet", "grandma", "elderly", "on behalf"),
    "vouch-fair": ("fair", "fairly", "fairness", "unfair", "even split", "split even*", "equal split", "50/50", "fifty fifty", "half each"),
    "uv-part": (
        "undervalu*", "under-valu*", "dismiss*", "belittl*", "downplay*", "disregard*",
        "you don't need", "you do not need", "insult*", "demean*",
    ),
    "non-strategic": ("logistic*", "filler", "housekeeping"),
}

_OUTCOME_TERMS: Dict[str, Tuple[str, ...]] = {
    OUTCOME_NO_AGREEMENT: (
        "no agreement", "no deal", "without a deal", "broke down", "breakdown", "break down",
        "walked away", "walk away", "walkaway", "fail*", "fell apart", "collaps*",
        "impasse", "deadlock*", "stalemate", "abandon*",
    ),
    OUTCOME_LOPSIDED: (
        "lopsided", "one-sided", "one sided", "favored", "favoured", "uneven", "imbalanc*",
        "exploit*", "dominat*", "steamroll*", "got the better",
    ),
    OUTCOME_BALANCED: (
        "balanced", "win-win", "win win", "even deal", "fair deal", "both satisfied",
        "equal points", "amicable", "both happy", "mutually satisfying",
    ),
}

_STRUCTURE_TERMS: Dict[str, Tuple[str, ...]] = {
    STRUCT_IDENTICAL: (
        "same priorit*", "identical priorit*", "same ranking*", "identical ranking*",
        "zero-sum", "zero sum", "directly opposed", "mirror*",
    ),
    STRUCT_HIGH_CLASH: (
        "both want*", "both need*", "same top priority", "contest*", "clash*", "compet*",
        "fight over", "fighting over", "conflict*", "head to head", "head-to-head", "scarce",
    ),
    STRUCT_COMPLEMENTARY: (
        "complementar*", "different priorit*", "opposite priorit*", "swap*", "trade*",
        "barter*", "exchange*",
    ),
}

#: Tone words that name no specific tactic but strongly imply a family of them.
#: This is the bridge that makes "find hostile precedents" answerable at all:
#: the corpus never uses the word, but it does record who played `uv-part` and
#: which negotiations collapsed.
_POLARITY_TERMS: Dict[str, Tuple[str, ...]] = {
    "adversarial": (
        "hostil*", "aggressiv*", "adversarial", "antagonist*", "combative", "confrontational",
        "attack*", "threat*", "ultimatum*", "stonewall*", "escalat*", "toxic", "rude",
        "abusive", "bad faith", "hardball", "coerc*",
    ),
    "integrative": (
        "collaborat*", "cooperat*", "win-win", "win win", "integrative", "value creat*",
        "amicable", "constructive", "friendly", "de-escalat*", "deescalat*",
    ),
    "distributive": ("zero-sum", "zero sum", "positional", "haggl*", "distributive", "hard bargain*", "anchor*"),
    "rapport": ("rapport", "small talk", "warm up", "icebreak*"),
}

#: A tone word also biases which *outcomes* and *structures* are plausible.
#: Stated as an explicit table rather than buried in code so the assumption is
#: reviewable — and it is an assumption, not a measured association.
_POLARITY_IMPLIES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "adversarial": {
        "outcomes": (OUTCOME_NO_AGREEMENT, OUTCOME_LOPSIDED),
        "structures": (STRUCT_HIGH_CLASH,),
    },
    "integrative": {
        "outcomes": (OUTCOME_BALANCED,),
        "structures": (STRUCT_COMPLEMENTARY,),
    },
    "distributive": {
        "outcomes": (OUTCOME_LOPSIDED,),
        "structures": (STRUCT_HIGH_CLASH,),
    },
    "rapport": {"outcomes": (), "structures": ()},
}

_POLARITY_TO_STRATEGIES: Dict[str, Tuple[str, ...]] = {
    polarity: tuple(sorted(name for name, p in STRATEGY_POLARITY.items() if p == polarity))
    for polarity in sorted(set(STRATEGY_POLARITY.values()))
}


def _compile(terms: Sequence[str]) -> List[Tuple[str, "re.Pattern"]]:
    compiled: List[Tuple[str, "re.Pattern"]] = []
    for term in terms:
        stem = term.endswith("*")
        core = term[:-1] if stem else term
        body = r"\s+".join(re.escape(word) for word in core.split())
        tail = r"\w*" if stem else r"\b"
        compiled.append((term, re.compile(r"\b" + body + tail, re.IGNORECASE)))
    return compiled


_COMPILED: Dict[str, Dict[str, List[Tuple[str, "re.Pattern"]]]] = {
    "issues": {anchor: _compile(terms) for anchor, terms in _ISSUE_TERMS.items()},
    "strategies": {anchor: _compile(terms) for anchor, terms in _STRATEGY_TERMS.items()},
    "outcomes": {anchor: _compile(terms) for anchor, terms in _OUTCOME_TERMS.items()},
    "structures": {anchor: _compile(terms) for anchor, terms in _STRUCTURE_TERMS.items()},
    "polarities": {anchor: _compile(terms) for anchor, terms in _POLARITY_TERMS.items()},
}


def _match_family(text: str, family: str) -> Dict[str, List[str]]:
    hits: Dict[str, List[str]] = {}
    for anchor, patterns in _COMPILED[family].items():
        matched = [term for term, pattern in patterns if pattern.search(text)]
        if matched:
            hits[anchor] = matched
    return hits


def _expand_structures(structures: Sequence[str]) -> List[str]:
    """`high_clash` should also surface the strictly-stronger `identical_rankings`."""
    out: List[str] = []
    for structure in structures:
        if structure not in out:
            out.append(structure)
        for implied in STRUCTURE_SUBSUMES.get(structure, ()):
            if implied not in out:
                out.append(implied)
    return out


class GraphQueryPlan(BaseModel):
    """What the planner decided to look up — inspectable, testable, loggable.

    Exposed rather than hidden inside `graph_retrieve` for the same reason the
    hits carry provenance: when retrieval returns something surprising, the
    first question is "what did it think I asked for?", and that should be
    answerable without a debugger.
    """

    query: str = ""
    origin: str = Field(default="text", description="'text' | 'transcript' | 'merged' | 'manual'.")
    issues: List[str] = Field(default_factory=list)
    strategies: List[str] = Field(default_factory=list)
    outcomes: List[str] = Field(default_factory=list)
    structures: List[str] = Field(default_factory=list)
    polarities: List[str] = Field(default_factory=list)
    matched_terms: Dict[str, List[str]] = Field(
        default_factory=dict, description="anchor -> the surface forms that triggered it."
    )
    notes: List[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.issues or self.strategies or self.outcomes or self.structures)

    def anchor_ids(self) -> Dict[str, List[str]]:
        return {
            "issue_ids": [issue_node_id(i) for i in self.issues],
            "strategy_ids": [strategy_node_id(s) for s in self.strategies],
            "outcome_ids": [outcome_node_id(o) for o in self.outcomes],
            "structure_ids": [conflict_node_id(s) for s in self.structures],
        }

    def describe(self) -> str:
        """One-line human summary — cheap to log next to a retrieval result."""
        parts: List[str] = []
        if self.issues:
            parts.append("issues=" + ",".join(self.issues))
        if self.strategies:
            parts.append("strategies=" + ",".join(self.strategies))
        if self.outcomes:
            parts.append("outcomes=" + ",".join(self.outcomes))
        if self.structures:
            parts.append("structures=" + ",".join(self.structures))
        if self.polarities:
            parts.append("tone=" + ",".join(self.polarities))
        return "plan[{}]({})".format(self.origin, "; ".join(parts) or "empty")


def plan_query(query: str) -> GraphQueryPlan:
    """Map free text onto graph anchors. Pure, deterministic, no LLM, no I/O."""
    text = query or ""
    issue_hits = _match_family(text, "issues")
    strategy_hits = _match_family(text, "strategies")
    outcome_hits = _match_family(text, "outcomes")
    structure_hits = _match_family(text, "structures")
    polarity_hits = _match_family(text, "polarities")

    matched: Dict[str, List[str]] = {}
    for family in (issue_hits, strategy_hits, outcome_hits, structure_hits, polarity_hits):
        matched.update(family)

    strategies = list(strategy_hits)
    outcomes = list(outcome_hits)
    structures = list(structure_hits)
    notes: List[str] = []

    for polarity in polarity_hits:
        for strategy in _POLARITY_TO_STRATEGIES.get(polarity, ()):
            if strategy not in strategies:
                strategies.append(strategy)
        implied = _POLARITY_IMPLIES.get(polarity, {})
        for outcome in implied.get("outcomes", ()):
            if outcome not in outcomes:
                outcomes.append(outcome)
        for structure in implied.get("structures", ()):
            if structure not in structures:
                structures.append(structure)
        notes.append(
            "tone '{}' expanded to strategies {} and outcomes {}".format(
                polarity,
                list(_POLARITY_TO_STRATEGIES.get(polarity, ())),
                list(implied.get("outcomes", ())),
            )
        )

    structures = _expand_structures(structures)

    plan = GraphQueryPlan(
        query=query,
        origin="text",
        issues=sorted(issue_hits),
        strategies=strategies,
        outcomes=outcomes,
        structures=structures,
        polarities=sorted(polarity_hits),
        matched_terms=matched,
        notes=notes,
    )
    if plan.is_empty():
        plan.notes.append(
            "no graph anchors found — graph_retrieve will fall back to vector-only"
        )
    return plan


def plan_from_transcript(
    transcript: Transcript,
    target_outcome: Optional[str] = None,
    max_strategies: int = 4,
) -> GraphQueryPlan:
    """Build a plan from a live transcript's *structure* rather than its words.

    Strictly stronger anchoring than `plan_query` where it applies: the parties'
    priority rankings are recorded facts, so the conflict structure and the
    contested/traded issues are computed exactly rather than guessed from
    phrasing. Strategy anchors come from annotations when present — a live,
    unlabeled transcript has none, which is stated in `notes` rather than
    papered over.

    `target_outcome` is opt-in and defaults to nothing: the outcome of the
    transcript under analysis is unknown (predicting it is a different node's
    job), so anchoring on one would beg the question. Pass
    `OUTCOME_NO_AGREEMENT` explicitly to ask "show me precedents that collapsed".
    """
    parties = list(transcript.parties)
    prio_a = parties[0].priorities if len(parties) > 0 else None
    prio_b = parties[1].priorities if len(parties) > 1 else None

    notes: List[str] = []
    structures: List[str] = []
    if len(parties) == 2:
        structures = _expand_structures([conflict_structure(prio_a, prio_b)])
    else:
        notes.append(
            "transcript has {} parties; conflict structure needs exactly 2".format(len(parties))
        )

    issues = sorted(set(contested_issues(prio_a, prio_b)) | set(traded_issues(prio_a, prio_b)))
    if not issues:
        notes.append("no contested or traded issue derivable from the priority rankings")

    counts: Dict[str, int] = {}
    for turn in transcript.turns:
        for strat in turn.strategies:
            # Only labels the graph actually has nodes for; an out-of-vocabulary
            # label would anchor on a node id that does not exist.
            if strat in STRATEGY_POLARITY:
                counts[strat] = counts.get(strat, 0) + 1
    strategies = [
        name for name, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_strategies]
    ]
    if not strategies:
        notes.append(
            "transcript carries no strategy annotations (only ~38% of CaSiNo does, and a "
            "live transcript never will) — anchoring on structure and issues only"
        )

    outcomes = [target_outcome] if target_outcome else []

    return GraphQueryPlan(
        query="",
        origin="transcript",
        issues=issues,
        strategies=strategies,
        outcomes=outcomes,
        structures=structures,
        polarities=[],
        matched_terms={},
        notes=notes,
    )


def merge_plans(*plans: GraphQueryPlan) -> GraphQueryPlan:
    """Union of anchors, order-preserving and duplicate-free."""
    merged = GraphQueryPlan(origin="merged")
    for plan in plans:
        merged.query = merged.query or plan.query
        for field in ("issues", "strategies", "outcomes", "structures", "polarities"):
            target = getattr(merged, field)
            for value in getattr(plan, field):
                if value not in target:
                    target.append(value)
        merged.matched_terms.update(plan.matched_terms)
        merged.notes.extend(plan.notes)
    return merged


# ==========================================================================
# 2. Traversal
# ==========================================================================


class RetrievalWeights(BaseModel):
    """Per-anchor-family scoring weights.

    **Hand-set priors, not fitted values.** They encode intuitions — an exactly
    matching persuasion strategy is worth more than a shared traded issue; a
    strategy reached by one expansion hop is worth half a directly matched one —
    that no experiment on this corpus has yet confirmed. Tuning them requires
    the retrieval benchmark that does not exist yet (`evals/retrieval_eval.py`),
    and tuning them by eyeballing a few queries would be exactly the kind of
    invisible overfitting this repo's README warns about.
    """

    strategy: float = Field(default=1.0, description="A party actually used an anchored strategy.")
    issue: float = Field(default=0.6, description="The negotiation contested an anchored issue.")
    outcome: float = Field(default=0.8, description="The negotiation ended in an anchored outcome class.")
    structure: float = Field(default=0.9, description="The negotiation has an anchored conflict structure.")
    strategy_doc: float = Field(default=0.5, description="Playbook definition of an anchored strategy.")
    traded_discount: float = Field(default=0.5, description="TRADED counts less than CONTESTED.")
    count_saturation: float = Field(default=3.0, description="Uses of a strategy beyond this add nothing.")
    hop_decay: float = Field(default=0.5, description="Multiplier per CO_OCCURS_WITH expansion hop.")


DEFAULT_WEIGHTS = RetrievalWeights()

# The recursive query expansion, shared by both statements below.
#
# Seeds are the strategies the planner anchored on; each hop follows a
# CO_OCCURS_WITH edge whose weight is P(dst | src) over annotated dialogues, so
# the walk strictly decays (weight <= 1, hop_decay < 1) and the depth guard
# bounds it regardless. Cycles are harmless for the same reason — the
# co-occurrence relation is stored in both directions.
_STRATEGY_WALK = """
WITH RECURSIVE strategy_walk(node_id, w, depth) AS (
    SELECT n.node_id, 1.0::double precision, 0
      FROM {nodes} n
     WHERE n.node_type = 'strategy'
       AND n.node_id = ANY(%(strategy_ids)s::text[])
    UNION ALL
    SELECT e.dst_id,
           sw.w * %(hop_decay)s::double precision * e.weight,
           sw.depth + 1
      FROM strategy_walk sw
      JOIN {edges} e
        ON e.src_id = sw.node_id
       AND e.rel = 'CO_OCCURS_WITH'
     WHERE sw.depth < %(max_hops)s
),
strategy_anchor AS (
    SELECT node_id, MAX(w) AS w, MIN(depth) AS depth
      FROM strategy_walk
     GROUP BY node_id
)
""".format(nodes=NODE_TABLE, edges=EDGE_TABLE)

_TRAVERSAL_SQL = _STRATEGY_WALK + """,
candidate AS (
    SELECT n.node_id, n.label, n.props
      FROM {nodes} n
     WHERE n.node_type = 'negotiation'
       AND n.props @> %(filters)s
),
strategy_raw AS (
    SELECT hp.src_id AS negotiation_id,
           sa.node_id AS anchor_id,
           us.src_id  AS party_id,
           sa.depth   AS hops,
           sa.w * LEAST(
               1.0,
               COALESCE((us.props->>'count')::double precision, 1.0)
               / %(count_saturation)s::double precision
           ) AS raw,
           COALESCE((us.props->>'count')::int, 1) AS use_count
      FROM strategy_anchor sa
      JOIN {edges} us ON us.dst_id = sa.node_id AND us.rel = 'USED_STRATEGY'
      JOIN {edges} hp ON hp.dst_id = us.src_id AND hp.rel = 'HAS_PARTY'
      JOIN candidate c ON c.node_id = hp.src_id
),
strategy_hits AS (
    -- Both parties may have used the same tactic; keep the stronger use so a
    -- two-party dialogue is not double-counted against a one-party one.
    SELECT DISTINCT ON (negotiation_id, anchor_id)
           negotiation_id,
           'strategy'::text AS kind,
           anchor_id,
           hops,
           raw * %(w_strategy)s::double precision AS contribution,
           jsonb_build_object('party_id', party_id, 'count', use_count) AS detail
      FROM strategy_raw
     ORDER BY negotiation_id, anchor_id, raw DESC, party_id
),
issue_hits AS (
    SELECT e.src_id AS negotiation_id,
           'issue'::text AS kind,
           e.dst_id AS anchor_id,
           0 AS hops,
           CASE WHEN e.rel = 'CONTESTED'
                THEN %(w_issue)s::double precision
                ELSE %(w_issue)s::double precision * %(traded_discount)s::double precision
           END AS contribution,
           jsonb_build_object('rel', e.rel) AS detail
      FROM {edges} e
      JOIN candidate c ON c.node_id = e.src_id
     WHERE e.rel IN ('CONTESTED', 'TRADED')
       AND e.dst_id = ANY(%(issue_ids)s::text[])
),
outcome_hits AS (
    SELECT e.src_id, 'outcome'::text, e.dst_id, 0,
           %(w_outcome)s::double precision,
           jsonb_build_object('rel', e.rel)
      FROM {edges} e
      JOIN candidate c ON c.node_id = e.src_id
     WHERE e.rel = 'RESULTED_IN'
       AND e.dst_id = ANY(%(outcome_ids)s::text[])
),
structure_hits AS (
    SELECT e.src_id, 'structure'::text, e.dst_id, 0,
           %(w_structure)s::double precision,
           jsonb_build_object('rel', e.rel)
      FROM {edges} e
      JOIN candidate c ON c.node_id = e.src_id
     WHERE e.rel = 'HAS_STRUCTURE'
       AND e.dst_id = ANY(%(structure_ids)s::text[])
),
all_hits AS (
    SELECT * FROM strategy_hits
    UNION ALL SELECT * FROM issue_hits
    UNION ALL SELECT * FROM outcome_hits
    UNION ALL SELECT * FROM structure_hits
),
agg AS (
    SELECT negotiation_id,
           SUM(contribution) AS graph_score,
           COUNT(*)          AS n_evidence,
           jsonb_agg(jsonb_build_object(
               'kind',         kind,
               'anchor_id',    anchor_id,
               'hops',         hops,
               'contribution', contribution,
               'detail',       detail
           ) ORDER BY contribution DESC, anchor_id) AS evidence
      FROM all_hits
     GROUP BY negotiation_id
)
SELECT a.negotiation_id AS node_id,
       c.label,
       c.props,
       a.graph_score,
       a.n_evidence,
       a.evidence
  FROM agg a
  JOIN candidate c ON c.node_id = a.negotiation_id
 ORDER BY a.graph_score DESC, a.negotiation_id
 LIMIT %(candidate_k)s
""".format(nodes=NODE_TABLE, edges=EDGE_TABLE)

# Playbook definitions for the anchored tactics. Returned alongside precedent
# cases because "what is uv-part?" is a legitimate retrieval need, and because
# the vector store already carries these same 10 documents — leaving them out
# would make the head-to-head comparison unfair to the graph.
_STRATEGY_DOC_SQL = _STRATEGY_WALK + """
SELECT n.node_id,
       n.label,
       n.props,
       sa.w * %(w_strategy_doc)s::double precision AS graph_score,
       sa.depth AS hops
  FROM strategy_anchor sa
  JOIN {nodes} n ON n.node_id = sa.node_id
 ORDER BY graph_score DESC, n.node_id
 LIMIT %(strategy_doc_k)s
""".format(nodes=NODE_TABLE)

_PRECEDED_SQL = """
SELECT e.src_id AS strategy_id,
       e.dst_id AS outcome_id,
       e.weight AS lift,
       e.props  AS props
  FROM {edges} e
 WHERE e.rel = 'PRECEDED'
   AND (e.props->>'support')::int >= %(min_support)s
 ORDER BY e.weight DESC, e.src_id
""".format(edges=EDGE_TABLE)


class GraphPathStep(BaseModel):
    """One hop of the path that justified a hit."""

    node_type: str
    node_id: str
    label: str
    rel: Optional[str] = Field(default=None, description="Relation traversed to reach this node.")


class GraphEvidence(BaseModel):
    """Why a case was retrieved: which anchor, along which path, worth how much."""

    kind: str = Field(..., description="'strategy' | 'issue' | 'outcome' | 'structure' | 'strategy_doc'.")
    anchor_id: str
    anchor_label: str
    contribution: float
    hops: int = Field(default=0, description="CO_OCCURS_WITH expansion hops from a seed strategy.")
    path: List[GraphPathStep] = Field(default_factory=list)
    detail: Dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        label = self.anchor_label
        if self.kind == "strategy":
            count = self.detail.get("count")
            party = self.detail.get("party_id", "")
            party_label = node_local_id(party).split(":")[-1] if party else "a party"
            base = "{} used strategy '{}'".format(party_label, label)
            if count:
                base += " x{}".format(count)
            if self.hops:
                base += " (reached by {} co-occurrence hop{})".format(
                    self.hops, "" if self.hops == 1 else "s"
                )
            return base
        if self.kind == "issue":
            rel = str(self.detail.get("rel", "")).lower() or "involved"
            return "{} issue {}".format(rel, label)
        if self.kind == "outcome":
            return "ended as {}".format(label)
        if self.kind == "structure":
            return "conflict structure {}".format(label)
        if self.kind == "strategy_doc":
            return "playbook definition of '{}'".format(label)
        return "{} {}".format(self.kind, label)


class GraphRetrievedCase(RetrievedCase):
    """A precedent plus the graph evidence that produced it.

    Subclasses [`RetrievedCase`](retriever.py) on purpose: `agent/graph.py`
    types its state as `List[RetrievedCase]`, so these are drop-in and the
    agent needs no state-schema change to consume them.

    `score` (inherited) is the **fused** score, not `1 - cosine`. The raw
    vector similarity is preserved separately in `vector_score`, which is
    `None` when the case was found by graph traversal alone.
    """

    graph_score: float = Field(default=0.0, description="Summed weighted anchor contributions.")
    vector_score: Optional[float] = Field(
        default=None, description="1 - cosine from the vector path; None if graph-only."
    )
    graph_rank: Optional[int] = Field(default=None, description="1-based rank in the graph list.")
    vector_rank: Optional[int] = Field(default=None, description="1-based rank in the vector list.")
    fusion: str = Field(default="weighted", description="Fusion method that produced `score`.")
    evidence: List[GraphEvidence] = Field(default_factory=list)
    matched_by: List[str] = Field(
        default_factory=list, description="Human-readable one-liners, one per evidence item."
    )


def _path_for(kind: str, negotiation_id: str, anchor_id: str, detail: Dict[str, Any]) -> List[GraphPathStep]:
    """Reconstruct the traversal path from ids alone — no extra database round trip."""

    def step(node_id: str, rel: Optional[str] = None) -> GraphPathStep:
        node_type, _, local = node_id.partition(":")
        return GraphPathStep(node_type=node_type, node_id=node_id, label=local or node_id, rel=rel)

    if kind == "strategy_doc":
        return [step(anchor_id)]
    start = [step(negotiation_id)]
    if kind == "strategy":
        party = detail.get("party_id")
        if party:
            start.append(step(str(party), rel="HAS_PARTY"))
        return start + [step(anchor_id, rel="USED_STRATEGY")]
    rel = detail.get("rel")
    return start + [step(anchor_id, rel=str(rel) if rel else None)]


def _traverse(
    plan: GraphQueryPlan,
    weights: RetrievalWeights,
    max_hops: int,
    candidate_k: int,
    include_strategy_docs: bool,
    filters: Optional[Dict[str, Any]],
    database_url: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run the traversal. Returns (negotiation rows, strategy-doc rows)."""
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb

    from rag.graph_db import connect

    anchors = plan.anchor_ids()
    params: Dict[str, Any] = {
        "strategy_ids": anchors["strategy_ids"],
        "issue_ids": anchors["issue_ids"],
        "outcome_ids": anchors["outcome_ids"],
        "structure_ids": anchors["structure_ids"],
        "filters": Jsonb(filters or {}),
        "max_hops": int(max_hops),
        "hop_decay": float(weights.hop_decay),
        "count_saturation": float(weights.count_saturation),
        "w_strategy": float(weights.strategy),
        "w_issue": float(weights.issue),
        "w_outcome": float(weights.outcome),
        "w_structure": float(weights.structure),
        "traded_discount": float(weights.traded_discount),
        "candidate_k": int(candidate_k),
    }

    with connect(database_url) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(_TRAVERSAL_SQL, params)
            negotiation_rows = [dict(row) for row in cur.fetchall()]

            doc_rows: List[Dict[str, Any]] = []
            if include_strategy_docs and anchors["strategy_ids"]:
                doc_params = {
                    "strategy_ids": anchors["strategy_ids"],
                    "max_hops": int(max_hops),
                    "hop_decay": float(weights.hop_decay),
                    "w_strategy_doc": float(weights.strategy_doc),
                    "strategy_doc_k": max(1, min(5, len(anchors["strategy_ids"]) + 1)),
                }
                cur.execute(_STRATEGY_DOC_SQL, doc_params)
                doc_rows = [dict(row) for row in cur.fetchall()]

    return negotiation_rows, doc_rows


def _rows_to_cases(
    negotiation_rows: Sequence[Dict[str, Any]],
    doc_rows: Sequence[Dict[str, Any]],
) -> List[GraphRetrievedCase]:
    """Turn raw SQL rows into `GraphRetrievedCase` objects with full provenance."""
    out: List[GraphRetrievedCase] = []

    for rank, row in enumerate(negotiation_rows, start=1):
        props = dict(row.get("props") or {})
        node_id = row["node_id"]
        dialogue_id = props.get("dialogue_id") or node_local_id(node_id)
        case_id = props.get("case_id") or "{}-{}".format(props.get("source", "casino"), dialogue_id)

        evidence: List[GraphEvidence] = []
        for item in row.get("evidence") or []:
            anchor_id = item.get("anchor_id", "")
            detail = dict(item.get("detail") or {})
            evidence.append(
                GraphEvidence(
                    kind=item.get("kind", ""),
                    anchor_id=anchor_id,
                    anchor_label=node_local_id(anchor_id),
                    contribution=float(item.get("contribution") or 0.0),
                    hops=int(item.get("hops") or 0),
                    path=_path_for(item.get("kind", ""), node_id, anchor_id, detail),
                    detail=detail,
                )
            )

        metadata = {key: value for key, value in props.items() if key != "text"}
        metadata["node_id"] = node_id
        out.append(
            GraphRetrievedCase(
                case_id=case_id,
                source=str(props.get("source") or ""),
                kind="case",
                text=props.get("text") or "",
                score=0.0,  # filled in by `fuse`
                metadata=metadata,
                graph_score=float(row.get("graph_score") or 0.0),
                graph_rank=rank,
                evidence=evidence,
                matched_by=[e.summary() for e in evidence],
            )
        )

    for rank, row in enumerate(doc_rows, start=1):
        props = dict(row.get("props") or {})
        node_id = row["node_id"]
        name = props.get("strategy") or node_local_id(node_id)
        evidence = [
            GraphEvidence(
                kind="strategy_doc",
                anchor_id=node_id,
                anchor_label=name,
                contribution=float(row.get("graph_score") or 0.0),
                hops=int(row.get("hops") or 0),
                path=_path_for("strategy_doc", node_id, node_id, {}),
                detail={"polarity": props.get("polarity")},
            )
        ]
        out.append(
            GraphRetrievedCase(
                case_id=str(props.get("case_id") or "strategy-{}".format(name)),
                source=str(props.get("source") or "playbook"),
                kind="strategy",
                text=props.get("text") or "",
                score=0.0,
                metadata={
                    "strategy": name,
                    "polarity": props.get("polarity"),
                    "node_id": node_id,
                    "n_annotated_dialogues": props.get("n_annotated_dialogues"),
                },
                graph_score=float(row.get("graph_score") or 0.0),
                graph_rank=len(negotiation_rows) + rank,
                evidence=evidence,
                matched_by=[e.summary() for e in evidence],
            )
        )

    out.sort(key=lambda c: (-c.graph_score, c.case_id))
    for rank, case in enumerate(out, start=1):
        case.graph_rank = rank
    return out


# ==========================================================================
# 3. Rerank / fusion
# ==========================================================================


def _minmax(values: Dict[str, float]) -> Dict[str, float]:
    """Scale to [0, 1] within the candidate set; all-equal collapses to 1.0."""
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi - lo < 1e-12:
        return {key: 1.0 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def fuse(
    graph_cases: Sequence[GraphRetrievedCase],
    vector_cases: Sequence[RetrievedCase],
    k: int = 5,
    fusion: str = "weighted",
    alpha: float = 0.6,
    rrf_k: int = 60,
) -> List[GraphRetrievedCase]:
    """Combine graph and vector rankings into one list, keyed by `case_id`.

    `alpha` is the graph's share of the blend (`alpha=1.0` → graph only,
    `alpha=0.0` → vector only), which makes the ablation a parameter sweep
    rather than a code change.

    Two methods, and **which is better here is unmeasured**:

    * ``"weighted"`` — min-max normalize each score within the candidate set,
      then blend. Known weakness on this corpus: the observed vector scores sit
      in a narrow 0.42–0.44 band, so min-max stretches a ~0.02 spread across
      the full [0, 1] range and amplifies whatever noise is in it.
    * ``"rrf"`` — reciprocal rank fusion. Ignores score magnitudes entirely, so
      the narrow-band problem cannot arise, and a candidate missing from one
      list simply contributes nothing from that side. Plausibly the better fit
      for this corpus; that is a prediction, not a result.

    **Known bias, stated rather than hidden:** a candidate found only by graph
    traversal has no vector score, because computing one would need a second
    similarity lookup by id that LangChain's `PGVector` does not expose
    cleanly. Under ``"weighted"`` it is treated as the minimum (0.0) on the
    vector axis, which systematically understates graph-only hits. ``"rrf"``
    does not have this problem, which is the second reason to prefer it once
    someone measures both.
    """
    graph_by_id: Dict[str, GraphRetrievedCase] = {}
    for case in graph_cases:
        if case.case_id not in graph_by_id:
            graph_by_id[case.case_id] = case

    vector_by_id: Dict[str, RetrievedCase] = {}
    vector_rank: Dict[str, int] = {}
    for rank, case in enumerate(vector_cases, start=1):
        if case.case_id in vector_by_id:
            continue
        vector_by_id[case.case_id] = case
        vector_rank[case.case_id] = rank

    keys = list(graph_by_id) + [key for key in vector_by_id if key not in graph_by_id]

    graph_norm = _minmax({key: graph_by_id[key].graph_score for key in graph_by_id})
    vector_norm = _minmax({key: float(vector_by_id[key].score) for key in vector_by_id})

    fused: List[GraphRetrievedCase] = []
    for key in keys:
        graph_case = graph_by_id.get(key)
        vector_case = vector_by_id.get(key)

        if graph_case is not None:
            case = graph_case.model_copy(deep=True)
            # Prefer the vector store's text when we have it: it is the exact
            # string that was embedded, so both retrievers hand the downstream
            # prompt identical bytes for the same case.
            if vector_case is not None and vector_case.text:
                case.text = vector_case.text
        elif vector_case is not None:
            case = GraphRetrievedCase(
                case_id=vector_case.case_id,
                source=vector_case.source,
                kind=vector_case.kind,
                text=vector_case.text,
                score=0.0,
                metadata=dict(vector_case.metadata),
                graph_score=0.0,
                graph_rank=None,
                evidence=[],
                matched_by=["vector similarity only — no graph anchor matched this case"],
            )
        else:  # pragma: no cover — every key came from one of the two maps
            continue

        case.vector_score = float(vector_case.score) if vector_case is not None else None
        case.vector_rank = vector_rank.get(key)
        case.fusion = fusion

        g_norm = graph_norm.get(key, 0.0)
        v_norm = vector_norm.get(key, 0.0)
        if fusion == "rrf":
            g_part = 1.0 / (rrf_k + case.graph_rank) if case.graph_rank else 0.0
            v_part = 1.0 / (rrf_k + case.vector_rank) if case.vector_rank else 0.0
            case.score = alpha * g_part + (1.0 - alpha) * v_part
        elif fusion == "weighted":
            case.score = alpha * g_norm + (1.0 - alpha) * v_norm
        else:
            raise ValueError("unknown fusion {!r} — expected 'weighted' or 'rrf'".format(fusion))

        fused.append(case)

    fused.sort(key=lambda c: (-c.score, c.case_id))
    return fused[:k]


# ==========================================================================
# 4. Public entry points
# ==========================================================================


def graph_retrieve(
    query: str,
    k: int = 5,
    plan: Optional[GraphQueryPlan] = None,
    fusion: str = "weighted",
    alpha: float = 0.6,
    max_hops: int = 1,
    candidate_k: int = 40,
    vector_k: int = 20,
    use_vector: bool = True,
    include_strategy_docs: bool = True,
    filters: Optional[Dict[str, Any]] = None,
    weights: Optional[RetrievalWeights] = None,
    database_url: str = "",
) -> List[GraphRetrievedCase]:
    """Retrieve the top-k precedents using graph structure **and** vector similarity.

    Signature-compatible with `rag.retriever.retrieve(query, k)` for the first
    two positional arguments, so a benchmark harness can swap one for the other.
    Returns `GraphRetrievedCase`, a subclass of `RetrievedCase` carrying the
    graph provenance — existing consumers that only read `case_id` / `text` /
    `score` keep working unchanged.

    Args:
        plan: skip text planning and use these anchors (see `plan_from_transcript`).
        alpha: graph's share of the blend. `alpha=1.0, use_vector=False` gives a
            graph-only ablation; `alpha=0.0` reduces to the vector baseline.
        max_hops: `CO_OCCURS_WITH` expansion depth. 0 disables query expansion.
        candidate_k: how many graph candidates to score before fusion.
        filters: JSONB containment predicate on negotiation props, e.g.
            `{"split": "test"}` or `{"has_strategy_annotations": True}`.

    Degradation (mirrors DESIGN.md §6): if the graph tables are missing or
    unreachable, this logs and returns the pure-vector result rather than
    raising — the graph layer is additive and must never take the existing
    retrieval path down with it. If the *vector* side fails instead, graph-only
    results are returned for the same reason.

    Whether the fused ranking beats the vector baseline on this corpus is
    **not yet measured**; see this module's docstring.
    """
    weights = weights or DEFAULT_WEIGHTS
    plan = plan if plan is not None else plan_query(query)

    query = query or ""
    graph_cases: List[GraphRetrievedCase] = []
    if plan.is_empty():
        logger.info("graph plan empty for query %r — vector-only", query[:120])
    else:
        try:
            negotiation_rows, doc_rows = _traverse(
                plan=plan,
                weights=weights,
                max_hops=max_hops,
                candidate_k=candidate_k,
                include_strategy_docs=include_strategy_docs,
                filters=filters,
                database_url=database_url,
            )
            graph_cases = _rows_to_cases(negotiation_rows, doc_rows)
            logger.info("%s -> %d graph candidates", plan.describe(), len(graph_cases))
        except Exception as exc:  # noqa: BLE001 — missing tables / unreachable DB / bad DSN
            logger.warning(
                "graph traversal failed (%s: %s) — degrading to vector-only",
                type(exc).__name__,
                exc,
            )

    vector_cases: List[RetrievedCase] = []
    if use_vector and query:
        try:
            from rag.retriever import retrieve as vector_retrieve

            vector_cases = list(vector_retrieve(query, k=vector_k))
        except Exception as exc:  # noqa: BLE001 — pgvector down should not kill graph results
            logger.warning(
                "vector retrieval failed (%s: %s) — degrading to graph-only",
                type(exc).__name__,
                exc,
            )

    if not graph_cases and not vector_cases:
        logger.warning("no results from either retrieval path for query %r", query[:120])
        return []

    return fuse(graph_cases, vector_cases, k=k, fusion=fusion, alpha=alpha)


def graph_retrieve_for_transcript(
    transcript: Transcript,
    query: Optional[str] = None,
    k: int = 5,
    target_outcome: Optional[str] = None,
    **kwargs: Any,
) -> List[GraphRetrievedCase]:
    """Retrieve precedent for a live transcript, anchoring on its actual structure.

    Merges `plan_from_transcript` (exact: priority rankings, contested issues,
    conflict structure) with `plan_query` over `query` (fuzzy: whatever the
    caller typed, or the last few turns). This is the call the LangGraph agent
    should make — see `infra/graph/README.md` for the integration point.
    """
    kwargs.pop("plan", None)  # this function owns planning; an override would conflict
    plan = plan_from_transcript(transcript, target_outcome=target_outcome)
    if query:
        plan = merge_plans(plan, plan_query(query))
    return graph_retrieve(query or "", k=k, plan=plan, **kwargs)


def strategy_outcome_associations(
    min_support: int = 5,
    database_url: str = "",
) -> List[Dict[str, Any]]:
    """Read the derived `(strategy)-[:PRECEDED]->(outcome)` edges. Analysis only.

    This is the query flat vector retrieval fundamentally cannot answer: "across
    the whole corpus, which tactics preceded which endings?" is an aggregate
    over relations, not a similarity lookup over documents.

    Read it with the sample sizes in view. Every row carries `support` (the raw
    dialogue count behind it), `base_rate`, and `n_annotated_dialogues`. CaSiNo
    reaches agreement in ~97.6% of dialogues and only ~396 carry strategy
    annotations, so the `no_agreement` rows rest on a couple of dozen dialogues
    and their lifts are noisy. That is why these edges are **not** used for
    ranking by default — an unvalidated association driving retrieval order
    would be a quiet overclaim.
    """
    from rag.graph_db import fetch_all

    rows = fetch_all(_PRECEDED_SQL, {"min_support": int(min_support)}, url=database_url)
    out: List[Dict[str, Any]] = []
    for row in rows:
        props = dict(row.get("props") or {})
        out.append(
            {
                "strategy": node_local_id(row["strategy_id"]),
                "outcome": node_local_id(row["outcome_id"]),
                "lift": float(row.get("lift") or 0.0),
                "support": props.get("support"),
                "confidence": props.get("confidence"),
                "base_rate": props.get("base_rate"),
                "n_strategy_dialogues": props.get("n_strategy_dialogues"),
                "n_annotated_dialogues": props.get("n_annotated_dialogues"),
            }
        )
    return out


__all__ = [
    "GraphQueryPlan",
    "GraphEvidence",
    "GraphPathStep",
    "GraphRetrievedCase",
    "RetrievalWeights",
    "DEFAULT_WEIGHTS",
    "plan_query",
    "plan_from_transcript",
    "merge_plans",
    "fuse",
    "graph_retrieve",
    "graph_retrieve_for_transcript",
    "strategy_outcome_associations",
]
