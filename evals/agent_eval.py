"""The signature experiment: RAG vs no-RAG, with a citation-grounding check.

Runs the same transcripts through `agent.graph.run(t, use_rag=True)` and
`use_rag=False`, then measures three different things — deliberately, because
they can disagree, and the disagreement is the finding:

1. **Which recommendation is better** (LLM-judge pairwise preference, blind,
   position-randomized, with a coin-flip baseline and an exact binomial test).
2. **Whether the RAG arm's citations are real** (citation grounding —
   see below). This is the metric this file exists for.
3. **What retrieval costs** (latency delta) and **how often it produces nothing
   usable** (`grounded_case_ids` empty).

Citation grounding — the metric that does not depend on the judge
-----------------------------------------------------------------
A recommendation can cite a genuinely retrieved case and still fabricate the
thing it claims about it. The reproducible instance in this project:

    "...as seen in [casino-204], where a walk-away strategy led to an
     agreement that balanced both parties' needs."

`casino-204` is real and was retrieved. It contains no walk-away — it is an
uneventful 18–18 trade whose text says only "Outcome: Agreement reached ...
Lesson: A balanced deal — both parties scored 18 points." The citation is
structurally valid and substantively invented, which is the failure mode
retrieval augmentation is supposed to *prevent*, wearing the costume of
groundedness.

So each cited case_id is checked two independent ways:

- **Lexical (deterministic, no LLM, cannot be gamed).** For every clause that
  attributes something to the cited case, any named tactic in it must appear in
  that case's retrieved text, and any asserted outcome must match the case's
  stated outcome. On the example above this fires on `walk-away`: absent from
  the case text, therefore unsupported — and it does *not* fire on
  "led to an agreement", which the case text supports. Attribution is scoped to
  clauses rather than sentences so a claim about the live negotiation earlier in
  the same sentence is not charged to the case, and negated references
  ("unlike case X, where nobody walked away") are exempt. Both choices bias
  toward **false negatives**: the reported rate is a floor, not a ceiling. The
  remaining known gap is paraphrase — "stormed off" for walk-away is missed,
  which is what the judge pass is for. Every flag lands in
  `results/agent_eval_citations.csv` with the exact clause that triggered it, so
  no verdict here has to be taken on trust.
- **Judge entailment.** The judge sees only that case's full text and the
  rationale, and labels the claim supported / unsupported / contradicted /
  no_claim. Catches paraphrase; inherits every LLM-judge weakness.

Two structural controls make the numbers hard to flatter:

- **The no-RAG arm is a pure-fabrication control.** With `use_rag=False`
  nothing is retrieved, so *any* `grounded_case_ids` entry in that arm is
  invented by construction — no judgement call, no threshold.
- Grounding is checked against each case's **full** text, while the agent's
  prompt truncates precedents to the first 400 characters
  (`agent/graph._format_analysis`, at time of writing). Checking against more
  context than the model saw makes every "unsupported" verdict a lower bound.

Judge model, and the bias it carries
------------------------------------
The judge should be a *different, larger* open model than the one under test
(DESIGN.md §7). Point it at a second SGLang endpoint with `--judge-model` /
`--judge-base-url`; it is still self-hosted open weights, never a hosted API
(DESIGN.md §1). When no judge is configured the harness falls back to the model
under test, sets `judge_is_same_model=True` in every output, and the preference
numbers must then be read as **self-evaluation**: a model scoring its own
output prefers it, and no amount of prompt wording removes that. Further
limitations, all reported rather than mitigated away:

- **Blinding is imperfect.** The RAG arm's rationale often names case ids, so a
  judge can infer which condition it is looking at even though the arm labels
  are hidden and the order is randomized. `judge_reason_mentions_citation_rate`
  is reported as a diagnostic for exactly this.
- **Position bias.** `position_a_pick_rate` near 0.5 means the judge is reading;
  near 0 or 1 means the preference column is measuring slot order, not quality.
- **n is tens, not thousands.** Every win rate ships with a Wilson 95% interval
  and an exact two-sided binomial p-value against the coin flip. A 12–8 split
  is not a result.

Cost
----
Each transcript costs 2 graph runs (~6 LLM calls) plus 1 pairwise judgement plus
one judgement per citation. At the default `--limit 20` that is roughly 140–180
calls; budget tens of minutes on a warm H100 and use `--limit` deliberately.

Outputs
-------
- `results/agent_eval.csv` — one summary row.
- `results/agent_eval_runs.csv` — one row per (transcript, arm): latency,
  tactic, citation count, degraded flag, judge verdict.
- `results/agent_eval_citations.csv` — one row per cited case_id with both
  grounding verdicts and the exact sentence that triggered any lexical flag.

Usage::

    # smoke test: two hand-written transcripts, ~4 graph runs
    python -m evals.agent_eval --transcripts-json \
        examples/test_hostile.json examples/test_cooperative.json

    # the real run
    python -m evals.agent_eval --limit 20
    python -m evals.agent_eval --limit 30 \
        --judge-model Qwen/Qwen2.5-32B-Instruct \
        --judge-base-url http://127.0.0.1:30001/v1

    # arms + deterministic grounding only, no judge calls
    python -m evals.agent_eval --limit 20 --skip-judge
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from data.schema import Outcome, Party, Transcript, Turn
from evals._common import (
    DEFAULT_RESULTS_DIR,
    DEFAULT_TRANSCRIPTS,
    EvalUnavailable,
    binomial_two_sided_p,
    emit,
    fail,
    judge_chat_model,
    load_transcripts,
    mean,
    percentile,
    preflight_llm,
    preflight_retrieval,
    safe_div,
    select_transcripts,
    text_turns,
    wilson_interval,
    write_csv,
)

# Observed in `agent/graph._format_analysis`: precedents are truncated before
# they reach the prompt. Recorded per citation so the grounding numbers can be
# read against what the model actually saw.
AGENT_PRECEDENT_CONTEXT_CHARS = 400

# The safe-default rationale `agent.graph._node_recommend` emits when the LLM
# call fails. Runs containing it are degraded, counted, and kept in the
# comparison — dropping them would flatter the system.
_DEGRADED_MARKER = "recommendation model failed"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT = re.compile(r"[;,]|\s[—–-]\s")
_CASE_REFERENCE_WORDS = ("case", "precedent", "prior negotiation", "past negotiation", "example")
# Negation cues suppress *tactic* flags only. "unlike case X, where nobody walked
# away" is an accurate statement about a case with no walk-away, and flagging it
# would be a false positive. Outcome claims keep their own polarity-aware phrase
# lists below and are not suppressed. Cost of the guard: a fabrication phrased
# negatively is missed — another push toward false negatives.
_NEGATION_CUE = re.compile(r"\b(?:no|not|never|nobody|none|without|nor|neither)\b|n't")

# Tactic vocabulary whose terms are checkable by string match against a case
# document. Includes the agent's own tactic enum plus CaSiNo's persuasion
# strategies, which appear verbatim in annotated case documents.
# Deliberately excluded: "label" and "other" (the two remaining tactic-enum
# values) — too generic to match without constant false positives.
TACTIC_SURFACE_FORMS: Dict[str, Tuple[str, ...]] = {
    # `walk_away` is how `data/build_case_corpus._outcome_line` renders the
    # Action enum ("ended via walk_away"), so a genuine walk-away case must
    # count as supporting evidence rather than reading as a fabrication.
    "walk-away": (
        "walk away", "walk-away", "walk_away", "walkaway", "walked away", "walking away",
    ),
    "ultimatum": ("ultimatum", "ultimatums", "take it or leave it"),
    "threat": ("threat", "threats", "threaten", "threatened", "threatening"),
    "stonewalling": ("stonewall", "stonewalling", "stonewalled"),
    "mirror": ("mirror", "mirroring", "mirrored"),
    "accusation-audit": ("accusation audit", "accusation-audit"),
    "calibrated-question": ("calibrated question", "calibrated-question"),
    "value-swap": ("value swap", "value-swap", "logroll", "logrolling"),
    "anchoring": ("anchoring", "anchored", "anchor"),
    "deadline": ("deadline", "time pressure"),
    "small-talk": ("small-talk", "small talk"),
    "elicit-pref": ("elicit-pref", "elicit preferences", "eliciting preferences"),
    "showing-empathy": ("showing-empathy", "empathy"),
    "promote-coordination": ("promote-coordination", "promote coordination"),
    "no-need": ("no-need",),
    "self-need": ("self-need",),
    "other-need": ("other-need",),
    "vouch-fair": ("vouch-fair", "vouch fair"),
    "uv-part": ("uv-part", "undervalue", "undervaluing", "undervalued"),
}

_AGREEMENT_CLAIMS = (
    "led to an agreement",
    "reached an agreement",
    "reached agreement",
    "ended in an agreement",
    "ended in agreement",
    "resulted in an agreement",
    "closed the deal",
    "secured a deal",
    "agreement was reached",
)
_BREAKDOWN_CLAIMS = (
    "broke down",
    "broke off",
    "no agreement",
    "without an agreement",
    "without a deal",
    "failed to reach",
    "fell apart",
    "no deal was reached",
    "ended in deadlock",
)

# Rendered by `data/build_case_corpus._outcome_line` — the ground truth for a
# case document's outcome, present verbatim in every case document.
_CASE_AGREEMENT_MARKER = "outcome: agreement reached"
_CASE_BREAKDOWN_MARKER = "outcome: no agreement"


# --------------------------------------------------------------------------
# Judge contracts
# --------------------------------------------------------------------------


class PairwiseVerdict(BaseModel):
    """Blind pairwise preference between two candidate recommendations."""

    winner: Literal["A", "B", "tie"] = Field(..., description="Which recommendation is better.")
    reason: str = Field(..., description="One or two sentences justifying the choice.")


class GroundingVerdict(BaseModel):
    """Whether a rationale's claim about one cited case survives that case's text."""

    claim: str = Field(..., description="The claim the rationale makes about this case.")
    verdict: Literal["supported", "unsupported", "contradicted", "no_claim"] = Field(
        ..., description="Whether the case text alone supports the claim."
    )
    reason: str = Field(..., description="One sentence citing the deciding part of the case text.")


# --------------------------------------------------------------------------
# Run records
# --------------------------------------------------------------------------


@dataclass
class ArmRun:
    """One arm's result for one transcript."""

    dialogue_id: str
    arm: str
    latency_s: float
    next_move: str = ""
    tactic: str = ""
    rationale: str = ""
    cited_ids: List[str] = field(default_factory=list)
    retrieved_ids: List[str] = field(default_factory=list)
    retrieved_texts: Dict[str, str] = field(default_factory=dict)
    retrieved_scores: List[float] = field(default_factory=list)
    degraded: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass
class CitationCheck:
    """One cited case_id, checked two ways."""

    dialogue_id: str
    arm: str
    cited_raw: str
    resolved_case_id: str
    resolution: str  # exact | normalized | unresolved
    lexical_unsupported_tactics: List[str] = field(default_factory=list)
    lexical_outcome_contradiction: bool = False
    lexical_evidence: str = ""
    judge_verdict: str = ""
    judge_claim: str = ""
    judge_reason: str = ""
    case_text_chars: int = 0

    @property
    def lexically_fabricated(self) -> bool:
        return (
            self.resolution == "unresolved"
            or bool(self.lexical_unsupported_tactics)
            or self.lexical_outcome_contradiction
        )

    @property
    def judge_fabricated(self) -> bool:
        return self.resolution == "unresolved" or self.judge_verdict in (
            "unsupported",
            "contradicted",
        )


# --------------------------------------------------------------------------
# Citation resolution + lexical grounding
# --------------------------------------------------------------------------


def normalize_case_id(raw: str) -> str:
    """Lowercase, strip citation punctuation, unify separators."""
    cleaned = raw.strip().strip("[](){}<>\"'`,.;: ").lower()
    return cleaned.replace("_", "-").replace(" ", "")


def resolve_citation(cited: str, retrieved_ids: Sequence[str]) -> Tuple[str, str]:
    """Map a cited id onto a retrieved case id. Returns (resolved_id, how).

    `how` is "exact", "normalized" (the id as written does not literally exist
    but unambiguously designates a retrieved case — e.g. `casino-204` for the
    corpus's `casino-casino-204`, which is what the observed failure emitted),
    or "unresolved" (no retrieved case matches: the id itself is invented).

    Normalized resolutions are counted separately and are *not* scored as
    fabrications — they point at a real retrieved case — but they are reported,
    because an id a reader cannot look up is its own defect.
    """
    if cited in retrieved_ids:
        return cited, "exact"
    target = normalize_case_id(cited)
    by_normalized = {normalize_case_id(rid): rid for rid in retrieved_ids}
    if target in by_normalized:
        return by_normalized[target], "normalized"
    if len(target) >= 4:
        for normalized, original in by_normalized.items():
            if normalized.endswith(target) or target.endswith(normalized):
                return original, "normalized"
    return "", "unresolved"


@lru_cache(maxsize=256)
def _term_pattern(form: str):
    """Word-ish boundary match, so `empathy` hits `showing-empathy` but not `empathetic`."""
    return re.compile(r"(?<![a-z0-9])" + re.escape(form) + r"(?![a-z0-9])", re.IGNORECASE)


def _contains_term(text: str, forms: Sequence[str]) -> bool:
    return any(_term_pattern(form).search(text) for form in forms)


def _attributing_spans(text: str, case_keys: Sequence[str], sole_citation: bool) -> List[str]:
    """Text spans that say something *about* the cited case.

    Scoped to clauses, not whole sentences, because one sentence routinely mixes
    a claim about the live negotiation with a claim about a precedent:

        "The partner has issued an ultimatum, so hold firm and prepare to walk
         away, as seen in [casino-204], where a walk-away strategy led to an
         agreement."

    Sentence-level scoping would attribute *ultimatum* to casino-204 as well,
    which the rationale never claimed. The span therefore starts at the clause
    naming the case and runs to the end of that sentence — the position an
    attributive relative clause ("…, where …") occupies. A claim placed before
    the citation inside the same clause is still captured; one placed in an
    earlier clause is not, which is a deliberate bias toward false negatives.

    When exactly one case is cited there is no ambiguity about the referent, so
    a sentence that says "the retrieved case shows…" without naming an id
    attributes as a whole.
    """
    spans: List[str] = []
    for sentence in _SENTENCE_SPLIT.split(text or ""):
        clauses = [c for c in _CLAUSE_SPLIT.split(sentence) if c.strip()]
        mention_at: Optional[int] = None
        for position, clause in enumerate(clauses):
            lowered = clause.lower()
            if any(key and key in lowered for key in case_keys):
                mention_at = position
                break
        if mention_at is None:
            if sole_citation and any(
                word in sentence.lower() for word in _CASE_REFERENCE_WORDS
            ):
                spans.append(sentence.strip())
            continue
        spans.append(" ".join(clauses[mention_at:]).strip())
    return spans


def _case_keys(resolved_id: str, cited_raw: str) -> List[str]:
    """Strings a rationale might use to name this case.

    Includes the id with a duplicated source prefix collapsed
    (`casino-casino-204` → `casino-204`), which is how the model actually wrote
    it in the observed failure.
    """
    keys = {normalize_case_id(resolved_id), normalize_case_id(cited_raw)}
    parts = normalize_case_id(resolved_id).split("-")
    if len(parts) >= 3 and parts[0] == parts[1]:
        keys.add("-".join(parts[1:]))
    return [k for k in keys if k]


def lexical_grounding_check(
    rationale: str,
    next_move: str,
    cited_raw: str,
    resolved_id: str,
    case_text: str,
    sole_citation: bool,
) -> Tuple[List[str], bool, str]:
    """Deterministic grounding check.

    Returns (unsupported_tactics, outcome_contradiction, evidence_span).

    For each sentence attributing something to this case: every named tactic
    must appear in the case's own text, and any asserted outcome must match the
    outcome the case document states. Pure string matching — no model, no
    threshold, reproducible run to run.
    """
    keys = _case_keys(resolved_id, cited_raw)
    spans = _attributing_spans(
        " ".join(part for part in (rationale, next_move) if part), keys, sole_citation
    )
    if not spans:
        return ([], False, "")

    case_lower = (case_text or "").lower()
    case_says_agreement = _CASE_AGREEMENT_MARKER in case_lower
    case_says_breakdown = _CASE_BREAKDOWN_MARKER in case_lower

    unsupported: List[str] = []
    contradiction = False
    evidence: List[str] = []

    for span in spans:
        flagged_here = False
        if not _NEGATION_CUE.search(span.lower()):
            for tactic, forms in TACTIC_SURFACE_FORMS.items():
                if _contains_term(span, forms) and not _contains_term(case_text, forms):
                    unsupported.append(tactic)
                    flagged_here = True
        lowered = span.lower()
        claims_agreement = any(phrase in lowered for phrase in _AGREEMENT_CLAIMS)
        claims_breakdown = any(phrase in lowered for phrase in _BREAKDOWN_CLAIMS)
        if (claims_agreement and case_says_breakdown) or (claims_breakdown and case_says_agreement):
            contradiction = True
            flagged_here = True
        if flagged_here:
            evidence.append(span.strip())

    # Preserve order, drop duplicates.
    seen = set()
    ordered_tactics = [t for t in unsupported if not (t in seen or seen.add(t))]
    return (ordered_tactics, contradiction, " | ".join(evidence))


# --------------------------------------------------------------------------
# Running the arms
# --------------------------------------------------------------------------


def _render_transcript(transcript: Transcript, max_turns: int = 40) -> str:
    turns = text_turns(transcript)[:max_turns]
    return "\n".join(f"{t.speaker}: {t.text}" for t in turns)


def _warmup_transcript() -> Transcript:
    """A throwaway two-turn dialogue used only to absorb cold start.

    Deliberately synthetic rather than `transcripts[0]`: warming on a real eval
    transcript would leave its sentiment/behavior prompts in SGLang's
    RadixAttention prefix cache, handing that transcript's first arm a latency
    advantage the other arm never gets.
    """
    return Transcript(
        dialogue_id="warmup",
        source="manual",
        domain="campsite_resources",
        parties=[
            Party(
                party_id="agent_1",
                priorities={"Food": "High", "Water": "Medium", "Firewood": "Low"},
            ),
            Party(
                party_id="agent_2",
                priorities={"Water": "High", "Firewood": "Medium", "Food": "Low"},
            ),
        ],
        turns=[
            Turn(index=0, speaker="agent_1", text="Warm-up: I could use extra food packages."),
            Turn(index=1, speaker="agent_2", text="Warm-up: water matters more to me."),
        ],
        outcome=Outcome(agreement_reached=False, final_deal=None, points={}),
        metadata={"split": "warmup"},
    )


def _run_arm(transcript: Transcript, use_rag: bool) -> ArmRun:
    """Run one arm and time it. Failures are recorded, not raised."""
    from agent.graph import run as run_graph

    arm = "rag" if use_rag else "norag"
    started = time.perf_counter()
    try:
        state = run_graph(transcript, use_rag=use_rag)
    except Exception as exc:  # noqa: BLE001 — one bad transcript must not kill the run
        return ArmRun(
            dialogue_id=transcript.dialogue_id,
            arm=arm,
            latency_s=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    latency = time.perf_counter() - started

    recommendation = state.get("recommendation")
    retrieved = state.get("retrieved") or []
    if recommendation is None:
        return ArmRun(
            dialogue_id=transcript.dialogue_id,
            arm=arm,
            latency_s=latency,
            error="graph returned no recommendation",
        )
    return ArmRun(
        dialogue_id=transcript.dialogue_id,
        arm=arm,
        latency_s=latency,
        next_move=recommendation.next_move,
        tactic=recommendation.tactic,
        rationale=recommendation.rationale,
        cited_ids=list(recommendation.grounded_case_ids or []),
        retrieved_ids=[r.case_id for r in retrieved],
        retrieved_texts={r.case_id: r.text for r in retrieved},
        retrieved_scores=[float(r.score) for r in retrieved],
        degraded=_DEGRADED_MARKER in (recommendation.rationale or ""),
    )


# --------------------------------------------------------------------------
# Judging
# --------------------------------------------------------------------------


_PAIRWISE_SYSTEM = (
    "You are evaluating two candidate coaching recommendations written for the SAME "
    "negotiation transcript. Pick the one that would help the negotiator more.\n\n"
    "Judge on:\n"
    "1. Specificity — does it name a concrete next action tied to THIS transcript, or "
    "could it have been written for any negotiation?\n"
    "2. Appropriateness — does the move fit the actual level of conflict and the parties' "
    "stated priorities?\n"
    "3. Verifiability — every factual claim in the rationale should be checkable against "
    "the transcript you are shown.\n\n"
    "Rules that matter:\n"
    "- Do NOT reward the mere presence of references to past cases. You cannot verify them "
    "from the transcript, and an unverifiable citation is WORSE than no citation.\n"
    "- Length is not quality. A shorter, sharper move beats a padded one.\n"
    "- Answer 'tie' when they are equally good or differ only in wording."
)

_GROUNDING_SYSTEM = (
    "You verify citations. You are given the FULL TEXT of one precedent case and the "
    "rationale of a recommendation that cited it.\n\n"
    "First state the specific claim the rationale makes ABOUT THIS CASE. Then judge that "
    "claim using the case text ALONE — not your own knowledge of negotiations, and not what "
    "is plausible.\n\n"
    "- supported: the case text states or directly implies the claim.\n"
    "- unsupported: the case text neither states nor implies it. Absence of evidence is "
    "'unsupported' — if the rationale says the case involved a tactic and the case text "
    "never mentions that tactic, the claim is unsupported.\n"
    "- contradicted: the case text says otherwise.\n"
    "- no_claim: the rationale mentions the case but asserts nothing specific about it."
)


def judge_pairwise(judge, transcript: Transcript, first: ArmRun, second: ArmRun):
    """Blind A/B comparison. `first`/`second` are already position-randomized."""
    from agent.callbacks import langfuse_callbacks

    def render(run: ArmRun) -> str:
        # grounded_case_ids is withheld: it is metadata, not advice. The
        # rationale text may still name cases — see the blinding limitation.
        return (
            f"next move: {run.next_move}\n"
            f"tactic: {run.tactic}\n"
            f"rationale: {run.rationale}"
        )

    prompt = [
        ("system", _PAIRWISE_SYSTEM),
        (
            "user",
            "TRANSCRIPT:\n"
            + _render_transcript(transcript)
            + "\n\nRECOMMENDATION A:\n"
            + render(first)
            + "\n\nRECOMMENDATION B:\n"
            + render(second)
            + "\n\nWhich is better?",
        ),
    ]
    return judge.with_structured_output(PairwiseVerdict).invoke(
        prompt, config={"callbacks": langfuse_callbacks()}
    )


def judge_grounding(judge, case_id: str, case_text: str, rationale: str, next_move: str):
    """Entailment check for one citation, against that case's text alone."""
    from agent.callbacks import langfuse_callbacks

    prompt = [
        ("system", _GROUNDING_SYSTEM),
        (
            "user",
            f"CASE {case_id} — FULL TEXT:\n{case_text}\n\n"
            f"RECOMMENDATION THAT CITED IT:\n"
            f"next move: {next_move}\nrationale: {rationale}\n\n"
            f"What does the rationale claim about case {case_id}, and does the case text "
            f"support it?",
        ),
    ]
    return judge.with_structured_output(GroundingVerdict).invoke(
        prompt, config={"callbacks": langfuse_callbacks()}
    )


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


def _load_transcripts_json(paths: Sequence[Path]) -> List[Transcript]:
    """Read transcripts from API-shaped example payloads (`{"transcript": {...}}`)."""
    out: List[Transcript] = []
    for path in paths:
        if not path.exists():
            raise EvalUnavailable(f"{path} not found.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("transcript", payload)
        out.append(Transcript.model_validate(raw))
    return out


def run_eval(
    transcripts: Sequence[Transcript],
    results_dir: Path,
    judge_model: Optional[str],
    judge_base_url: Optional[str],
    seed: int,
    warmup: bool,
    skip_judge: bool,
) -> dict:
    if not transcripts:
        raise EvalUnavailable("No transcripts selected — loosen --split/--limit.")

    ids = [t.dialogue_id for t in transcripts]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise EvalUnavailable(
            f"Duplicate dialogue_id(s) in the selection: {duplicates}.\n"
            f"Both arms are paired by dialogue_id, so duplicates would silently overwrite each "
            f"other. Give each transcript a unique id — note that several examples/*.json "
            f"payloads ship with dialogue_id 'example-1'."
        )

    llm_info = preflight_llm()
    retrieval_info = preflight_retrieval()
    model_under_test = llm_info.get("model_id") or "unknown"

    warmup_s: Optional[float] = None
    if warmup:
        # Absorbs Modal cold start (~60-90 s per DESIGN.md §6) so it is reported
        # once instead of being smeared into the first arm's latency.
        started = time.perf_counter()
        _run_arm(_warmup_transcript(), use_rag=True)
        warmup_s = time.perf_counter() - started

    rng = random.Random(seed)  # noqa: S311 — arm/position shuffling, not cryptography
    runs: Dict[str, Dict[str, ArmRun]] = {}
    arm_orders: Dict[str, str] = {}
    for index, transcript in enumerate(transcripts):
        # Alternate which arm runs first: SGLang's RadixAttention prefix cache
        # makes the second arm's sentiment/behavior calls cheaper (identical
        # prompts), so a fixed order would hand one arm a latency advantage.
        rag_first = index % 2 == 0
        arm_orders[transcript.dialogue_id] = "rag_first" if rag_first else "norag_first"
        order = [True, False] if rag_first else [False, True]
        runs[transcript.dialogue_id] = {}
        for use_rag in order:
            record = _run_arm(transcript, use_rag=use_rag)
            runs[transcript.dialogue_id][record.arm] = record

    rag_runs = [runs[t.dialogue_id]["rag"] for t in transcripts]
    norag_runs = [runs[t.dialogue_id]["norag"] for t in transcripts]
    paired = [
        t
        for t in transcripts
        if runs[t.dialogue_id]["rag"].ok and runs[t.dialogue_id]["norag"].ok
    ]

    # --- citation grounding: lexical pass (no LLM) -------------------------
    checks: List[CitationCheck] = []
    for record in rag_runs + norag_runs:
        if not record.ok:
            continue
        sole = len(record.cited_ids) == 1
        for cited in record.cited_ids:
            resolved, how = resolve_citation(cited, record.retrieved_ids)
            case_text = record.retrieved_texts.get(resolved, "")
            tactics, contradiction, evidence = (
                lexical_grounding_check(
                    record.rationale, record.next_move, cited, resolved, case_text, sole
                )
                if how != "unresolved"
                else ([], False, "")
            )
            checks.append(
                CitationCheck(
                    dialogue_id=record.dialogue_id,
                    arm=record.arm,
                    cited_raw=cited,
                    resolved_case_id=resolved,
                    resolution=how,
                    lexical_unsupported_tactics=tactics,
                    lexical_outcome_contradiction=contradiction,
                    lexical_evidence=evidence,
                    case_text_chars=len(case_text),
                )
            )

    # --- judging (all generation is finished before the judge is built) ----
    judge_info: Dict[str, object] = {
        "requested_model": judge_model,
        "base_url": judge_base_url,
        "used": False,
    }
    verdicts: Dict[str, dict] = {}
    judge_errors = 0
    if not skip_judge:
        effective_model = judge_model or model_under_test
        judge = judge_chat_model(effective_model, base_url=judge_base_url, max_tokens=512)
        judge_info.update({"used": True, "model": effective_model})

        for transcript in paired:
            rag_run = runs[transcript.dialogue_id]["rag"]
            norag_run = runs[transcript.dialogue_id]["norag"]
            rag_is_a = rng.random() < 0.5
            first, second = (rag_run, norag_run) if rag_is_a else (norag_run, rag_run)
            try:
                verdict = judge_pairwise(judge, transcript, first, second)
                winner_arm = (
                    "tie"
                    if verdict.winner == "tie"
                    else (first.arm if verdict.winner == "A" else second.arm)
                )
                verdicts[transcript.dialogue_id] = {
                    "winner_arm": winner_arm,
                    "picked_position": verdict.winner,
                    "rag_position": "A" if rag_is_a else "B",
                    "reason": verdict.reason,
                }
            except Exception as exc:  # noqa: BLE001 — a judge failure is data, not a crash
                judge_errors += 1
                verdicts[transcript.dialogue_id] = {
                    "winner_arm": "error",
                    "picked_position": "",
                    "rag_position": "A" if rag_is_a else "B",
                    "reason": f"{type(exc).__name__}: {exc}",
                }

        for check in checks:
            if check.resolution == "unresolved":
                check.judge_verdict = "skipped_unresolved_id"
                continue
            record = runs[check.dialogue_id][check.arm]
            case_text = record.retrieved_texts.get(check.resolved_case_id, "")
            try:
                grounding = judge_grounding(
                    judge, check.resolved_case_id, case_text, record.rationale, record.next_move
                )
                check.judge_verdict = grounding.verdict
                check.judge_claim = grounding.claim
                check.judge_reason = grounding.reason
            except Exception as exc:  # noqa: BLE001
                judge_errors += 1
                check.judge_verdict = "error"
                check.judge_reason = f"{type(exc).__name__}: {exc}"

    # --- aggregate --------------------------------------------------------
    decisive = [v for v in verdicts.values() if v["winner_arm"] in ("rag", "norag")]
    rag_wins = sum(1 for v in decisive if v["winner_arm"] == "rag")
    ties = sum(1 for v in verdicts.values() if v["winner_arm"] == "tie")
    n_judged = len([v for v in verdicts.values() if v["winner_arm"] != "error"])
    picked_a = sum(1 for v in verdicts.values() if v["picked_position"] == "A")
    # Diagnostic for imperfect blinding: if the judge keeps justifying its pick
    # by pointing at citations, the preference column is partly measuring "has
    # citations", which is the thing under suspicion.
    mentions_citation = sum(
        1
        for v in verdicts.values()
        if v["winner_arm"] != "error"
        and any(word in str(v["reason"]).lower() for word in ("case", "precedent", "citation"))
    )
    wilson_low, wilson_high = wilson_interval(rag_wins, len(decisive))

    rag_latencies = [r.latency_s for r in rag_runs if r.ok]
    norag_latencies = [r.latency_s for r in norag_runs if r.ok]

    rag_checks = [c for c in checks if c.arm == "rag"]
    norag_checks = [c for c in checks if c.arm == "norag"]
    judged_checks = [
        c
        for c in rag_checks
        if c.judge_verdict in ("supported", "unsupported", "contradicted", "no_claim")
    ]

    all_scores = [s for r in rag_runs if r.ok for s in r.retrieved_scores]
    top1_scores = [r.retrieved_scores[0] for r in rag_runs if r.ok and r.retrieved_scores]

    summary = {
        "n_transcripts": len(transcripts),
        "n_paired_ok": len(paired),
        "n_failed_runs": sum(1 for r in rag_runs + norag_runs if not r.ok),
        "model_under_test": model_under_test,
        "judge_model": judge_info.get("model", ""),
        "judge_is_same_model": bool(judge_info.get("used"))
        and judge_info.get("model") == model_under_test,
        "judge_errors": judge_errors,
        # --- preference ---
        "n_judged": n_judged,
        "n_decisive": len(decisive),
        "rag_win_rate": safe_div(rag_wins, len(decisive)),
        "norag_win_rate": safe_div(len(decisive) - rag_wins, len(decisive)),
        "tie_rate": safe_div(ties, n_judged),
        "coin_flip_baseline": 0.5,
        "rag_win_rate_wilson_lo": wilson_low,
        "rag_win_rate_wilson_hi": wilson_high,
        "binomial_p_vs_coin_flip": binomial_two_sided_p(rag_wins, len(decisive)),
        "position_a_pick_rate": safe_div(picked_a, n_judged),
        "judge_reason_mentions_citation_rate": safe_div(mentions_citation, n_judged),
        # --- latency ---
        "warmup_s": warmup_s,
        "rag_latency_p50_s": percentile(rag_latencies, 50),
        "rag_latency_p95_s": percentile(rag_latencies, 95),
        "norag_latency_p50_s": percentile(norag_latencies, 50),
        "norag_latency_p95_s": percentile(norag_latencies, 95),
        "latency_delta_p50_s": percentile(rag_latencies, 50) - percentile(norag_latencies, 50),
        # --- retrieval usage ---
        "rag_empty_citations_rate": safe_div(
            sum(1 for r in rag_runs if r.ok and not r.cited_ids),
            sum(1 for r in rag_runs if r.ok),
        ),
        "norag_empty_citations_rate": safe_div(
            sum(1 for r in norag_runs if r.ok and not r.cited_ids),
            sum(1 for r in norag_runs if r.ok),
        ),
        "mean_top1_retrieval_score": mean(top1_scores) if top1_scores else float("nan"),
        "mean_retrieval_score": mean(all_scores) if all_scores else float("nan"),
        # --- citation grounding ---
        "n_citations_rag": len(rag_checks),
        "citation_unresolvable_rate": safe_div(
            sum(1 for c in rag_checks if c.resolution == "unresolved"), len(rag_checks)
        ),
        "citation_needed_normalization_rate": safe_div(
            sum(1 for c in rag_checks if c.resolution == "normalized"), len(rag_checks)
        ),
        "lexical_unsupported_tactic_rate": safe_div(
            sum(1 for c in rag_checks if c.lexical_unsupported_tactics), len(rag_checks)
        ),
        "lexical_outcome_contradiction_rate": safe_div(
            sum(1 for c in rag_checks if c.lexical_outcome_contradiction), len(rag_checks)
        ),
        "fabrication_rate_lexical": safe_div(
            sum(1 for c in rag_checks if c.lexically_fabricated), len(rag_checks)
        ),
        "n_citations_judged": len(judged_checks),
        "judge_unsupported_rate": safe_div(
            sum(1 for c in judged_checks if c.judge_verdict == "unsupported"), len(judged_checks)
        ),
        "judge_contradicted_rate": safe_div(
            sum(1 for c in judged_checks if c.judge_verdict == "contradicted"), len(judged_checks)
        ),
        "fabrication_rate_judge": safe_div(
            sum(1 for c in rag_checks if c.judge_fabricated),
            len([c for c in rag_checks if c.judge_verdict not in ("", "error")]),
        ),
        # --- pure-fabrication control: no retrieval happened at all ---
        "norag_runs_with_citations": sum(1 for r in norag_runs if r.ok and r.cited_ids),
        "norag_phantom_citations": len(norag_checks),
        "norag_phantom_citation_rate": safe_div(
            sum(1 for r in norag_runs if r.ok and r.cited_ids),
            sum(1 for r in norag_runs if r.ok),
        ),
        # --- degradation ---
        "rag_degraded_runs": sum(1 for r in rag_runs if r.degraded),
        "norag_degraded_runs": sum(1 for r in norag_runs if r.degraded),
        "agent_precedent_context_chars": AGENT_PRECEDENT_CONTEXT_CHARS,
        "seed": seed,
    }

    # --- write artifacts --------------------------------------------------
    summary_path = write_csv(
        results_dir / "agent_eval.csv",
        list(summary.keys()),
        [list(summary.values())],
    )

    run_rows: List[List[object]] = []
    for record in rag_runs + norag_runs:
        verdict = verdicts.get(record.dialogue_id, {})
        run_rows.append(
            [
                record.dialogue_id,
                record.arm,
                arm_orders.get(record.dialogue_id, ""),
                round(record.latency_s, 4),
                record.tactic,
                len(record.cited_ids),
                len(record.retrieved_ids),
                round(record.retrieved_scores[0], 6) if record.retrieved_scores else None,
                int(record.degraded),
                verdict.get("winner_arm", ""),
                verdict.get("rag_position", ""),
                verdict.get("reason", ""),
                record.next_move,
                record.rationale,
                record.error,
            ]
        )
    runs_path = write_csv(
        results_dir / "agent_eval_runs.csv",
        [
            "dialogue_id",
            "arm",
            "arm_order",
            "latency_s",
            "tactic",
            "n_cited",
            "n_retrieved",
            "top1_retrieval_score",
            "degraded",
            "judge_winner_arm",
            "rag_position",
            "judge_reason",
            "next_move",
            "rationale",
            "error",
        ],
        run_rows,
    )

    citations_path = write_csv(
        results_dir / "agent_eval_citations.csv",
        [
            "dialogue_id",
            "arm",
            "cited_raw",
            "resolved_case_id",
            "resolution",
            "lexical_unsupported_tactics",
            "lexical_outcome_contradiction",
            "lexical_evidence_span",
            "judge_verdict",
            "judge_claim",
            "judge_reason",
            "case_text_chars",
            "agent_saw_first_n_chars",
        ],
        [
            [
                c.dialogue_id,
                c.arm,
                c.cited_raw,
                c.resolved_case_id,
                c.resolution,
                ";".join(c.lexical_unsupported_tactics),
                int(c.lexical_outcome_contradiction),
                c.lexical_evidence,
                c.judge_verdict,
                c.judge_claim,
                c.judge_reason,
                c.case_text_chars,
                AGENT_PRECEDENT_CONTEXT_CHARS,
            ]
            for c in checks
        ],
    )

    summary.update(
        {
            "summary_csv": str(summary_path),
            "runs_csv": str(runs_path),
            "citations_csv": str(citations_path),
            "retrieval_probe": retrieval_info,
            "judge": judge_info,
            "caveats": [
                "LLM-judge preference is not ground truth; self-preference bias applies in "
                "full when judge_is_same_model is true",
                "blinding is imperfect — rationales may name case ids; see "
                "judge_reason_mentions_citation_rate",
                "grounding is checked against each case's FULL text while the agent saw only "
                f"the first {AGENT_PRECEDENT_CONTEXT_CHARS} chars, so unsupported verdicts "
                "are a lower bound",
                "lexical grounding misses paraphrase and can false-positive on negation — "
                "every flag is auditable in agent_eval_citations.csv",
            ],
        }
    )
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="RAG vs no-RAG ablation with an LLM judge and a citation-grounding check."
    )
    parser.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS)
    parser.add_argument(
        "--transcripts-json",
        type=Path,
        nargs="+",
        default=None,
        help="Use API-shaped example payloads (examples/*.json) instead of the corpus.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--limit", type=int, default=20, help="Transcripts to evaluate (2 graph runs each)."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Open-weight judge model id. Should differ from the model under test "
        "(DESIGN.md §7); defaults to the model under test with the bias recorded.",
    )
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help="Second SGLang endpoint serving the judge, e.g. http://127.0.0.1:30001/v1.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the warm-up run that absorbs Modal cold start into its own number.",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Run the arms and the deterministic grounding check only (no judge calls).",
    )
    args = parser.parse_args(argv)

    try:
        if args.transcripts_json:
            transcripts = _load_transcripts_json(args.transcripts_json)
        else:
            transcripts = select_transcripts(
                load_transcripts(args.transcripts),
                split=args.split,
                limit=args.limit or None,
                seed=args.seed,
            )
        summary = run_eval(
            transcripts=transcripts,
            results_dir=args.results_dir,
            judge_model=args.judge_model,
            judge_base_url=args.judge_base_url,
            seed=args.seed,
            warmup=not args.no_warmup,
            skip_judge=args.skip_judge,
        )
    except EvalUnavailable as exc:
        return fail(exc)

    emit(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
