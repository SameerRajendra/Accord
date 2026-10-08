"""Reference-free RAG triad — faithfulness, answer-relevance, context-precision.

The retrieval evals (`retrieval_eval.py`) score against a single gold document,
which only exists for the CaSiNo benchmark. That cannot measure the path that
matters most for a product: a user's **own** ingested corpus, where there is no
gold label. This module fills that gap with the RAGAS-style triad — three
metrics that need **no reference answer**, so they run on any namespace,
including live user data:

1. **Faithfulness (groundedness).** Decompose the recommendation's rationale
   into atomic claims; judge each claim against the retrieved context alone.
   Score = supported / total. This is the RAG failure that motivated the whole
   provenance layer — a recommendation asserting things its precedent never
   said. (`agent_eval.py` also computes a *deterministic* lexical version of
   this; report both — the deterministic one is a floor, this one has recall
   the string-matcher lacks.)
2. **Answer relevance.** Does the recommended next move actually address the
   negotiation state? Rubric-scored 1–5. Catches fluent, grounded, but
   off-topic advice.
3. **Context precision@k.** Of the retrieved precedents, how many are relevant
   to the query? Judged per chunk. This is measurable on the live path where
   recall@k is not, because "is this chunk relevant" needs no gold set.

Judge trustworthiness — the part that makes the above credible
--------------------------------------------------------------
Every metric here is **LLM-judged**, so it inherits the circularity `agent_eval`
already warns about: if the judge is the same 7B under test, the numbers grade
their own homework. Two defences, both reported so a reader can discount
accordingly:

* **A stronger judge by default is *requested*** via `--judge-model`
  (e.g. `Qwen/Qwen2.5-32B-Instruct`). If you leave it unset the judge is the
  model under test and the summary says so in `judge_is_model_under_test`.
* **Judge self-consistency is measured**, not assumed. On a subsample every
  judgement is repeated `--consistency-runs` times at non-zero temperature and
  the agreement is reported (`self_consistency`). A metric whose judge flips
  under resampling is not a metric; this surfaces that instead of hiding it
  behind a single deterministic-looking number.

No human calibration set exists here, so none is claimed. Judge-vs-human
agreement (Cohen's κ) is the honest next step and is called out in the summary
rather than faked.

Usage::

    python -m evals.rag_triad_eval --arms vector graph hybrid --limit 10
    python -m evals.rag_triad_eval --namespace acme-legal --limit 10   # live path
    python -m evals.rag_triad_eval --judge-model Qwen/Qwen2.5-32B-Instruct
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from data.schema import Transcript
from evals._common import (
    DEFAULT_RESULTS_DIR,
    DEFAULT_TRANSCRIPTS,
    EvalUnavailable,
    emit,
    fail,
    judge_chat_model,
    load_transcripts,
    mean,
    normalized_entropy,
    preflight_llm,
    preflight_retrieval,
    safe_div,
    select_transcripts,
    stdev,
    text_turns,
    write_csv,
)

ARMS = ("vector", "graph", "hybrid")


# --------------------------------------------------------------------------
# Judge output contracts
# --------------------------------------------------------------------------


class ClaimSet(BaseModel):
    """Atomic factual claims extracted from a rationale."""

    claims: List[str] = Field(
        default_factory=list,
        description="Each standalone factual assertion the rationale makes about precedent "
        "or the negotiation. Split compound sentences. Empty if it makes no checkable claim.",
    )


class ClaimVerdict(BaseModel):
    verdict: str = Field(
        ...,
        description="'supported' | 'unsupported' | 'contradicted' | 'no_context' — judged "
        "against the provided context ONLY, not world knowledge.",
    )
    reason: str = Field(
        ..., description="One sentence pointing at the deciding span, or its absence."
    )


class ClaimVerdicts(BaseModel):
    verdicts: List[ClaimVerdict] = Field(default_factory=list)


class RelevanceVerdict(BaseModel):
    score: int = Field(
        ..., ge=1, le=5, description="1=off-topic, 5=directly addresses the negotiation."
    )
    reason: str = Field(..., description="One sentence.")


class ChunkVerdict(BaseModel):
    relevant: bool = Field(..., description="Is this precedent relevant to the negotiation query?")


class ChunkVerdicts(BaseModel):
    verdicts: List[ChunkVerdict] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Prompts (kept terse; the judge is instructed to use context only)
# --------------------------------------------------------------------------

_DECOMPOSE_SYS = (
    "Extract the atomic factual claims a negotiation rationale makes. Each claim is one "
    "checkable assertion (about a precedent, a party, or the negotiation). Do not invent "
    "claims; split compound ones. If the text makes no checkable claim, return an empty list."
)
_CLAIM_JUDGE_SYS = (
    "You judge whether each claim is supported by the CONTEXT ALONE — the retrieved precedent "
    "text below. Ignore world knowledge. 'supported' = the context states it; 'contradicted' = "
    "the context states the opposite; 'no_context' = the context is silent; 'unsupported' = the "
    "claim goes beyond anything in the context. Return one verdict per claim, in order."
)
_RELEVANCE_SYS = (
    "Rate 1–5 how well the recommended NEXT MOVE addresses the CURRENT negotiation state. "
    "5 = a directly on-point, actionable move for this exact situation; 1 = generic or off-topic. "
    "Judge relevance to the situation, not whether you personally agree with the tactic."
)
_CHUNK_SYS = (
    "For each retrieved precedent, judge whether it is RELEVANT to the negotiation query — "
    "i.e. a reasonable person analysing this negotiation would find it pertinent. Return one "
    "boolean per precedent, in order."
)


# --------------------------------------------------------------------------
# Per-transcript scoring
# --------------------------------------------------------------------------


@dataclass
class TripletScore:
    dialogue_id: str
    arm: str
    faithfulness: float = float("nan")
    n_claims: int = 0
    answer_relevance: float = float("nan")   # 1–5
    context_precision: float = float("nan")
    n_retrieved: int = 0
    error: str = ""


def _render_state(transcript: Transcript, max_turns: int = 12) -> str:
    turns = text_turns(transcript)[-max_turns:]
    return "\n".join(f"{t.speaker}: {t.text}" for t in turns)


def _context_block(retrieved: Sequence) -> str:
    parts: List[str] = []
    for i, r in enumerate(retrieved, start=1):
        text = r.text if len(r.text) <= 600 else r.text[:600] + "…"
        parts.append(f"[{i}] {text}")
    return "\n\n".join(parts) if parts else "(no precedent retrieved)"


def _score_faithfulness(judge, rationale: str, context: str) -> tuple:
    """Return (faithfulness in [0,1] or nan, n_claims)."""
    if not rationale.strip():
        return (float("nan"), 0)
    claims = judge.with_structured_output(ClaimSet).invoke(
        [("system", _DECOMPOSE_SYS), ("user", rationale)]
    ).claims
    if not claims:
        return (float("nan"), 0)
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    verdicts = judge.with_structured_output(ClaimVerdicts).invoke(
        [("system", _CLAIM_JUDGE_SYS), ("user", f"CONTEXT:\n{context}\n\nCLAIMS:\n{numbered}")]
    ).verdicts
    # Only claims that COULD be grounded count against faithfulness; 'no_context'
    # (the claim is about the live negotiation, not precedent) is excluded from
    # the denominator rather than scored as a failure.
    gradable = [v for v in verdicts if v.verdict != "no_context"]
    if not gradable:
        return (float("nan"), len(claims))
    supported = sum(1 for v in gradable if v.verdict == "supported")
    return (safe_div(supported, len(gradable)), len(claims))


def _score_relevance(judge, state: str, next_move: str) -> float:
    if not next_move.strip():
        return float("nan")
    v = judge.with_structured_output(RelevanceVerdict).invoke(
        [("system", _RELEVANCE_SYS), ("user", f"NEGOTIATION:\n{state}\n\nNEXT MOVE:\n{next_move}")]
    )
    return float(v.score)


def _score_context_precision(judge, state: str, retrieved: Sequence) -> float:
    if not retrieved:
        return float("nan")
    numbered = "\n\n".join(
        f"[{i+1}] {(r.text[:400] + '…') if len(r.text) > 400 else r.text}"
        for i, r in enumerate(retrieved)
    )
    verdicts = judge.with_structured_output(ChunkVerdicts).invoke(
        [("system", _CHUNK_SYS), ("user", f"QUERY:\n{state}\n\nPRECEDENTS:\n{numbered}")]
    ).verdicts
    if not verdicts:
        return float("nan")
    relevant = sum(1 for v in verdicts if v.relevant)
    return safe_div(relevant, len(verdicts))


# --------------------------------------------------------------------------
# Judge self-consistency
# --------------------------------------------------------------------------


def _self_consistency(judge_hot, state: str, next_move: str, runs: int) -> Optional[dict]:
    """Repeat the relevance judgement `runs` times; report agreement.

    Uses the answer-relevance rubric as the probe because it is the cheapest
    single-call judgement. A judge whose score swings across identical inputs is
    not measuring a stable property — this quantifies that without needing any
    human label.
    """
    if runs < 2 or not next_move.strip():
        return None
    scores: List[int] = []
    for _ in range(runs):
        v = judge_hot.with_structured_output(RelevanceVerdict).invoke(
            [
                ("system", _RELEVANCE_SYS),
                ("user", f"NEGOTIATION:\n{state}\n\nNEXT MOVE:\n{next_move}"),
            ]
        )
        scores.append(int(v.score))
    counts = list(Counter(scores).values())
    return {
        "scores": scores,
        "unanimous": len(set(scores)) == 1,
        "stdev": stdev(scores),
        "entropy": normalized_entropy(counts) if len(counts) > 1 else 0.0,
    }


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


def run_eval(
    transcripts_path: Path,
    results_dir: Path,
    arms: Sequence[str],
    limit: int,
    split: str,
    namespace: Optional[str],
    judge_model: Optional[str],
    judge_base_url: Optional[str],
    consistency_runs: int,
    consistency_sample: int,
    seed: int,
) -> dict:
    llm = preflight_llm()
    if namespace is None:
        # Benchmark path reads the CaSiNo vector store; fail early with a fix if
        # it's empty. The live path probes its own namespace at run time instead.
        preflight_retrieval()

    from agent.graph import run as run_graph

    transcripts = select_transcripts(
        load_transcripts(transcripts_path), split=split, limit=limit or None, seed=seed
    )
    if not transcripts:
        raise EvalUnavailable("No transcripts selected — loosen --split/--limit.")

    effective_judge = judge_model or llm.get("model_id") or ""
    judge = judge_chat_model(effective_judge, base_url=judge_base_url, temperature=0.0)
    judge_hot = (
        judge_chat_model(effective_judge, base_url=judge_base_url, temperature=0.7)
        if consistency_runs >= 2 else None
    )

    scores: List[TripletScore] = []
    consistency: List[dict] = []

    for arm in arms:
        for i, transcript in enumerate(transcripts):
            ts = TripletScore(dialogue_id=transcript.dialogue_id, arm=arm)
            try:
                state = run_graph(transcript, use_rag=True, retrieval_mode=arm, namespace=namespace)
                rec = state.get("recommendation")
                retrieved = state.get("retrieved") or []
                rationale = getattr(rec, "rationale", "") if rec else ""
                next_move = getattr(rec, "next_move", "") if rec else ""
                rendered = _render_state(transcript)

                ts.faithfulness, ts.n_claims = _score_faithfulness(
                    judge, rationale, _context_block(retrieved)
                )
                ts.answer_relevance = _score_relevance(judge, rendered, next_move)
                ts.context_precision = _score_context_precision(judge, rendered, retrieved)
                ts.n_retrieved = len(retrieved)

                if judge_hot is not None and i < consistency_sample:
                    sc = _self_consistency(judge_hot, rendered, next_move, consistency_runs)
                    if sc is not None:
                        consistency.append(
                            {"dialogue_id": transcript.dialogue_id, "arm": arm, **sc}
                        )
            except Exception as exc:  # noqa: BLE001
                ts.error = f"{type(exc).__name__}: {exc}"
            scores.append(ts)

    # Aggregate per arm.
    per_arm: Dict[str, dict] = {}
    rows: List[List[object]] = []
    for arm in arms:
        arm_scores = [s for s in scores if s.arm == arm and not s.error]
        faith = [s.faithfulness for s in arm_scores if s.faithfulness == s.faithfulness]
        rel = [s.answer_relevance for s in arm_scores if s.answer_relevance == s.answer_relevance]
        prec = [
            s.context_precision for s in arm_scores if s.context_precision == s.context_precision
        ]
        per_arm[arm] = {
            "n": len(arm_scores),
            "n_errors": sum(1 for s in scores if s.arm == arm and s.error),
            "faithfulness_mean": mean(faith) if faith else None,
            "faithfulness_n": len(faith),
            "answer_relevance_mean_1to5": mean(rel) if rel else None,
            "context_precision_mean": mean(prec) if prec else None,
        }
        for s in scores:
            if s.arm == arm:
                rows.append([
                    s.dialogue_id, s.arm, s.faithfulness, s.n_claims,
                    s.answer_relevance, s.context_precision, s.n_retrieved, s.error,
                ])

    write_csv(
        results_dir / "rag_triad.csv",
        ["dialogue_id", "arm", "faithfulness", "n_claims",
         "answer_relevance_1to5", "context_precision", "n_retrieved", "error"],
        rows,
    )
    if consistency:
        write_csv(
            results_dir / "rag_triad_consistency.csv",
            ["dialogue_id", "arm", "scores", "unanimous", "stdev", "entropy"],
            [[c["dialogue_id"], c["arm"], c["scores"], c["unanimous"], c["stdev"], c["entropy"]]
             for c in consistency],
        )

    unanimous_rate = (
        safe_div(sum(1 for c in consistency if c["unanimous"]), len(consistency))
        if consistency else None
    )
    return {
        "corpus": "live namespace: " + namespace if namespace else "benchmark (CaSiNo)",
        "arms": list(arms),
        "n_transcripts": len(transcripts),
        "judge_model": effective_judge,
        "judge_is_model_under_test": (judge_model is None or judge_model == llm.get("model_id")),
        "per_arm": per_arm,
        "self_consistency": {
            "runs_per_item": consistency_runs,
            "n_items": len(consistency),
            "unanimous_rate": unanimous_rate,
            "note": "fraction of resampled relevance judgements that were unanimous; low = the "
                    "judge is noisy and the metrics above should be discounted accordingly",
        } if consistency else {"note": "self-consistency not run (--consistency-runs < 2)"},
        "caveats": [
            "all three metrics are LLM-judged and reference-free — no gold labels involved",
            "if judge_is_model_under_test is true, the judge grades its own output; use "
            "--judge-model for a stronger, independent judge",
            "no human-vs-judge calibration (Cohen's kappa) exists yet — that is the honest "
            "next step, not claimed here",
            "faithfulness excludes 'no_context' claims (statements about the live negotiation, "
            "not precedent) from the denominator rather than scoring them as failures",
        ],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Reference-free RAG triad (RAGAS-style).")
    parser.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument(
        "--limit", type=int, default=10, help="Transcripts per arm (each = ~3-4 judge calls)."
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--namespace",
        default=None,
        help="Score the LIVE path for this namespace instead of the CaSiNo benchmark.",
    )
    parser.add_argument(
        "--judge-model", default=None, help="A STRONGER model than the one under test."
    )
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument(
        "--consistency-runs",
        type=int,
        default=3,
        help="Resamples for judge self-consistency (0/1 to skip).",
    )
    parser.add_argument(
        "--consistency-sample", type=int, default=5, help="How many transcripts to resample."
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        summary = run_eval(
            transcripts_path=args.transcripts,
            results_dir=args.results_dir,
            arms=args.arms,
            limit=args.limit,
            split=args.split,
            namespace=args.namespace,
            judge_model=args.judge_model,
            judge_base_url=args.judge_base_url,
            consistency_runs=args.consistency_runs,
            consistency_sample=args.consistency_sample,
            seed=args.seed,
        )
    except EvalUnavailable as exc:
        return fail(exc)

    emit(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
