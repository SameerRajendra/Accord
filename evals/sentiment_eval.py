"""Sentiment eval — and an upfront admission: **this is not accuracy.**

There is no gold-labeled dataset for the 6-class emotion taxonomy in
`analysis/sentiment.py`, and this harness does not invent one.

Why not, concretely:

- The taxonomy (neutral / collaborative / frustrated / anxious / distrustful /
  urgent) is a demo subset defined *in this repository* — a design choice, not a
  published annotation scheme with an annotated corpus behind it.
- CaSiNo carries no emotion labels at any granularity. Its per-turn human labels
  are **persuasion strategies** (small-talk, uv-part, vouch-fair, …), a
  different construct: "what move is this utterance making", not "what does this
  turn feel like".
- Labeling a few hundred turns with a stronger LLM and calling the result ground
  truth would produce a number that *looks* like accuracy and actually measures
  agreement with that other model. DESIGN.md §7 promises "sentiment F1 vs
  labels"; on this corpus that promise cannot be kept, and the honest move is to
  say so in the artifact rather than ship a number that reads as F1.
  `results/sentiment.csv` therefore carries `has_gold_labels=false` and an empty
  `f1_vs_gold` column, and `evals/report.py` renders it as *not measurable*.

What is measured instead — three weaker claims, each with a name for what it is
--------------------------------------------------------------------------------
``proxy`` (default; the strongest available evidence)
    **Convergent validity** against labels a human actually wrote. CaSiNo's
    annotators marked utterances with persuasion strategies, one of which —
    ``uv-part`` — the corpus itself defines as adversarial ("downplay or dismiss
    their stated needs … an adversarial tactic"), while others (small-talk,
    showing-empathy, promote-coordination, elicit-pref, no-need) are
    cooperative. If the escalation score measures anything, it should rank
    adversarial turns above cooperative ones. Reported as AUC, whose null is
    exactly 0.5. Also: emotion × strategy-group contingency with Cramér's V, and
    whether dialogue-level escalation separates the dialogues that actually
    broke down.
    This validates the *escalation* dimension and the emotion labels only
    indirectly. It says nothing about whether "anxious" vs "distrustful" is
    correct on any individual turn — nothing here can.

``consistency``
    **Test–retest reliability**: the same transcripts scored repeatedly, exact
    label agreement and escalation spread across runs. `analysis/sentiment.py`
    pins temperature to 0.0, so this is not sampling variance — it is runtime
    nondeterminism (continuous batching changes reduction order between runs).
    Note the asymmetry: high agreement is weak evidence (a deterministic engine
    scores ~1.0 trivially, and so does a model that answers "neutral" every
    time); low agreement is strong evidence of a problem. Reported next to
    `majority_emotion_share` and `emotion_entropy_normalized` precisely so a
    degenerate constant labeler cannot hide behind a perfect consistency score.

``judge``
    **Inter-model agreement** with a second, different open-weight model given
    the identical prompt (Cohen's κ on emotion, Spearman on escalation). Two
    models agreeing means they share a bias as often as it means they are right.
    Requires `--judge-model`; still self-hosted (DESIGN.md §1).

Reused verbatim from `analysis/sentiment.py`: the system prompt and the turn
renderer, imported rather than copied so the judge is scored on the same task
the system performs, and so a change upstream breaks this eval loudly instead of
letting it drift.

Outputs
-------
- `results/sentiment.csv` — one summary row (with `has_gold_labels=false`).
- `results/sentiment_strategy_contingency.csv` — emotion × human strategy group.
- `results/sentiment_turns.csv` — every per-turn label from every run.

Cost: one batched LLM call per dialogue per run. Defaults (40 annotated
dialogues × 3 runs) are ~120 calls.

Usage::

    python -m evals.sentiment_eval                          # proxy + consistency
    python -m evals.sentiment_eval --mode proxy --consistency-runs 1
    python -m evals.sentiment_eval --mode proxy consistency judge \
        --judge-model Qwen/Qwen2.5-32B-Instruct \
        --judge-base-url http://127.0.0.1:30001/v1
    python -m evals.sentiment_eval --split all --limit 150  # bigger, slower
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from analysis.sentiment import Emotion, PerTurnSentiment
from data.schema import Transcript
from evals._common import (
    DEFAULT_RESULTS_DIR,
    DEFAULT_TRANSCRIPTS,
    EvalUnavailable,
    cramers_v,
    emit,
    fail,
    judge_chat_model,
    load_transcripts,
    mean,
    normalized_entropy,
    preflight_llm,
    safe_auc,
    safe_div,
    select_transcripts,
    spearman,
    stdev,
    write_csv,
)

MODES = ("proxy", "consistency", "judge")

# CaSiNo strategy groups. Membership follows the playbook definitions committed
# in `data/build_case_corpus.STRATEGY_PLAYBOOK` — not a judgement call made here.
ADVERSARIAL_STRATEGIES = frozenset({"uv-part"})
DISTRIBUTIVE_STRATEGIES = frozenset({"self-need", "other-need", "vouch-fair"})
COOPERATIVE_STRATEGIES = frozenset(
    {"small-talk", "showing-empathy", "promote-coordination", "elicit-pref", "no-need"}
)
STRATEGY_GROUPS = ("adversarial", "distributive", "cooperative", "unclassified")

# Emitted by `analysis.sentiment._neutral_default` when the model drops a turn
# or the call fails outright. A high rate here means the "labels" being analyzed
# below are mostly fallbacks.
_DEFAULT_RATIONALE = "model omitted this turn — neutral default"

_MIN_MINORITY_FOR_AUC = 5


def _sentiment_internals():
    """Fetch the prompt + turn renderer the production path uses.

    Private symbols on purpose: the judge must be given the *same* task, and a
    local copy would silently diverge the day `analysis/sentiment.py` changes.
    """
    from analysis import sentiment as sentiment_module

    system_prompt = getattr(sentiment_module, "_SYSTEM", None)
    render_turns = getattr(sentiment_module, "_render_turns", None)
    if system_prompt is None or render_turns is None:  # pragma: no cover - upstream rename
        raise EvalUnavailable(
            "analysis.sentiment no longer exposes `_SYSTEM` / `_render_turns`.\n"
            "`judge` mode scores a second model on the *identical* task, so it refuses to "
            "fall back to a copied prompt.\n"
            "Fix: update evals/sentiment_eval.py._sentiment_internals, or drop `judge` from "
            "--mode."
        )
    return system_prompt, render_turns


def strategy_group(strategies: Sequence[str]) -> str:
    """Classify one turn's human strategy labels into a single group.

    A turn spanning two contrasting groups is `unclassified` rather than
    assigned to one — mixed evidence should not become a label.
    """
    labels = set(strategies)
    adversarial = bool(labels & ADVERSARIAL_STRATEGIES)
    distributive = bool(labels & DISTRIBUTIVE_STRATEGIES)
    cooperative = bool(labels & COOPERATIVE_STRATEGIES)
    if adversarial and not cooperative:
        return "adversarial"
    if cooperative and not adversarial and not distributive:
        return "cooperative"
    if distributive and not adversarial and not cooperative:
        return "distributive"
    return "unclassified"


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _score_transcripts(
    transcripts: Sequence[Transcript], runs: int
) -> List[Dict[str, List[PerTurnSentiment]]]:
    """Run the production sentiment path `runs` times over every transcript."""
    from analysis.sentiment import analyze

    collected: List[Dict[str, List[PerTurnSentiment]]] = []
    for _ in range(runs):
        this_run: Dict[str, List[PerTurnSentiment]] = {}
        for transcript in transcripts:
            this_run[transcript.dialogue_id] = analyze(transcript)
        collected.append(this_run)
    return collected


def _strategy_by_turn(transcript: Transcript) -> Dict[int, List[str]]:
    return {turn.index: list(turn.strategies) for turn in transcript.turns}


def proxy_metrics(
    transcripts: Sequence[Transcript], run: Dict[str, List[PerTurnSentiment]]
) -> Tuple[dict, List[List[int]]]:
    """Convergent validity against human strategy annotations and dialogue outcome."""
    escalations: List[float] = []
    groups: List[str] = []
    emotions: List[str] = []

    dialogue_scores: List[float] = []
    dialogue_broke_down: List[int] = []

    for transcript in transcripts:
        entries = run.get(transcript.dialogue_id) or []
        if not entries:
            continue
        by_turn = _strategy_by_turn(transcript)
        for entry in entries:
            escalations.append(float(entry.escalation))
            emotions.append(entry.emotion.value)
            groups.append(strategy_group(by_turn.get(entry.turn_index, [])))
        dialogue_scores.append(mean([float(e.escalation) for e in entries]))
        dialogue_broke_down.append(0 if transcript.outcome.agreement_reached else 1)

    def auc_between(positive: str, negative: str) -> Tuple[float, int, int]:
        labels: List[int] = []
        scores: List[float] = []
        for group, escalation in zip(groups, escalations):
            if group == positive:
                labels.append(1)
                scores.append(escalation)
            elif group == negative:
                labels.append(0)
                scores.append(escalation)
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        if min(n_pos, n_neg) < _MIN_MINORITY_FOR_AUC:
            return (float("nan"), n_pos, n_neg)
        return (safe_auc(labels, scores), n_pos, n_neg)

    auc_adversarial, n_adversarial, n_cooperative = auc_between("adversarial", "cooperative")
    auc_distributive, n_distributive, _ = auc_between("distributive", "cooperative")

    n_broke_down = sum(dialogue_broke_down)
    if min(n_broke_down, len(dialogue_broke_down) - n_broke_down) < _MIN_MINORITY_FOR_AUC:
        # CaSiNo is ~97.6% agreement; on a test-split-sized sample this is
        # routinely 2-4 dialogues. An AUC over that is noise wearing a metric's
        # clothes.
        auc_breakdown = float("nan")
    else:
        auc_breakdown = safe_auc(dialogue_broke_down, dialogue_scores)

    emotion_order = [e.value for e in Emotion]
    contingency = [
        [
            sum(1 for g, em in zip(groups, emotions) if g == group and em == emotion)
            for emotion in emotion_order
        ]
        for group in STRATEGY_GROUPS
    ]
    # Cramér's V over the three classified groups only — `unclassified` is a
    # residue category, not a construct.
    classified = [
        row for group, row in zip(STRATEGY_GROUPS, contingency) if group != "unclassified"
    ]

    escalation_counts = Counter(round(v, 4) for v in escalations)
    emotion_counts = Counter(emotions)
    modal_escalation = escalation_counts.most_common(1)[0] if escalation_counts else (None, 0)
    modal_emotion = emotion_counts.most_common(1)[0] if emotion_counts else ("", 0)

    return (
        {
            "n_turns_scored": len(escalations),
            "n_adversarial_turns": n_adversarial,
            "n_cooperative_turns": n_cooperative,
            "n_distributive_turns": n_distributive,
            "auc_escalation_adversarial_vs_cooperative": auc_adversarial,
            "auc_escalation_distributive_vs_cooperative": auc_distributive,
            "auc_null_baseline": 0.5,
            "cramers_v_emotion_by_strategy_group": cramers_v(classified),
            "auc_escalation_predicts_breakdown": auc_breakdown,
            "n_dialogues_broke_down": n_broke_down,
            # Degenerate-labeler detectors: an "always neutral, escalation 0.0"
            # model scores 0.5 on every AUC above and 1.0 on consistency below.
            # These are the columns that expose it.
            "modal_escalation_value": modal_escalation[0],
            "modal_escalation_share": safe_div(modal_escalation[1], len(escalations)),
            "n_distinct_escalation_values": len(escalation_counts),
            "majority_emotion": modal_emotion[0],
            "majority_emotion_share": safe_div(modal_emotion[1], len(emotions)),
            "emotion_entropy_normalized": normalized_entropy(
                [emotion_counts.get(e, 0) for e in emotion_order]
            ),
        },
        contingency,
    )


def consistency_metrics(
    runs: Sequence[Dict[str, List[PerTurnSentiment]]]
) -> dict:
    """Test-retest reliability across repeated runs of the same transcripts."""
    if len(runs) < 2:
        return {}
    unanimous = 0
    compared = 0
    spreads: List[float] = []
    pairwise_agreements: List[float] = []

    for dialogue_id in runs[0]:
        per_run = [{e.turn_index: e for e in (run.get(dialogue_id) or [])} for run in runs]
        shared = set(per_run[0])
        for mapping in per_run[1:]:
            shared &= set(mapping)
        for turn_index in sorted(shared):
            labels = [mapping[turn_index].emotion.value for mapping in per_run]
            values = [float(mapping[turn_index].escalation) for mapping in per_run]
            compared += 1
            if len(set(labels)) == 1:
                unanimous += 1
            spreads.append(stdev(values))
            agreements = [
                1.0 if labels[i] == labels[j] else 0.0
                for i in range(len(labels))
                for j in range(i + 1, len(labels))
            ]
            pairwise_agreements.append(mean(agreements))

    return {
        "consistency_runs": len(runs),
        "n_turns_compared": compared,
        "unanimous_emotion_rate": safe_div(unanimous, compared),
        "pairwise_emotion_agreement": mean(pairwise_agreements) if pairwise_agreements else None,
        "mean_escalation_stdev_across_runs": mean(spreads) if spreads else None,
        "max_escalation_stdev_across_runs": max(spreads) if spreads else None,
    }


def judge_metrics(
    transcripts: Sequence[Transcript],
    run: Dict[str, List[PerTurnSentiment]],
    judge_model: str,
    judge_base_url: Optional[str],
) -> Tuple[dict, List[List[object]]]:
    """Inter-model agreement with a second open-weight model on the same prompt."""
    from agent.callbacks import langfuse_callbacks
    from analysis.sentiment import SentimentBatch

    system_prompt, render_turns = _sentiment_internals()
    judge = judge_chat_model(judge_model, base_url=judge_base_url, max_tokens=1024)
    model = judge.with_structured_output(SentimentBatch)

    under_test_emotions: List[str] = []
    judge_emotions: List[str] = []
    under_test_escalations: List[float] = []
    judge_escalations: List[float] = []
    rows: List[List[object]] = []
    errors = 0
    missing = 0

    for transcript in transcripts:
        entries = run.get(transcript.dialogue_id) or []
        if not entries:
            continue
        rendered, _ = render_turns(transcript)
        if not rendered:
            continue
        prompt = [
            ("system", system_prompt),
            (
                "user",
                "Score the following turns. Return exactly one entry per turn, in input "
                "order.\n\n" + rendered,
            ),
        ]
        try:
            result = model.invoke(prompt, config={"callbacks": langfuse_callbacks()})
        except Exception:  # noqa: BLE001 — a judge failure is a counted datum, not a crash
            errors += 1
            continue
        by_index = {entry.turn_index: entry for entry in result.turns}
        for entry in entries:
            other = by_index.get(entry.turn_index)
            if other is None:
                missing += 1
                continue
            under_test_emotions.append(entry.emotion.value)
            judge_emotions.append(other.emotion.value)
            under_test_escalations.append(float(entry.escalation))
            judge_escalations.append(float(other.escalation))
            rows.append(
                [
                    transcript.dialogue_id,
                    entry.turn_index,
                    entry.emotion.value,
                    other.emotion.value,
                    round(float(entry.escalation), 4),
                    round(float(other.escalation), 4),
                ]
            )

    kappa: Optional[float] = None
    if under_test_emotions and len(set(under_test_emotions + judge_emotions)) > 1:
        from sklearn.metrics import cohen_kappa_score

        kappa = float(cohen_kappa_score(under_test_emotions, judge_emotions))

    return (
        {
            "judge_model": judge_model,
            "n_turns_judged": len(under_test_emotions),
            "judge_emotion_agreement_rate": safe_div(
                sum(1 for a, b in zip(under_test_emotions, judge_emotions) if a == b),
                len(under_test_emotions),
            ),
            "judge_emotion_cohen_kappa": kappa,
            "judge_escalation_spearman": spearman(under_test_escalations, judge_escalations),
            "judge_turns_missing": missing,
            "judge_errors": errors,
        },
        rows,
    )


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


def run_eval(
    transcripts_path: Path,
    results_dir: Path,
    modes: Sequence[str],
    split: str,
    limit: int,
    seed: int,
    consistency_runs: int,
    judge_model: Optional[str],
    judge_base_url: Optional[str],
) -> dict:
    llm_info = preflight_llm()
    model_under_test = llm_info.get("model_id") or "unknown"

    transcripts = select_transcripts(
        load_transcripts(transcripts_path),
        split=split,
        limit=limit or None,
        seed=seed,
        annotated_only="proxy" in modes,
    )
    if not transcripts:
        raise EvalUnavailable(
            "No transcripts selected. `proxy` mode needs dialogues carrying human strategy "
            "annotations (396 of 1,030 in CaSiNo; 40 of them in the test split).\n"
            "Try --split all, raise --limit, or drop `proxy` from --mode."
        )

    n_runs = consistency_runs if "consistency" in modes else 1
    runs = _score_transcripts(transcripts, n_runs)

    summary: Dict[str, object] = {
        "model_under_test": model_under_test,
        "modes": "+".join(modes),
        "n_dialogues": len(transcripts),
        "split": split,
        # The headline caveat, carried in the artifact itself so it survives
        # being copy-pasted out of context.
        "has_gold_labels": False,
        "f1_vs_gold": None,
        "gold_label_status": "NOT MEASURABLE — CaSiNo has no emotion labels; the 6-class "
        "taxonomy is defined in this repo. Numbers below are convergent validity, "
        "reliability and inter-model agreement, NOT accuracy.",
    }

    # Share of labels that are `analysis.sentiment`'s neutral fallback rather
    # than a real model output — if this is high, everything below describes the
    # fallback path, not the model.
    total_entries = sum(len(entries) for entries in runs[0].values())
    defaulted = sum(
        1
        for entries in runs[0].values()
        for entry in entries
        if entry.rationale == _DEFAULT_RATIONALE
    )
    summary["neutral_default_rate"] = safe_div(defaulted, total_entries)

    contingency: List[List[int]] = []
    if "proxy" in modes:
        proxy, contingency = proxy_metrics(transcripts, runs[0])
        summary.update(proxy)

    if "consistency" in modes:
        summary.update(consistency_metrics(runs))

    judge_rows: List[List[object]] = []
    if "judge" in modes:
        if not judge_model:
            raise EvalUnavailable(
                "`judge` mode needs a second model: --judge-model <open-weight-model-id> "
                "[--judge-base-url http://host:port/v1].\n"
                "Pointing it at the model under test would measure self-agreement, which is "
                "not evidence."
            )
        judge_summary, judge_rows = judge_metrics(
            transcripts, runs[0], judge_model, judge_base_url
        )
        judge_summary["judge_is_same_model"] = judge_model == model_under_test
        summary.update(judge_summary)

    summary_path = write_csv(
        results_dir / "sentiment.csv", list(summary.keys()), [list(summary.values())]
    )

    contingency_path = None
    if contingency:
        emotion_order = [e.value for e in Emotion]
        contingency_path = write_csv(
            results_dir / "sentiment_strategy_contingency.csv",
            ["human_strategy_group", *emotion_order, "row_total"],
            [
                [group, *row, sum(row)]
                for group, row in zip(STRATEGY_GROUPS, contingency)
            ],
        )

    turn_rows: List[List[object]] = []
    for run_index, run in enumerate(runs):
        for transcript in transcripts:
            by_turn = _strategy_by_turn(transcript)
            for entry in run.get(transcript.dialogue_id) or []:
                turn_rows.append(
                    [
                        transcript.dialogue_id,
                        run_index,
                        entry.turn_index,
                        entry.emotion.value,
                        round(float(entry.escalation), 4),
                        ";".join(by_turn.get(entry.turn_index, [])),
                        strategy_group(by_turn.get(entry.turn_index, [])),
                        int(entry.rationale == _DEFAULT_RATIONALE),
                    ]
                )
    turns_path = write_csv(
        results_dir / "sentiment_turns.csv",
        [
            "dialogue_id",
            "run_index",
            "turn_index",
            "emotion",
            "escalation",
            "human_strategies",
            "human_strategy_group",
            "is_neutral_default",
        ],
        turn_rows,
    )

    if judge_rows:
        write_csv(
            results_dir / "sentiment_judge_turns.csv",
            [
                "dialogue_id",
                "turn_index",
                "emotion_under_test",
                "emotion_judge",
                "escalation_under_test",
                "escalation_judge",
            ],
            judge_rows,
        )

    result = dict(summary)
    result.update(
        {
            "summary_csv": str(summary_path),
            "contingency_csv": str(contingency_path) if contingency_path else None,
            "turns_csv": str(turns_path),
            "caveats": [
                "no gold labels exist for this taxonomy — nothing here is accuracy or F1",
                "AUC null is 0.5; a constant scorer achieves exactly that, so read it next to "
                "n_distinct_escalation_values and modal_escalation_share",
                "consistency is measured at temperature 0.0 (fixed in analysis/sentiment.py), "
                "so it reflects runtime nondeterminism, not sampling variance; a constant "
                "labeler scores 1.0",
                "inter-model agreement is agreement, not correctness — two models can share a "
                "bias",
            ],
        }
    )
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sentiment reliability and convergent validity (no gold labels exist)."
    )
    parser.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=list(MODES),
        default=["proxy", "consistency"],
        help="Which evidence to collect (default: proxy + consistency).",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=40, help="Dialogues to score (0 = all).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--consistency-runs",
        type=int,
        default=3,
        help="Repeat scorings for the test-retest metric (>= 2 when `consistency` is on).",
    )
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-base-url", default=None)
    args = parser.parse_args(argv)

    if "consistency" in args.mode and args.consistency_runs < 2:
        parser.error("--consistency-runs must be >= 2 for `consistency` mode (or drop the mode)")

    try:
        summary = run_eval(
            transcripts_path=args.transcripts,
            results_dir=args.results_dir,
            modes=args.mode,
            split=args.split,
            limit=args.limit,
            seed=args.seed,
            consistency_runs=args.consistency_runs,
            judge_model=args.judge_model,
            judge_base_url=args.judge_base_url,
        )
    except EvalUnavailable as exc:
        return fail(exc)

    emit(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
