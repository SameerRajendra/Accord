"""Assemble everything in `results/` into one markdown report.

The report is built from a **manifest of every artifact DESIGN.md §7 promises**,
not from a directory listing. That ordering matters: a directory listing can
only show what exists, so a promised-but-unrun benchmark would vanish from the
summary and the reader would never know to ask. Here, a missing artifact gets a
row marked `MISSING` and the exact command that produces it, and an artifact
that exists but left a metric blank renders that metric as `not measured` rather
than dropping it.

The same principle applies inside each block: metrics that can be gamed are
printed next to what they must be read against — accuracy next to the
majority-class base rate, a win rate next to the coin flip, an AUC next to 0.5.

Deterministic given the CSVs: the report carries each source file's modification
time rather than a wall-clock "generated at", so re-running it without re-running
an eval produces a byte-identical file.

Usage::

    python -m evals.report                       # writes results/REPORT.md, prints it
    python -m evals.report --stdout-only
    python -m evals.report --results-dir results --output docs/RESULTS.md
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from evals._common import DEFAULT_RESULTS_DIR

MISSING = "**MISSING**"
NOT_MEASURED = "*not measured*"
UNDEFINED = "n/a (undefined)"


@dataclass
class Metric:
    """One number to surface, plus whatever keeps it honest."""

    label: str
    column: str
    kind: str = "float"  # float | pct | int | seconds | raw
    baseline_label: str = ""
    baseline_column: str = ""
    note: str = ""


@dataclass
class Artifact:
    """An expected file in `results/` and how to render (or miss) it."""

    filename: str
    title: str
    layer: str
    command: str
    metrics: List[Metric] = field(default_factory=list)
    table_columns: List[str] = field(default_factory=list)
    max_table_rows: int = 12
    supporting: bool = False
    blurb: str = ""
    renderer: str = "metrics"  # metrics | table | retrieval | none


MANIFEST: List[Artifact] = [
    # ---------------- serving ----------------
    Artifact(
        filename="coldstart.csv",
        title="Cold start vs warm",
        layer="Serving",
        command="modal run benchmarks/bench_serving.py::bench --gpu-type H100 "
        "--model Qwen/Qwen2.5-7B-Instruct",
        blurb="The scale-to-zero tradeoff, reported as two numbers rather than one average.",
        metrics=[
            Metric("Cold start (container enter → /v1/models 200)", "cold_start_s", "seconds"),
            Metric("First warm request p50", "first_warm_request_p50_s", "seconds"),
            Metric("Cost of one cold hit", "cold_start_usd", "float"),
            Metric("GPU", "gpu", "raw"),
            Metric("Model", "model", "raw"),
        ],
    ),
    Artifact(
        filename="batching_curve.csv",
        title="Continuous-batching curve (latency ↔ throughput)",
        layer="Serving",
        command="modal run benchmarks/bench_serving.py::bench --gpu-type H100 "
        "--model Qwen/Qwen2.5-7B-Instruct --concurrencies 1,2,4,8,16,32,64",
        blurb="Aggregate throughput climbs with concurrency while per-request latency degrades "
        "— the trade the number exists to quantify.",
        renderer="table",
        table_columns=[
            "concurrency",
            "ttft_p50_ms",
            "ttft_p95_ms",
            "e2e_p50_ms",
            "e2e_p95_ms",
            "output_tok_s",
            "usd_per_1m_output_tokens",
        ],
    ),
    Artifact(
        filename="latency_cost.csv",
        title="End-to-end /analyze latency and $/request",
        layer="Serving",
        command="modal run benchmarks/bench_serving.py::bench --gpu-type H100 "
        "--model Qwen/Qwen2.5-7B-Instruct",
        blurb="p50/p95 of the full pipeline and Modal GPU-second cost amortized per request.",
        renderer="table",
        table_columns=[],
    ),
    Artifact(
        filename="per_request_trace.csv",
        title="Per-request trace (raw)",
        layer="Serving",
        command="modal run benchmarks/bench_serving.py::bench --gpu-type H100 "
        "--model Qwen/Qwen2.5-7B-Instruct",
        supporting=True,
        renderer="none",
        blurb="Raw per-request TTFT/e2e samples behind the batching curve.",
    ),
    # ---------------- task quality ----------------
    Artifact(
        filename="outcome.csv",
        title="Breakdown-risk model (XGBoost)",
        layer="Task quality",
        command="python -m evals.outcome_eval",
        blurb="CaSiNo is cooperative (~97.6% agreement), so accuracy here is mostly the class "
        "ratio. Read the lift and the minority-class recall, not the accuracy.",
        metrics=[
            Metric("Test dialogues", "n_test", "int"),
            Metric(
                "Accuracy",
                "accuracy",
                "pct",
                baseline_label="majority-class base rate",
                baseline_column="base_rate",
            ),
            Metric("Accuracy lift over base rate", "accuracy_lift", "pct"),
            Metric(
                "Breakdown recall (minority class)",
                "breakdown_recall",
                "pct",
                note="of the negotiations that actually broke down, how many were caught",
            ),
            Metric("F1", "f1", "float", note="dominated by the majority class"),
            Metric("ROC-AUC", "roc_auc", "float", note="computed over very few negatives"),
            Metric("Confusion tn / fp / fn / tp", "tn", "raw"),
        ],
    ),
    Artifact(
        filename="retrieval.csv",
        title="Retrieval (recall@k / MRR@k)",
        layer="Task quality",
        command="python -m evals.retrieval_eval",
        blurb="Single-gold labeling built from corpus metadata — see the module docstring for "
        "how relevance is assigned and where it is weak. `random` is the chance baseline; "
        "`tfidf` is bag-of-words on the same corpus; `summary` mode is a control, not a "
        "quality claim.",
        renderer="retrieval",
    ),
    Artifact(
        filename="agent_eval.csv",
        title="RAG vs no-RAG ablation + citation grounding",
        layer="Task quality",
        command="python -m evals.agent_eval --limit 20",
        blurb="Preference is LLM-judged and carries every LLM-judge caveat. The citation "
        "grounding numbers do not depend on the judge: the lexical check is deterministic, "
        "and the no-RAG arm is a pure-fabrication control (nothing was retrieved, so any "
        "citation there is invented by construction).",
        metrics=[
            Metric("Transcripts (both arms succeeded)", "n_paired_ok", "int"),
            Metric(
                "RAG win rate",
                "rag_win_rate",
                "pct",
                baseline_label="coin flip",
                baseline_column="coin_flip_baseline",
            ),
            Metric("… 95% Wilson interval (low)", "rag_win_rate_wilson_lo", "pct"),
            Metric("… 95% Wilson interval (high)", "rag_win_rate_wilson_hi", "pct"),
            Metric(
                "… exact binomial p vs coin flip",
                "binomial_p_vs_coin_flip",
                "float",
                note="on n_decisive comparisons; ties excluded",
            ),
            Metric("Tie rate", "tie_rate", "pct"),
            Metric(
                "Judge position-A pick rate",
                "position_a_pick_rate",
                "pct",
                baseline_label="unbiased judge",
                note="far from 50% means the preference column is measuring slot order",
            ),
            Metric("Judge is the model under test", "judge_is_same_model", "raw",
                   note="true ⇒ self-preference bias applies in full"),
            Metric("Latency p50, RAG", "rag_latency_p50_s", "seconds"),
            Metric("Latency p50, no-RAG", "norag_latency_p50_s", "seconds"),
            Metric("Latency delta (p50)", "latency_delta_p50_s", "seconds"),
            Metric("Citations, RAG arm", "n_citations_rag", "int"),
            Metric(
                "Fabrication rate (lexical, deterministic)",
                "fabrication_rate_lexical",
                "pct",
                note="cited id unresolvable, or a tactic/outcome claim absent from the case text",
            ),
            Metric("Fabrication rate (judge entailment)", "fabrication_rate_judge", "pct"),
            Metric("Cited id not in the retrieved set", "citation_unresolvable_rate", "pct"),
            Metric(
                "Cited id needed normalization",
                "citation_needed_normalization_rate",
                "pct",
                note="resolves to a real case but the id as written does not exist",
            ),
            Metric("`grounded_case_ids` empty, RAG arm", "rag_empty_citations_rate", "pct"),
            Metric(
                "no-RAG runs that cited a case anyway",
                "norag_phantom_citation_rate",
                "pct",
                note="pure-fabrication control: nothing was retrieved in this arm",
            ),
            Metric("Mean top-1 retrieval similarity", "mean_top1_retrieval_score", "float"),
        ],
    ),
    Artifact(
        filename="sentiment.csv",
        title="Sentiment / escalation",
        layer="Task quality",
        command="python -m evals.sentiment_eval",
        blurb="No gold labels exist for this taxonomy, so there is no accuracy row to print. "
        "What follows is convergent validity against human strategy annotations, "
        "test–retest reliability, and inter-model agreement.",
        metrics=[
            Metric("Dialogues scored", "n_dialogues", "int"),
            Metric("Turns scored", "n_turns_scored", "int"),
            Metric(
                "F1 vs gold labels",
                "f1_vs_gold",
                "float",
                note="not measurable — CaSiNo carries no emotion labels (see gold_label_status)",
            ),
            Metric(
                "AUC: escalation ranks adversarial (uv-part) above cooperative turns",
                "auc_escalation_adversarial_vs_cooperative",
                "float",
                baseline_label="null",
                baseline_column="auc_null_baseline",
            ),
            Metric("… adversarial (uv-part) turns in sample", "n_adversarial_turns", "int"),
            Metric("… cooperative turns in sample", "n_cooperative_turns", "int"),
            Metric(
                "Cramér's V, emotion × human strategy group",
                "cramers_v_emotion_by_strategy_group",
                "float",
            ),
            Metric(
                "AUC: escalation predicts breakdown",
                "auc_escalation_predicts_breakdown",
                "float",
                note="NaN when the split holds too few breakdowns to support the statistic",
            ),
            Metric("Unanimous emotion across repeat runs", "unanimous_emotion_rate", "pct"),
            Metric("Mean escalation stdev across runs", "mean_escalation_stdev_across_runs",
                   "float"),
            Metric(
                "Majority emotion share",
                "majority_emotion_share",
                "pct",
                note="degeneracy check — a constant labeler scores 100% here and 1.0 on "
                "consistency",
            ),
            Metric("Emotion entropy (normalized)", "emotion_entropy_normalized", "float"),
            Metric(
                "Neutral-default rate",
                "neutral_default_rate",
                "pct",
                note="share of labels that are the fallback, not a model output",
            ),
            Metric("Judge agreement (Cohen's κ)", "judge_emotion_cohen_kappa", "float"),
            Metric("Judge escalation correlation (Spearman)", "judge_escalation_spearman",
                   "float"),
        ],
    ),
    Artifact(
        filename="outcome_calibration.csv",
        title="Outcome-model reliability diagram (raw)",
        layer="Task quality",
        command="python -m evals.outcome_eval",
        supporting=True,
        renderer="table",
        table_columns=["bin_index", "mean_predicted_probability", "actual_positive_fraction"],
        blurb="Per-bin predicted probability vs observed frequency.",
    ),
    Artifact(
        filename="retrieval_queries.csv",
        title="Per-query retrieval detail (raw)",
        layer="Task quality",
        command="python -m evals.retrieval_eval",
        supporting=True,
        renderer="none",
        blurb="Gold rank and top-1 hit for every query — every aggregate above is recomputable "
        "from this.",
    ),
    Artifact(
        filename="agent_eval_runs.csv",
        title="Per-transcript ablation detail (raw)",
        layer="Task quality",
        command="python -m evals.agent_eval",
        supporting=True,
        renderer="none",
        blurb="Both arms per transcript: latency, tactic, judge verdict and its stated reason.",
    ),
    Artifact(
        filename="agent_eval_citations.csv",
        title="Per-citation grounding detail (raw)",
        layer="Task quality",
        command="python -m evals.agent_eval",
        supporting=True,
        renderer="none",
        blurb="Every cited case_id with both verdicts and the sentence that triggered any "
        "lexical flag — each fabrication finding is auditable by hand.",
    ),
    Artifact(
        filename="sentiment_turns.csv",
        title="Per-turn sentiment labels (raw)",
        layer="Task quality",
        command="python -m evals.sentiment_eval",
        supporting=True,
        renderer="none",
        blurb="Every label from every repeat run, with the human strategy annotation alongside.",
    ),
    Artifact(
        filename="sentiment_strategy_contingency.csv",
        title="Emotion × human strategy group",
        layer="Task quality",
        command="python -m evals.sentiment_eval",
        supporting=True,
        renderer="table",
        table_columns=[],
        blurb="Counts, so any association measure can be recomputed independently.",
    ),
]


# --------------------------------------------------------------------------
# Reading + formatting
# --------------------------------------------------------------------------


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _is_blank(value: Optional[str]) -> bool:
    return value is None or str(value).strip() == ""


def _is_nan(value: str) -> bool:
    return str(value).strip().lower() in ("nan", "-nan", "none")


def format_value(value: Optional[str], kind: str) -> str:
    """Render one cell, distinguishing 'not measured' from 'undefined' from a number."""
    if _is_blank(value):
        return NOT_MEASURED
    text = str(value).strip()
    if _is_nan(text):
        return UNDEFINED
    if kind == "raw":
        return f"`{text}`"
    try:
        number = float(text)
    except ValueError:
        return f"`{text}`"
    if kind == "pct":
        return f"{number * 100:.1f}%"
    if kind == "int":
        return f"{int(round(number)):,}"
    if kind == "seconds":
        return f"{number:.2f} s"
    if number == int(number) and abs(number) < 1e15:
        return f"{int(number):,}"
    if abs(number) >= 1000:
        return f"{number:,.1f}"
    if abs(number) < 0.001:
        return f"{number:.2e}"
    return f"{number:.4g}"


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(h) for h in header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(c) for c in row) + " |")
    return lines


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------


def render_metrics(artifact: Artifact, rows: List[Dict[str, str]]) -> List[str]:
    if not rows:
        return [f"_{artifact.filename} exists but has no data rows._", ""]
    row = rows[0]
    table_rows: List[Sequence[str]] = []
    for metric in artifact.metrics:
        if metric.column not in row:
            # The eval that wrote this CSV predates the metric (or never
            # computed it). Say so instead of dropping the line.
            value = NOT_MEASURED
        else:
            value = format_value(row.get(metric.column), metric.kind)
        # Special case: the outcome model's confusion counts read as one cell.
        if metric.column == "tn" and all(k in row for k in ("tn", "fp", "fn", "tp")):
            value = f"`{row['tn']} / {row['fp']} / {row['fn']} / {row['tp']}`"

        against = ""
        if metric.baseline_column and metric.baseline_column in row:
            against = f"{metric.baseline_label or 'baseline'}: " + format_value(
                row.get(metric.baseline_column), metric.kind
            )
        elif metric.baseline_label:
            against = metric.baseline_label
        if metric.note:
            against = f"{against} — {metric.note}" if against else metric.note
        table_rows.append([metric.label, value, against])
    return markdown_table(["Metric", "Value", "Read alongside"], table_rows) + [""]


def render_table(artifact: Artifact, rows: List[Dict[str, str]]) -> List[str]:
    if not rows:
        return [f"_{artifact.filename} exists but has no data rows._", ""]
    columns = [c for c in artifact.table_columns if c in rows[0]] or list(rows[0].keys())
    shown = rows[: artifact.max_table_rows]
    lines = markdown_table(
        columns, [[format_value(row.get(c), "raw" if not _numeric(row.get(c)) else "float")
                   for c in columns] for row in shown]
    )
    if len(rows) > len(shown):
        lines.append("")
        lines.append(f"_{len(rows) - len(shown)} further rows in `{artifact.filename}`._")
    return lines + [""]


def _numeric(value: Optional[str]) -> bool:
    if _is_blank(value):
        return False
    try:
        float(str(value))
    except ValueError:
        return False
    return True


def render_retrieval(artifact: Artifact, rows: List[Dict[str, str]]) -> List[str]:
    """Pivot retrieval.csv to (query_mode, retriever) × recall@k, MRR at max k."""
    if not rows:
        return [f"_{artifact.filename} exists but has no data rows._", ""]
    ks: List[str] = []
    for row in rows:
        if row.get("k") and row["k"] not in ks:
            ks.append(row["k"])
    ks.sort(key=lambda k: int(k) if k.isdigit() else 0)

    grouped: Dict[str, Dict[str, Dict[str, str]]] = {}
    for row in rows:
        key = f"{row.get('query_mode', '?')} / {row.get('retriever', '?')}"
        grouped.setdefault(key, {})[row.get("k", "")] = row

    max_k = ks[-1] if ks else "?"
    header = ["query mode / retriever", "queries", *[f"recall@{k}" for k in ks], f"MRR@{max_k}"]
    table_rows: List[Sequence[str]] = []
    for key, by_k in grouped.items():
        any_row = next(iter(by_k.values()))
        cells = [key, format_value(any_row.get("n_queries"), "int")]
        for k in ks:
            cells.append(format_value((by_k.get(k) or {}).get("recall_at_k"), "pct"))
        deepest = by_k.get(max_k) or {}
        cells.append(format_value(deepest.get("mrr_at_k"), "float"))
        table_rows.append(cells)

    lines = markdown_table(header, table_rows)
    notes = {row.get("labeling_note", "") for row in rows if row.get("labeling_note")}
    if notes:
        lines.append("")
        for note in sorted(notes):
            lines.append(f"- {note}")
    return lines + [""]


RENDERERS = {
    "metrics": render_metrics,
    "table": render_table,
    "retrieval": render_retrieval,
}


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------


def _mtime(path: Path) -> str:
    stamp = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
    return stamp.strftime("%Y-%m-%d %H:%M UTC")


def build_report(results_dir: Path) -> str:
    lines: List[str] = [
        "# Accord — measured results",
        "",
        f"Assembled from `{results_dir.as_posix()}/` by `python -m evals.report`.",
        "",
        "Every artifact DESIGN.md §7 promises is listed below, present or not. A benchmark "
        "that has not been run is marked " + MISSING + " with the command that produces it; a "
        "metric an eval left blank is marked " + NOT_MEASURED + ". Nothing promised is "
        "silently omitted, and numbers that a degenerate model could score well on are "
        "printed next to what they must be read against.",
        "",
        "## Status",
        "",
    ]

    status_rows: List[Sequence[str]] = []
    present: Dict[str, List[Dict[str, str]]] = {}
    missing: List[Artifact] = []
    for artifact in MANIFEST:
        path = results_dir / artifact.filename
        if path.exists():
            rows = read_csv(path)
            present[artifact.filename] = rows
            status = "present" if rows else "present (empty)"
            status_rows.append(
                [
                    f"`{artifact.filename}`",
                    artifact.layer,
                    status,
                    f"{len(rows):,}",
                    _mtime(path),
                    f"`{artifact.command}`",
                ]
            )
        else:
            missing.append(artifact)
            status_rows.append(
                [f"`{artifact.filename}`", artifact.layer, MISSING, "—", "—",
                 f"`{artifact.command}`"]
            )
    lines += markdown_table(
        ["Artifact", "Layer", "Status", "Rows", "Last written", "Produced by"], status_rows
    )
    lines.append("")

    for layer in ("Serving", "Task quality"):
        layer_artifacts = [a for a in MANIFEST if a.layer == layer and not a.supporting]
        if not layer_artifacts:
            continue
        lines += [f"## {layer}", ""]
        for artifact in layer_artifacts:
            lines += [f"### {artifact.title}", ""]
            if artifact.blurb:
                lines += [artifact.blurb, ""]
            rows = present.get(artifact.filename)
            if rows is None:
                lines += [
                    f"{MISSING} — `{artifact.filename}` has not been produced.",
                    "",
                    "```bash",
                    artifact.command,
                    "```",
                    "",
                ]
                continue
            renderer = RENDERERS.get(artifact.renderer)
            lines += renderer(artifact, rows) if renderer else [""]

    supporting = [a for a in MANIFEST if a.supporting]
    if supporting:
        lines += ["## Supporting raw artifacts", ""]
        support_rows: List[Sequence[str]] = []
        for artifact in supporting:
            rows = present.get(artifact.filename)
            support_rows.append(
                [
                    f"`{artifact.filename}`",
                    "present" if rows is not None else MISSING,
                    f"{len(rows):,}" if rows is not None else "—",
                    artifact.blurb,
                ]
            )
        lines += markdown_table(["File", "Status", "Rows", "What it holds"], support_rows)
        lines.append("")

    known = {a.filename for a in MANIFEST}
    extras = sorted(
        p.name for p in results_dir.glob("*.csv") if p.name not in known
    ) if results_dir.exists() else []
    if extras:
        lines += [
            "## Other files present in `results/`",
            "",
            "Not part of the DESIGN.md §7 manifest — listed so the directory is fully "
            "accounted for.",
            "",
        ]
        lines += [f"- `{name}`" for name in extras]
        lines.append("")

    if missing:
        lines += [
            "## Not measured yet",
            "",
            "These are promised in DESIGN.md §7 and have no committed artifact. Until they "
            "exist, the corresponding claims are unsupported.",
            "",
        ]
        for artifact in missing:
            lines.append(f"- **{artifact.title}** (`{artifact.filename}`) — `{artifact.command}`")
        lines.append("")

    lines += [
        "## How to read this",
        "",
        "- **Accuracy on this corpus is mostly the class ratio.** CaSiNo reaches agreement in "
        "~97.6% of dialogues, so any accuracy figure is printed next to the majority-class "
        "base rate and the lift over it.",
        "- **Retrieval relevance labels are constructed, not human** (except the `strategy` "
        "query mode). One document counts as relevant per query, so recall@k understates "
        "usefulness; see `evals/retrieval_eval.py` for the labeling strategy and its "
        "weaknesses.",
        "- **The RAG win rate is an LLM judgement**, with a coin-flip baseline, a Wilson "
        "interval and an exact binomial p-value beside it. The citation-grounding numbers are "
        "the ones that do not depend on the judge.",
        "- **Sentiment has no gold labels and therefore no accuracy.** What is reported is "
        "convergent validity against human strategy annotations, test–retest reliability, and "
        "inter-model agreement — each named for what it is.",
        "- **Stretch items are absent on purpose.** The TensorRT-LLM quantization ablation "
        "(DESIGN.md §7) is not built; no number for it appears here.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble results/ into one markdown report.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown destination (default: <results-dir>/REPORT.md).",
    )
    parser.add_argument("--stdout-only", action="store_true", help="Print without writing a file.")
    args = parser.parse_args(argv)

    if not args.results_dir.exists():
        # Not an EvalUnavailable: an empty results dir is a legitimate state, and
        # the report's whole job is to say what is missing.
        args.results_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(args.results_dir)
    if not args.stdout_only:
        destination = args.output or (args.results_dir / "REPORT.md")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
