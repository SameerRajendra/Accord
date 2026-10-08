"""One eval scorecard — the model-card-style artifact a product team reads.

Every other module here writes its own CSV. This reads all of them and renders
**one page**: per dimension, the headline metric, its status against a stated
threshold, and its source file. It is deliberately honest about three states
that a vanity dashboard collapses into a green tick:

* **not measured** — the eval hasn't been run (its CSV is absent). Shown as a
  gap, never as a pass. This is the common state before a GPU run, and hiding
  it would make the scorecard lie by omission.
* **by design** — the metric is *not measurable on this corpus* and that is a
  documented finding, not a failure: sentiment has no gold labels (no F1), the
  outcome target is near-degenerate. Rendered with the caveat attached.
* **pass / warn / fail** — a real thresholded judgement, only where a real
  threshold exists.

Thresholds are conservative and stated inline so a reader can disagree with the
call while trusting the number. This is a reporting layer — it runs nothing and
needs no GPU or database; it only reads `results/*.csv`.

Usage::

    python -m evals.scorecard                    # render from whatever CSVs exist
    python -m evals.scorecard --out results/SCORECARD.md
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_RESULTS_DIR = Path("results")

PASS, WARN, FAIL, INFO, BY_DESIGN, NOT_MEASURED = (
    "pass", "warn", "fail", "info", "by-design", "not-measured"
)
_ICON = {
    PASS: "✅", WARN: "⚠️", FAIL: "❌", INFO: "•",
    BY_DESIGN: "◐", NOT_MEASURED: "—",
}


@dataclass
class Metric:
    dimension: str
    name: str
    value: str
    status: str
    note: str = ""
    source: str = ""


# --------------------------------------------------------------------------
# CSV reading helpers (defensive — a missing file or column is 'not measured')
# --------------------------------------------------------------------------


def _rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _num(row: Dict[str, str], key: str) -> Optional[float]:
    raw = (row or {}).get(key, "")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _fmt(x: Optional[float], places: int = 3) -> str:
    return "—" if x is None else f"{x:.{places}f}"


def _missing(dimension: str, name: str, source: str, hint: str) -> Metric:
    return Metric(dimension, name, "—", NOT_MEASURED, f"run: {hint}", source)


# --------------------------------------------------------------------------
# Per-source extractors
# --------------------------------------------------------------------------


def _serving(results_dir: Path) -> List[Metric]:
    rows = _rows(results_dir / "batching_curve.csv")
    cold = _rows(results_dir / "coldstart.csv")
    out: List[Metric] = []
    if rows:
        peak = max(rows, key=lambda r: _num(r, "output_tok_s") or 0.0)
        tok = _num(peak, "output_tok_s")
        usd = _num(peak, "usd_per_1m_output_tokens")
        conc = peak.get("concurrency", "?")
        out.append(Metric("Serving", "Peak throughput", f"{tok:.0f} tok/s @ c={conc}", INFO,
                          "characterization, not a pass/fail", "batching_curve.csv"))
        out.append(Metric("Serving", "Cost at saturation", f"${usd:.2f} / 1M out-tok", INFO,
                          "amortized GPU-second", "batching_curve.csv"))
    else:
        out.append(
            _missing("Serving", "Throughput", "batching_curve.csv", "benchmarks/bench_serving.py")
        )
    if cold:
        cs = _num(cold[0], "cold_start_s")
        out.append(Metric("Serving", "Cold start", f"{cs:.0f} s" if cs else "—", INFO,
                          "scale-to-zero wake; first-visit UX caveat", "coldstart.csv"))
    return out


def _retrieval(results_dir: Path) -> List[Metric]:
    rows = _rows(results_dir / "retrieval.csv")
    if not rows:
        return [_missing("Retrieval", "recall@5 (pgvector)", "retrieval.csv",
                         "python -m evals.retrieval_eval")]
    out: List[Metric] = []
    for retriever in ("pgvector", "graph", "hybrid"):
        match = [r for r in rows if r.get("query_mode") == "dialogue"
                 and r.get("retriever") == retriever and r.get("k") == "5"]
        if not match:
            continue
        recall = _num(match[0], "recall_at_k")
        rand = _num(match[0], "random_recall_at_k")
        struct = _num(match[0], "structural_match_at_k")
        base = _num(match[0], "structural_match_baseline")
        note = f"vs random {_pct(rand)}"
        if struct is not None:
            note += f"; struct-match {_pct(struct)} vs chance {_pct(base)}"
        # Single-gold self-retrieval — a lower bound, so INFO not pass/fail.
        out.append(Metric("Retrieval", f"recall@5 ({retriever}, dialogue)",
                          _pct(recall), INFO, note, "retrieval.csv"))
    return out


def _agent(results_dir: Path) -> List[Metric]:
    rows = _rows(results_dir / "agent_eval.csv")
    if not rows:
        return [_missing("RAG ablation", "RAG win-rate", "agent_eval.csv",
                         "python -m evals.agent_eval --limit 20")]
    row = rows[0]
    out: List[Metric] = []

    win = _num(row, "rag_win_rate")
    lo = _num(row, "rag_win_rate_wilson_lo")
    hi = _num(row, "rag_win_rate_wilson_hi")
    if win is not None:
        if lo is not None and lo > 0.5:
            status = PASS
        elif hi is not None and hi < 0.5:
            status = FAIL
        else:
            status = WARN
        ci = f" (95% CI {_pct(lo)}–{_pct(hi)})" if lo is not None else ""
        out.append(Metric("RAG ablation", "RAG vs no-RAG win-rate", _pct(win) + ci, status,
                          "pass only if the CI clears 50%", "agent_eval.csv"))

    fab = _num(row, "fabrication_rate_lexical")
    if fab is not None:
        status = PASS if fab <= 0.05 else WARN if fab <= 0.2 else FAIL
        out.append(Metric("RAG ablation", "Citation fabrication (deterministic)", _pct(fab), status,
                          "lexical grounding floor; lower is better", "agent_eval.csv"))
    return out


def _sentiment(results_dir: Path) -> List[Metric]:
    rows = _rows(results_dir / "sentiment.csv")
    if not rows:
        return [_missing("Sentiment", "reliability", "sentiment.csv",
                         "python -m evals.sentiment_eval")]
    row = rows[0]
    out: List[Metric] = [
        Metric("Sentiment", "Accuracy / F1", "not measurable", BY_DESIGN,
               "CaSiNo has no emotion labels — reliability & validity only, never an F1",
               "sentiment.csv"),
    ]
    nd = _num(row, "neutral_default_rate")
    if nd is not None:
        status = PASS if nd <= 0.15 else WARN if nd <= 0.4 else FAIL
        out.append(Metric("Sentiment", "Neutral-fallback rate", _pct(nd), status,
                          "high = metrics describe the fallback, not the model", "sentiment.csv"))
    return out


def _outcome(results_dir: Path) -> List[Metric]:
    rows = _rows(results_dir / "outcome.csv")
    if not rows:
        return [_missing("Outcome model", "accuracy lift", "outcome.csv",
                         "python -m evals.outcome_eval")]
    row = rows[0]
    out: List[Metric] = []
    lift = _num(row, "accuracy_lift")
    base = _num(row, "base_rate")
    rec = _num(row, "breakdown_recall")
    out.append(Metric("Outcome model", "Accuracy lift over base rate",
                      _fmt(lift), BY_DESIGN,
                      f"near-degenerate target (base rate {_pct(base)}); reported, not claimed",
                      "outcome.csv"))
    if rec is not None:
        out.append(
            Metric(
                "Outcome model",
                "Breakdown recall (minority class)",
                _pct(rec),
                BY_DESIGN,
                "few positives — read the confusion counts, not this alone",
                "outcome.csv",
            )
        )
    return out


def _safety(results_dir: Path) -> List[Metric]:
    out: List[Metric] = []

    inj = _rows(results_dir / "safety_injection.csv")
    if inj:
        complied = sum(1 for r in inj if (r.get("result") or "").upper() == "COMPLIED")
        total = sum(1 for r in inj if (r.get("result") or "").upper() != "ERROR")
        status = PASS if complied == 0 else FAIL
        out.append(Metric("Safety", "Prompt-injection compliance", f"{complied}/{total}", status,
                          "0 = resisted all tested payloads (a floor, not a proof)",
                          "safety_injection.csv"))
    else:
        out.append(_missing("Safety", "Prompt-injection", "safety_injection.csv",
                            "python -m evals.safety_eval --checks injection"))

    iso = _rows(results_dir / "safety_isolation.csv")
    if iso:
        viol = next(
            (r.get("result") for r in iso if r.get("check") == "isolation_violations"), None
        )
        try:
            vi = int(float(viol)) if viol not in (None, "") else None
        except ValueError:
            vi = None
        status = PASS if vi == 0 else FAIL if vi else NOT_MEASURED
        out.append(Metric("Safety", "Tenant/corpus isolation violations",
                          "—" if vi is None else str(vi), status,
                          "MUST be 0 — any cross-namespace leak is a shipping blocker",
                          "safety_isolation.csv"))
    else:
        out.append(_missing("Safety", "Tenant isolation", "safety_isolation.csv",
                            "python -m evals.safety_eval --checks isolation"))

    pii = _rows(results_dir / "safety_pii.csv")
    if pii is not None and (results_dir / "safety_pii.csv").exists():
        n = len(pii)
        status = PASS if n == 0 else WARN
        out.append(Metric("Safety", "High-severity PII hits", str(n), status,
                          "SSN/card/IBAN/account patterns in scanned surface", "safety_pii.csv"))
    return out


def _rag_triad(results_dir: Path) -> List[Metric]:
    rows = _rows(results_dir / "rag_triad.csv")
    if not rows:
        return [_missing("RAG triad", "faithfulness", "rag_triad.csv",
                         "python -m evals.rag_triad_eval")]

    def _mean(col: str) -> Optional[float]:
        vals = [_num(r, col) for r in rows if r.get("error", "") == ""]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    out: List[Metric] = []
    faith = _mean("faithfulness")
    if faith is not None:
        status = PASS if faith >= 0.8 else WARN if faith >= 0.6 else FAIL
        out.append(Metric("RAG triad", "Faithfulness (reference-free)", _pct(faith), status,
                          "LLM-judged — trust only with a strong judge + self-consistency",
                          "rag_triad.csv"))
    rel = _mean("answer_relevance_1to5")
    if rel is not None:
        status = PASS if rel >= 4.0 else WARN if rel >= 3.0 else FAIL
        out.append(Metric("RAG triad", "Answer relevance (1–5)", _fmt(rel, 2), status,
                          "LLM-judged rubric", "rag_triad.csv"))
    prec = _mean("context_precision")
    if prec is not None:
        status = PASS if prec >= 0.6 else WARN if prec >= 0.4 else FAIL
        out.append(Metric("RAG triad", "Context precision (reference-free)", _pct(prec), status,
                          "works on the live path where recall@k cannot", "rag_triad.csv"))
    return out


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------


def collect(results_dir: Path) -> List[Metric]:
    metrics: List[Metric] = []
    for fn in (_serving, _retrieval, _agent, _rag_triad, _sentiment, _outcome, _safety):
        metrics.extend(fn(results_dir))
    return metrics


def render_markdown(metrics: Sequence[Metric]) -> str:
    measured = [m for m in metrics if m.status not in (NOT_MEASURED,)]
    failing = [m for m in metrics if m.status == FAIL]
    warning = [m for m in metrics if m.status == WARN]
    not_run = [m for m in metrics if m.status == NOT_MEASURED]

    lines: List[str] = ["# Accord — eval scorecard", ""]
    lines.append(
        f"**{len(measured)}/{len(metrics)} metrics measured** · "
        f"{len(failing)} failing · {len(warning)} warnings · {len(not_run)} not yet run."
    )
    lines.append("")
    lines.append(
        "Statuses: ✅ pass · ⚠️ warn · ❌ fail · ◐ by-design (not measurable, documented) · "
        "• info (characterization) · — not measured. Thresholds are stated per row and are "
        "conservative; disagree with the call, trust the number."
    )
    lines.append("")
    lines.append("| Dimension | Metric | Value | Status | Threshold / note | Source |")
    lines.append("|---|---|---|:---:|---|---|")
    for m in metrics:
        icon = _ICON.get(m.status, m.status)
        note = m.note.replace("|", "\\|")
        lines.append(f"| {m.dimension} | {m.name} | {m.value} | {icon} | {note} | `{m.source}` |")
    lines.append("")

    if failing:
        lines.append("## ❌ Blockers")
        for m in failing:
            lines.append(f"- **{m.dimension} / {m.name}** = {m.value} — {m.note}")
        lines.append("")
    if not_run:
        lines.append("## — Not yet run")
        lines.append("These are gaps, not passes. See RUN_EVALS.md.")
        for m in not_run:
            lines.append(f"- {m.dimension} / {m.name} — {m.note}")
        lines.append("")

    lines.append(
        "> Honesty note: LLM-judged rows (RAG triad, some agent-eval) are only as trustworthy "
        "as the judge. Use a stronger self-hosted judge than the model under test and read the "
        "self-consistency number before quoting them. No human-vs-judge calibration exists yet."
    )
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render the unified eval scorecard.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--out", type=Path, default=None, help="Write markdown here (default: stdout)."
    )
    args = parser.parse_args(argv)

    metrics = collect(args.results_dir)

    # Machine-readable companion, always written next to the results.
    csv_path = args.results_dir / "scorecard.csv"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dimension", "metric", "value", "status", "note", "source"])
        for m in metrics:
            writer.writerow([m.dimension, m.name, m.value, m.status, m.note, m.source])

    md = render_markdown(metrics)
    if args.out:
        args.out.write_text(md, encoding="utf-8")
        print(f"wrote {args.out} and {csv_path}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
