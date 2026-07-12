"""Outcome model evaluation: F1, ROC-AUC, and a calibration curve.

Trains the XGBoost breakdown-risk model on the `train` split, calibrates on
`validation`, and reports metrics on the held-out `test` split — the split
CraigslistBargain already carries per-dialogue (see `Transcript.metadata["split"]`,
set by `data/ingest_craigslist.py`). Test is touched exactly once, at the end.

Ships two committed artifacts:
- `results/outcome.csv` — one summary row (n, F1, accuracy, ROC-AUC).
- `results/outcome_calibration.csv` — the reliability-diagram bins (mean
  predicted probability vs. actual positive fraction per bin), so the
  calibration claim is independently checkable, not just asserted.

The trained model itself is saved to `models/outcome_model.joblib` —
gitignored (a regeneratable binary, not an eval output) but is what Phase 4's
API would load for real-time inference.

Usage::

    python -m evals.outcome_eval
    python -m evals.outcome_eval --input data/processed/craigslist_bargain.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from analysis.outcome_model import (
    build_feature_matrix,
    calibrate,
    predict_calibrated,
    save_model,
    train_outcome_model,
)
from data.schema import Transcript

DEFAULT_INPUT = Path("data/processed/craigslist_bargain.jsonl")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_MODEL_PATH = Path("models/outcome_model.joblib")

N_CALIBRATION_BINS = 10


def _load_transcripts(path: Path) -> list[Transcript]:
    return [
        Transcript.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _split_by(transcripts: list[Transcript], split: str) -> list[Transcript]:
    return [t for t in transcripts if t.metadata.get("split") == split]


def run_eval(input_path: Path, results_dir: Path, model_path: Path) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run ingestion first:\n"
            f"  python -m data.ingest_craigslist --download"
        )

    transcripts = _load_transcripts(input_path)
    train = _split_by(transcripts, "train")
    validation = _split_by(transcripts, "validation")
    test = _split_by(transcripts, "test")
    if not train or not validation or not test:
        raise ValueError(
            f"Expected all three splits present; got train={len(train)}, "
            f"validation={len(validation)}, test={len(test)}. Did ingestion "
            f"run with --split all (the default)?"
        )

    X_train, y_train = build_feature_matrix(train)
    X_val, y_val = build_feature_matrix(validation)
    X_test, y_test = build_feature_matrix(test)

    model = train_outcome_model(X_train, y_train)
    calibrator = calibrate(model, X_val, y_val)

    test_proba = predict_calibrated(model, calibrator, X_test)
    test_pred = (test_proba >= 0.5).astype(int)

    f1 = f1_score(y_test, test_pred)
    accuracy = accuracy_score(y_test, test_pred)
    roc_auc = roc_auc_score(y_test, test_proba)
    prob_true, prob_pred = calibration_curve(y_test, test_proba, n_bins=N_CALIBRATION_BINS, strategy="uniform")

    save_model(model, calibrator, list(X_train.columns), model_path)

    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "outcome.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["n_train", "n_validation", "n_test", "f1", "accuracy", "roc_auc"])
        writer.writerow([len(train), len(validation), len(test), f1, accuracy, roc_auc])

    calibration_path = results_dir / "outcome_calibration.csv"
    with calibration_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["bin_index", "mean_predicted_probability", "actual_positive_fraction"])
        for i, (true_frac, pred_mean) in enumerate(zip(prob_true, prob_pred)):
            writer.writerow([i, pred_mean, true_frac])

    return {
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "f1": f1,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "summary_csv": str(summary_path),
        "calibration_csv": str(calibration_path),
        "model_path": str(model_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the XGBoost outcome model.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args(argv)

    summary = run_eval(args.input, args.results_dir, args.model_path)
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
