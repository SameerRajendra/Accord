"""Serving-time wrapper around `outcome_model` — loads a trained artifact once.

The training pipeline (train + calibrate + save) lives in `outcome_model.py`
and is driven by `evals/outcome_eval.py`. This module is the *inference*
surface used by the API/agent/MCP layer: `predict_from_transcript(transcript)`
returns a calibrated breakdown-risk probability.

Model artifact path defaults to `models/outcome_model.joblib` — the path
`evals/outcome_eval.py` and Modal's `train_outcome` function both write to.
`models/` is gitignored (large binary, regeneratable). Override with env
`OUTCOME_MODEL_PATH`. Missing artifact → callers get a `MissingModelError`
so the API can degrade gracefully rather than 500.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from data.schema import Transcript

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/outcome_model.joblib")


class MissingModelError(RuntimeError):
    """Raised when the outcome-model artifact is not on disk."""


@lru_cache(maxsize=1)
def _load():
    from analysis.outcome_model import load_model

    path_env = os.environ.get("OUTCOME_MODEL_PATH")
    path = Path(path_env) if path_env else DEFAULT_MODEL_PATH
    if not path.exists():
        raise MissingModelError(
            f"outcome model not found at {path}; "
            "run `python -m evals.outcome_eval` first "
            "(or `modal run infra/modal/app.py::train_outcome` for the cloud path)"
        )
    logger.info("Loading outcome model from %s", path)
    return load_model(path)


def predict_from_transcript(transcript: Transcript) -> Optional[float]:
    """Return P(agreement_reached) in [0, 1], or None if the model isn't loaded.

    Callers should treat `None` as "outcome prediction unavailable this run"
    and continue — the API contract still returns sentiment + behavior +
    retrieval + recommendation even when this fails.
    """
    import pandas as pd

    from analysis.outcome_model import extract_features, predict_calibrated

    try:
        model, calibrator, feature_columns = _load()
    except MissingModelError as exc:
        logger.warning("outcome model unavailable: %s", exc)
        return None

    features = extract_features(transcript)
    # Reindex to the trained column set so category one-hots (present at
    # train time but absent from this single-row inference) don't drop
    # features silently.
    row = pd.DataFrame([features])
    # Apply the same category one-hot expansion as build_feature_matrix.
    from analysis.outcome_model import CATEGORY_VALUES

    category = row.pop("category").iloc[0] if "category" in row.columns else "unknown"
    for cat in CATEGORY_VALUES:
        row[f"category_{cat}"] = 1.0 if category == cat else 0.0
    row = row.reindex(columns=feature_columns, fill_value=0.0)

    proba = predict_calibrated(model, calibrator, row)
    # P(agreement_reached=1); breakdown risk is (1 - P). Return the agreement
    # probability so the API is symmetric with the trained label.
    return float(proba[0])
