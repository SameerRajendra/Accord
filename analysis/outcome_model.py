"""XGBoost breakdown-risk predictor for CaSiNo campsite negotiations.

Predicts `Outcome.agreement_reached` (did the two campers close a deal?) from
features available during or before the negotiation — never from the
resolution itself. This is the one non-LLM analysis component, deliberately:
cheap, fast (~ms inference), and it gives a genuinely calibrated probability
that can be benchmarked against an LLM zero-shot baseline (see
`evals/outcome_eval.py`).

Feature design — no leakage, order-invariant
--------------------------------------------
CaSiNo's two negotiators are labeled `agent_1`/`agent_2` **arbitrarily** (which
MTurk worker is "1" carries no meaning), so every feature here is symmetric
across the two parties — min/mean aggregations and conflict flags, never a raw
`agent_1_x` column that would let the model learn a labeling artifact.

Features come from three leakage-free places:

1. **Preferences** (known before any turn): the priority rankings — in
   particular whether both parties rank the *same* issue High (a structural
   conflict that should predict harder negotiations).
2. **Personality** (pre-negotiation survey): SVO (proself/prosocial) and the
   Big-Five traits most plausibly tied to reaching agreement (agreeableness,
   emotional-stability, openness), aggregated order-invariantly.
3. **Process** (the shape of the dialogue, not its resolution): message-turn
   count and offer (Submit-Deal) count.

Explicitly excluded: `Outcome.final_deal`, `Outcome.points`,
`Party.satisfaction`/`opponent_likeness` (all outcomes), and any count of the
resolution actions themselves (accept/reject/quit) — counting "was there an
accept" would just re-encode the label. **Strategy annotations are also
excluded**: only ~38% of dialogues carry them, so their presence is a dataset
*selection* artifact, not a property of the negotiation — including them would
leak the annotated-subset boundary. Strategies stay on the RAG/analysis side.

Calibration
-----------
Hand-rolled isotonic regression on held-out validation predictions, not
sklearn's `CalibratedClassifierCV(cv="prefit")`. That parameter path has
churned across sklearn versions (`base_estimator` -> `estimator`, and
`cv="prefit"` deprecated in 1.6 for `FrozenEstimator`) — `IsotonicRegression`
is a simpler, version-stable surface that does the same thing here.

Class-balance caveat
--------------------
CaSiNo is a cooperative task, so most dialogues reach agreement — the positive
class dominates. `evals/outcome_eval.py` reports the base rate alongside F1/AUC
so a high accuracy that just tracks the majority class is visible, not hidden.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from data.schema import Action, Transcript

# Big-Five traits used as features (order-invariant aggregates below).
_BIG_FIVE_KEYS = ("agreeableness", "emotional-stability", "openness-to-experiences")

DEFAULT_XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


def _high_issue(party) -> str:
    """The issue this party ranks 'High' (priorities is issue->priority)."""
    for issue, level in (party.priorities or {}).items():
        if level == "High":
            return issue
    return ""


def _svo(party) -> str:
    return ((party.metadata or {}).get("personality") or {}).get("svo", "")


def _big_five(party, trait: str):
    bf = ((party.metadata or {}).get("personality") or {}).get("big-five") or {}
    val = bf.get(trait)
    return float(val) if val is not None else float("nan")


def _agg(values: list, fn) -> float:
    vals = [v for v in values if v == v]  # drop NaN
    return float(fn(vals)) if vals else float("nan")


def extract_features(transcript: Transcript) -> dict:
    """Engineer one feature row (pure — no ML deps, unit-testable alone).

    Symmetric across parties: which negotiator is `agent_1` is arbitrary."""
    parties = transcript.parties
    message_turns = [t for t in transcript.turns if t.action is None]
    offer_turns = [t for t in transcript.turns if t.action is Action.SUBMIT_DEAL]

    high_issues = [_high_issue(p) for p in parties if _high_issue(p)]
    high_conflict = 1.0 if len(high_issues) == 2 and high_issues[0] == high_issues[1] else 0.0

    svos = [_svo(p) for p in parties]
    num_proself = float(sum(1 for s in svos if s == "proself"))

    features = {
        "num_message_turns": float(len(message_turns)),
        "num_offers": float(len(offer_turns)),
        "high_conflict": high_conflict,
        "num_proself": num_proself,
        "both_proself": 1.0 if num_proself == 2 else 0.0,
    }
    for trait in _BIG_FIVE_KEYS:
        vals = [_big_five(p, trait) for p in parties]
        key = trait.replace("-", "_")
        features[f"mean_{key}"] = _agg(vals, np.mean)
        features[f"min_{key}"] = _agg(vals, np.min)
    return features


def build_feature_matrix(transcripts: list) -> tuple:
    """Rows -> (X, y). All features are numeric and share a fixed schema across
    splits (no categorical one-hots — CaSiNo's domain is fixed)."""
    rows = [extract_features(t) for t in transcripts]
    df = pd.DataFrame(rows)
    y = pd.Series([int(t.outcome.agreement_reached) for t in transcripts], name="agreement_reached")
    return df, y


def train_outcome_model(X_train: pd.DataFrame, y_train: pd.Series, **xgb_overrides) -> xgb.XGBClassifier:
    params = {**DEFAULT_XGB_PARAMS, **xgb_overrides}
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def calibrate(model: xgb.XGBClassifier, X_val: pd.DataFrame, y_val: pd.Series) -> IsotonicRegression:
    """Fit isotonic calibration on the model's raw validation-set scores."""
    raw_proba = model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_proba, y_val)
    return calibrator


def predict_calibrated(model: xgb.XGBClassifier, calibrator: IsotonicRegression, X: pd.DataFrame) -> np.ndarray:
    raw_proba = model.predict_proba(X)[:, 1]
    return calibrator.predict(raw_proba)


def save_model(
    model: xgb.XGBClassifier, calibrator: IsotonicRegression, feature_columns: list[str], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "calibrator": calibrator, "feature_columns": feature_columns}, path)


def load_model(path: Path) -> tuple[xgb.XGBClassifier, IsotonicRegression, list[str]]:
    obj = joblib.load(path)
    return obj["model"], obj["calibrator"], obj["feature_columns"]
