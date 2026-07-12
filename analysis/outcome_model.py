"""XGBoost breakdown-risk predictor for CraigslistBargain negotiations.

Predicts `Outcome.agreement_reached` (did the negotiation close a deal?) from
features available during or before the negotiation — never from the
resolution itself. This is the one non-LLM analysis component, deliberately:
cheap, fast (~ms inference), and it gives a genuinely calibrated probability
that can be benchmarked against an LLM zero-shot baseline (see
`evals/outcome_eval.py`).

Feature design — no leakage
----------------------------
Features come from two places only:

1. **Setup** (known before any turn is spoken): buyer/seller `Target` price,
   the public listed price, item category.
2. **Process** (the shape of the dialogue, not its resolution): message-turn
   count, offer count, who spoke first, counts of *meaningful* dialogue-act
   intents (disagree/agree/inquiry/inform/vague-price/init-price/
   counter-price/intro — see `MEANINGFUL_INTENTS` in `build_case_corpus.py`).

Explicitly excluded: `Outcome.final_deal`, and any count of the resolution
actions themselves (accept/reject/quit). Counting "was there an accept turn"
would just be re-encoding the label — degenerate, not predictive. `num_offers`
is fine to keep, since an offer can still be rejected (verified in the
ingestion tests) — it doesn't trivially determine the outcome.

Calibration
-----------
Hand-rolled isotonic regression on held-out validation predictions, not
sklearn's `CalibratedClassifierCV(cv="prefit")`. That parameter path has
churned across sklearn versions (`base_estimator` -> `estimator`, and
`cv="prefit"` deprecated in 1.6 for `FrozenEstimator`) — since the exact
sklearn version on the target cluster isn't known ahead of time,
`IsotonicRegression.fit`/`.predict` is a simpler, version-stable surface that
does the same thing this project needs, transparently.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from data.build_case_corpus import MEANINGFUL_INTENTS
from data.schema import Action, Transcript

CATEGORY_VALUES = ["housing", "furniture", "electronics", "bike", "car", "phone"]

DEFAULT_XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


def _party_meta(transcript: Transcript, role: str, key: str):
    party = next((p for p in transcript.parties if p.party_id == role), None)
    return party.metadata.get(key) if party else None


def extract_features(transcript: Transcript) -> dict:
    """Engineer one feature row (pure — no ML deps, unit-testable alone)."""
    buyer_target = _party_meta(transcript, "buyer", "target")
    seller_target = _party_meta(transcript, "seller", "target")
    listed_price = _party_meta(transcript, "buyer", "item_listed_price")
    if listed_price is None:
        listed_price = _party_meta(transcript, "seller", "item_listed_price")

    message_turns = [t for t in transcript.turns if t.action is None]
    offer_turns = [t for t in transcript.turns if t.action is Action.SUBMIT_DEAL]

    intent_counts = dict.fromkeys(MEANINGFUL_INTENTS, 0)
    for turn in message_turns:
        intent = turn.metadata.get("intent")
        if intent in intent_counts:
            intent_counts[intent] += 1

    features = {
        "buyer_target": float(buyer_target) if buyer_target is not None else float("nan"),
        "seller_target": float(seller_target) if seller_target is not None else float("nan"),
        "listed_price": float(listed_price) if listed_price is not None else float("nan"),
        "target_gap": (
            float(seller_target - buyer_target)
            if buyer_target is not None and seller_target is not None
            else float("nan")
        ),
        "num_message_turns": float(len(message_turns)),
        "num_offers": float(len(offer_turns)),
        "first_mover_is_buyer": (
            1.0 if transcript.turns and transcript.turns[0].speaker == "buyer" else 0.0
        ),
        "category": transcript.metadata.get("category") or "unknown",
    }
    for name, count in intent_counts.items():
        features[f"intent_{name.replace('-', '_')}"] = float(count)
    return features


def build_feature_matrix(transcripts: list[Transcript]) -> tuple[pd.DataFrame, pd.Series]:
    """Rows -> (X, y). Category is one-hot encoded against a FIXED column set
    so train/validation/test always share the same schema, even if a
    category happens to be absent from one split."""
    rows = [extract_features(t) for t in transcripts]
    df = pd.DataFrame(rows)
    for cat in CATEGORY_VALUES:
        df[f"category_{cat}"] = (df["category"] == cat).astype(float)
    df = df.drop(columns=["category"])
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
