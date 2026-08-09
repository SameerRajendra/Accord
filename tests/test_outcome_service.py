"""Tests for `analysis/outcome_service.predict_from_transcript`.

Covers the two things a serving-time wrapper actually has to get right:
1. Missing artifact → return `None`, not crash (bug 2 regression: the path
   default was pointing at a file no training pipeline actually wrote).
2. Single-row inference reindexes to the trained feature-column set so
   category one-hots line up (train time may include categories a single
   inference row doesn't).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis.outcome_service import _load, predict_from_transcript
from data.schema import Outcome, Party, Transcript, Turn


def _make_transcript(category: str = "bike") -> Transcript:
    return Transcript(
        dialogue_id="t-1",
        source="test",
        domain="unit-test",
        parties=[
            Party(party_id="buyer", metadata={"target": 300, "item_listed_price": 450}),
            Party(party_id="seller", metadata={"target": 400, "item_listed_price": 450}),
        ],
        turns=[
            Turn(index=0, speaker="seller", text="hi"),
            Turn(index=1, speaker="buyer", text="hello"),
        ],
        outcome=Outcome(agreement_reached=False, final_deal=None, points={}),
        has_strategy_annotations=False,
        metadata={"category": category},
    )


def test_missing_artifact_returns_none(monkeypatch, tmp_path):
    """Bug-2 regression: no artifact at the default path must not raise."""
    _load.cache_clear()
    monkeypatch.setenv("OUTCOME_MODEL_PATH", str(tmp_path / "does_not_exist.joblib"))
    result = predict_from_transcript(_make_transcript())
    assert result is None


# Module-level fakes — pickle can't serialize classes defined inside a
# function (Py3.12 is strict), and `joblib.dump` goes through pickle.
# `_EXPECTED_COLUMNS` is set by the test before dumping the fake artifact;
# the fake asserts against it inside predict_proba.
_EXPECTED_COLUMNS: list = []


class _FakeClassifier:
    def predict_proba(self, X):
        import numpy as np

        assert list(X.columns) == _EXPECTED_COLUMNS, (
            f"column mismatch: got {list(X.columns)}, expected {_EXPECTED_COLUMNS}"
        )
        # Real code does `predict_proba(X)[:, 1]` — must be a 2-D ndarray.
        n = len(X)
        return np.tile([0.3, 0.7], (n, 1))


class _FakeCalibrator:
    def predict(self, raw):
        import numpy as np

        return np.asarray(raw)


def test_single_row_reindex_matches_trained_columns(monkeypatch, tmp_path):
    """The feature matrix at inference must have the same columns as at train time,
    even if the inference row's category value isn't the one it was trained on."""
    global _EXPECTED_COLUMNS
    import joblib

    from analysis.outcome_model import build_feature_matrix

    _load.cache_clear()

    # Build a two-row training matrix so the trained column set covers both
    # buyer_target etc. and the one-hot category columns.
    train_transcripts = [_make_transcript("bike"), _make_transcript("housing")]
    X_train, _ = build_feature_matrix(train_transcripts)
    trained_columns = list(X_train.columns)
    _EXPECTED_COLUMNS = trained_columns

    artifact_path = tmp_path / "outcome_model.joblib"
    joblib.dump(
        {"model": _FakeClassifier(), "calibrator": _FakeCalibrator(), "feature_columns": trained_columns},
        artifact_path,
    )
    monkeypatch.setenv("OUTCOME_MODEL_PATH", str(artifact_path))

    # Inference on a transcript whose category is one the trained model saw.
    # The fake classifier asserts column alignment; a mismatch would raise.
    prob = predict_from_transcript(_make_transcript("bike"))
    assert prob is not None
    assert 0.0 <= prob <= 1.0

    # And on a transcript with a category NOT in the trained set — the reindex
    # must still line up (all category_* one-hots become 0.0).
    _load.cache_clear()
    prob2 = predict_from_transcript(_make_transcript("unknown-category"))
    assert prob2 is not None


def test_env_override_wins_over_default(monkeypatch, tmp_path):
    """`OUTCOME_MODEL_PATH` env must be honored over the module default."""
    from analysis.outcome_service import MissingModelError, _load as loader

    loader.cache_clear()  # prior tests may have cached a successful load
    fake = tmp_path / "elsewhere.joblib"
    monkeypatch.setenv("OUTCOME_MODEL_PATH", str(fake))

    with pytest.raises(MissingModelError) as exc:
        loader()
    assert str(fake) in str(exc.value)
