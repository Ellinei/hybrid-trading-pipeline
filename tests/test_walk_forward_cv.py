"""Causal CV folds for MLSignalGenerator.train_in_memory's walk-forward
hyperparameter search.

Phase 11a (commit 7b73f06) had train_in_memory's GridSearchCV tuning path
run ``cv=TimeSeriesSplit(n_splits=5)`` over whatever row order the caller's
frame arrived in. Real walk-forward frames (backtest.signal_cache's
available_training_rows / backtest.engine's _load_all_features) are sorted
(symbol, timestamp) — symbol-grouped, not globally time-ordered — and carry
no label-reveal purge. A later fold could therefore train on one symbol's
newest rows while validating on another symbol's older, contemporaneous
rows, and a training row's label could be revealed after its own fold's
validation start. train_walk_forward freezes the winning params at the
first successful boundary for the whole run, so this biased hyperparameters
used by every walk-forward experiment.

These tests pin the fix: MLSignalGenerator._time_grouped_purged_cv sorts
globally by (timestamp, symbol), groups folds by distinct timestamp, and
purges each fold's training rows whose label reveal time exceeds that
fold's own validation start — and train_in_memory's tuning path (the
params=None branch) actually uses it whenever the caller's frame carries
timestamps.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from signals.ml_model import MLSignalGenerator

UTC = UTC


def _symbol_grouped_frame(hours=8, symbols=("AAA", "BBB")):
    """Two (or more) symbols, each with `hours` distinct hourly timestamps,
    sorted (symbol, timestamp) — the exact row order
    backtest.signal_cache.available_training_rows / backtest.engine's
    walk-forward pre-loader hand to train_walk_forward. future_timestamp is
    one bar ahead of timestamp (the label candle's own open time); its
    reveal time (see label_reveal_time) is one bar later still.
    """
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for symbol in symbols:
        for hour in range(hours):
            row = {name: float(hour) for name in MLSignalGenerator.FEATURES}
            row.update(
                symbol=symbol,
                timestamp=start + timedelta(hours=hour),
                future_timestamp=start + timedelta(hours=hour + 1),
                target=hour % 2,
            )
            rows.append(row)
    return pd.DataFrame(rows)


class TestTimeGroupedPurgedCV:
    """Direct tests of the shared helper both partition_training_rows and
    train_in_memory's tuning path build folds with."""

    def test_folds_are_causal_and_group_simultaneous_symbols(self):
        frame = _symbol_grouped_frame(hours=8, symbols=("AAA", "BBB"))

        sorted_frame, cv = MLSignalGenerator._time_grouped_purged_cv(
            frame, "future_timestamp", n_splits=5,
        )

        assert len(cv) == 5
        for train_idx, val_idx in cv:
            train = sorted_frame.iloc[train_idx]
            val = sorted_frame.iloc[val_idx]
            assert train["timestamp"].max() < val["timestamp"].min()
            reveal = MLSignalGenerator.label_reveal_time(train["future_timestamp"])
            assert reveal.max() <= val["timestamp"].min()
            # Both symbols at a given validation timestamp stay in the same
            # fold — no cross-symbol time scrambling.
            assert val.groupby("timestamp")["symbol"].nunique().eq(2).all()

    def test_purges_a_label_that_reveals_after_validation_start(self):
        """A label opening before a fold's validation start but revealing
        (open + BAR_DURATION) after it must be purged from that fold's
        training rows — the leak this fix closes."""
        start = datetime(2026, 1, 1, tzinfo=UTC)
        rows = []
        for hour in range(7):
            if hour == 1:
                # Opens at 1:30 (before fold 0's validation_start of hour 2)
                # but reveals at 2:30 — a real leak into fold 0.
                future_ts = start + timedelta(hours=1, minutes=30)
            else:
                future_ts = start + timedelta(hours=hour + 1)
            row = {name: float(hour) for name in MLSignalGenerator.FEATURES}
            row.update(symbol="AAA", timestamp=start + timedelta(hours=hour),
                       future_timestamp=future_ts, target=hour % 2)
            rows.append(row)
        frame = pd.DataFrame(rows)

        sorted_frame, cv = MLSignalGenerator._time_grouped_purged_cv(
            frame, "future_timestamp", n_splits=5,
        )

        fold0_train_idx, _ = cv[0]
        fold0_timestamps = set(sorted_frame.loc[fold0_train_idx, "timestamp"])
        assert (start + timedelta(hours=1)) not in fold0_timestamps
        assert start in fold0_timestamps

    def test_raises_when_too_few_distinct_timestamps(self):
        frame = _symbol_grouped_frame(hours=3, symbols=("AAA", "BBB"))

        with pytest.raises(ValueError):
            MLSignalGenerator._time_grouped_purged_cv(frame, "future_timestamp", n_splits=5)


class TestOldSplitWasNonCausal:

    def test_naive_row_order_split_violates_time_ordering(self):
        """Demonstrates the bug: TimeSeriesSplit over the frame's raw
        (symbol, timestamp)-sorted row order — the old
        cv=TimeSeriesSplit(n_splits=5) behavior — lets a fold train on one
        symbol's later rows while validating on another symbol's earlier,
        contemporaneous rows. The new time-grouped, purged split does not.
        """
        frame = _symbol_grouped_frame(hours=8, symbols=("AAA", "BBB"))

        old_cv = list(TimeSeriesSplit(n_splits=5).split(frame))
        violation_found = any(
            frame.iloc[train_idx]["timestamp"].max() >= frame.iloc[val_idx]["timestamp"].min()
            for train_idx, val_idx in old_cv
        )
        assert violation_found, "expected the naive row-order split to violate time ordering"

        sorted_frame, new_cv = MLSignalGenerator._time_grouped_purged_cv(
            frame, "future_timestamp", n_splits=5,
        )
        for train_idx, val_idx in new_cv:
            train_max = sorted_frame.iloc[train_idx]["timestamp"].max()
            val_min = sorted_frame.iloc[val_idx]["timestamp"].min()
            assert train_max < val_min


class TestTrainInMemoryCausalCV:

    def test_uses_time_grouped_cv_when_frame_has_timestamps(self):
        """train_in_memory's tuning path (params=None) must pass the causal
        (train_idx, val_idx) list to GridSearchCV, not a bare
        TimeSeriesSplit over row order, when the frame carries timestamps."""
        frame = _symbol_grouped_frame(hours=8, symbols=("AAA", "BBB"))
        gen = MLSignalGenerator(MagicMock())

        with patch("signals.ml_model.GridSearchCV") as mock_grid:
            mock_grid.return_value.best_estimator_.named_steps = {
                "scaler": MagicMock(), "clf": MagicMock(),
            }
            mock_grid.return_value.best_params_ = {
                "clf__max_depth": 3, "clf__n_estimators": 100, "clf__learning_rate": 0.1,
            }
            gen.train_in_memory(frame)
            cv_arg = mock_grid.call_args.kwargs["cv"]

        assert isinstance(cv_arg, list)
        assert len(cv_arg) == 5
        for train_idx, val_idx in cv_arg:
            assert len(train_idx) > 0
            assert len(val_idx) > 0

    def test_legacy_frame_without_timestamps_falls_back_with_warning(self, caplog):
        """A bare FEATURES+target frame (no timestamp/label-open column)
        can't be ordered causally, so it keeps the old plain
        TimeSeriesSplit-over-row-order behavior instead of failing closed —
        every real walk-forward caller supplies timestamps, so this path is
        only reachable from synthetic test frames."""
        import numpy as np

        gen = MLSignalGenerator(MagicMock())
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            rng.standard_normal((200, len(MLSignalGenerator.FEATURES))),
            columns=MLSignalGenerator.FEATURES,
        )
        df["target"] = (rng.random(200) > 0.5).astype(int)

        with patch("signals.ml_model.GridSearchCV") as mock_grid:
            mock_grid.return_value.best_estimator_.named_steps = {
                "scaler": MagicMock(), "clf": MagicMock(),
            }
            mock_grid.return_value.best_params_ = {
                "clf__max_depth": 3, "clf__n_estimators": 100, "clf__learning_rate": 0.1,
            }
            result = gen.train_in_memory(df)
            cv_arg = mock_grid.call_args.kwargs["cv"]

        assert result is True
        assert isinstance(cv_arg, TimeSeriesSplit)

    def test_frozen_params_path_skips_causal_cv_and_succeeds_with_few_timestamps(self):
        """Frozen-params fits (used on every walk-forward boundary after
        the first) must not build CV folds at all — a thin window with
        fewer than six distinct timestamps still succeeds via the frozen
        path even though it would fail causal-CV tuning (which needs at
        least n_splits+1=6 distinct timestamps)."""
        frame = _symbol_grouped_frame(hours=4, symbols=("AAA", "BBB"))  # 4 < 6 required by CV
        gen = MLSignalGenerator(MagicMock())
        params = {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.1}

        with patch("signals.ml_model.GridSearchCV") as mock_grid:
            result = gen.train_in_memory(frame, params=params)

        mock_grid.assert_not_called()
        assert result is True
        assert gen.is_trained_in_memory() is True

    def test_causal_cv_unavailable_returns_false(self):
        """A causal frame (has timestamps) with too few distinct
        timestamps for 5-fold CV must fail closed on the tuning path,
        not silently fall back to a non-causal split."""
        frame = _symbol_grouped_frame(hours=4, symbols=("AAA", "BBB"))  # 4 < 6 required
        gen = MLSignalGenerator(MagicMock())

        result = gen.train_in_memory(frame)

        assert result is False
        assert gen.is_trained_in_memory() is False
