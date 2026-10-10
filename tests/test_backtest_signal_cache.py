"""Unit tests for backtest.signal_cache — replay fusion parity and the
walk-forward probability precompute schedule, all on synthetic data.

The full-fidelity parity check (replay vs live SignalAggregator over real
DB data) is run as part of the sweep-v2 workflow; these tests pin the
arithmetic and scheduling contracts.

Run:
    poetry run pytest tests/test_backtest_signal_cache.py -v
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV

from backtest.signal_cache import (
    FusionParams,
    ReplayAggregator,
    precompute_walk_forward_probs,
)
from signals.base import SignalDirection
from signals.ml_model import MLSignalGenerator

_START = datetime(2026, 1, 1, tzinfo=UTC)


def _feature_df(n_rows: int, symbol: str = "BTCUSDT", **rule_overrides) -> pd.DataFrame:
    """Synthetic feature matrix: hourly rows, all 18 ML features populated."""
    rng = np.random.default_rng(7)
    data = {
        col: rng.standard_normal(n_rows)
        for col in MLSignalGenerator.FEATURES
    }
    data["symbol"] = symbol
    data["timestamp"] = [_START + timedelta(hours=h) for h in range(n_rows)]
    data["close"] = 100.0 + rng.standard_normal(n_rows).cumsum()
    df = pd.DataFrame(data)
    for col, value in rule_overrides.items():
        df[col] = value
    horizon = MLSignalGenerator.LABEL_HORIZON_BARS
    df["future_close"] = df.groupby("symbol")["close"].shift(-horizon)
    df["target"] = MLSignalGenerator.compute_target(df["close"], df["future_close"])
    return df


class TestReplayAggregatorFusion:
    """Hand-computed fused scores through the default weights 0.4/0.4/0.2."""

    def _row_df(self, **rule_values) -> pd.DataFrame:
        neutral = {
            "rsi_14_proxy": 50.0, "macd_line": 0.0, "bb_zscore": 0.0,
            "pct_change_1h": 0.0, "pct_change_24h": 0.0, "close_vs_sma50": 0.0,
        }
        neutral.update(rule_values)
        df = _feature_df(1)
        for col, val in neutral.items():
            df[col] = val
        return df

    def test_strong_technical_buy_alone(self):
        # tech BUY(0.8)*0.4 = 0.32 > 0.15 → BUY, conf 0.32
        df = self._row_df(rsi_14_proxy=25.0, macd_line=0.5, pct_change_1h=0.5)
        agg = ReplayAggregator(df, prob_map={})
        result = agg.aggregate("BTCUSDT", as_of=_START)

        assert result.direction == SignalDirection.BUY
        assert result.confidence == pytest.approx(0.32)

    def test_moderate_technical_sell_alone(self):
        # tech SELL(0.55)*0.4 = -0.22 < -0.15 → SELL
        df = self._row_df(rsi_14_proxy=65.0, pct_change_1h=-0.5)
        agg = ReplayAggregator(df, prob_map={})
        result = agg.aggregate("BTCUSDT", as_of=_START)

        assert result.direction == SignalDirection.SELL
        assert result.confidence == pytest.approx(0.22)

    def test_ml_and_technical_combine(self):
        # tech BUY(0.8)*0.4 + ml BUY(0.9)*0.4 = 0.32 + 0.36 = 0.68
        df = self._row_df(rsi_14_proxy=25.0, macd_line=0.5, pct_change_1h=0.5)
        agg = ReplayAggregator(df, prob_map={("BTCUSDT", _START): 0.9})
        result = agg.aggregate("BTCUSDT", as_of=_START)

        assert result.direction == SignalDirection.BUY
        assert result.confidence == pytest.approx(0.68)

    def test_ml_sell_below_threshold(self):
        # prob 0.2 < 0.35 → ml SELL conf 0.8 → score -0.32 → SELL
        df = self._row_df()
        agg = ReplayAggregator(df, prob_map={("BTCUSDT", _START): 0.2})
        result = agg.aggregate("BTCUSDT", as_of=_START)

        assert result.direction == SignalDirection.SELL
        assert result.confidence == pytest.approx(0.32)

    def test_ml_neutral_band_holds(self):
        df = self._row_df()
        agg = ReplayAggregator(df, prob_map={("BTCUSDT", _START): 0.5})
        result = agg.aggregate("BTCUSDT", as_of=_START)

        assert result.direction == SignalDirection.HOLD

    def test_no_feature_row_before_as_of_holds(self):
        df = self._row_df(rsi_14_proxy=25.0, macd_line=0.5, pct_change_1h=0.5)
        agg = ReplayAggregator(df, prob_map={})
        result = agg.aggregate("BTCUSDT", as_of=_START - timedelta(hours=1))

        assert result.direction == SignalDirection.HOLD

    def test_latest_row_at_or_before_is_used(self):
        """A bar between feature rows uses the earlier row, like the live SQL."""
        df = self._row_df(rsi_14_proxy=25.0, macd_line=0.5, pct_change_1h=0.5)
        agg = ReplayAggregator(df, prob_map={})
        result = agg.aggregate("BTCUSDT", as_of=_START + timedelta(minutes=30))

        assert result.direction == SignalDirection.BUY

    def test_custom_gates_and_thresholds(self):
        # Moderate SELL scores -0.22: blocked by sell_gate=-0.30
        df = self._row_df(rsi_14_proxy=65.0, pct_change_1h=-0.5)
        agg = ReplayAggregator(
            df, prob_map={}, params=FusionParams(sell_gate=-0.30)
        )
        assert agg.aggregate("BTCUSDT", as_of=_START).direction == SignalDirection.HOLD

        # prob 0.66 passes default 0.65 but fails ml_buy_threshold=0.70
        df2 = self._row_df()
        agg2 = ReplayAggregator(
            df2, prob_map={("BTCUSDT", _START): 0.66},
            params=FusionParams(ml_buy_threshold=0.70),
        )
        assert agg2.aggregate("BTCUSDT", as_of=_START).direction == SignalDirection.HOLD

    def test_rule_toggles_flow_through(self):
        # Falling-knife bb dip: BUY by default, HOLD with guard on
        df = self._row_df(bb_zscore=-2.5, pct_change_1h=-0.5)
        default_agg = ReplayAggregator(df, prob_map={})
        guarded_agg = ReplayAggregator(
            df, prob_map={}, params=FusionParams(guard_bb_dip_buy=True)
        )

        assert default_agg.aggregate("BTCUSDT", as_of=_START).direction == SignalDirection.BUY
        assert guarded_agg.aggregate("BTCUSDT", as_of=_START).direction == SignalDirection.HOLD


class TestReplayAggregatorRegimeGate:
    """Phase 12: an optional post-fusion regime gate — blocks BUY only,
    never SELL/HOLD, and is fully backward compatible (regime=None is the
    existing no-gate behavior)."""

    def _row_df(self, **rule_values):
        neutral = {
            "rsi_14_proxy": 50.0, "macd_line": 0.0, "bb_zscore": 0.0,
            "pct_change_1h": 0.0, "pct_change_24h": 0.0, "close_vs_sma50": 0.0,
        }
        neutral.update(rule_values)
        df = _feature_df(1)
        for col, val in neutral.items():
            df[col] = val
        return df

    def _buy_signal_df(self):
        # tech BUY(0.8)*0.4 = 0.32 > 0.15 -> BUY, matches test_strong_technical_buy_alone
        return self._row_df(rsi_14_proxy=25.0, macd_line=0.5, pct_change_1h=0.5)

    def test_risk_off_blocks_a_buy(self):
        from signals.regime import RegimeState

        df = self._buy_signal_df()
        regime = {_START.date(): RegimeState.RISK_OFF}
        agg = ReplayAggregator(df, prob_map={}, regime=regime)

        result = agg.aggregate("BTCUSDT", as_of=_START)

        assert result.direction == SignalDirection.HOLD

    def test_risk_on_allows_a_buy(self):
        from signals.regime import RegimeState

        df = self._buy_signal_df()
        regime = {_START.date(): RegimeState.RISK_ON}
        agg = ReplayAggregator(df, prob_map={}, regime=regime)

        result = agg.aggregate("BTCUSDT", as_of=_START)

        assert result.direction == SignalDirection.BUY

    def test_no_regime_dict_is_unaffected_default_behavior(self):
        df = self._buy_signal_df()
        agg = ReplayAggregator(df, prob_map={})  # regime not passed

        result = agg.aggregate("BTCUSDT", as_of=_START)

        assert result.direction == SignalDirection.BUY

    def test_missing_date_in_enabled_regime_dict_blocks_buy(self):

        df = self._buy_signal_df()
        regime = {}  # no entry for this date at all
        agg = ReplayAggregator(df, prob_map={}, regime=regime)

        result = agg.aggregate("BTCUSDT", as_of=_START)

        assert result.direction == SignalDirection.HOLD
        assert "classification_unavailable" in result.reasoning

    def test_never_blocks_a_sell(self):
        from signals.regime import RegimeState

        # tech SELL(0.55)*0.4 = -0.22 < -0.15 -> SELL, matches test_moderate_technical_sell_alone
        df = self._row_df(rsi_14_proxy=65.0, pct_change_1h=-0.5)
        regime = {_START.date(): RegimeState.RISK_OFF}
        agg = ReplayAggregator(df, prob_map={}, regime=regime)

        result = agg.aggregate("BTCUSDT", as_of=_START)

        assert result.direction == SignalDirection.SELL


class TestPrecomputeWalkForwardProbs:

    def test_schedule_and_warmup(self):
        """Bars before the first retrain get no prob; bars after all do."""
        df = _feature_df(72)
        bar_stream = [
            (_START + timedelta(hours=h), "BTCUSDT") for h in range(72)
        ]

        probs = precompute_walk_forward_probs(
            df, bar_stream, min_train_days=1, walk_forward_months=1,
        )

        first_retrain = _START + timedelta(days=1)
        warmup_keys = [(s, ts) for ts, s in bar_stream if ts < first_retrain]
        post_keys = [(s, ts) for ts, s in bar_stream if ts >= first_retrain]

        assert all(k not in probs for k in warmup_keys)
        assert all(k in probs for k in post_keys)
        assert all(0.0 <= v <= 1.0 for v in probs.values())

    def test_empty_inputs_return_empty(self):
        assert precompute_walk_forward_probs(pd.DataFrame(), [], 1, 1) == {}

    def test_nan_feature_rows_get_no_prob(self):
        """Rows with a NaN feature must be skipped — live ML returns HOLD."""
        df = _feature_df(72)
        df.loc[30:, "rsi_14_proxy"] = np.nan  # NaN from bar 30 onward
        bar_stream = [
            (_START + timedelta(hours=h), "BTCUSDT") for h in range(72)
        ]

        probs = precompute_walk_forward_probs(
            df, bar_stream, min_train_days=1, walk_forward_months=1,
        )

        nan_keys = [("BTCUSDT", _START + timedelta(hours=h)) for h in range(30, 72)]
        assert all(k not in probs for k in nan_keys)

    def test_reuses_frozen_params_after_first_retrain(self):
        """First retrain boundary tunes via GridSearchCV; every later
        boundary reuses those frozen params — one search per run, not one
        per window (~35 windows/run in a real walk-forward experiment)."""
        n_rows = 33 * 24  # 33 days hourly: forces 2 retrain boundaries
        df = _feature_df(n_rows)
        bar_stream = [
            (_START + timedelta(hours=h), "BTCUSDT") for h in range(n_rows)
        ]

        with patch("signals.ml_model.GridSearchCV", wraps=GridSearchCV) as mock_grid:
            probs = precompute_walk_forward_probs(
                df, bar_stream, min_train_days=1, walk_forward_months=1,
            )

        assert mock_grid.call_count == 1
        assert probs
        assert all(0.0 <= v <= 1.0 for v in probs.values())
