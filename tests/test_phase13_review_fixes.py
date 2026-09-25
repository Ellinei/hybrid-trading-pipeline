"""Regressions for causal labels, cache identity, regime startup and scoring."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backtest.engine import BacktestConfig, _load_all_features, run_backtest
from backtest.exit_simulation import OHLCVBar
from backtest.experiment import _compute_metrics
from backtest.run_backtest import _compute_cache_key
from backtest.signal_cache import precompute_walk_forward_probs
from config.strategy import StrategyConfig
from execution import bot_runner
from execution.bot_runner import run_once
from signals.base import AggregatedSignal, SignalDirection
from signals.regime import RegimeState


UTC = timezone.utc
START = datetime(2026, 1, 1, tzinfo=UTC)


def _sparse_features():
    times = [START + timedelta(hours=h) for h in range(8)]
    times.append(START + timedelta(days=1, hours=1))
    return pd.DataFrame({
        "symbol": ["BTCUSDT"] * 9, "timestamp": times,
        "close": [100.0] * 8 + [200.0], "feature": [1.0] * 9,
    })


def test_sparse_label_reveal_timestamp_survives_both_feature_loaders():
    frame = _sparse_features()
    with patch("backtest.data_store.load_features", return_value=frame):
        direct = _load_all_features(None, ["BTCUSDT"])
        from backtest.signal_cache import load_feature_matrix
        cached = load_feature_matrix(None, ["BTCUSDT"])
    expected = START + timedelta(days=1, hours=1)
    assert direct.iloc[0]["future_timestamp"] == expected
    assert cached.iloc[0]["future_timestamp"] == expected


def test_sparse_future_label_is_excluded_from_cached_retrain(monkeypatch):
    frame = _sparse_features()
    frame["future_close"] = frame["close"].shift(-8)
    frame["target"] = 1
    seen = []

    class CaptureML:
        FEATURES = ["feature"]
        LABEL_HORIZON_BARS = 8

        def __init__(self, db_engine):
            pass

        def train_walk_forward(self, rows, frozen):
            seen.append(rows.copy())
            return False, frozen

    monkeypatch.setattr("backtest.signal_cache.MLSignalGenerator", CaptureML)
    precompute_walk_forward_probs(
        frame, [(START, "BTCUSDT"), (START + timedelta(days=1), "BTCUSDT")],
        min_train_days=1,
    )
    assert len(seen) == 1
    assert seen[0].empty


def test_sparse_future_label_is_excluded_from_direct_retrain():
    frame = _sparse_features()
    frame["future_close"] = frame["close"].shift(-8)
    frame["target"] = 1
    ml = MagicMock()
    ml.train_walk_forward.return_value = (False, None)
    hold = AggregatedSignal(SignalDirection.HOLD, 0.0, "BTCUSDT", START, [], "hold", {})
    aggregator = SimpleNamespace(ml=ml, aggregate=lambda *args, **kwargs: hold)
    bars = [OHLCVBar(timestamp=ts, open=100, high=101, low=99, close=100)
            for ts in (START, START + timedelta(days=1))]
    run_backtest(
        {"BTCUSDT": bars}, {"BTCUSDT": {}}, {}, aggregator,
        BacktestConfig(["BTCUSDT"], START, START + timedelta(days=1)),
        walk_forward_months=1, min_train_days=1, feature_df=frame,
    )
    assert ml.train_walk_forward.call_count == 1
    assert ml.train_walk_forward.call_args.args[0].empty


def test_cache_key_changes_with_training_bar_stream_and_algorithm_version():
    args = dict(label_horizon=8, label_min_return=0.003, min_train_days=60,
                walk_forward_months=1, train_window_days=540,
                symbols=["BTCUSDT"], fingerprint={"BTCUSDT": (2, "end", "features")})
    old = _compute_cache_key(**args)
    a = _compute_cache_key(**args, algorithm_version=2,
                           bar_stream=[(START, "BTCUSDT")])
    b = _compute_cache_key(**args, algorithm_version=2,
                           bar_stream=[(START - timedelta(hours=1), "BTCUSDT"),
                                       (START, "BTCUSDT")])
    assert len({old, a, b}) == 3


def test_missing_regime_classifications_count_as_blocked_scored_days():
    result = SimpleNamespace(trades=[], metrics={
        "trade_count": 0, "win_rate": 0.0, "profit_factor": 0.0,
        "max_drawdown": 0.0, "sharpe_ratio": 0.0, "total_return": 0.0,
    })
    metrics = _compute_metrics(result, regime={START.date(): RegimeState.RISK_ON},
                               replay_days={START.date(), (START + timedelta(days=1)).date()})
    assert metrics["regime_days_blocked_pct"] == 50.0
    assert _compute_metrics(result, regime={}, replay_days={START.date()})[
        "regime_days_blocked_pct"] == 100.0
    assert "regime_days_blocked_pct" not in _compute_metrics(result, regime=None,
                                                               replay_days={START.date()})


def test_startup_regime_query_failure_keeps_cycle_and_exits_alive():
    cfg = StrategyConfig(regime_candidate="r1_sma200")
    with patch("execution.bot_runner.resolve_regime", side_effect=RuntimeError("db down")):
        regime, error = bot_runner.initial_regime(MagicMock(), cfg)
        aggregator = MagicMock()
        aggregator._regime = regime
        aggregator._regime_refresh_error = error
        executor = MagicMock()
        run_once([], executor, aggregator, engine=MagicMock(), strategy_cfg=cfg,
                 max_hold_bars=24)
    assert regime == {}
    assert error == "refresh_failed"
    executor.close_expired_positions.assert_called_once_with(24, dry_run=False)


def test_invalid_regime_config_still_fails_at_startup():
    with pytest.raises(ValueError, match="Unknown regime candidate"):
        bot_runner.initial_regime(MagicMock(), StrategyConfig(regime_candidate="unknown"))
