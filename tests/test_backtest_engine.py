"""End-to-end test of backtest.engine.run_backtest against synthetic,
in-memory data — no real DB or exchange. Proves the engine wiring (signal
-> risk gate -> fill simulation -> exit check -> metrics) is correct.

Run:
    poetry run pytest tests/test_backtest_engine.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from backtest.engine import BacktestConfig, run_backtest
from backtest.exit_simulation import OHLCVBar
from execution.risk_manager import RiskConfig
from signals.base import AggregatedSignal, Signal, SignalDirection


class _FakeAggregator:
    """Returns BUY on the first call for a symbol, HOLD on every call after."""

    def __init__(self):
        self._called = {}

    def aggregate(self, symbol: str, as_of=None) -> AggregatedSignal:
        first_call = symbol not in self._called
        self._called[symbol] = True
        direction = SignalDirection.BUY if first_call else SignalDirection.HOLD
        confidence = 0.9 if first_call else 0.5
        return AggregatedSignal(
            direction=direction, confidence=confidence, symbol=symbol,
            timestamp=as_of or datetime.now(timezone.utc),
            signals=[Signal(direction, confidence, "technical", symbol, as_of)],
            reasoning="fake", weights_used={},
        )


def _bars(closes_highs_lows, start=None):
    start = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = []
    for i, (close, high, low) in enumerate(closes_highs_lows):
        ts = start + timedelta(hours=i)
        bars.append(OHLCVBar(timestamp=ts, open=close, high=high, low=low, close=close))
    return bars


class TestRunBacktestBuyThenStopLossHit:
    def test_buy_entry_then_stop_loss_close_produces_one_losing_trade(self):
        # bar0: BUY entry at close=100, atr=5 -> stop_loss=92.5, take_profit=115
        # bar1: low=90 dips through stop_loss -> closes as a loss
        # bar2: HOLD, no further action
        bars = _bars([
            (100.0, 101.0, 99.0),
            (93.0, 95.0, 90.0),
            (93.0, 94.0, 92.0),
        ])
        config = BacktestConfig(
            symbols=["BTCUSDT"],
            start=bars[0].timestamp, end=bars[-1].timestamp,
            initial_balance=10_000.0,
            risk_config=RiskConfig(min_confidence=0.0),
        )
        atr_by_symbol = {"BTCUSDT": {b.timestamp: 5.0 for b in bars}}

        result = run_backtest(
            bars_by_symbol={"BTCUSDT": bars},
            atr_by_symbol=atr_by_symbol,
            filters_by_symbol={},
            aggregator=_FakeAggregator(),
            config=config,
        )

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade["reason"] == "stop_loss"
        assert trade["pnl"] < 0
        assert len(result.equity_curve) == 1
        assert result.metrics["trade_count"] == 1
        assert result.metrics["win_rate"] == 0.0
        assert any("training-on-test" in w for w in result.warnings)

class TestRunBacktestMultiSymbolChronologicalOrder:
    def test_equity_curve_is_globally_chronological_across_symbols(self):
        """AAA's position stays open across 5 hourly bars (closing at t4);
        BBB opens and closes within that same window (closing at t1.5). A
        symbol-outer loop would process all of AAA first then all of BBB,
        producing equity_curve timestamps [t4, t1.5] — backwards. Bars must
        be processed in one global, timestamp-sorted stream instead."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)

        aaa_bars = [
            OHLCVBar(timestamp=start + timedelta(hours=h), open=100.0, high=101.0, low=99.0, close=100.0)
            for h in range(4)
        ] + [
            OHLCVBar(timestamp=start + timedelta(hours=4), open=100.0, high=101.0, low=90.0, close=92.0),
        ]
        bbb_bars = [
            OHLCVBar(timestamp=start + timedelta(hours=0.5), open=100.0, high=101.0, low=99.0, close=100.0),
            OHLCVBar(timestamp=start + timedelta(hours=1.5), open=100.0, high=101.0, low=85.0, close=92.0),
        ]

        config = BacktestConfig(
            symbols=["AAA", "BBB"],
            start=start, end=start + timedelta(hours=5),
            risk_config=RiskConfig(min_confidence=0.0, max_open_positions=5),
        )
        atr_by_symbol = {
            "AAA": {b.timestamp: 5.0 for b in aaa_bars},
            "BBB": {b.timestamp: 5.0 for b in bbb_bars},
        }

        result = run_backtest(
            bars_by_symbol={"AAA": aaa_bars, "BBB": bbb_bars},
            atr_by_symbol=atr_by_symbol,
            filters_by_symbol={},
            aggregator=_FakeAggregator(),
            config=config,
        )

        timestamps = [ts for ts, _ in result.equity_curve]
        assert timestamps == sorted(timestamps), (
            f"equity_curve timestamps are not in chronological order: {timestamps}"
        )
        assert len(result.trades) == 2

    def test_no_data_for_symbol_produces_no_trades(self):
        config = BacktestConfig(
            symbols=["ETHUSDT"],
            start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
        result = run_backtest(
            bars_by_symbol={}, atr_by_symbol={}, filters_by_symbol={},
            aggregator=_FakeAggregator(), config=config,
        )
        assert result.trades == []
        assert result.metrics["trade_count"] == 0


class TestTimeStop:
    def test_time_stop_closes_position_after_max_hold_bars(self):
        """max_hold_bars=2: BUY on bar0, no SL/TP touch, forced close at bar2."""
        bars = _bars([(100.0, 101.0, 99.0)] * 5)
        config = BacktestConfig(
            symbols=["BTCUSDT"],
            start=bars[0].timestamp, end=bars[-1].timestamp,
            risk_config=RiskConfig(min_confidence=0.0, max_hold_bars=2),
        )
        result = run_backtest(
            bars_by_symbol={"BTCUSDT": bars},
            atr_by_symbol={"BTCUSDT": {b.timestamp: 5.0 for b in bars}},
            filters_by_symbol={},
            aggregator=_FakeAggregator(),
            config=config,
        )

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade["reason"] == "time_stop"
        assert trade["exit_time"] - trade["entry_time"] == timedelta(hours=2)

    def test_time_stop_disabled_by_default(self):
        """max_hold_bars=0 (default): position stays open to the end."""
        bars = _bars([(100.0, 101.0, 99.0)] * 5)
        config = BacktestConfig(
            symbols=["BTCUSDT"],
            start=bars[0].timestamp, end=bars[-1].timestamp,
            risk_config=RiskConfig(min_confidence=0.0),
        )
        result = run_backtest(
            bars_by_symbol={"BTCUSDT": bars},
            atr_by_symbol={"BTCUSDT": {b.timestamp: 5.0 for b in bars}},
            filters_by_symbol={},
            aggregator=_FakeAggregator(),
            config=config,
        )

        assert result.trades == []

    def test_stop_loss_takes_priority_over_time_stop(self):
        """A bar that both breaches the SL and exceeds max hold closes as stop_loss."""
        bars = _bars([
            (100.0, 101.0, 99.0),
            (100.0, 101.0, 99.0),
            (93.0, 95.0, 90.0),  # SL 92.5 hit at bar2 == max_hold boundary
        ])
        config = BacktestConfig(
            symbols=["BTCUSDT"],
            start=bars[0].timestamp, end=bars[-1].timestamp,
            risk_config=RiskConfig(min_confidence=0.0, max_hold_bars=2),
        )
        result = run_backtest(
            bars_by_symbol={"BTCUSDT": bars},
            atr_by_symbol={"BTCUSDT": {b.timestamp: 5.0 for b in bars}},
            filters_by_symbol={},
            aggregator=_FakeAggregator(),
            config=config,
        )

        assert len(result.trades) == 1
        assert result.trades[0]["reason"] == "stop_loss"


# ===========================================================================
# Walk-forward retraining tests
# ===========================================================================

class _HoldAggregatorWithML:
    """Always returns HOLD; exposes a mock ml attribute for walk-forward tests."""

    def __init__(self, ml_mock):
        self.ml = ml_mock

    def aggregate(self, symbol: str, as_of=None) -> AggregatedSignal:
        return AggregatedSignal(
            direction=SignalDirection.HOLD, confidence=0.5, symbol=symbol,
            timestamp=as_of or datetime.now(timezone.utc),
            signals=[], reasoning="hold", weights_used={},
        )


class TestWalkForwardMode:

    def test_walk_forward_retraining_boundary_calls_train_walk_forward(self):
        """With 91 bars at 1-day intervals, 30-day warm-up, monthly retraining:
        train_walk_forward fires at day-30 and day-60 boundaries (≥ 2 calls)."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = [
            OHLCVBar(
                timestamp=start + timedelta(days=d),
                open=100.0, high=101.0, low=99.0, close=100.0,
            )
            for d in range(91)
        ]
        config = BacktestConfig(
            symbols=["BTCUSDT"],
            start=bars[0].timestamp, end=bars[-1].timestamp,
        )
        atr_by_symbol = {"BTCUSDT": {b.timestamp: 5.0 for b in bars}}

        mock_ml = MagicMock()
        mock_ml.train_walk_forward.return_value = (True, {"max_depth": 3})

        feature_df = pd.DataFrame({
            "timestamp": [start + timedelta(days=d) for d in range(91)],
        })

        result = run_backtest(
            bars_by_symbol={"BTCUSDT": bars},
            atr_by_symbol=atr_by_symbol,
            filters_by_symbol={},
            aggregator=_HoldAggregatorWithML(mock_ml),
            config=config,
            walk_forward_months=1,
            min_train_days=30,
            feature_df=feature_df,
        )

        mock_ml.set_walk_forward_mode.assert_called_once_with(True)
        assert mock_ml.train_walk_forward.call_count >= 2
        assert len(result.retrain_events) == mock_ml.train_walk_forward.call_count
        assert all(e["trained"] for e in result.retrain_events)
        assert all("at" in e and "rows" in e for e in result.retrain_events)

    def test_retrain_slice_excludes_label_horizon(self):
        """Rows within LABEL_HORIZON_BARS hours of the retrain time have labels
        that peek past 'now' — the training slice must exclude them."""
        from signals.ml_model import MLSignalGenerator

        horizon = MLSignalGenerator.LABEL_HORIZON_BARS
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = [
            OHLCVBar(timestamp=start + timedelta(hours=h),
                     open=100.0, high=101.0, low=99.0, close=100.0)
            for h in range(72)
        ]
        config = BacktestConfig(
            symbols=["BTCUSDT"],
            start=bars[0].timestamp, end=bars[-1].timestamp,
        )
        mock_ml = MagicMock()
        mock_ml.train_walk_forward.return_value = (True, {"max_depth": 3})
        feature_df = pd.DataFrame({
            "timestamp": [b.timestamp for b in bars],
        })

        run_backtest(
            bars_by_symbol={"BTCUSDT": bars},
            atr_by_symbol={"BTCUSDT": {b.timestamp: 5.0 for b in bars}},
            filters_by_symbol={},
            aggregator=_HoldAggregatorWithML(mock_ml),
            config=config,
            walk_forward_months=1,
            min_train_days=1,
            feature_df=feature_df,
        )

        # First retrain fires at start+1day; slice must stop 'horizon' hours earlier
        first_slice = mock_ml.train_walk_forward.call_args_list[0][0][0]
        retrain_time = start + timedelta(days=1)
        assert first_slice["timestamp"].max() < retrain_time - timedelta(hours=horizon)
        assert len(first_slice) == 24 - horizon

    def test_train_window_days_caps_slice_to_sliding_window(self):
        """train_window_days > 0 drops rows older than the sliding window."""
        from signals.ml_model import MLSignalGenerator

        horizon = MLSignalGenerator.LABEL_HORIZON_BARS
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = [
            OHLCVBar(timestamp=start + timedelta(hours=h),
                     open=100.0, high=101.0, low=99.0, close=100.0)
            for h in range(120)
        ]
        config = BacktestConfig(
            symbols=["BTCUSDT"],
            start=bars[0].timestamp, end=bars[-1].timestamp,
        )
        mock_ml = MagicMock()
        mock_ml.train_walk_forward.return_value = (True, {"max_depth": 3})
        feature_df = pd.DataFrame({
            "timestamp": [b.timestamp for b in bars],
        })

        run_backtest(
            bars_by_symbol={"BTCUSDT": bars},
            atr_by_symbol={"BTCUSDT": {b.timestamp: 5.0 for b in bars}},
            filters_by_symbol={},
            aggregator=_HoldAggregatorWithML(mock_ml),
            config=config,
            walk_forward_months=1,
            min_train_days=2,
            feature_df=feature_df,
            train_window_days=1,
        )

        # Retrain at start+2d, label cutoff = 2d - horizon, window = 24h before that
        first_slice = mock_ml.train_walk_forward.call_args_list[0][0][0]
        label_cutoff = start + timedelta(days=2) - timedelta(hours=horizon)
        assert len(first_slice) == 24
        assert first_slice["timestamp"].min() == label_cutoff - timedelta(days=1)
        assert first_slice["timestamp"].max() < label_cutoff

    def test_retrain_events_empty_without_walk_forward(self):
        bars = _bars([(100.0, 101.0, 99.0), (100.0, 101.0, 99.0)])
        config = BacktestConfig(
            symbols=["BTCUSDT"],
            start=bars[0].timestamp, end=bars[-1].timestamp,
        )
        result = run_backtest(
            bars_by_symbol={"BTCUSDT": bars},
            atr_by_symbol={"BTCUSDT": {b.timestamp: 5.0 for b in bars}},
            filters_by_symbol={},
            aggregator=_FakeAggregator(),
            config=config,
        )
        assert result.retrain_events == []

    def test_walk_forward_warning_replaces_lookahead_warning(self):
        """Walk-forward result carries an OOS warning, not the in-sample warning."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = [
            OHLCVBar(timestamp=start + timedelta(hours=h),
                     open=100.0, high=101.0, low=99.0, close=100.0)
            for h in range(5)
        ]
        config = BacktestConfig(
            symbols=["BTCUSDT"],
            start=bars[0].timestamp, end=bars[-1].timestamp,
        )
        mock_ml = MagicMock()
        mock_ml.train_walk_forward.return_value = (False, None)
        feature_df = pd.DataFrame({"timestamp": [b.timestamp for b in bars]})

        result = run_backtest(
            bars_by_symbol={"BTCUSDT": bars},
            atr_by_symbol={"BTCUSDT": {b.timestamp: 5.0 for b in bars}},
            filters_by_symbol={},
            aggregator=_HoldAggregatorWithML(mock_ml),
            config=config,
            walk_forward_months=1,
            min_train_days=30,
            feature_df=feature_df,
        )

        assert any("out-of-sample" in w for w in result.warnings)
        assert not any("training-on-test" in w for w in result.warnings)

    def test_reuses_frozen_params_after_first_retrain(self):
        """The inline retrain loop must tune once (first boundary) and reuse
        those params on every later boundary — mirrors the signal_cache
        precompute path so both walk-forward code paths behave consistently
        and pay the ~40-fit GridSearchCV cost once per run, not once per
        window (~35 windows/run in a real experiment)."""
        from unittest.mock import patch

        import numpy as np
        from sklearn.model_selection import GridSearchCV

        from signals.ml_model import MLSignalGenerator

        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        n_rows = 33 * 24  # 33 days hourly: forces 2 retrain boundaries
        bars = [
            OHLCVBar(timestamp=start + timedelta(hours=h),
                     open=100.0, high=101.0, low=99.0, close=100.0)
            for h in range(n_rows)
        ]
        config = BacktestConfig(
            symbols=["BTCUSDT"],
            start=bars[0].timestamp, end=bars[-1].timestamp,
        )
        rng = np.random.default_rng(7)
        feature_df = pd.DataFrame({
            col: rng.standard_normal(n_rows) for col in MLSignalGenerator.FEATURES
        })
        feature_df["symbol"] = "BTCUSDT"
        feature_df["timestamp"] = [b.timestamp for b in bars]
        feature_df["close"] = 100.0 + rng.standard_normal(n_rows).cumsum()
        horizon = MLSignalGenerator.LABEL_HORIZON_BARS
        feature_df["future_close"] = feature_df["close"].shift(-horizon)
        feature_df["target"] = MLSignalGenerator.compute_target(
            feature_df["close"], feature_df["future_close"]
        )

        real_ml = MLSignalGenerator(db_engine=None)

        with patch("signals.ml_model.GridSearchCV", wraps=GridSearchCV) as mock_grid:
            run_backtest(
                bars_by_symbol={"BTCUSDT": bars},
                atr_by_symbol={"BTCUSDT": {b.timestamp: 5.0 for b in bars}},
                filters_by_symbol={},
                aggregator=_HoldAggregatorWithML(real_ml),
                config=config,
                walk_forward_months=1,
                min_train_days=1,
                feature_df=feature_df,
            )

        assert mock_grid.call_count == 1
