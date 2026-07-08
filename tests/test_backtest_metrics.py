"""Unit tests for backtest.metrics — pure functions over synthetic data.

Run:
    poetry run pytest tests/test_backtest_metrics.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backtest.metrics import (
    exit_reason_breakdown,
    expectancy,
    holding_hours_stats,
    max_drawdown,
    max_single_trade_profit_share,
    per_symbol_metrics,
    per_window_metrics,
    profit_factor,
    sharpe_ratio,
    total_return,
    win_rate,
)


def _equity_curve(values):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [(start + timedelta(hours=i), v) for i, v in enumerate(values)]


class TestWinRate:
    def test_mixed_trades(self):
        trades = [{"pnl": 10.0}, {"pnl": -5.0}, {"pnl": 20.0}]
        assert win_rate(trades) == pytest.approx(2 / 3)

    def test_no_trades_returns_zero(self):
        assert win_rate([]) == 0.0

    def test_all_wins(self):
        assert win_rate([{"pnl": 1.0}, {"pnl": 2.0}]) == pytest.approx(1.0)


class TestProfitFactor:
    def test_mixed_trades(self):
        trades = [{"pnl": 10.0}, {"pnl": -5.0}, {"pnl": 20.0}]
        assert profit_factor(trades) == pytest.approx(30.0 / 5.0)

    def test_no_losses_returns_inf(self):
        trades = [{"pnl": 10.0}, {"pnl": 5.0}]
        assert profit_factor(trades) == float("inf")

    def test_no_trades_returns_zero(self):
        assert profit_factor([]) == 0.0

    def test_no_profits_returns_zero(self):
        trades = [{"pnl": -10.0}, {"pnl": -5.0}]
        assert profit_factor(trades) == 0.0


class TestMaxDrawdown:
    def test_drawdown_from_peak(self):
        equity_curve = _equity_curve([1000.0, 1200.0, 900.0, 1100.0])
        assert max_drawdown(equity_curve) == pytest.approx(0.25)

    def test_monotonic_increase_has_zero_drawdown(self):
        equity_curve = _equity_curve([1000.0, 1100.0, 1200.0])
        assert max_drawdown(equity_curve) == pytest.approx(0.0)

    def test_empty_curve_returns_zero(self):
        assert max_drawdown([]) == 0.0


class TestTotalReturn:
    def test_positive_return(self):
        equity_curve = _equity_curve([1000.0, 1500.0])
        assert total_return(equity_curve) == pytest.approx(0.5)

    def test_negative_return(self):
        equity_curve = _equity_curve([1000.0, 800.0])
        assert total_return(equity_curve) == pytest.approx(-0.2)

    def test_empty_curve_returns_zero(self):
        assert total_return([]) == 0.0

    def test_single_point_returns_zero(self):
        assert total_return(_equity_curve([1000.0])) == 0.0


class TestExpectancy:
    def test_mean_pnl_per_trade(self):
        trades = [{"pnl": 10.0}, {"pnl": -5.0}, {"pnl": 25.0}]
        assert expectancy(trades) == pytest.approx(10.0)

    def test_no_trades_returns_zero(self):
        assert expectancy([]) == 0.0


class TestExitReasonBreakdown:
    def test_counts_per_reason(self):
        trades = [
            {"reason": "signal_close"}, {"reason": "stop_loss"},
            {"reason": "signal_close"}, {},
        ]
        assert exit_reason_breakdown(trades) == {
            "signal_close": 2, "stop_loss": 1, "unknown": 1,
        }

    def test_no_trades_returns_empty(self):
        assert exit_reason_breakdown([]) == {}


class TestHoldingHoursStats:
    def test_median_mean_p90(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        trades = [
            {"entry_time": start, "exit_time": start + timedelta(hours=h)}
            for h in (2.0, 4.0, 6.0, 8.0, 20.0)
        ]
        stats = holding_hours_stats(trades)
        assert stats["median"] == pytest.approx(6.0)
        assert stats["mean"] == pytest.approx(8.0)
        assert stats["p90"] == pytest.approx(20.0)

    def test_no_trades_returns_zeros(self):
        assert holding_hours_stats([]) == {"median": 0.0, "mean": 0.0, "p90": 0.0}


class TestPerWindowMetrics:
    def test_trades_bucketed_by_entry_time(self):
        b0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        b1 = b0 + timedelta(days=30)
        trades = [
            # warm-up trade — excluded
            {"entry_time": b0 - timedelta(hours=1), "pnl": 99.0},
            # window 0: one win, one loss
            {"entry_time": b0, "pnl": 10.0},
            {"entry_time": b0 + timedelta(days=1), "pnl": -5.0},
            # window 1 (open-ended): one win
            {"entry_time": b1 + timedelta(hours=1), "pnl": 7.0},
        ]
        windows = per_window_metrics(trades, [b0, b1])
        assert len(windows) == 2
        assert windows[0]["trade_count"] == 2
        assert windows[0]["pnl"] == pytest.approx(5.0)
        assert windows[0]["profit_factor"] == pytest.approx(2.0)
        assert windows[0]["win_rate"] == pytest.approx(0.5)
        assert windows[1]["trade_count"] == 1
        assert windows[1]["window_end"] is None

    def test_no_boundaries_returns_empty(self):
        assert per_window_metrics([{"entry_time": datetime.now(timezone.utc), "pnl": 1.0}], []) == []


class TestSharpeRatio:
    def test_constant_equity_has_zero_sharpe(self):
        equity_curve = _equity_curve([1000.0, 1000.0, 1000.0])
        assert sharpe_ratio(equity_curve, periods_per_year=24 * 365) == pytest.approx(0.0)

    def test_rising_equity_has_positive_sharpe(self):
        equity_curve = _equity_curve([1000.0, 1010.0, 1025.0, 1015.0, 1040.0])
        assert sharpe_ratio(equity_curve, periods_per_year=24 * 365) > 0

    def test_falling_equity_has_negative_sharpe(self):
        equity_curve = _equity_curve([1000.0, 990.0, 975.0, 985.0, 960.0])
        assert sharpe_ratio(equity_curve, periods_per_year=24 * 365) < 0

    def test_fewer_than_two_points_returns_zero(self):
        assert sharpe_ratio(_equity_curve([1000.0]), periods_per_year=24 * 365) == 0.0
        assert sharpe_ratio([], periods_per_year=24 * 365) == 0.0


class TestPerSymbolMetrics:
    """The LOSO breakdown used throughout the experiment ledger."""

    def test_splits_trades_by_symbol(self):
        trades = [
            {"symbol": "BTCUSDT", "pnl": 10.0},
            {"symbol": "BTCUSDT", "pnl": -4.0},
            {"symbol": "ETHUSDT", "pnl": 5.0},
        ]
        result = per_symbol_metrics(trades)

        assert set(result) == {"BTCUSDT", "ETHUSDT"}
        assert result["BTCUSDT"]["trade_count"] == 2
        assert result["BTCUSDT"]["pnl"] == pytest.approx(6.0)
        assert result["ETHUSDT"]["trade_count"] == 1
        assert result["ETHUSDT"]["profit_factor"] == float("inf")

    def test_empty_trades_returns_empty_dict(self):
        assert per_symbol_metrics([]) == {}


class TestMaxSingleTradeProfitShare:
    """The acceptance rule's '>40% of gross profit from one trade' guard."""

    def test_one_dominant_trade(self):
        trades = [{"pnl": 100.0}, {"pnl": 10.0}, {"pnl": 5.0}, {"pnl": -20.0}]
        assert max_single_trade_profit_share(trades) == pytest.approx(100.0 / 115.0)

    def test_evenly_split_profit(self):
        trades = [{"pnl": 10.0}, {"pnl": 10.0}]
        assert max_single_trade_profit_share(trades) == pytest.approx(0.5)

    def test_no_winning_trades_returns_zero(self):
        assert max_single_trade_profit_share([{"pnl": -5.0}, {"pnl": -1.0}]) == 0.0

    def test_no_trades_returns_zero(self):
        assert max_single_trade_profit_share([]) == 0.0
