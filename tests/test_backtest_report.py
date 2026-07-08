"""Unit tests for backtest.report — the self-contained Plotly HTML report.

Pure data-shaping helpers (monthly bucketing, drawdown series, color
assignment) are tested directly. The full HTML assembly is tested only
for structural properties (valid string output, no crash on empty data,
Plotly bundled inline) — Plotly's own figure JSON isn't re-tested here.

Run:
    poetry run pytest tests/test_backtest_report.py -v
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backtest.report import (
    _drawdown_series,
    _monthly_windows,
    _symbol_color_map,
    build_report_html,
)


def _dt(y, m, d=1):
    return datetime(y, m, d, tzinfo=timezone.utc)


class TestMonthlyWindows:

    def test_buckets_trades_by_entry_month(self):
        trades = [
            {"symbol": "BTCUSDT", "entry_time": _dt(2026, 1, 5), "pnl": 10.0},
            {"symbol": "BTCUSDT", "entry_time": _dt(2026, 1, 20), "pnl": -3.0},
            {"symbol": "BTCUSDT", "entry_time": _dt(2026, 2, 2), "pnl": 5.0},
        ]
        windows = _monthly_windows(trades)

        assert [w["month"] for w in windows] == ["2026-01", "2026-02"]
        assert windows[0]["trade_count"] == 2
        assert windows[1]["trade_count"] == 1

    def test_empty_trades_returns_empty_list(self):
        assert _monthly_windows([]) == []

    def test_each_window_has_profit_factor(self):
        trades = [
            {"symbol": "BTCUSDT", "entry_time": _dt(2026, 1, 5), "pnl": 10.0},
            {"symbol": "BTCUSDT", "entry_time": _dt(2026, 1, 20), "pnl": -5.0},
        ]
        windows = _monthly_windows(trades)
        assert windows[0]["profit_factor"] == pytest.approx(2.0)


class TestDrawdownSeries:

    def test_computes_running_drawdown_percent(self):
        equity_curve = [
            (_dt(2026, 1, 1), 1000.0),
            (_dt(2026, 1, 2), 1100.0),
            (_dt(2026, 1, 3), 990.0),
            (_dt(2026, 1, 4), 1050.0),
        ]
        dd = _drawdown_series(equity_curve)

        assert dd[0] == pytest.approx(0.0)
        assert dd[1] == pytest.approx(0.0)  # new peak
        assert dd[2] == pytest.approx(-10.0)  # 10% below the 1100 peak
        assert dd[3] == pytest.approx(-(1100 - 1050) / 1100 * 100)

    def test_empty_curve_returns_empty_list(self):
        assert _drawdown_series([]) == []


class TestSymbolColorMap:

    def test_assigns_fixed_order_never_random(self):
        m1 = _symbol_color_map(["SOLUSDT", "BTCUSDT", "ETHUSDT"])
        m2 = _symbol_color_map(["ETHUSDT", "BTCUSDT", "SOLUSDT"])
        assert m1 == m2  # same symbols -> same colors regardless of input order

    def test_distinct_colors_for_distinct_symbols(self):
        m = _symbol_color_map(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        assert len(set(m.values())) == 3


class TestBuildReportHtml:

    def _trades(self):
        return [
            {"symbol": "BTCUSDT", "entry_price": 100.0, "exit_price": 105.0,
             "quantity": 1.0, "entry_time": _dt(2026, 1, 5), "exit_time": _dt(2026, 1, 5),
             "pnl": 5.0, "reason": "signal_close"},
            {"symbol": "ETHUSDT", "entry_price": 50.0, "exit_price": 48.0,
             "quantity": 1.0, "entry_time": _dt(2026, 1, 10), "exit_time": _dt(2026, 1, 10),
             "pnl": -2.0, "reason": "stop_loss"},
        ]

    def _metrics(self):
        return {
            "trade_count": 2, "win_rate": 0.5, "profit_factor": 2.5,
            "expectancy": 1.5, "max_drawdown": 0.02, "sharpe_ratio": 0.4,
            "total_return": 0.0003, "exit_reason_breakdown": {"signal_close": 1, "stop_loss": 1},
            "per_symbol": {"BTCUSDT": {"trade_count": 1, "profit_factor": float("inf"),
                                         "win_rate": 1.0, "pnl": 5.0}},
            "max_single_trade_profit_share": 1.0,
        }

    def test_returns_self_contained_html_string(self):
        html = build_report_html(
            exp_id="exp999", metrics=self._metrics(), trades=self._trades(),
            equity_curve=[(_dt(2026, 1, 5), 10000.0), (_dt(2026, 1, 10), 10003.0)],
        )

        assert html.strip().startswith("<!DOCTYPE html") or "<html" in html
        assert "exp999" in html
        assert "plotly" in html.lower()

    def test_no_crash_on_empty_trades_and_equity_curve(self):
        html = build_report_html(
            exp_id="exp999", metrics={
                "trade_count": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "expectancy": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0,
                "total_return": 0.0, "exit_reason_breakdown": {}, "per_symbol": {},
                "max_single_trade_profit_share": 0.0,
            },
            trades=[], equity_curve=[],
        )
        assert "exp999" in html
