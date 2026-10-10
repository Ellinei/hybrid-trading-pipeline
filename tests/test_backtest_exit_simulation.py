"""Unit tests for backtest.exit_simulation — pure functions, no mocks needed.

Run:
    poetry run pytest tests/test_backtest_exit_simulation.py -v
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from backtest.exit_simulation import (
    OHLCVBar,
    apply_slippage,
    check_exit,
    simulate_entry_fill,
    simulate_exit_fill,
)
from backtest.state import OpenPosition


def _position(stop_loss=90.0, take_profit=120.0, entry_price=100.0, qty=1.0):
    return OpenPosition(
        symbol="BTCUSDT", quantity=qty, entry_price=entry_price,
        entry_time=datetime.now(UTC),
        stop_loss=stop_loss, take_profit=take_profit,
    )


def _bar(low, high, close=None, open_=None, ts=None):
    return OHLCVBar(
        timestamp=ts or datetime.now(UTC),
        open=open_ if open_ is not None else low,
        high=high, low=low,
        close=close if close is not None else high,
    )


class TestApplySlippage:
    def test_buy_slippage_worsens_price_upward(self):
        assert apply_slippage(100.0, slippage_bps=10.0, side="BUY") == pytest.approx(100.1)

    def test_sell_slippage_worsens_price_downward(self):
        assert apply_slippage(100.0, slippage_bps=10.0, side="SELL") == pytest.approx(99.9)

    def test_zero_slippage_is_noop(self):
        assert apply_slippage(100.0, slippage_bps=0.0, side="BUY") == pytest.approx(100.0)


class TestSimulateEntryFill:
    def test_returns_fill_price_and_fee(self):
        fill_price, fee = simulate_entry_fill(
            price=100.0, qty=2.0, fee_rate=0.001, slippage_bps=10.0,
        )
        assert fill_price == pytest.approx(100.1)
        assert fee == pytest.approx(100.1 * 2.0 * 0.001)


class TestSimulateExitFill:
    def test_returns_fill_price_and_fee(self):
        fill_price, fee = simulate_exit_fill(
            price=100.0, qty=2.0, fee_rate=0.001, slippage_bps=10.0,
        )
        assert fill_price == pytest.approx(99.9)
        assert fee == pytest.approx(99.9 * 2.0 * 0.001)


class TestCheckExit:
    def test_no_hit_returns_none(self):
        position = _position(stop_loss=90.0, take_profit=120.0)
        bar = _bar(low=95.0, high=110.0)
        assert check_exit(bar, position) is None

    def test_stop_loss_hit(self):
        position = _position(stop_loss=90.0, take_profit=120.0)
        bar = _bar(low=85.0, high=95.0)
        result = check_exit(bar, position)
        assert result == ("stop_loss", 90.0)

    def test_take_profit_hit(self):
        position = _position(stop_loss=90.0, take_profit=120.0)
        bar = _bar(low=105.0, high=125.0)
        result = check_exit(bar, position)
        assert result == ("take_profit", 120.0)

    def test_both_hit_in_same_bar_assumes_stop_loss_first(self):
        """Conservative tie-break — within-bar order is unknowable without
        tick data, so assume the worse outcome (stop_loss) happened first."""
        position = _position(stop_loss=90.0, take_profit=120.0)
        bar = _bar(low=80.0, high=130.0)
        result = check_exit(bar, position)
        assert result == ("stop_loss", 90.0)

    def test_exact_boundary_low_equals_stop_loss_counts_as_hit(self):
        position = _position(stop_loss=90.0, take_profit=120.0)
        bar = _bar(low=90.0, high=95.0)
        result = check_exit(bar, position)
        assert result == ("stop_loss", 90.0)

    def test_exact_boundary_high_equals_take_profit_counts_as_hit(self):
        position = _position(stop_loss=90.0, take_profit=120.0)
        bar = _bar(low=105.0, high=120.0)
        result = check_exit(bar, position)
        assert result == ("take_profit", 120.0)
