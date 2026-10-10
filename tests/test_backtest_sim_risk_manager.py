"""Unit tests for backtest.state and backtest.sim_risk_manager.

Proves SimulatedRiskManager reuses RiskManager.approve_trade/
calculate_position_size/calculate_stops verbatim — only the 5 I/O methods
are swapped for in-memory state. No DB, no exchange, no mocks needed.

Run:
    poetry run pytest tests/test_backtest_sim_risk_manager.py -v
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from backtest.sim_risk_manager import SimulatedRiskManager
from backtest.state import BacktestState, OpenPosition
from execution.risk_manager import RiskConfig
from signals.base import AggregatedSignal, Signal, SignalDirection


def _signal(direction=SignalDirection.BUY, confidence=0.9, symbol="BTCUSDT"):
    return AggregatedSignal(
        direction=direction, confidence=confidence, symbol=symbol,
        timestamp=datetime.now(UTC),
        signals=[Signal(direction, confidence, "technical", symbol, datetime.now(UTC))],
        reasoning="test", weights_used={},
    )


class TestBacktestState:
    def test_starts_empty(self):
        state = BacktestState(balance=1_000.0)
        assert state.positions == {}
        assert state.closed_trades == []
        assert state.equity_curve == []
        assert state.current_atr == {}


class TestSimulatedRiskManagerIOOverrides:
    def test_get_account_balance_reads_state(self):
        state = BacktestState(balance=1_234.0)
        risk = SimulatedRiskManager(state, initial_balance=1_234.0)
        assert risk.get_account_balance() == 1_234.0

    def test_has_open_position_reads_state_positions(self):
        state = BacktestState(balance=1_000.0)
        risk = SimulatedRiskManager(state, initial_balance=1_000.0)
        assert risk.has_open_position("BTCUSDT") is False

        state.positions["BTCUSDT"] = OpenPosition(
            symbol="BTCUSDT", quantity=1.0, entry_price=100.0,
            entry_time=datetime.now(UTC), stop_loss=90.0, take_profit=120.0,
        )
        assert risk.has_open_position("BTCUSDT") is True

    def test_get_open_position_count_reads_state(self):
        state = BacktestState(balance=1_000.0)
        risk = SimulatedRiskManager(state, initial_balance=1_000.0)
        assert risk.get_open_position_count() == 0
        state.positions["BTCUSDT"] = OpenPosition(
            symbol="BTCUSDT", quantity=1.0, entry_price=100.0,
            entry_time=datetime.now(UTC), stop_loss=90.0, take_profit=120.0,
        )
        assert risk.get_open_position_count() == 1

    def test_get_atr_reads_state_current_atr(self):
        state = BacktestState(balance=1_000.0)
        risk = SimulatedRiskManager(state, initial_balance=1_000.0)
        assert risk.get_atr("BTCUSDT") == 0.0
        state.current_atr["BTCUSDT"] = 5.0
        assert risk.get_atr("BTCUSDT") == 5.0

    def test_check_drawdown_sums_closed_trade_pnl(self):
        state = BacktestState(balance=900.0)
        risk = SimulatedRiskManager(state, initial_balance=1_000.0,
                                     config=RiskConfig(max_drawdown_pct=0.05))
        state.closed_trades.append({"pnl": -100.0})
        drawdown, breached = risk.check_drawdown()
        assert drawdown == pytest.approx(0.10)
        assert breached is True

    def test_check_drawdown_tracks_peak_after_gain_then_pullback(self):
        """Mirrors RiskManager's fix: a gain followed by a partial pullback
        must be detected from the peak, not read as 0% just because
        cumulative pnl is still net positive."""
        state = BacktestState(balance=1_500.0)
        risk = SimulatedRiskManager(state, initial_balance=1_000.0,
                                     config=RiskConfig(max_drawdown_pct=0.05))

        state.closed_trades.append({"pnl": 500.0})  # equity peaks at 1500
        drawdown, breached = risk.check_drawdown()
        assert drawdown == pytest.approx(0.0)
        assert breached is False

        state.closed_trades.append({"pnl": -100.0})  # equity now 1400, down from 1500 peak
        drawdown, breached = risk.check_drawdown()
        assert drawdown == pytest.approx((1500.0 - 1400.0) / 1500.0)
        assert breached is True  # ~6.7% >= 5% threshold

    def test_check_drawdown_peak_never_decreases(self):
        state = BacktestState(balance=1_000.0)
        risk = SimulatedRiskManager(state, initial_balance=1_000.0)

        state.closed_trades.append({"pnl": 500.0})
        risk.check_drawdown()
        state.closed_trades.append({"pnl": -400.0})  # net pnl now 100, still above initial
        drawdown, _ = risk.check_drawdown()

        # peak stayed at 1500 (1000+500), not re-lowered to 1100 (1000+100)
        assert drawdown == pytest.approx((1500.0 - 1100.0) / 1500.0)

    def test_get_open_position_risk_sums_entry_minus_stop_times_qty(self):
        state = BacktestState(balance=1_000.0)
        risk = SimulatedRiskManager(state, initial_balance=1_000.0)
        assert risk.get_open_position_risk() == 0.0

        state.positions["BTCUSDT"] = OpenPosition(
            symbol="BTCUSDT", quantity=2.0, entry_price=100.0,
            entry_time=datetime.now(UTC), stop_loss=90.0, take_profit=120.0,
        )
        state.positions["ETHUSDT"] = OpenPosition(
            symbol="ETHUSDT", quantity=1.0, entry_price=50.0,
            entry_time=datetime.now(UTC), stop_loss=45.0, take_profit=60.0,
        )
        # BTC: (100-90)*2 = 20; ETH: (50-45)*1 = 5
        assert risk.get_open_position_risk() == pytest.approx(25.0)


class TestSimulatedRiskManagerReusesLiveGating:
    """approve_trade/calculate_position_size/calculate_stops are inherited,
    unmodified, from RiskManager — this proves they actually run, not just
    that the class is constructible."""

    def test_approve_trade_rejects_hold_same_as_live(self):
        state = BacktestState(balance=1_000.0)
        risk = SimulatedRiskManager(state, initial_balance=1_000.0)
        approved, reason = risk.approve_trade(_signal(direction=SignalDirection.HOLD), price=100.0)
        assert approved is False
        assert "HOLD" in reason

    def test_approve_trade_rejects_sell_with_no_open_position(self):
        state = BacktestState(balance=1_000.0)
        risk = SimulatedRiskManager(state, initial_balance=1_000.0)
        approved, reason = risk.approve_trade(_signal(direction=SignalDirection.SELL), price=100.0)
        assert approved is False
        assert "no open position" in reason

    def test_approve_trade_approves_valid_buy(self):
        state = BacktestState(balance=1_000.0)
        risk = SimulatedRiskManager(state, initial_balance=1_000.0,
                                     config=RiskConfig(min_confidence=0.0))
        state.current_atr["BTCUSDT"] = 5.0
        approved, reason = risk.approve_trade(_signal(confidence=0.9), price=100.0)
        assert approved is True

    def test_calculate_position_size_and_stops_inherited_unmodified(self):
        state = BacktestState(balance=1_000.0)
        risk = SimulatedRiskManager(state, initial_balance=1_000.0)
        size = risk.calculate_position_size(price=100.0, atr=5.0, balance=1_000.0)
        assert size > 0
        stop_loss, take_profit = risk.calculate_stops(price=100.0, atr=5.0)
        assert stop_loss < 100.0 < take_profit

    def test_approve_trade_rejects_when_portfolio_risk_cap_exceeded(self):
        state = BacktestState(balance=1_000.0)
        risk = SimulatedRiskManager(
            state, initial_balance=1_000.0,
            config=RiskConfig(min_confidence=0.0, max_portfolio_risk_pct=0.04),
        )
        state.current_atr["BTCUSDT"] = 5.0
        # existing exposure already at $30; a new ~2%-of-balance ($20) entry
        # would push total past the $40 cap (4% of $1000).
        state.positions["ETHUSDT"] = OpenPosition(
            symbol="ETHUSDT", quantity=1.0, entry_price=100.0,
            entry_time=datetime.now(UTC), stop_loss=70.0, take_profit=160.0,
        )

        approved, reason = risk.approve_trade(
            _signal(confidence=0.9, symbol="BTCUSDT"), price=100.0,
        )

        assert approved is False
        assert "risk cap" in reason

    def test_approve_trade_disabled_cap_ignores_existing_exposure(self):
        state = BacktestState(balance=1_000.0)
        risk = SimulatedRiskManager(state, initial_balance=1_000.0,
                                     config=RiskConfig(min_confidence=0.0))
        state.current_atr["BTCUSDT"] = 5.0
        state.positions["ETHUSDT"] = OpenPosition(
            symbol="ETHUSDT", quantity=1.0, entry_price=100.0,
            entry_time=datetime.now(UTC), stop_loss=70.0, take_profit=160.0,
        )

        approved, _ = risk.approve_trade(_signal(confidence=0.9, symbol="BTCUSDT"), price=100.0)

        assert approved is True
