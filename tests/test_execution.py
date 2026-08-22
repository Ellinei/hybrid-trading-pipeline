"""Unit tests for the execution package.

No real exchange calls or DB connections — everything is mocked.

Run:
    poetry run pytest tests/test_execution.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from execution.risk_manager import RiskConfig, RiskManager
from execution.trade_executor import TradeExecutor
from signals.base import AggregatedSignal, Signal, SignalDirection


def _make_signal(direction=SignalDirection.BUY, confidence=0.9, symbol="BTCUSDT"):
    return AggregatedSignal(
        direction=direction,
        confidence=confidence,
        symbol=symbol,
        timestamp=datetime.now(timezone.utc),
        signals=[
            Signal(direction, confidence, "technical", symbol, datetime.now(timezone.utc)),
            Signal(direction, confidence, "ml", symbol, datetime.now(timezone.utc)),
            Signal(direction, confidence, "sentiment", symbol, datetime.now(timezone.utc)),
        ],
        reasoning="test",
        weights_used={},
    )


def _make_engine_returning(scalar_value):
    """A fake SQLAlchemy engine whose connect().execute(...).scalar() returns scalar_value."""
    conn = MagicMock()
    conn.execute.return_value.scalar.return_value = scalar_value
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    return engine


def _symbol_info(step_size="0.0001", min_qty="0.0001", tick_size="0.01", min_notional="10"):
    return {
        "symbol": "BTCUSDT",
        "filters": [
            {"filterType": "LOT_SIZE", "stepSize": step_size, "minQty": min_qty, "maxQty": "9000"},
            {"filterType": "PRICE_FILTER", "tickSize": tick_size, "minPrice": "0.01", "maxPrice": "1000000"},
            {"filterType": "MIN_NOTIONAL", "minNotional": min_notional},
        ],
    }


class _FakeConn:
    """Routes execute() calls by SQL content so a single fake connection can
    back the whole lifecycle (table bootstrap, open-trade lookup, insert,
    update) within one execute_signal() call."""

    def __init__(self, open_trade_row=None, scalar_value=0, expired_rows=None):
        self.open_trade_row = open_trade_row
        self.scalar_value = scalar_value
        self.expired_rows = expired_rows or []
        self.calls = []  # list of (sql, params)

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.calls.append((sql, params))
        result = MagicMock()
        if "executed_at <=" in sql:
            result.fetchall.return_value = [
                MagicMock(_mapping=row) for row in self.expired_rows
            ]
        elif "SELECT id, quantity" in sql:
            result.fetchone.return_value = (
                MagicMock(_mapping=self.open_trade_row) if self.open_trade_row else None
            )
        elif "INSERT INTO raw.trades" in sql:
            result.fetchone.return_value = MagicMock(
                _mapping={"id": 1, "executed_at": datetime.now(timezone.utc)}
            )
        else:
            result.scalar.return_value = self.scalar_value
        return result

    def commit(self):
        pass


def _make_fake_engine(open_trade_row=None, scalar_value=0, expired_rows=None):
    conn = _FakeConn(open_trade_row=open_trade_row, scalar_value=scalar_value,
                      expired_rows=expired_rows)
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    return engine, conn


def _make_approved_risk(client, engine, atr=5.0, has_open_position=True):
    risk = RiskManager(client=client, db_engine=engine, config=RiskConfig(min_confidence=0.0))
    risk.get_atr = MagicMock(return_value=atr)
    risk.get_open_position_count = MagicMock(return_value=0)
    risk.check_drawdown = MagicMock(return_value=(0.0, False))
    risk.has_open_position = MagicMock(return_value=has_open_position)
    return risk


# ===========================================================================
# 1. RiskManager.calculate_position_size — notional cannot exceed balance
# ===========================================================================

class TestCalculatePositionSize:
    def test_caps_size_to_available_balance_when_atr_is_tiny(self):
        risk = RiskManager(client=MagicMock(), db_engine=MagicMock())
        size = risk.calculate_position_size(price=100.0, atr=0.0001, balance=1_000.0)
        assert size * 100.0 <= 1_000.0 + 1e-6

    def test_returns_zero_when_atr_is_zero(self):
        risk = RiskManager(client=MagicMock(), db_engine=MagicMock())
        assert risk.calculate_position_size(price=100.0, atr=0.0, balance=1_000.0) == 0.0

    def test_returns_zero_when_notional_below_minimum(self):
        risk = RiskManager(client=MagicMock(), db_engine=MagicMock())
        size = risk.calculate_position_size(price=100.0, atr=1000.0, balance=1_000.0)
        assert size == 0.0


# ===========================================================================
# 1b. RiskManager.calculate_stops — shared pure formula (live + backtest)
# ===========================================================================

class TestCalculateStops:
    def test_buy_stop_below_and_target_above_price(self):
        risk = RiskManager(client=MagicMock(), db_engine=MagicMock())
        stop_loss, take_profit = risk.calculate_stops(price=100.0, atr=5.0)
        # defaults: stop_loss_atr_mult=1.5, take_profit_atr_mult=3.0
        assert stop_loss == pytest.approx(100.0 - 5.0 * 1.5)
        assert take_profit == pytest.approx(100.0 + 5.0 * 3.0)

    def test_uses_configured_multiples(self):
        risk = RiskManager(
            client=MagicMock(), db_engine=MagicMock(),
            config=RiskConfig(stop_loss_atr_mult=2.0, take_profit_atr_mult=4.0),
        )
        stop_loss, take_profit = risk.calculate_stops(price=100.0, atr=10.0)
        assert stop_loss == pytest.approx(80.0)
        assert take_profit == pytest.approx(140.0)


# ===========================================================================
# 2. RiskManager.approve_trade — fail-closed gates
# ===========================================================================

class TestApproveTrade:
    def test_rejects_when_balance_unavailable(self):
        client = MagicMock()
        client.get_account_balance.side_effect = Exception("api down")
        engine = _make_engine_returning(0)
        risk = RiskManager(client=client, db_engine=engine)

        approved, reason = risk.approve_trade(_make_signal(), price=100.0)

        assert approved is False
        assert "balance" in reason

    def test_rejects_sell_with_no_open_position(self):
        client = MagicMock()
        client.get_account_balance.return_value = 1_000.0
        engine = _make_engine_returning(0)  # COUNT(*) == 0 -> no open position
        risk = RiskManager(client=client, db_engine=engine)

        approved, reason = risk.approve_trade(
            _make_signal(direction=SignalDirection.SELL), price=100.0
        )

        assert approved is False
        assert "no open position" in reason

    def test_rejects_hold_signal(self):
        risk = RiskManager(client=MagicMock(), db_engine=MagicMock())
        approved, reason = risk.approve_trade(
            _make_signal(direction=SignalDirection.HOLD), price=100.0
        )
        assert approved is False
        assert "HOLD" in reason

    def test_rejects_low_confidence(self):
        risk = RiskManager(client=MagicMock(), db_engine=MagicMock())
        approved, reason = risk.approve_trade(
            _make_signal(confidence=0.1), price=100.0
        )
        assert approved is False
        assert "confidence" in reason


# ===========================================================================
# 2a. RiskManager.get_open_position_risk / portfolio concentration cap —
# max_portfolio_risk_pct defaults to None (disabled): the mechanism is
# opt-in plumbing for a future backtest A/B campaign, not a live default,
# so enabling it here cannot silently change golden-parity replay output.
# ===========================================================================

class TestGetOpenPositionRisk:
    def test_returns_scalar_from_db(self):
        engine = _make_engine_returning(42.5)
        risk = RiskManager(client=MagicMock(), db_engine=engine)
        assert risk.get_open_position_risk() == 42.5

    def test_returns_zero_on_db_failure(self):
        engine, _ = _make_fake_engine()
        risk = RiskManager(client=MagicMock(), db_engine=engine)
        engine.connect.side_effect = Exception("db down")
        assert risk.get_open_position_risk() == 0.0


class TestPortfolioConcentrationCap:
    def _risk(self, balance=1_000.0, existing_risk=0.0, max_portfolio_risk_pct=0.04):
        client = MagicMock()
        client.get_account_balance.return_value = balance
        engine = _make_engine_returning(0)
        risk = RiskManager(
            client=client, db_engine=engine,
            config=RiskConfig(min_confidence=0.0, max_portfolio_risk_pct=max_portfolio_risk_pct),
        )
        risk.get_atr = MagicMock(return_value=5.0)
        risk.get_open_position_count = MagicMock(return_value=0)
        risk.check_drawdown = MagicMock(return_value=(0.0, False))
        risk.get_open_position_risk = MagicMock(return_value=existing_risk)
        return risk

    def test_disabled_by_default(self):
        cfg = RiskConfig()
        assert cfg.max_portfolio_risk_pct is None

    def test_disabled_cap_never_calls_get_open_position_risk(self):
        risk = self._risk(max_portfolio_risk_pct=None)
        risk.get_open_position_risk = MagicMock(return_value=1_000_000.0)

        approved, _ = risk.approve_trade(_make_signal(), price=100.0)

        assert approved is True
        risk.get_open_position_risk.assert_not_called()

    def test_rejects_when_existing_plus_new_risk_exceeds_cap(self):
        # own risk of a single entry ~= balance * max_position_pct (2% = $20);
        # $25 existing + ~$20 new > $40 cap (4% of $1000).
        risk = self._risk(existing_risk=25.0)

        approved, reason = risk.approve_trade(_make_signal(), price=100.0)

        assert approved is False
        assert "risk cap" in reason

    def test_approves_when_existing_plus_new_risk_within_cap(self):
        # $10 existing + ~$20 new = ~$30 <= $40 cap.
        risk = self._risk(existing_risk=10.0)

        approved, reason = risk.approve_trade(_make_signal(), price=100.0)

        assert approved is True


# ===========================================================================
# 2b. RiskManager.check_drawdown — peak-equity tracking (was: hardcoded
# _INITIAL_BALANCE floor, realized-pnl-sum-only, never reset, so a gain
# followed by a partial pullback silently read as 0% drawdown).
# ===========================================================================

class _FakeDrawdownConn:
    """Routes the realized-pnl SUM query to a fixed value and raw.bot_state
    get/set to an in-memory dict, so peak persistence can be exercised
    across multiple check_drawdown() calls on the same fake engine."""

    def __init__(self, total_pnl=0.0, bot_state=None):
        self.total_pnl = total_pnl
        self.bot_state = dict(bot_state or {})
        self.calls = []

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.calls.append((sql, params))
        result = MagicMock()
        if "SUM(pnl)" in sql:
            result.scalar.return_value = self.total_pnl
        elif "SELECT value FROM raw.bot_state" in sql:
            value = self.bot_state.get(params["key"])
            result.fetchone.return_value = (
                MagicMock(_mapping={"value": value}) if value is not None else None
            )
        elif "INSERT INTO raw.bot_state" in sql:
            self.bot_state[params["key"]] = params["value"]
        return result

    def commit(self):
        pass


def _make_drawdown_engine(total_pnl=0.0, bot_state=None):
    conn = _FakeDrawdownConn(total_pnl=total_pnl, bot_state=bot_state)
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    return engine, conn


class TestCheckDrawdownPeakTracking:
    def test_first_call_seeds_baseline_and_peak_at_default(self):
        engine, conn = _make_drawdown_engine(total_pnl=0.0)
        risk = RiskManager(client=MagicMock(), db_engine=engine)

        drawdown, breached = risk.check_drawdown()

        assert drawdown == pytest.approx(0.0)
        assert breached is False
        assert conn.bot_state["drawdown_baseline"] == 10_000.0
        assert conn.bot_state["drawdown_peak"] == 10_000.0

    def test_loss_from_fresh_baseline_matches_old_behavior(self):
        engine, conn = _make_drawdown_engine(
            total_pnl=-1000.0,
            bot_state={"drawdown_baseline": 10_000.0, "drawdown_peak": 10_000.0},
        )
        risk = RiskManager(client=MagicMock(), db_engine=engine,
                            config=RiskConfig(max_drawdown_pct=0.05))

        drawdown, breached = risk.check_drawdown()

        assert drawdown == pytest.approx(0.10)
        assert breached is True

    def test_gain_then_pullback_is_detected_unlike_old_realized_only_formula(self):
        """After a $2000 gain, peak=12000. A later pullback to +$1500
        (still net POSITIVE overall) must show drawdown from the 12000
        peak, not 0.0 — this is the bug being fixed: the old formula read
        0% drawdown any time cumulative pnl was still positive."""
        engine, conn = _make_drawdown_engine(
            total_pnl=2000.0,
            bot_state={"drawdown_baseline": 10_000.0, "drawdown_peak": 10_000.0},
        )
        risk = RiskManager(client=MagicMock(), db_engine=engine)
        risk.check_drawdown()
        assert conn.bot_state["drawdown_peak"] == 12_000.0

        conn.total_pnl = 1500.0
        drawdown, breached = risk.check_drawdown()

        assert drawdown == pytest.approx((12_000 - 11_500) / 12_000)
        assert breached is False  # ~4.2%, under the default 10% trip

    def test_peak_never_decreases_across_calls(self):
        engine, conn = _make_drawdown_engine(
            total_pnl=500.0,
            bot_state={"drawdown_baseline": 10_000.0, "drawdown_peak": 10_000.0},
        )
        risk = RiskManager(client=MagicMock(), db_engine=engine)
        risk.check_drawdown()
        assert conn.bot_state["drawdown_peak"] == 10_500.0

        conn.total_pnl = 100.0  # pullback — must not lower the recorded peak
        risk.check_drawdown()

        assert conn.bot_state["drawdown_peak"] == 10_500.0

    def test_db_failure_fails_open_same_as_before(self):
        engine, conn = _make_drawdown_engine()
        risk = RiskManager(client=MagicMock(), db_engine=engine)
        engine.connect.side_effect = Exception("db unreachable")  # outage starts after construction

        drawdown, breached = risk.check_drawdown()

        assert drawdown == 0.0
        assert breached is False


class TestResetDrawdownBaseline:
    def test_reset_reanchors_baseline_and_peak_to_explicit_value(self):
        engine, conn = _make_drawdown_engine(
            bot_state={"drawdown_baseline": 10_000.0, "drawdown_peak": 12_000.0},
        )
        risk = RiskManager(client=MagicMock(), db_engine=engine)

        risk.reset_drawdown_baseline(new_baseline=9_000.0)

        assert conn.bot_state["drawdown_baseline"] == 9_000.0
        assert conn.bot_state["drawdown_peak"] == 9_000.0

    def test_reset_defaults_to_current_account_balance(self):
        engine, conn = _make_drawdown_engine()
        client = MagicMock()
        client.get_account_balance.return_value = 8_500.0
        risk = RiskManager(client=client, db_engine=engine)

        risk.reset_drawdown_baseline()

        assert conn.bot_state["drawdown_baseline"] == 8_500.0
        assert conn.bot_state["drawdown_peak"] == 8_500.0

    def test_reset_raises_when_balance_unavailable_and_no_explicit_value(self):
        engine, conn = _make_drawdown_engine()
        client = MagicMock()
        client.get_account_balance.side_effect = Exception("api down")
        risk = RiskManager(client=client, db_engine=engine)

        with pytest.raises(RuntimeError):
            risk.reset_drawdown_baseline()


# ===========================================================================
# 3. TradeExecutor — live_orders_enabled defaults
# ===========================================================================

class TestLiveOrdersEnabledDefault:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("LIVE_ORDERS_ENABLED", raising=False)
        engine, _ = _make_fake_engine()
        executor = TradeExecutor(client=MagicMock(), risk_manager=MagicMock(), db_engine=engine)
        assert executor.live_orders_enabled is False

    def test_enabled_when_env_set_true(self, monkeypatch):
        monkeypatch.setenv("LIVE_ORDERS_ENABLED", "true")
        engine, _ = _make_fake_engine()
        executor = TradeExecutor(client=MagicMock(), risk_manager=MagicMock(), db_engine=engine)
        assert executor.live_orders_enabled is True

    def test_explicit_constructor_arg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("LIVE_ORDERS_ENABLED", "true")
        engine, _ = _make_fake_engine()
        executor = TradeExecutor(
            client=MagicMock(), risk_manager=MagicMock(), db_engine=engine,
            live_orders_enabled=False,
        )
        assert executor.live_orders_enabled is False


# ===========================================================================
# 4. TradeExecutor._execute_buy_entry — OCO placement after fill
# ===========================================================================

class TestBuyEntryWithOCO:
    def _client(self, price=100.0, balance=1_000.0):
        client = MagicMock()
        client.get_current_price.return_value = price
        client.get_account_balance.return_value = balance
        client.get_symbol_info_cached.return_value = _symbol_info()
        client.client.order_market_buy.return_value = {
            "symbol": "BTCUSDT",
            "executedQty": "2.6666",
            "fills": [
                {"price": "100.0", "qty": "2.6666", "commission": "0.0026666", "commissionAsset": "BTC"}
            ],
        }
        client.client.create_oco_order.return_value = {"orderListId": 12345}
        return client

    def test_successful_oco_placement_records_order_list_id(self, monkeypatch):
        monkeypatch.setenv("LIVE_ORDERS_ENABLED", "true")
        client = self._client()
        engine, conn = _make_fake_engine()
        risk = _make_approved_risk(client, engine)
        executor = TradeExecutor(client=client, risk_manager=risk, db_engine=engine)

        result = executor.execute_signal(_make_signal(), dry_run=False)

        assert result is not None
        assert result["status"] == "open"
        assert result["oco_order_list_id"] == 12345
        insert_params = next(p for sql, p in conn.calls if "INSERT INTO raw.trades" in sql)
        assert insert_params["oco_order_list_id"] == 12345

    def test_oco_failure_after_fill_logs_critical_and_records_unprotected_open_trade(self, monkeypatch):
        monkeypatch.setenv("LIVE_ORDERS_ENABLED", "true")
        client = self._client()
        client.client.create_oco_order.side_effect = Exception("exchange rejected OCO")
        engine, conn = _make_fake_engine()
        risk = _make_approved_risk(client, engine)
        executor = TradeExecutor(client=client, risk_manager=risk, db_engine=engine)

        result = executor.execute_signal(_make_signal(), dry_run=False)

        # The dangerous case must stay visible, not silently dropped.
        assert result is not None
        assert result["status"] == "open"
        assert result["oco_order_list_id"] is None

    def test_qty_rounded_down_to_lot_size_before_buy_order(self, monkeypatch):
        monkeypatch.setenv("LIVE_ORDERS_ENABLED", "true")
        client = self._client()
        client.get_symbol_info_cached.return_value = _symbol_info(step_size="0.001")
        engine, conn = _make_fake_engine()
        risk = _make_approved_risk(client, engine)
        executor = TradeExecutor(client=client, risk_manager=risk, db_engine=engine)

        executor.execute_signal(_make_signal(), dry_run=False)

        _, kwargs = client.client.order_market_buy.call_args
        # step 0.001 -> at most 3 decimal places, never the raw 6-decimal calc.
        assert round(kwargs["quantity"], 3) == kwargs["quantity"]

    def test_rejects_trade_when_notional_below_minimum_after_rounding(self, monkeypatch):
        monkeypatch.setenv("LIVE_ORDERS_ENABLED", "true")
        client = self._client(balance=5.0)  # tiny balance -> tiny qty -> below MIN_NOTIONAL
        engine, conn = _make_fake_engine()
        risk = _make_approved_risk(client, engine)
        executor = TradeExecutor(client=client, risk_manager=risk, db_engine=engine)

        result = executor.execute_signal(_make_signal(), dry_run=False)

        assert result is None
        client.client.order_market_buy.assert_not_called()

    def test_dry_run_records_without_touching_exchange(self):
        client = self._client()
        engine, conn = _make_fake_engine()
        risk = _make_approved_risk(client, engine)
        executor = TradeExecutor(client=client, risk_manager=risk, db_engine=engine)

        result = executor.execute_signal(_make_signal(), dry_run=True)

        assert result is not None
        assert result["status"] == "dry_run"
        client.client.order_market_buy.assert_not_called()

    def test_live_orders_disabled_blocks_order_placement(self):
        client = self._client()
        engine, conn = _make_fake_engine()
        risk = _make_approved_risk(client, engine)
        executor = TradeExecutor(
            client=client, risk_manager=risk, db_engine=engine, live_orders_enabled=False
        )

        result = executor.execute_signal(_make_signal(), dry_run=False)

        assert result is None
        client.client.order_market_buy.assert_not_called()


# ===========================================================================
# 5. TradeExecutor._execute_sell_close — close the ACTUAL held quantity
# ===========================================================================

class TestSellCloseUsesActualHeldQuantity:
    def _open_trade_row(self, order_list_id=999):
        return {"id": 5, "quantity": 0.05, "entry_price": 100.0, "oco_order_list_id": order_list_id}

    def test_close_sells_recorded_quantity_not_recomputed_size(self, monkeypatch):
        monkeypatch.setenv("LIVE_ORDERS_ENABLED", "true")
        client = MagicMock()
        client.get_current_price.return_value = 110.0
        client.client.order_market_sell.return_value = {
            "symbol": "BTCUSDT", "executedQty": "0.05",
            "fills": [{"price": "110.0", "qty": "0.05", "commission": "0.0", "commissionAsset": "USDT"}],
        }
        engine, conn = _make_fake_engine(open_trade_row=self._open_trade_row())
        risk = _make_approved_risk(client, engine, has_open_position=True)
        risk.calculate_position_size = MagicMock(return_value=0.5)  # must NOT be used
        executor = TradeExecutor(client=client, risk_manager=risk, db_engine=engine)

        executor.execute_signal(_make_signal(direction=SignalDirection.SELL), dry_run=False)

        risk.calculate_position_size.assert_not_called()
        _, kwargs = client.client.order_market_sell.call_args
        assert kwargs["quantity"] == 0.05

    def test_cancels_oco_before_market_sell_when_order_list_id_present(self, monkeypatch):
        monkeypatch.setenv("LIVE_ORDERS_ENABLED", "true")
        client = MagicMock()
        client.get_current_price.return_value = 110.0
        client.client.order_market_sell.return_value = {
            "symbol": "BTCUSDT", "executedQty": "0.05", "fills": [],
        }
        engine, conn = _make_fake_engine(open_trade_row=self._open_trade_row(order_list_id=999))
        risk = _make_approved_risk(client, engine, has_open_position=True)
        executor = TradeExecutor(client=client, risk_manager=risk, db_engine=engine)

        executor.execute_signal(_make_signal(direction=SignalDirection.SELL), dry_run=False)

        names = [c[0] for c in client.client.mock_calls]
        assert names.index("v3_delete_order_list") < names.index("order_market_sell")
        _, kwargs = client.client.v3_delete_order_list.call_args
        assert kwargs["orderListId"] == 999

    def test_oco_cancel_failure_treated_as_already_closed_externally(self, monkeypatch):
        monkeypatch.setenv("LIVE_ORDERS_ENABLED", "true")
        client = MagicMock()
        client.get_current_price.return_value = 110.0
        client.client.v3_delete_order_list.side_effect = Exception("Unknown order list")
        engine, conn = _make_fake_engine(open_trade_row=self._open_trade_row(order_list_id=999))
        risk = _make_approved_risk(client, engine, has_open_position=True)
        executor = TradeExecutor(client=client, risk_manager=risk, db_engine=engine)

        result = executor.execute_signal(_make_signal(direction=SignalDirection.SELL), dry_run=False)

        client.client.order_market_sell.assert_not_called()
        assert result["status"] == "closed_externally"

    def test_close_updates_existing_row_instead_of_inserting_new_one(self, monkeypatch):
        monkeypatch.setenv("LIVE_ORDERS_ENABLED", "true")
        client = MagicMock()
        client.get_current_price.return_value = 110.0
        client.client.order_market_sell.return_value = {
            "symbol": "BTCUSDT", "executedQty": "0.05", "fills": [],
        }
        engine, conn = _make_fake_engine(open_trade_row=self._open_trade_row(order_list_id=None))
        risk = _make_approved_risk(client, engine, has_open_position=True)
        executor = TradeExecutor(client=client, risk_manager=risk, db_engine=engine)

        executor.execute_signal(_make_signal(direction=SignalDirection.SELL), dry_run=False)

        sqls = [sql for sql, _ in conn.calls]
        assert any("UPDATE raw.trades" in sql for sql in sqls)
        assert not any("INSERT INTO raw.trades" in sql for sql in sqls)

    def test_pnl_computed_correctly_on_close(self, monkeypatch):
        monkeypatch.setenv("LIVE_ORDERS_ENABLED", "true")
        client = MagicMock()
        client.get_current_price.return_value = 110.0
        client.client.order_market_sell.return_value = {
            "symbol": "BTCUSDT", "executedQty": "2", "fills": [],
        }
        row = {"id": 5, "quantity": 2.0, "entry_price": 100.0, "oco_order_list_id": None}
        engine, conn = _make_fake_engine(open_trade_row=row)
        risk = _make_approved_risk(client, engine, has_open_position=True)
        executor = TradeExecutor(client=client, risk_manager=risk, db_engine=engine)

        result = executor.execute_signal(_make_signal(direction=SignalDirection.SELL), dry_run=False)

        assert result["pnl"] == pytest.approx(20.0)

    def test_signal_close_records_close_reason(self, monkeypatch):
        monkeypatch.setenv("LIVE_ORDERS_ENABLED", "true")
        client = MagicMock()
        client.get_current_price.return_value = 110.0
        client.client.order_market_sell.return_value = {
            "symbol": "BTCUSDT", "executedQty": "0.05", "fills": [],
        }
        engine, conn = _make_fake_engine(open_trade_row=self._open_trade_row(order_list_id=None))
        risk = _make_approved_risk(client, engine, has_open_position=True)
        executor = TradeExecutor(client=client, risk_manager=risk, db_engine=engine)

        executor.execute_signal(_make_signal(direction=SignalDirection.SELL), dry_run=False)

        update_params = next(p for sql, p in conn.calls if "UPDATE raw.trades" in sql)
        assert update_params["close_reason"] == "signal_close"


# ===========================================================================
# 6. TradeExecutor.close_expired_positions — live time-stop (the backtest's
# max_hold_bars mechanism, engine.py:273-282, had no live counterpart).
# ===========================================================================

class TestCloseExpiredPositions:
    def _expired_row(self, oco_order_list_id=None, hours_old=30):
        return {
            "id": 7, "symbol": "BTCUSDT", "quantity": 0.05, "entry_price": 100.0,
            "oco_order_list_id": oco_order_list_id,
            "executed_at": datetime.now(timezone.utc) - timedelta(hours=hours_old),
        }

    def test_max_hold_bars_zero_disables_time_stop_no_db_touch(self):
        client = MagicMock()
        engine, conn = _make_fake_engine()
        executor = TradeExecutor(client=client, risk_manager=MagicMock(), db_engine=engine)
        conn.calls.clear()

        result = executor.close_expired_positions(max_hold_bars=0)

        assert result == []
        assert conn.calls == []

    def test_expired_position_market_sold_and_recorded_as_time_stop(self, monkeypatch):
        monkeypatch.setenv("LIVE_ORDERS_ENABLED", "true")
        client = MagicMock()
        client.get_current_price.return_value = 105.0
        client.client.order_market_sell.return_value = {
            "symbol": "BTCUSDT", "executedQty": "0.05", "fills": [],
        }
        engine, conn = _make_fake_engine(expired_rows=[self._expired_row()])
        executor = TradeExecutor(client=client, risk_manager=MagicMock(), db_engine=engine)

        results = executor.close_expired_positions(max_hold_bars=24)

        assert len(results) == 1
        assert results[0]["status"] == "closed"
        client.client.order_market_sell.assert_called_once()
        update_params = next(p for sql, p in conn.calls if "UPDATE raw.trades" in sql)
        assert update_params["close_reason"] == "time_stop"

    def test_cancels_resting_oco_before_market_sell(self, monkeypatch):
        monkeypatch.setenv("LIVE_ORDERS_ENABLED", "true")
        client = MagicMock()
        client.get_current_price.return_value = 105.0
        client.client.order_market_sell.return_value = {
            "symbol": "BTCUSDT", "executedQty": "0.05", "fills": [],
        }
        engine, conn = _make_fake_engine(expired_rows=[self._expired_row(oco_order_list_id=999)])
        executor = TradeExecutor(client=client, risk_manager=MagicMock(), db_engine=engine)

        executor.close_expired_positions(max_hold_bars=24)

        names = [c[0] for c in client.client.mock_calls]
        assert names.index("v3_delete_order_list") < names.index("order_market_sell")

    def test_already_filled_oco_marked_closed_externally_not_double_sold(self, monkeypatch):
        monkeypatch.setenv("LIVE_ORDERS_ENABLED", "true")
        client = MagicMock()
        client.get_current_price.return_value = 105.0
        client.client.v3_delete_order_list.side_effect = Exception("Unknown order list")
        engine, conn = _make_fake_engine(expired_rows=[self._expired_row(oco_order_list_id=999)])
        executor = TradeExecutor(client=client, risk_manager=MagicMock(), db_engine=engine)

        results = executor.close_expired_positions(max_hold_bars=24)

        client.client.order_market_sell.assert_not_called()
        assert results[0]["status"] == "closed_externally"

    def test_dry_run_records_without_touching_exchange(self):
        client = MagicMock()
        client.get_current_price.return_value = 105.0
        engine, conn = _make_fake_engine(expired_rows=[self._expired_row()])
        executor = TradeExecutor(client=client, risk_manager=MagicMock(), db_engine=engine)

        results = executor.close_expired_positions(max_hold_bars=24, dry_run=True)

        assert results[0]["status"] == "dry_run"
        client.client.order_market_sell.assert_not_called()
        update_params = next(p for sql, p in conn.calls if "UPDATE raw.trades" in sql)
        assert update_params["close_reason"] == "time_stop"

    def test_only_fetches_trades_older_than_max_hold_bars(self, monkeypatch):
        """The cutoff bind param must reflect max_hold_bars — not a query
        that returns everything and filters in Python."""
        monkeypatch.setenv("LIVE_ORDERS_ENABLED", "true")
        client = MagicMock()
        client.get_current_price.return_value = 105.0
        client.client.order_market_sell.return_value = {
            "symbol": "BTCUSDT", "executedQty": "0.05", "fills": [],
        }
        engine, conn = _make_fake_engine(expired_rows=[self._expired_row()])
        executor = TradeExecutor(client=client, risk_manager=MagicMock(), db_engine=engine)

        before = datetime.now(timezone.utc)
        executor.close_expired_positions(max_hold_bars=24)

        select_params = next(p for sql, p in conn.calls if "executed_at <=" in sql)
        expected_cutoff = before - timedelta(hours=24)
        assert abs((select_params["cutoff"] - expected_cutoff).total_seconds()) < 5
