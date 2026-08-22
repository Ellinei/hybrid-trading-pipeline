"""Executes approved signals: places orders on Binance testnet and records them.

dry_run=True logs everything and writes a 'dry_run' status row to raw.trades
without touching the exchange. Use for testing the full pipeline end-to-end.

Live order placement is gated by `live_orders_enabled` (default: read from
the LIVE_ORDERS_ENABLED env var). A BUY entry places a market buy followed
immediately by a protective OCO sell (stop-loss + take-profit) sized to the
ACTUAL filled quantity. A SELL closes the existing open position using its
RECORDED quantity — never a freshly computed size — cancelling any resting
OCO first so the manual sell isn't blocked by locked balance.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import text

from execution.exchange_filters import (
    FilterViolation,
    parse_symbol_filters,
    round_price_to_tick,
    round_qty_down,
    validate_notional,
)
from execution.risk_manager import RiskManager
from ingestion.binance_client import BinanceClientWrapper
from signals.base import AggregatedSignal, SignalDirection

log = structlog.get_logger()

_INSERT_TRADE = text("""
    INSERT INTO raw.trades (
        symbol, direction, entry_price, quantity,
        stop_loss, take_profit, signal_confidence,
        signal_reasoning, technical_signal, ml_signal,
        sentiment_signal, status, oco_order_list_id
    ) VALUES (
        :symbol, :direction, :entry_price, :quantity,
        :stop_loss, :take_profit, :signal_confidence,
        :signal_reasoning, :technical_signal, :ml_signal,
        :sentiment_signal, :status, :oco_order_list_id
    ) RETURNING id, executed_at
""")

_SELECT_OPEN_TRADE = text("""
    SELECT id, quantity, entry_price, oco_order_list_id
    FROM raw.trades
    WHERE symbol = :symbol AND status = 'open' AND direction = 'BUY'
    ORDER BY executed_at DESC LIMIT 1
""")

_SELECT_EXPIRED_OPEN_TRADES = text("""
    SELECT id, symbol, quantity, entry_price, oco_order_list_id, executed_at
    FROM raw.trades
    WHERE status = 'open' AND direction = 'BUY' AND executed_at <= :cutoff
""")

_UPDATE_CLOSED_TRADE = text("""
    UPDATE raw.trades
    SET status = :status, closed_at = now(), pnl = :pnl, close_reason = :close_reason
    WHERE id = :id
""")


class TradeExecutor:
    """Orchestrates order placement and trade recording."""

    def __init__(
        self,
        client: BinanceClientWrapper,
        risk_manager: RiskManager,
        db_engine,
        live_orders_enabled: bool | None = None,
    ) -> None:
        self.client = client
        self.risk   = risk_manager
        self.engine = db_engine
        self.live_orders_enabled = (
            live_orders_enabled
            if live_orders_enabled is not None
            else os.getenv("LIVE_ORDERS_ENABLED", "false").lower() == "true"
        )
        self._ensure_trades_table()

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _ensure_trades_table(self) -> None:
        """Create/upgrade raw.trades — idempotent."""
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS raw.trades (
                    id                SERIAL        PRIMARY KEY,
                    symbol            VARCHAR(20)   NOT NULL,
                    direction         VARCHAR(10)   NOT NULL,
                    entry_price       NUMERIC(20,8),
                    quantity          NUMERIC(20,8),
                    stop_loss         NUMERIC(20,8),
                    take_profit       NUMERIC(20,8),
                    signal_confidence NUMERIC(6,4),
                    signal_reasoning  TEXT,
                    technical_signal  VARCHAR(10),
                    ml_signal         VARCHAR(10),
                    sentiment_signal  VARCHAR(10),
                    status            VARCHAR(20)   DEFAULT 'open',
                    executed_at       TIMESTAMPTZ   DEFAULT now(),
                    closed_at         TIMESTAMPTZ,
                    pnl               NUMERIC(20,8)
                )
            """))
            conn.execute(text(
                "ALTER TABLE raw.trades "
                "ADD COLUMN IF NOT EXISTS oco_order_list_id BIGINT"
            ))
            conn.execute(text(
                "ALTER TABLE raw.trades "
                "ADD COLUMN IF NOT EXISTS close_reason VARCHAR(20)"
            ))
            conn.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_sub_signals(
        self, signal: AggregatedSignal
    ) -> tuple[str, str, str]:
        """Extract per-source directions from the aggregated signal."""
        dirs = {s.source: s.direction.value for s in signal.signals}
        return (
            dirs.get("technical", "HOLD"),
            dirs.get("ml",        "HOLD"),
            dirs.get("sentiment", "HOLD"),
        )

    def _parse_fill(
        self, order_resp: dict, fallback_qty: float, fallback_price: float
    ) -> tuple[float, float]:
        """Net filled qty (minus base-asset commission) + avg fill price.

        Requires newOrderRespType='FULL' on the order call. Falls back to
        the requested qty/price if the response has no fill detail.
        """
        fills = order_resp.get("fills") or []
        if not fills:
            return fallback_qty, fallback_price

        executed_qty = float(order_resp.get("executedQty", fallback_qty))
        # USDT-quoted pairs only — see module docstring / Phase 7 plan.
        base_asset = order_resp["symbol"].replace("USDT", "")
        commission_in_base = sum(
            float(f["commission"]) for f in fills
            if f.get("commissionAsset") == base_asset
        )
        net_qty = executed_qty - commission_in_base

        notional = sum(float(f["price"]) * float(f["qty"]) for f in fills)
        avg_price = notional / executed_qty if executed_qty else fallback_price

        return net_qty, avg_price

    def _insert_trade(
        self, signal, price, qty, stop_loss, take_profit,
        tech, ml, sent, status, oco_order_list_id,
    ) -> dict:
        params = {
            "symbol":            signal.symbol,
            "direction":         signal.direction.value,
            "entry_price":       price,
            "quantity":          qty,
            "stop_loss":         stop_loss,
            "take_profit":       take_profit,
            "signal_confidence": signal.confidence,
            "signal_reasoning":  signal.reasoning,
            "technical_signal":  tech,
            "ml_signal":         ml,
            "sentiment_signal":  sent,
            "status":            status,
            "oco_order_list_id": oco_order_list_id,
        }
        with self.engine.connect() as conn:
            row = conn.execute(_INSERT_TRADE, params).fetchone()
            conn.commit()

        trade = {**params,
                 "id":          row._mapping["id"],
                 "executed_at": row._mapping["executed_at"]}
        log.info("trade_recorded",
                 id=trade["id"], symbol=signal.symbol,
                 direction=signal.direction.value,
                 price=round(price, 2), qty=qty, status=status)
        return trade

    def _get_open_trade(self, symbol: str) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(_SELECT_OPEN_TRADE, {"symbol": symbol}).fetchone()
        return dict(row._mapping) if row else None

    def _close_trade(self, trade_id, entry_price, exit_price, qty, status,
                      close_reason: str | None = None) -> dict:
        pnl = (exit_price - entry_price) * qty
        with self.engine.connect() as conn:
            conn.execute(_UPDATE_CLOSED_TRADE,
                         {"status": status, "pnl": pnl, "id": trade_id,
                          "close_reason": close_reason})
            conn.commit()
        log.info("trade_closed", id=trade_id, exit_price=round(exit_price, 2),
                  pnl=round(pnl, 2), status=status, close_reason=close_reason)
        return {"id": trade_id, "status": status, "pnl": pnl, "exit_price": exit_price,
                "close_reason": close_reason}

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_signal(
        self, signal: AggregatedSignal, dry_run: bool = False
    ) -> dict | None:
        """Validate → dispatch to entry or close. None if rejected or on error."""
        try:
            price = self.client.get_current_price(signal.symbol)
        except Exception as exc:
            log.error("price_fetch_failed", symbol=signal.symbol, error=str(exc))
            return None

        approved, reason = self.risk.approve_trade(signal, price)
        if not approved:
            log.info("trade_rejected", symbol=signal.symbol, reason=reason,
                     direction=signal.direction.value, confidence=signal.confidence)
            return None

        if signal.direction == SignalDirection.BUY:
            return self._execute_buy_entry(signal, price, dry_run)
        # approve_trade already rejected SELL with no open position.
        return self._execute_sell_close(signal, price, dry_run)

    def _execute_buy_entry(
        self, signal: AggregatedSignal, price: float, dry_run: bool
    ) -> dict | None:
        atr     = self.risk.get_atr(signal.symbol)
        balance = self.risk.get_account_balance()
        if balance is None:
            log.error("balance_unavailable_after_approval", symbol=signal.symbol)
            return None

        info    = self.client.get_symbol_info_cached(signal.symbol)
        filters = parse_symbol_filters(info)

        qty = self.risk.calculate_position_size(price, atr, balance)
        qty = round_qty_down(qty, filters)
        if qty <= 0:
            log.info("trade_rejected", symbol=signal.symbol,
                     reason="qty rounds to 0 after LOT_SIZE filter")
            return None

        stop_loss, take_profit = self.risk.calculate_stops(price, atr)
        stop_loss   = round_price_to_tick(stop_loss, filters)
        take_profit = round_price_to_tick(take_profit, filters)
        stop_limit  = round_price_to_tick(stop_loss * 0.999, filters)

        try:
            validate_notional(qty, price, filters)
        except FilterViolation as exc:
            log.info("trade_rejected", symbol=signal.symbol, reason=str(exc))
            return None

        tech, ml, sent = self._get_sub_signals(signal)

        if dry_run:
            log.info("DRY_RUN", symbol=signal.symbol, direction="BUY",
                      price=round(price, 2), quantity=qty,
                      stop_loss=round(stop_loss, 2), take_profit=round(take_profit, 2),
                      confidence=round(signal.confidence, 3), reasoning=signal.reasoning)
            return self._insert_trade(signal, price, qty, stop_loss, take_profit,
                                       tech, ml, sent, "dry_run", oco_order_list_id=None)

        if not self.live_orders_enabled:
            log.warning("live_orders_disabled", symbol=signal.symbol)
            return None

        try:
            buy_resp = self.client.client.order_market_buy(
                symbol=signal.symbol, quantity=qty, newOrderRespType="FULL",
            )
        except Exception as exc:
            log.error("buy_order_failed", symbol=signal.symbol, error=str(exc))
            return None

        filled_qty, avg_fill_price = self._parse_fill(buy_resp, qty, price)
        sell_qty = round_qty_down(filled_qty, filters)

        if sell_qty <= 0:
            log.critical("UNPROTECTED_POSITION_NO_OCO_POSSIBLE",
                          symbol=signal.symbol, filled_qty=filled_qty,
                          reason="sell_qty rounds to 0 after fee + LOT_SIZE")
            return self._insert_trade(signal, avg_fill_price, filled_qty, stop_loss,
                                       take_profit, tech, ml, sent, "open",
                                       oco_order_list_id=None)

        try:
            # Current Binance OCO API: "above" order fires when price rises
            # (take-profit, plain LIMIT_MAKER), "below" order fires when
            # price falls (stop-loss, STOP_LOSS_LIMIT with separate trigger/
            # limit price). Legacy price/stopPrice/stopLimitPrice params
            # don't exist on this client version.
            oco_resp = self.client.client.create_oco_order(
                symbol=signal.symbol, side="SELL", quantity=sell_qty,
                aboveType="LIMIT_MAKER", abovePrice=f"{take_profit:.8f}",
                belowType="STOP_LOSS_LIMIT", belowStopPrice=f"{stop_loss:.8f}",
                belowPrice=f"{stop_limit:.8f}", belowTimeInForce="GTC",
            )
            order_list_id = oco_resp["orderListId"]
        except Exception as exc:
            log.critical("UNPROTECTED_POSITION_OCO_FAILED",
                          symbol=signal.symbol, filled_qty=filled_qty,
                          avg_fill_price=avg_fill_price, error=str(exc),
                          action_required="manually place SL/TP or close position")
            return self._insert_trade(signal, avg_fill_price, filled_qty, stop_loss,
                                       take_profit, tech, ml, sent, "open",
                                       oco_order_list_id=None)

        return self._insert_trade(signal, avg_fill_price, filled_qty, stop_loss,
                                   take_profit, tech, ml, sent, "open",
                                   oco_order_list_id=order_list_id)

    def _execute_sell_close(
        self, signal: AggregatedSignal, price: float, dry_run: bool
    ) -> dict | None:
        open_trade = self._get_open_trade(signal.symbol)
        if open_trade is None:
            log.error("sell_close_no_open_trade", symbol=signal.symbol)
            return None
        return self._close_open_trade(signal.symbol, open_trade, price, dry_run,
                                       close_reason="signal_close")

    def _close_open_trade(
        self, symbol: str, open_trade: dict, price: float, dry_run: bool,
        close_reason: str,
    ) -> dict | None:
        """Cancel any resting OCO then market-close *open_trade* at *price*,
        closing the ACTUAL recorded quantity — never a freshly computed
        size. Shared by _execute_sell_close (signal-driven) and
        close_expired_positions (time-stop): an OCO that already filled
        externally is recorded as closed_externally with no close_reason
        (which leg fired isn't knowable without an extra order-status
        call), not double-sold.
        """
        qty           = float(open_trade["quantity"])
        entry_price   = float(open_trade["entry_price"])
        order_list_id = open_trade["oco_order_list_id"]

        if dry_run:
            log.info("DRY_RUN_CLOSE", symbol=symbol, quantity=qty,
                      exit_price=round(price, 2), close_reason=close_reason)
            return self._close_trade(open_trade["id"], entry_price, price, qty,
                                      "dry_run", close_reason=close_reason)

        if not self.live_orders_enabled:
            log.warning("live_orders_disabled", symbol=symbol)
            return None

        if order_list_id is not None:
            try:
                # No cancel_oco_order on this client version — the spot OCO
                # cancel endpoint (DELETE /api/v3/orderList) is exposed as
                # the generic v3_delete_order_list.
                self.client.client.v3_delete_order_list(
                    symbol=symbol, orderListId=order_list_id
                )
            except Exception as exc:
                log.warning("oco_cancel_failed_assuming_already_closed",
                            symbol=symbol, order_list_id=order_list_id,
                            error=str(exc))
                return self._close_trade(open_trade["id"], entry_price, price, qty,
                                          "closed_externally", close_reason=None)

        try:
            sell_resp = self.client.client.order_market_sell(
                symbol=symbol, quantity=qty, newOrderRespType="FULL",
            )
        except Exception as exc:
            log.error("sell_close_order_failed", symbol=symbol, error=str(exc))
            return None

        _, avg_exit_price = self._parse_fill(sell_resp, qty, price)
        return self._close_trade(open_trade["id"], entry_price, avg_exit_price, qty,
                                  "closed", close_reason=close_reason)

    def close_expired_positions(
        self, max_hold_bars: int, dry_run: bool = False
    ) -> list[dict]:
        """Force-close open BUY positions held longer than *max_hold_bars*
        hours — the live counterpart to backtest.engine's time-stop
        (engine.py: "the ML edge is defined over LABEL_HORIZON_BARS;
        holding past max_hold_bars is unmodeled exposure"). No-op
        (doesn't touch the DB) when max_hold_bars <= 0 — the disabled
        default. Call at the top of each bot_runner cycle, before
        generating new entry signals.
        """
        if max_hold_bars <= 0:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_hold_bars)
        with self.engine.connect() as conn:
            rows = conn.execute(_SELECT_EXPIRED_OPEN_TRADES, {"cutoff": cutoff}).fetchall()

        closed = []
        for row in rows:
            trade = dict(row._mapping)
            try:
                price = self.client.get_current_price(trade["symbol"])
            except Exception as exc:
                log.error("price_fetch_failed", symbol=trade["symbol"], error=str(exc))
                continue
            result = self._close_open_trade(trade["symbol"], trade, price, dry_run,
                                             close_reason="time_stop")
            if result is not None:
                closed.append(result)
        return closed
