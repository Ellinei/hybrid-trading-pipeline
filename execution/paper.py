"""Persistent spot simulation. Uses quotes only; never submits exchange orders.

Stops are evaluated at each bot-cycle quote, not intrabar. Fills include
adverse slippage and quote-currency fees on both entry and exit.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import isfinite

import structlog

from execution.exchange_filters import (
    FilterViolation,
    parse_symbol_filters,
    round_qty_down,
    validate_notional,
)
from execution.journal import connect_store, operations_path
from execution.risk_manager import RiskConfig, RiskManager
from signals.base import SignalDirection


@dataclass(frozen=True)
class PaperConfig:
    initial_balance: float = 10000.0
    fee_rate: float = 0.001
    slippage_bps: float = 5.0

    def __post_init__(self):
        if not isfinite(self.initial_balance) or self.initial_balance <= 0:
            raise ValueError("Initial paper balance must be finite and positive")
        if not isfinite(self.fee_rate) or not 0 <= self.fee_rate < 1:
            raise ValueError("Paper fee rate must be in [0, 1)")
        if not isfinite(self.slippage_bps) or not 0 <= self.slippage_bps < 10000:
            raise ValueError("Paper slippage must be in [0, 10000) basis points")


class PaperStore:
    def __init__(self, path=None, initial_balance=10000.0):
        PaperConfig(initial_balance=initial_balance)
        self.conn = connect_store(path if path is not None else operations_path())
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS paper_account (
                id INTEGER PRIMARY KEY CHECK(id = 1), initial_balance REAL NOT NULL,
                cash REAL NOT NULL CHECK(cash >= 0), peak REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY, symbol TEXT NOT NULL,
                entry_price REAL NOT NULL, quantity REAL NOT NULL CHECK(quantity > 0),
                stop_loss REAL NOT NULL, take_profit REAL NOT NULL,
                entry_fee REAL NOT NULL, fee_rate REAL NOT NULL, slippage_bps REAL NOT NULL,
                executed_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open',
                exit_price REAL, exit_fee REAL, closed_at TEXT, close_reason TEXT, pnl REAL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS paper_one_open_symbol
                ON paper_trades(symbol) WHERE status = 'open';
        """)
        self.conn.execute(
            "INSERT OR IGNORE INTO paper_account VALUES (1, ?, ?, ?)",
            (initial_balance, initial_balance, initial_balance),
        )

    @contextmanager
    def transaction(self):
        # Serializes approval + cash/position updates across concurrent bot processes.
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise

    def positions(self):
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM paper_trades WHERE status = 'open' ORDER BY id"
            )
        ]

    def close(self):
        self.conn.close()


class PaperRiskManager(RiskManager):
    """Reuse live approval/sizing rules with account state from the paper ledger."""

    def __init__(self, store, engine, config=None):
        self.store = store
        self.engine = engine
        self.config = config or RiskConfig()

    def get_account_balance(self):
        return self.store.conn.execute("SELECT cash FROM paper_account WHERE id = 1").fetchone()[0]

    def has_open_position(self, symbol):
        return any(p["symbol"] == symbol for p in self.store.positions())

    def get_open_position_count(self):
        return len(self.store.positions())

    def get_open_position_risk(self):
        return sum(
            max(0, p["entry_price"] - p["stop_loss"]) * p["quantity"]
            for p in self.store.positions()
        )

    def check_drawdown(self):
        # Realized equity, including entry fees; open positions valued at entry cost.
        equity = self.get_account_balance() + sum(
            p["entry_price"] * p["quantity"] for p in self.store.positions()
        )
        peak = self.store.conn.execute("SELECT peak FROM paper_account WHERE id = 1").fetchone()[0]
        peak = max(peak, equity)
        self.store.conn.execute("UPDATE paper_account SET peak = ? WHERE id = 1", (peak,))
        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        return drawdown, drawdown >= self.config.max_drawdown_pct


class PaperExecutor:
    mode = "paper"

    def __init__(self, client, store, risk_manager, config=None, clock=None):
        self.client = client
        self.store = store
        self.risk = risk_manager
        self.config = config or PaperConfig()
        self.clock = clock or (lambda: datetime.now(UTC))
        self.last_decision = {}

    def _reject(self, reason):
        self.last_decision = {"outcome": "rejected", "reason": reason}
        return None

    def _price(self, symbol):
        price = float(self.client.get_current_price(symbol))
        if not isfinite(price) or price <= 0:
            raise ValueError("Quote must be finite and positive")
        return price

    def execute_signal(self, signal, dry_run=False):
        if dry_run:
            raise ValueError("Paper mode and dry-run are mutually exclusive")
        self.last_decision = {}
        if signal.direction == SignalDirection.HOLD:
            self.last_decision = {"outcome": "hold", "reason": "signal is HOLD"}
            return None
        quote = self._price(signal.symbol)
        with self.store.transaction():
            if signal.direction == SignalDirection.BUY and self.risk.has_open_position(
                signal.symbol
            ):
                return self._reject("paper position already open")
            approved, reason = self.risk.approve_trade(signal, quote)
            if not approved:
                return self._reject(reason)
            if signal.direction == SignalDirection.SELL:
                position = next(p for p in self.store.positions() if p["symbol"] == signal.symbol)
                return self._close(position, quote, "signal_close")

            fill = quote * (1 + self.config.slippage_bps / 10000)
            atr = self.risk.get_atr(signal.symbol)
            cash = self.risk.get_account_balance()
            filters = parse_symbol_filters(self.client.get_symbol_info_cached(signal.symbol))
            qty = min(
                self.risk.calculate_position_size(fill, atr, cash),
                cash / (fill * (1 + self.config.fee_rate)),
            )
            qty = round_qty_down(qty, filters)
            if not isfinite(qty) or qty <= 0:
                return self._reject("paper quantity rounds to zero")
            try:
                validate_notional(qty, fill, filters)
            except FilterViolation as exc:
                return self._reject(str(exc))
            stop, take = self.risk.calculate_stops(fill, atr)
            if not all(isfinite(v) for v in (fill, stop, take)) or not 0 < stop < fill < take:
                return self._reject("invalid paper stop/take-profit prices")
            fee = fill * qty * self.config.fee_rate
            remaining = cash - fill * qty - fee
            if remaining < -1e-8:
                return self._reject("insufficient paper cash including fees")
            row = self.store.conn.execute(
                """INSERT INTO paper_trades
                (symbol, entry_price, quantity, stop_loss, take_profit, entry_fee,
                 fee_rate, slippage_bps, executed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    signal.symbol,
                    fill,
                    qty,
                    stop,
                    take,
                    fee,
                    self.config.fee_rate,
                    self.config.slippage_bps,
                    self.clock().isoformat(),
                ),
            )
            self.store.conn.execute(
                "UPDATE paper_account SET cash = ? WHERE id = 1", (max(0, remaining),)
            )
            return dict(
                self.store.conn.execute(
                    "SELECT * FROM paper_trades WHERE id = ?", (row.lastrowid,)
                ).fetchone()
            )

    def _close(self, position, quote, reason):
        fill = quote * (1 - position["slippage_bps"] / 10000)
        proceeds = fill * position["quantity"]
        fee = proceeds * position["fee_rate"]
        pnl = (fill - position["entry_price"]) * position["quantity"] - position["entry_fee"] - fee
        self.store.conn.execute(
            """UPDATE paper_trades SET status = 'closed',
            exit_price = ?, exit_fee = ?, closed_at = ?, close_reason = ?, pnl = ?
            WHERE id = ? AND status = 'open'""",
            (fill, fee, self.clock().isoformat(), reason, pnl, position["id"]),
        )
        self.store.conn.execute(
            "UPDATE paper_account SET cash = cash + ? WHERE id = 1", (proceeds - fee,)
        )
        return {
            **position,
            "status": "closed",
            "exit_price": fill,
            "exit_fee": fee,
            "pnl": pnl,
            "close_reason": reason,
        }

    def close_expired_positions(self, max_hold_bars, dry_run=False):
        if dry_run:
            raise ValueError("Paper mode and dry-run are mutually exclusive")
        closed = []
        for position in self.store.positions():
            try:
                quote = self._price(position["symbol"])
                with self.store.transaction():
                    # Re-read after taking the write lock: another process may have closed it.
                    row = self.store.conn.execute(
                        "SELECT * FROM paper_trades WHERE id = ? AND status = 'open'",
                        (position["id"],),
                    ).fetchone()
                    if row is None:
                        continue
                    position = dict(row)
                    reason = None
                    if quote <= position["stop_loss"]:
                        reason = "stop_loss"
                    elif quote >= position["take_profit"]:
                        reason = "take_profit"
                    elif max_hold_bars > 0 and self.clock() >= (
                        datetime.fromisoformat(position["executed_at"])
                        + timedelta(hours=max_hold_bars)
                    ):
                        reason = "time_stop"
                    if reason:
                        closed.append(self._close(position, quote, reason))
            except Exception:
                structlog.get_logger().exception("paper_exit_failed", symbol=position["symbol"])
        return closed
