"""Risk gate: approves or rejects trade signals before execution.

All position sizing, drawdown checks, and confidence thresholds live here
so the executor stays focused on mechanics.
"""
from __future__ import annotations

from dataclasses import dataclass

import structlog
from sqlalchemy import text

from ingestion.binance_client import BinanceClientWrapper
from signals.base import AggregatedSignal, SignalDirection

log = structlog.get_logger()

_INITIAL_BALANCE = 10_000.0   # baseline for drawdown calculation


@dataclass
class RiskConfig:
    max_position_pct:      float = 0.02   # 2 % of balance per trade
    max_open_positions:    int   = 3
    stop_loss_atr_mult:    float = 1.5
    take_profit_atr_mult:  float = 3.0
    max_drawdown_pct:      float = 0.10   # 10 % circuit-breaker
    min_confidence:        float = 0.55
    max_hold_bars:         int   = 0      # close after N hourly bars (0 = disabled)


class RiskManager:
    """Validates trade signals against risk parameters before execution."""

    def __init__(
        self,
        client: BinanceClientWrapper,
        db_engine,
        config: RiskConfig | None = None,
    ) -> None:
        self.client = client
        self.engine = db_engine
        self.config = config or RiskConfig()

    # ------------------------------------------------------------------
    # Account state
    # ------------------------------------------------------------------

    def get_account_balance(self) -> float | None:
        """USDT free balance; None if the exchange call fails."""
        try:
            return self.client.get_account_balance(asset="USDT")
        except Exception as exc:
            log.warning("balance_fetch_failed", error=str(exc))
            return None

    def has_open_position(self, symbol: str) -> bool:
        """True if there's an open BUY position on *symbol* (spot can't short)."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT COUNT(*) FROM raw.trades"
                        " WHERE symbol = :symbol AND status = 'open'"
                        " AND direction = 'BUY'"
                    ),
                    {"symbol": symbol},
                )
                return int(result.scalar() or 0) > 0
        except Exception as exc:
            log.warning("open_position_lookup_failed", symbol=symbol, error=str(exc))
            return False

    def get_open_position_count(self) -> int:
        """Number of rows in raw.trades with status = 'open'."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT COUNT(*) FROM raw.trades WHERE status = 'open'")
                )
                return int(result.scalar() or 0)
        except Exception as exc:
            log.warning("open_position_count_failed", error=str(exc))
            return 0

    def get_atr(self, symbol: str) -> float:
        """Latest range_14 from features.feat_technical; 0.0 if not found."""
        try:
            with self.engine.connect() as conn:
                row = conn.execute(
                    text("""
                        SELECT range_14
                        FROM features.feat_technical
                        WHERE symbol = :symbol
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """),
                    {"symbol": symbol},
                ).fetchone()
            return float(row._mapping["range_14"]) if row else 0.0
        except Exception as exc:
            log.warning("atr_fetch_failed", symbol=symbol, error=str(exc))
            return 0.0

    # ------------------------------------------------------------------
    # Calculations
    # ------------------------------------------------------------------

    def calculate_position_size(
        self, price: float, atr: float, balance: float
    ) -> float:
        """Risk-based position size in base-asset units.

        risk_amount = balance * max_position_pct
        size = risk_amount / (stop_loss_atr_mult * atr)

        Returns 0.0 if ATR is 0 or notional value < $10.
        """
        if atr <= 0:
            return 0.0
        risk_amount = balance * self.config.max_position_pct
        size = round(risk_amount / (self.config.stop_loss_atr_mult * atr), 6)

        # Spot has no margin — a position can never cost more than the
        # available balance, however small ATR makes the risk-based size.
        max_size_for_balance = round(balance / price, 6) if price > 0 else 0.0
        size = min(size, max_size_for_balance)

        if size * price < 10:
            return 0.0
        return size

    def calculate_stops(self, price: float, atr: float) -> tuple[float, float]:
        """(stop_loss, take_profit) before exchange tick rounding.

        Shared by live execution and the backtester so both compute
        identical stops from the same ATR-multiple formula.
        """
        stop_loss   = price - atr * self.config.stop_loss_atr_mult
        take_profit = price + atr * self.config.take_profit_atr_mult
        return stop_loss, take_profit

    def check_drawdown(self) -> tuple[float, bool]:
        """Return (drawdown_pct, is_breached) based on realised P&L.

        Uses _INITIAL_BALANCE as the denominator so a fresh account
        doesn't divide by zero.
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT COALESCE(SUM(pnl), 0)"
                        " FROM raw.trades WHERE pnl IS NOT NULL"
                    )
                )
                total_pnl = float(result.scalar() or 0)
        except Exception:
            return 0.0, False

        drawdown = abs(total_pnl) / _INITIAL_BALANCE if total_pnl < 0 else 0.0
        return drawdown, drawdown >= self.config.max_drawdown_pct

    # ------------------------------------------------------------------
    # Gate
    # ------------------------------------------------------------------

    def approve_trade(
        self, signal: AggregatedSignal, price: float
    ) -> tuple[bool, str]:
        """Run all checks in order; first failure short-circuits.

        Returns (True, "approved") or (False, reason_string).
        """
        if signal.direction == SignalDirection.HOLD:
            return False, "signal is HOLD"

        if signal.direction == SignalDirection.SELL:
            # Closing a held position is never blocked by sizing/drawdown/
            # confidence gates — those only govern opening NEW exposure.
            if not self.has_open_position(signal.symbol):
                return False, "SELL rejected — no open position to close (spot, no margin)"
            return True, "approved (closing)"

        if signal.confidence < self.config.min_confidence:
            return (
                False,
                f"confidence {signal.confidence:.2f} < {self.config.min_confidence}",
            )

        _, breached = self.check_drawdown()
        if breached:
            return False, "max drawdown circuit breaker triggered"

        if self.get_open_position_count() >= self.config.max_open_positions:
            return False, f"max open positions ({self.config.max_open_positions}) reached"

        atr = self.get_atr(signal.symbol)
        if atr <= 0:
            return False, "ATR = 0 — run dbt first to populate features.feat_technical"

        balance = self.get_account_balance()
        if balance is None:
            return False, "account balance unavailable — refusing to size trade"

        size = self.calculate_position_size(price, atr, balance)
        if size <= 0:
            return False, "position size < $10 notional — trade too small"

        return True, "approved"
