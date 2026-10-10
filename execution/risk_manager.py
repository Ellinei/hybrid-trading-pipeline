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
    # cap on sum(entry-stop)*qty across open positions; None = disabled
    max_portfolio_risk_pct: float | None = None


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
        self._ensure_bot_state_table()

    # ------------------------------------------------------------------
    # raw.bot_state — small key/value table for state that must survive
    # process restarts (currently just the drawdown baseline/peak).
    # ------------------------------------------------------------------

    def _ensure_bot_state_table(self) -> None:
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS raw.bot_state (
                    key        VARCHAR(50) PRIMARY KEY,
                    value      NUMERIC,
                    updated_at TIMESTAMPTZ DEFAULT now()
                )
            """))
            conn.commit()

    def _get_bot_state(self, key: str) -> float | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT value FROM raw.bot_state WHERE key = :key"), {"key": key}
            ).fetchone()
        return float(row._mapping["value"]) if row else None

    def _set_bot_state(self, key: str, value: float) -> None:
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO raw.bot_state (key, value, updated_at)
                VALUES (:key, :value, now())
                ON CONFLICT (key) DO UPDATE SET value = :value, updated_at = now()
            """), {"key": key, "value": value})
            conn.commit()

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

    def get_open_position_risk(self) -> float:
        """Sum of (entry_price - stop_loss) * quantity across open BUY
        positions — dollar exposure already committed if every stop-loss
        fires. Feeds the portfolio concentration cap in approve_trade."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT COALESCE(SUM((entry_price - stop_loss) * quantity), 0)"
                        " FROM raw.trades WHERE status = 'open' AND direction = 'BUY'"
                    )
                )
                return float(result.scalar() or 0)
        except Exception as exc:
            log.warning("open_position_risk_lookup_failed", error=str(exc))
            return 0.0

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
        """Return (drawdown_pct, is_breached) — peak-tracked realized
        equity (raw.bot_state's persisted baseline + cumulative closed
        pnl), not a fixed floor. The old formula (abs(total_pnl) /
        _INITIAL_BALANCE, 0 whenever total_pnl was positive) never
        detected a pullback after a gain — a run that peaked at +$2000
        then dropped to +$1500 read as 0% drawdown instead of the real
        ~4% pullback from its peak. Baseline defaults to _INITIAL_BALANCE
        on first use and is only mutable via reset_drawdown_baseline
        (an explicit, human-initiated action after reviewing a breach).
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

            baseline = self._get_bot_state("drawdown_baseline")
            if baseline is None:
                baseline = _INITIAL_BALANCE
                self._set_bot_state("drawdown_baseline", baseline)

            current_equity = baseline + total_pnl

            peak = self._get_bot_state("drawdown_peak")
            if peak is None or current_equity > peak:
                peak = current_equity
                self._set_bot_state("drawdown_peak", peak)
        except Exception:
            return 0.0, False

        drawdown = (peak - current_equity) / peak if peak > 0 else 0.0
        return drawdown, drawdown >= self.config.max_drawdown_pct

    def reset_drawdown_baseline(self, new_baseline: float | None = None) -> None:
        """Explicit, human-initiated reset — re-anchors both the baseline
        and peak to *new_baseline* (default: current live account
        balance). Use after reviewing a tripped circuit breaker; nothing
        else in this class calls this automatically."""
        if new_baseline is None:
            new_baseline = self.get_account_balance()
            if new_baseline is None:
                raise RuntimeError(
                    "cannot reset drawdown baseline: account balance unavailable"
                )
        self._set_bot_state("drawdown_baseline", new_baseline)
        self._set_bot_state("drawdown_peak", new_baseline)

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

        if self.config.max_portfolio_risk_pct is not None:
            stop_loss, _ = self.calculate_stops(price, atr)
            new_risk = (price - stop_loss) * size
            existing_risk = self.get_open_position_risk()
            max_risk = balance * self.config.max_portfolio_risk_pct
            if existing_risk + new_risk > max_risk:
                return False, (
                    f"portfolio risk cap exceeded: existing ${existing_risk:.2f} + "
                    f"new ${new_risk:.2f} > {self.config.max_portfolio_risk_pct:.0%} "
                    f"of balance (${max_risk:.2f})"
                )

        return True, "approved"
