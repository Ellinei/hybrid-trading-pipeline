"""Rule-based signal generator from dbt feature table.

Rules are evaluated top-to-bottom; first match wins.
All thresholds come from standard technical analysis literature.
"""
from __future__ import annotations

from datetime import datetime, timezone

import structlog
from sqlalchemy import text

from signals.base import Signal, SignalDirection

log = structlog.get_logger()


class TechnicalSignalGenerator:
    """Generates BUY / SELL / HOLD signals from ``features.feat_technical``."""

    def __init__(
        self,
        db_engine,
        require_uptrend_for_buys: bool = False,
        guard_bb_dip_buy: bool = False,
    ) -> None:
        self.engine = db_engine
        self.require_uptrend_for_buys = require_uptrend_for_buys
        self.guard_bb_dip_buy = guard_bb_dip_buy

    # ------------------------------------------------------------------
    # DB access
    # ------------------------------------------------------------------

    def get_latest_features(
        self, symbol: str, as_of: datetime | None = None
    ) -> dict | None:
        """Return the most recent feature row for *symbol*, or None.

        When *as_of* is given, returns the most recent row at or before that
        timestamp instead of the live latest row — used by the backtester to
        replay signals without leaking future data.
        """
        params: dict = {"symbol": symbol}
        cutoff_clause = ""
        if as_of is not None:
            params["as_of"] = as_of
            cutoff_clause = "AND timestamp <= :as_of"

        query = text(f"""
            SELECT
                rsi_14_proxy, macd_line, bb_zscore,
                pct_change_1h, pct_change_24h, log_return_1h,
                range_14, volume, close_vs_sma50
            FROM features.feat_technical
            WHERE symbol = :symbol
            {cutoff_clause}
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        with self.engine.connect() as conn:
            row = conn.execute(query, params).fetchone()
        return dict(row._mapping) if row else None

    # ------------------------------------------------------------------
    # Signal logic
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_rules(
        features: dict,
        require_uptrend_for_buys: bool = False,
        guard_bb_dip_buy: bool = False,
    ) -> tuple[SignalDirection, float, dict]:
        """Pure rule evaluation: features dict → (direction, confidence, meta).

        No DB, no clock — the backtest sweep replays this function over
        pre-loaded feature rows, so any rule change here automatically
        applies to both live signals and sweep replays.
        """
        rsi    = v if (v := features.get("rsi_14_proxy"))  is not None else 50.0
        macd   = v if (v := features.get("macd_line"))     is not None else 0.0
        bz     = v if (v := features.get("bb_zscore"))     is not None else 0.0
        pct_1h = v if (v := features.get("pct_change_1h")) is not None else 0.0
        pct_24 = v if (v := features.get("pct_change_24h"))is not None else 0.0
        sma50  = v if (v := features.get("close_vs_sma50"))is not None else 0.0

        meta = {
            "rsi": rsi, "macd": macd, "bb_zscore": bz,
            "pct_change_1h": pct_1h, "pct_change_24h": pct_24,
            "close_vs_sma50": sma50,
        }

        # Regime filter: when enabled, strong BUY rules also require price
        # above its 50-bar SMA — no dip-buying inside a downtrend. A missing
        # sma50 value blocks the BUY (conservative).
        uptrend_ok = (not require_uptrend_for_buys) or sma50 > 0

        # ── Strong BUY ──────────────────────────────────────────────
        # pct_1h > 0 guards against falling-knife entries: RSI<30 in crypto
        # historically precedes multi-month consolidation, not V-reversals.
        # Require positive 1h momentum to confirm the dip is already bouncing.
        if rsi < 30 and macd > 0 and pct_1h > 0 and uptrend_ok:
            return SignalDirection.BUY, 0.8, meta
        # guard_bb_dip_buy applies the same falling-knife guard as the RSI
        # rule: a -2σ Bollinger break during a crash keeps crashing more
        # often than it mean-reverts. Off by default (legacy behavior);
        # measured as an A/B toggle before adoption.
        if bz < -2.0 and uptrend_ok and (pct_1h > 0 or not guard_bb_dip_buy):
            return SignalDirection.BUY, 0.75, meta

        # ── Strong SELL ─────────────────────────────────────────────
        if rsi > 70 and macd < 0:
            return SignalDirection.SELL, 0.8, meta
        if bz > 2.0:
            return SignalDirection.SELL, 0.75, meta

        # ── Moderate BUY ────────────────────────────────────────────
        # NOTE: with default aggregator weights these BUY rules can never
        # clear the risk gate (0.55-0.60 × 0.40 < min_confidence 0.55 even
        # with max ML agreement) — kept for signal telemetry and for sweep
        # configs that lower the gates.
        if rsi < 40 and pct_1h > 0 and uptrend_ok:
            return SignalDirection.BUY, 0.55, meta
        if macd > 0 and pct_24 > 2.0 and uptrend_ok:
            return SignalDirection.BUY, 0.6, meta

        # ── Moderate SELL ───────────────────────────────────────────
        # These ARE load-bearing: SELL bypasses the risk gate's confidence
        # check, so a lone moderate SELL (0.55×0.40 = 0.22 > 0.15) closes
        # positions — they are the strategy's de facto exit engine.
        if rsi > 60 and pct_1h < 0:
            return SignalDirection.SELL, 0.55, meta
        if macd < 0 and pct_24 < -2.0:
            return SignalDirection.SELL, 0.6, meta

        # ── Default ─────────────────────────────────────────────────
        return SignalDirection.HOLD, 0.5, meta

    def generate_signal(self, symbol: str, as_of: datetime | None = None) -> Signal:
        """Apply rule set and return a Signal."""
        now = as_of if as_of is not None else datetime.now(timezone.utc)
        features = self.get_latest_features(symbol, as_of=as_of)

        if not features:
            return Signal(
                direction=SignalDirection.HOLD, confidence=0.5,
                source="technical", symbol=symbol, timestamp=now,
                metadata={},
            )

        direction, confidence, meta = self.evaluate_rules(
            features,
            require_uptrend_for_buys=self.require_uptrend_for_buys,
            guard_bb_dip_buy=self.guard_bb_dip_buy,
        )
        return Signal(
            direction=direction, confidence=confidence,
            source="technical", symbol=symbol, timestamp=now,
            metadata=meta,
        )
