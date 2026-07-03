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

    def __init__(self, db_engine) -> None:
        self.engine = db_engine

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
                range_14, volume
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

        rsi    = v if (v := features.get("rsi_14_proxy"))  is not None else 50.0
        macd   = v if (v := features.get("macd_line"))     is not None else 0.0
        bz     = v if (v := features.get("bb_zscore"))     is not None else 0.0
        pct_1h = v if (v := features.get("pct_change_1h")) is not None else 0.0
        pct_24 = v if (v := features.get("pct_change_24h"))is not None else 0.0

        meta = {
            "rsi": rsi, "macd": macd, "bb_zscore": bz,
            "pct_change_1h": pct_1h, "pct_change_24h": pct_24,
        }

        def sig(direction: SignalDirection, confidence: float) -> Signal:
            return Signal(
                direction=direction, confidence=confidence,
                source="technical", symbol=symbol, timestamp=now,
                metadata=meta,
            )

        # ── Strong BUY ──────────────────────────────────────────────
        # pct_1h > 0 guards against falling-knife entries: RSI<30 in crypto
        # historically precedes multi-month consolidation, not V-reversals.
        # Require positive 1h momentum to confirm the dip is already bouncing.
        if rsi < 30 and macd > 0 and pct_1h > 0:
            return sig(SignalDirection.BUY, 0.8)
        if bz < -2.0:
            return sig(SignalDirection.BUY, 0.75)

        # ── Strong SELL ─────────────────────────────────────────────
        if rsi > 70 and macd < 0:
            return sig(SignalDirection.SELL, 0.8)
        if bz > 2.0:
            return sig(SignalDirection.SELL, 0.75)

        # ── Moderate BUY ────────────────────────────────────────────
        if rsi < 40 and pct_1h > 0:
            return sig(SignalDirection.BUY, 0.55)
        if macd > 0 and pct_24 > 2.0:
            return sig(SignalDirection.BUY, 0.6)

        # ── Moderate SELL ───────────────────────────────────────────
        if rsi > 60 and pct_1h < 0:
            return sig(SignalDirection.SELL, 0.55)
        if macd < 0 and pct_24 < -2.0:
            return sig(SignalDirection.SELL, 0.6)

        # ── Default ─────────────────────────────────────────────────
        return sig(SignalDirection.HOLD, 0.5)
