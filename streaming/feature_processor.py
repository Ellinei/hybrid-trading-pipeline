"""Kafka consumer → real-time feature computation → TimescaleDB.

Consumes the ``price-ticks`` topic, maintains a per-symbol rolling price
buffer, and flushes computed features to ``raw.realtime_features`` every 60
seconds.

Pure functions (TickRecord, filter_window, compute_vwap, compute_features)
are module-level so they can be imported and unit-tested without any I/O.

Usage:
    python -m streaming.feature_processor
"""
from __future__ import annotations

import json
import os
import signal
from bisect import insort_right
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import isfinite
from statistics import StatisticsError, stdev
from typing import Any
from urllib.parse import quote_plus

import psycopg2
import structlog
from dotenv import load_dotenv
from kafka import KafkaConsumer
from sqlalchemy import create_engine

from ingestion.models import create_all_tables

log = structlog.get_logger()

_FLUSH_INTERVAL_S = 60

FEATURE_UPSERT_SQL = """
    INSERT INTO raw.realtime_features
        (symbol, timestamp, last_price, vwap_1m, vwap_5m,
         price_change_1m, price_change_5m, volatility_1m, trade_count_1m)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (symbol, timestamp) DO UPDATE SET
        last_price      = EXCLUDED.last_price,
        vwap_1m         = EXCLUDED.vwap_1m,
        vwap_5m         = EXCLUDED.vwap_5m,
        price_change_1m = EXCLUDED.price_change_1m,
        price_change_5m = EXCLUDED.price_change_5m,
        volatility_1m   = EXCLUDED.volatility_1m,
        trade_count_1m  = EXCLUDED.trade_count_1m
"""


# ---------------------------------------------------------------------------
# Pure data structures and computation functions
# ---------------------------------------------------------------------------

@dataclass
class TickRecord:
    timestamp: datetime
    price: float
    quantity: float


def _positive_float(value: Any) -> float:
    """Parse a trade amount, rejecting booleans and non-finite values."""
    if isinstance(value, bool):
        raise ValueError("Trade amounts cannot be booleans")
    number = float(value)
    if not isfinite(number) or number <= 0:
        raise ValueError("Trade amounts must be finite and positive")
    return number


def filter_window(
    buffer: list[TickRecord],
    minutes: int,
    now: datetime | None = None,
) -> list[TickRecord]:
    """Return ticks whose timestamp >= (now - minutes). Preserves order."""
    if now is None:
        now = datetime.now(UTC)
    cutoff = now - timedelta(minutes=minutes)
    return [t for t in buffer if t.timestamp >= cutoff]


def compute_vwap(ticks: list[TickRecord]) -> float | None:
    """Volume-weighted average price. Returns None for empty or zero-volume input."""
    if not ticks:
        return None
    total_volume = sum(t.quantity for t in ticks)
    if total_volume == 0:
        return None
    return sum(t.price * t.quantity for t in ticks) / total_volume


def compute_features(
    symbol: str,
    buffer: list[TickRecord],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Compute all real-time features from the tick buffer.

    Args:
        symbol: Trading pair symbol (e.g. "BTCUSDT").
        buffer: Ordered list of TickRecords, oldest first.
        now: Timestamp to use as «current time» (defaults to UTC now).

    Returns:
        Dict with all feature columns + symbol and timestamp.
    """
    if now is None:
        now = datetime.now(UTC)

    ticks_1m = filter_window(buffer, minutes=1, now=now)
    ticks_5m = filter_window(buffer, minutes=5, now=now)

    last_price = buffer[-1].price if buffer else None

    # VWAP
    vwap_1m = compute_vwap(ticks_1m)
    vwap_5m = compute_vwap(ticks_5m)

    # % price change (oldest → newest within window)
    def _pct_change(ticks: list[TickRecord]) -> float | None:
        if len(ticks) < 2:
            return None
        old, new = ticks[0].price, ticks[-1].price
        return (new - old) / old * 100 if old != 0 else None

    price_change_1m = _pct_change(ticks_1m)
    price_change_5m = _pct_change(ticks_5m)

    # Volatility = std dev of prices in 1m window
    volatility_1m: float | None = None
    prices_1m = [t.price for t in ticks_1m]
    if len(prices_1m) >= 2:
        try:
            volatility_1m = stdev(prices_1m)
        except StatisticsError:
            volatility_1m = None

    return {
        "symbol": symbol,
        "timestamp": now,
        "last_price": last_price,
        "vwap_1m": vwap_1m,
        "vwap_5m": vwap_5m,
        "price_change_1m": price_change_1m,
        "price_change_5m": price_change_5m,
        "volatility_1m": volatility_1m,
        "trade_count_1m": len(ticks_1m),
    }


# ---------------------------------------------------------------------------
# Consumer class
# ---------------------------------------------------------------------------

class FeatureProcessor:
    """Kafka consumer that computes and persists real-time features."""

    def __init__(self) -> None:
        load_dotenv()
        bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

        self._stopping = False
        self._buffers: dict[str, list[TickRecord]] = {}
        self._last_flush: dict[str, datetime] = {}
        self._log = structlog.get_logger()

        # Ensure raw.realtime_features (and the rest of the schema) exists.
        # The processor owns this table — don't rely on rest_backfill running first.
        self._ensure_schema()

        self._consumer = KafkaConsumer(
            "price-ticks",
            bootstrap_servers=bootstrap,
            group_id="feature-processor",
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )

        self._conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "trading"),
            user=os.getenv("POSTGRES_USER", "trader"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
        )

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _ensure_schema(self) -> None:
        """Run create_all_tables() — idempotent, safe to call every startup."""
        engine = create_engine(
            "postgresql+psycopg2://{u}:{p}@{h}:{port}/{db}".format(
                u=quote_plus(os.getenv("POSTGRES_USER", "trader")),
                p=quote_plus(os.getenv("POSTGRES_PASSWORD", "")),
                h=os.getenv("POSTGRES_HOST", "localhost"),
                port=os.getenv("POSTGRES_PORT", "5432"),
                db=os.getenv("POSTGRES_DB", "trading"),
            )
        )
        try:
            create_all_tables(engine)
        finally:
            engine.dispose()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._log.info("feature_processor_starting")
        try:
            while not self._stopping:
                raw = self._consumer.poll(timeout_ms=1000)
                for _, messages in raw.items():
                    for msg in messages:
                        # One malformed payload must not crash the consumer.
                        try:
                            self._process_message(msg.value)
                        except Exception:
                            self._log.exception(
                                "process_message_failed",
                                offset=getattr(msg, "offset", None),
                                partition=getattr(msg, "partition", None),
                            )
                self._maybe_flush_all()
        finally:
            self._consumer.close()
            self._conn.close()
            self._log.info("feature_processor_stopped")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_message(self, msg: Any) -> None:
        if not isinstance(msg, dict):
            self._log.warning("bad_message", msg=msg)
            return
        if msg.get("stream_type") != "trade":
            return

        try:
            symbol = msg["symbol"]
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("Trade symbol must be a non-empty string")
            ts = datetime.fromisoformat(msg["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            tick = TickRecord(
                timestamp=ts,
                price=_positive_float(msg["price"]),
                quantity=_positive_float(msg["quantity"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            self._log.warning("bad_message", msg=msg)
            return

        buf = self._buffers.setdefault(symbol, [])
        # Price changes and last_price depend on event order, not arrival order.
        if not buf or tick.timestamp >= buf[-1].timestamp:
            buf.append(tick)
        else:
            insort_right(buf, tick, key=lambda item: item.timestamp)

    def _maybe_flush_all(self) -> None:
        now = datetime.now(UTC)
        for symbol, buf in self._buffers.items():
            last = self._last_flush.get(symbol)
            if last is None or (now - last).total_seconds() >= _FLUSH_INTERVAL_S:
                # Retain the full feature window regardless of trade volume.
                # For idle symbols, keep one tick for the last known price.
                buf[:] = filter_window(buf, minutes=5, now=now) or buf[-1:]
                try:
                    features = compute_features(symbol, buf, now=now)
                    if features["last_price"] is not None:
                        self._write_features(features)
                    self._last_flush[symbol] = now
                except psycopg2.Error:
                    # DB hiccup — roll back this txn and try again on next tick.
                    # If we don't roll back, the connection stays in a failed
                    # state and every future cursor errors out.
                    self._log.exception("write_features_failed", symbol=symbol)
                    try:
                        self._conn.rollback()
                    except psycopg2.Error:
                        self._log.exception("rollback_failed", symbol=symbol)

    def _write_features(self, f: dict[str, Any]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(FEATURE_UPSERT_SQL, (
                f["symbol"], f["timestamp"], f["last_price"],
                f["vwap_1m"], f["vwap_5m"],
                f["price_change_1m"], f["price_change_5m"],
                f["volatility_1m"], f["trade_count_1m"],
            ))
        self._conn.commit()
        self._log.info(
            "features_written",
            symbol=f["symbol"],
            last_price=f["last_price"],
            vwap_1m=f["vwap_1m"],
            trade_count_1m=f["trade_count_1m"],
        )

    def _handle_shutdown(self, signum: int, frame: object) -> None:
        self._log.info("shutdown_requested", signal=signum)
        self._stopping = True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    FeatureProcessor().run()
