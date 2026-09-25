"""Historical OHLCV backfill from Binance REST API into TimescaleDB.

Idempotent: corrects changed candle values on conflict.
Incremental: refetches a 48-hour overlap to repair provisional rows and gaps.

Usage (from project root):
    python -m ingestion.rest_backfill
    python -m ingestion.rest_backfill --symbols BTCUSDT,ETHUSDT --days 30
    python -m ingestion.rest_backfill --interval 15m --days 7
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote_plus

import structlog
from dotenv import load_dotenv
from psycopg2.extras import execute_values

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

OHLCV_INSERT_SQL = """
    INSERT INTO raw.ohlcv
        (symbol, timestamp, open, high, low, close,
         volume, quote_volume, trades, taker_buy_vol)
    VALUES %s
    ON CONFLICT (symbol, timestamp) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        quote_volume = EXCLUDED.quote_volume,
        trades = EXCLUDED.trades,
        taker_buy_vol = EXCLUDED.taker_buy_vol
    WHERE (raw.ohlcv.open, raw.ohlcv.high, raw.ohlcv.low, raw.ohlcv.close,
           raw.ohlcv.volume, raw.ohlcv.quote_volume, raw.ohlcv.trades,
           raw.ohlcv.taker_buy_vol)
          IS DISTINCT FROM
          (EXCLUDED.open, EXCLUDED.high, EXCLUDED.low, EXCLUDED.close,
           EXCLUDED.volume, EXCLUDED.quote_volume, EXCLUDED.trades,
           EXCLUDED.taker_buy_vol)
"""


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_db_connection():
    """Open a psycopg2 connection from env vars."""
    import psycopg2

    load_dotenv()
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "trading"),
        user=os.getenv("POSTGRES_USER", "trader"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )


def upsert_ohlcv_rows(conn, rows: list[dict[str, Any]]) -> int:
    """Insert OHLCV rows and repair differing values on conflict.

    Args:
        conn: psycopg2 connection.
        rows: list of dicts with keys matching OHLCV column names.

    Returns:
        Number of rows inserted or updated (0 when all values are unchanged).
    """
    if not rows:
        return 0

    values = [
        (
            r["symbol"],
            r["timestamp"],
            r["open"],
            r["high"],
            r["low"],
            r["close"],
            r["volume"],
            r["quote_volume"],
            r["trades"],
            r["taker_buy_vol"],
        )
        for r in rows
    ]

    with conn.cursor() as cur:
        execute_values(cur, OHLCV_INSERT_SQL, values)
        inserted = cur.rowcount

    conn.commit()
    return max(inserted, 0)


def _get_latest_timestamp(conn, symbol: str) -> datetime | None:
    """Return the most recent candle timestamp for symbol, or None."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MAX(timestamp) FROM raw.ohlcv WHERE symbol = %s",
            (symbol,),
        )
        row = cur.fetchone()
    return row[0] if row and row[0] else None


# ---------------------------------------------------------------------------
# Backfill logic
# ---------------------------------------------------------------------------

def backfill_symbol(
    conn,
    client,
    symbol: str,
    days: int = 90,
    interval: str = "1h",
    no_incremental: bool = False,
    now: datetime | None = None,
) -> int:
    """Fetch completed candles, including a 48-hour repair overlap.

    Returns rows inserted or updated across all chunks.
    """
    bound_log = log.bind(symbol=symbol, interval=interval)

    snapshot = now or datetime.now(timezone.utc)
    if snapshot.tzinfo is None or snapshot.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    latest = _get_latest_timestamp(conn, symbol)
    if latest and not no_incremental:
        chunk_start: datetime = latest - timedelta(hours=48)
        bound_log.info("incremental_backfill", from_ts=str(chunk_start))
    else:
        chunk_start = snapshot - timedelta(days=days)
        bound_log.info("full_backfill", from_ts=str(chunk_start), days=days)

    total_inserted = 0

    while chunk_start < snapshot:
        t0 = time.monotonic()
        df = client.fetch_ohlcv(
            symbol, interval=interval, start_date=chunk_start,
            end_date=snapshot, limit=1000, as_of=snapshot,
        )

        if df.empty:
            break

        rows = df.to_dict("records")
        inserted = upsert_ohlcv_rows(conn, rows)
        total_inserted += inserted
        elapsed = time.monotonic() - t0

        bound_log.info(
            "chunk_done",
            fetched=len(rows),
            inserted=inserted,
            elapsed_s=round(elapsed, 2),
        )

        # A stale or unordered page must never cause an endless request loop.
        next_start = (
            df["timestamp"].max().to_pydatetime() + timedelta(milliseconds=1)
        )
        if next_start <= chunk_start:
            raise RuntimeError(f"{symbol} response did not advance the backfill cursor")
        chunk_start = next_start

        if len(rows) < 1000:
            break  # No more historical data available

    return total_inserted


def run_backfill(
    symbols: list[str],
    days: int = 90,
    interval: str = "1h",
    no_incremental: bool = False,
    source: str = "mainnet",
    invalidate_features: bool = True,
) -> None:
    """Orchestrate full backfill: init DB, iterate symbols, print summary."""
    from sqlalchemy import create_engine

    from ingestion.binance_client import BinanceClientWrapper
    from ingestion.feature_validation import (
        build_validation_engine, feature_refresh_lock, mark_features_invalid,
    )
    from ingestion.models import create_all_tables

    if not symbols:
        raise ValueError("At least one symbol is required for backfill")

    if invalidate_features:
        # A standalone backfill also changes the source under dbt. Keep it
        # serialized with Airflow and manual feature publication, and leave
        # features invalid until a subsequent validated build succeeds.
        lock_engine = build_validation_engine()
        try:
            with feature_refresh_lock(lock_engine):
                mark_features_invalid(lock_engine, symbols)
                return run_backfill(
                    symbols, days=days, interval=interval,
                    no_incremental=no_incremental, source=source,
                    invalidate_features=False,
                )
        finally:
            lock_engine.dispose()

    load_dotenv()

    # 1. Ensure schema + hypertable exist.
    # quote_plus the password so symbols like @, /, :, # don't break the URL.
    engine = create_engine(
        "postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}".format(
            user=quote_plus(os.getenv("POSTGRES_USER", "trader")),
            pw=quote_plus(os.getenv("POSTGRES_PASSWORD", "")),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            db=os.getenv("POSTGRES_DB", "trading"),
        ),
        # future=True is a no-op under SQLAlchemy 2.0 (the main project env)
        # but switches SQLAlchemy 1.4 (pinned inside the Airflow image,
        # which can't take 2.0) into the 2.0-style Connection API that
        # create_all_tables relies on (conn.commit()).
        future=True,
    )
    try:
        create_all_tables(engine)
    finally:
        engine.dispose()

    # Market data comes from mainnet by default (real prices, full history,
    # no API key needed); order execution elsewhere stays on testnet.
    client = BinanceClientWrapper(testnet=(source == "testnet"))
    conn = _get_db_connection()
    summary: dict[str, dict[str, int]] = {}
    failed_symbols: list[str] = []

    try:
        for symbol in symbols:
            try:
                inserted = backfill_symbol(
                    conn, client, symbol, days=days, interval=interval,
                    no_incremental=no_incremental,
                )
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM raw.ohlcv WHERE symbol = %s",
                        (symbol,),
                    )
                    total = cur.fetchone()[0]

                summary[symbol] = {"inserted": inserted, "total": total}
                log.info(
                    "symbol_complete",
                    symbol=symbol,
                    changed_rows=inserted,
                    total_in_db=total,
                )
            except Exception:
                log.exception("symbol_failed", symbol=symbol)
                summary[symbol] = {"inserted": -1, "total": -1}
                failed_symbols.append(symbol)
    finally:
        conn.close()

    # 3. Print human-readable summary
    print("\n=== Backfill Summary ===")
    for sym, stats in summary.items():
        if stats["inserted"] < 0:
            print(f"  {sym:<15} FAILED")
        else:
            print(
                f"  {sym:<15} {stats['inserted']:>6} inserted/updated rows"
                f"  |  {stats['total']:>7,} total in DB"
            )

    if failed_symbols:
        raise RuntimeError(f"Backfill failed for {', '.join(failed_symbols)}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill historical OHLCV candles from Binance into TimescaleDB"
    )
    parser.add_argument(
        "--symbols",
        help="Comma-separated symbols (overrides TRADING_SYMBOLS env var)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Lookback window in days for a full backfill (default: 90)",
    )
    parser.add_argument(
        "--interval",
        default="1h",
        help="Candle interval, e.g. 1m, 5m, 15m, 1h, 4h, 1d (default: 1h)",
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        default=False,
        help="Ignore existing data and backfill from now−days, re-inserting "
             "older bars (safe: changed values are updated; use with --days 180 "
             "to extend an existing 90-day dataset backwards)",
    )
    parser.add_argument(
        "--source",
        choices=["mainnet", "testnet"],
        default="mainnet",
        help="Market-data source (default: mainnet — public klines, no key "
             "needed, years of history; testnet only keeps legacy behavior)",
    )
    args = parser.parse_args()

    load_dotenv()

    if args.symbols:
        target_symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        env_syms = os.getenv("TRADING_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT")
        target_symbols = [s.strip() for s in env_syms.split(",") if s.strip()]

    run_backfill(
        symbols=target_symbols, days=args.days, interval=args.interval,
        no_incremental=args.no_incremental, source=args.source,
    )
