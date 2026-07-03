"""Historical OHLCV backfill from Binance REST API into TimescaleDB.

Idempotent: uses INSERT … ON CONFLICT (symbol, timestamp) DO NOTHING.
Incremental: picks up from the latest stored timestamp for each symbol.

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
    ON CONFLICT (symbol, timestamp) DO NOTHING
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
    """Bulk-insert OHLCV rows with ON CONFLICT DO NOTHING.

    Args:
        conn: psycopg2 connection.
        rows: list of dicts with keys matching OHLCV column names.

    Returns:
        Number of rows inserted (0 when rows is empty or all conflicted).
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
) -> int:
    """Fetch and insert all missing candles for one symbol.

    Returns total rows inserted across all chunks.
    """
    bound_log = log.bind(symbol=symbol, interval=interval)

    latest = _get_latest_timestamp(conn, symbol)
    if latest and not no_incremental:
        # Advance 1 ms past the last stored candle to avoid re-fetching it
        chunk_start: datetime = latest + timedelta(milliseconds=1)
        bound_log.info("incremental_backfill", from_ts=str(chunk_start))
    else:
        chunk_start = datetime.now(timezone.utc) - timedelta(days=days)
        bound_log.info("full_backfill", from_ts=str(chunk_start), days=days)

    total_inserted = 0

    while chunk_start < datetime.now(timezone.utc):
        t0 = time.monotonic()
        df = client.fetch_ohlcv(
            symbol, interval=interval, start_date=chunk_start, limit=1000
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

        # Advance past the last returned candle
        chunk_start = (
            df["timestamp"].max().to_pydatetime() + timedelta(milliseconds=1)
        )

        if len(rows) < 1000:
            break  # No more historical data available

    return total_inserted


def run_backfill(
    symbols: list[str],
    days: int = 90,
    interval: str = "1h",
    no_incremental: bool = False,
) -> None:
    """Orchestrate full backfill: init DB, iterate symbols, print summary."""
    from sqlalchemy import create_engine

    from ingestion.binance_client import BinanceClientWrapper
    from ingestion.models import create_all_tables

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

    client = BinanceClientWrapper()
    conn = _get_db_connection()
    summary: dict[str, dict[str, int]] = {}

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
                    new_rows=inserted,
                    total_in_db=total,
                )
            except Exception:
                log.exception("symbol_failed", symbol=symbol)
                summary[symbol] = {"inserted": -1, "total": -1}
    finally:
        conn.close()

    # 3. Print human-readable summary
    print("\n=== Backfill Summary ===")
    for sym, stats in summary.items():
        if stats["inserted"] < 0:
            print(f"  {sym:<15} FAILED")
        else:
            print(
                f"  {sym:<15} {stats['inserted']:>6} new rows"
                f"  |  {stats['total']:>7,} total in DB"
            )


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
             "older bars (safe: ON CONFLICT DO NOTHING; use with --days 180 "
             "to extend an existing 90-day dataset backwards)",
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
        no_incremental=args.no_incremental,
    )
