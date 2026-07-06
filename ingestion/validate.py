"""Quick validation script — Binance connectivity + TimescaleDB data quality.

Exits 0 if all checks pass, 1 if any fail.

Usage (from project root):
    python -m ingestion.validate
"""
from __future__ import annotations

import os
import sys

import structlog
from dotenv import load_dotenv

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Binance checks
# ---------------------------------------------------------------------------

def _check_binance(client) -> bool:
    ok = True

    reachable = client.ping()
    status = "OK" if reachable else "FAILED"
    print(f"  Binance ping:        {status}")
    if not reachable:
        return False

    for symbol in client.get_symbols():
        try:
            price = client.get_current_price(symbol)
            print(f"  {symbol:<16} ${price:>14,.2f}")
        except Exception as exc:
            print(f"  {symbol:<16} ERROR — {exc}")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# Database checks
# ---------------------------------------------------------------------------

def _check_database(conn) -> bool:
    ok = True

    # TimescaleDB
    with conn.cursor() as cur:
        cur.execute(
            "SELECT installed_version"
            " FROM pg_available_extensions"
            " WHERE name = 'timescaledb'"
        )
        row = cur.fetchone()

    ts_version = row[0] if row else None
    ts_label = f"enabled  (v{ts_version})" if ts_version else "NOT FOUND"
    print(f"  TimescaleDB:         {ts_label}")
    if not ts_version:
        ok = False

    # Row counts + time range per symbol, for each populated raw.* table
    for table, hint in (
        ("raw.ohlcv",             "run rest_backfill first"),
        ("raw.realtime_features", "run streaming.feature_processor first"),
    ):
        ok &= _print_table_stats(conn, table, hint)

    return ok


def _print_table_stats(conn, table: str, empty_hint: str) -> bool:
    """Print row counts and time ranges for a (symbol, timestamp) table.

    Returns True if the table exists; False if it's missing entirely.
    """
    with conn.cursor() as cur:
        # Note: f-string interpolation here is safe — `table` is a hardcoded
        # constant from this module, never user input. We can't parameterise
        # an identifier in psycopg2.
        try:
            cur.execute(
                f"""
                SELECT symbol, COUNT(*) AS rows,
                       MIN(timestamp) AS earliest,
                       MAX(timestamp) AS latest
                FROM {table}
                GROUP BY symbol
                ORDER BY symbol
                """
            )
            rows = cur.fetchall()
        except Exception as exc:
            print(f"\n  {table}:  TABLE MISSING — {exc}")
            conn.rollback()
            return False

    print(f"\n  {table}")
    if not rows:
        print(f"    (empty — {empty_hint})")
        return True

    print(f"    {'Symbol':<16} {'Rows':>8}  {'Earliest':<32} {'Latest':<32}")
    print(f"    {'-'*16} {'-'*8}  {'-'*32} {'-'*32}")
    for sym, count, earliest, latest in rows:
        print(f"    {sym:<16} {count:>8,}  {str(earliest):<32} {str(latest):<32}")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    load_dotenv()
    all_ok = True

    from ingestion.binance_client import BinanceClientWrapper

    print("\n=== Binance Connectivity ===")
    try:
        client = BinanceClientWrapper()
        all_ok &= _check_binance(client)
    except Exception as exc:
        print(f"  Binance init FAILED: {exc}")
        all_ok = False

    print("\n=== Database Status ===")
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "trading"),
            user=os.getenv("POSTGRES_USER", "trader"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
        )
        try:
            all_ok &= _check_database(conn)
        finally:
            conn.close()
    except Exception as exc:
        print(f"  Database connection FAILED: {exc}")
        all_ok = False

    # ASCII only — Windows consoles default to cp1252, which can't encode ✓/✗
    verdict = "All checks passed [OK]" if all_ok else "SOME CHECKS FAILED [X]"
    print(f"\n{verdict}\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
