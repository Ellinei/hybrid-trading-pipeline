"""Read-only summary of the forward paper trial, used by scripts/trial_*.ps1.

Uptime is counted from raw.bot_heartbeat, where each completed cycle writes
one row. Since activation only the trial bot writes heartbeats, so the number
of distinct UTC hours with a heartbeat is the number of evaluated hours. The
protocol amendment in docs/forward-trial.md targets 168 (7 days of cumulative
uptime) and reports every gap.

    python scripts/trial_report.py            # account + heartbeat uptime
    python scripts/trial_report.py --no-db    # account only (Postgres down)
    python scripts/trial_report.py --open-only
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRIAL_DB = REPO / "results" / "forward_20260924.sqlite3"
ACTIVATION = datetime(2026, 9, 25, 12, 37, 51, tzinfo=UTC)
TARGET_HOURS = 168
# Hourly cycles plus cycle time; a longer interval is downtime.
GAP_THRESHOLD = timedelta(minutes=70)


def _paper(conn: sqlite3.Connection) -> None:
    initial, cash, peak = conn.execute(
        "SELECT initial_balance, cash, peak FROM paper_account WHERE id = 1").fetchone()
    closed = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(pnl), 0) FROM paper_trades WHERE status = 'closed'"
    ).fetchone()
    print(f"Account: cash {cash:,.2f} USDT (initial {initial:,.2f}, peak {peak:,.2f})")
    print(f"Closed trades: {closed[0]}, net P&L {closed[1]:+,.2f} USDT")
    outcomes = conn.execute(
        "SELECT outcome, COUNT(*) FROM decisions GROUP BY outcome ORDER BY outcome").fetchall()
    print("Decisions: " + (", ".join(f"{o} {n}" for o, n in outcomes) or "none"))


def _open_positions(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute(
        "SELECT symbol, quantity, entry_price, stop_loss, take_profit, executed_at "
        "FROM paper_trades WHERE status = 'open' ORDER BY id").fetchall()


def _print_open(rows: list[tuple]) -> None:
    if not rows:
        print("Open positions: none")
        return
    print(f"Open positions: {len(rows)}")
    for sym, qty, entry, sl, tp, opened in rows:
        print(f"  {sym} qty={qty:g} entry={entry:g} stop={sl:g} target={tp:g} opened={opened}")


def _heartbeats() -> None:
    from sqlalchemy import text

    from ingestion.feature_validation import build_validation_engine

    engine = build_validation_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT ts, errors FROM raw.bot_heartbeat WHERE ts >= :t ORDER BY ts"
            ), {"t": ACTIVATION}).all()
    finally:
        engine.dispose()
    if not rows:
        print("Heartbeats since activation: none")
        return
    stamps = [r[0] for r in rows]
    gaps = [(a, b) for a, b in zip(stamps, stamps[1:], strict=False) if b - a > GAP_THRESHOLD]
    since_last = datetime.now(UTC) - stamps[-1]
    # A restart runs one cycle immediately, so two heartbeats can share an
    # hour. Count distinct UTC hours with a heartbeat instead of rows.
    hours = len({ts.replace(minute=0, second=0, microsecond=0) for ts in stamps})
    print(f"Hours evaluated since activation: {hours} / {TARGET_HOURS} target "
          f"({100 * hours / TARGET_HOURS:.0f}%); {len(rows)} cycles, "
          f"cycle errors {sum(r[1] for r in rows)}")
    print(f"Last heartbeat: {stamps[-1]:%Y-%m-%d %H:%M} UTC "
          f"({since_last.total_seconds() / 60:.0f} min ago)")
    print(f"Downtime gaps (> {GAP_THRESHOLD.seconds // 60} min): {len(gaps)}")
    for a, b in gaps[-10:]:
        print(f"  {a:%Y-%m-%d %H:%M} -> {b:%Y-%m-%d %H:%M} UTC "
              f"({(b - a).total_seconds() / 3600:.1f} h)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-db", action="store_true", help="skip Postgres heartbeat queries")
    parser.add_argument("--open-only", action="store_true", help="only list open positions")
    args = parser.parse_args()

    if not TRIAL_DB.exists():
        print(f"Trial account not found: {TRIAL_DB}")
        return 1
    conn = sqlite3.connect(f"file:{TRIAL_DB.as_posix()}?mode=ro", uri=True)
    try:
        if args.open_only:
            _print_open(_open_positions(conn))
            return 0
        _paper(conn)
        _print_open(_open_positions(conn))
    finally:
        conn.close()
    if not args.no_db:
        sys.path.insert(0, str(REPO))
        try:
            _heartbeats()
        except Exception as exc:  # Postgres down or unreachable
            print(f"Heartbeats: unavailable ({type(exc).__name__})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
