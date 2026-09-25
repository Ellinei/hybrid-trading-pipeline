"""Local, persistent decision history shared by live, dry-run and paper modes."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog


def operations_path():
    return Path(os.getenv("BOT_OPERATIONS_DB", "results/operations.sqlite3"))


def connect_store(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


class DecisionJournal:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else operations_path()
        conn = connect_store(self.path)
        try:
            conn.execute("""CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY, recorded_at TEXT NOT NULL,
                symbol TEXT NOT NULL, mode TEXT NOT NULL, direction TEXT,
                outcome TEXT NOT NULL, reason TEXT NOT NULL, details TEXT NOT NULL
            )""")
            conn.execute("CREATE INDEX IF NOT EXISTS decisions_symbol ON decisions(symbol, id)")
        finally:
            conn.close()

    def record(self, *, symbol, mode, outcome, reason, signal=None, config=None, **details):
        """Return success; a journal outage must not interrupt position exits."""
        try:
            payload = {
                "signal": asdict(signal) if signal is not None else None,
                "config": asdict(config) if is_dataclass(config) else config,
                **details,
            }
            encoded = json.dumps(payload, default=lambda value: getattr(value, "value", str(value)))
            conn = connect_store(self.path)
            try:
                conn.execute(
                    """INSERT INTO decisions
                    (recorded_at, symbol, mode, direction, outcome, reason, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        datetime.now(UTC).isoformat(),
                        symbol,
                        mode,
                        signal.direction.value if signal is not None else None,
                        outcome,
                        reason,
                        encoded,
                    ),
                )
            finally:
                conn.close()
            return True
        except Exception:
            structlog.get_logger().exception("decision_journal_failed", symbol=symbol)
            return False
