"""Per-cycle liveness/activity record for the live bot — raw.bot_heartbeat.

Written once per bot_runner.run_once cycle so ops/dashboard can tell a
hung bot (stale last-heartbeat timestamp) from a quiet market (fresh
timestamp, zero actions).
"""
from __future__ import annotations

import structlog
from sqlalchemy import text

log = structlog.get_logger()


class HeartbeatRecorder:
    def __init__(self, db_engine) -> None:
        self.engine = db_engine
        self._ensure_table()

    def _ensure_table(self) -> None:
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS raw.bot_heartbeat (
                    id      SERIAL      PRIMARY KEY,
                    ts      TIMESTAMPTZ DEFAULT now(),
                    signals INT         NOT NULL,
                    actions INT         NOT NULL,
                    errors  INT         NOT NULL
                )
            """))
            conn.commit()

    def record(self, signals: int, actions: int, errors: int) -> None:
        """Insert one heartbeat row. Never raises — a heartbeat failure
        must not take down the trading cycle that produced it."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO raw.bot_heartbeat (signals, actions, errors)
                    VALUES (:signals, :actions, :errors)
                """), {"signals": signals, "actions": actions, "errors": errors})
                conn.commit()
        except Exception as exc:
            log.warning("heartbeat_record_failed", error=str(exc))
