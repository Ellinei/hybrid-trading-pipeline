"""Unit tests for execution/heartbeat.py — per-cycle liveness/activity
record (raw.bot_heartbeat) so ops/dashboard can tell a hung bot (stale
last-heartbeat timestamp) from a quiet market (fresh timestamp, zero
actions).

No real DB — all I/O is mocked.

Run:
    poetry run pytest tests/test_heartbeat.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock

from execution.heartbeat import HeartbeatRecorder


def _make_fake_engine():
    conn = MagicMock()
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    return engine, conn


class TestHeartbeatRecorderConstruction:
    def test_ensures_bot_heartbeat_table_on_construction(self):
        engine, conn = _make_fake_engine()
        HeartbeatRecorder(engine)

        sqls = [str(call.args[0]) for call in conn.execute.call_args_list]
        assert any("CREATE TABLE IF NOT EXISTS raw.bot_heartbeat" in sql for sql in sqls)


class TestHeartbeatRecorderRecord:
    def test_inserts_row_with_signal_action_error_counts(self):
        engine, conn = _make_fake_engine()
        recorder = HeartbeatRecorder(engine)
        conn.execute.reset_mock()

        recorder.record(signals=3, actions=1, errors=0)

        sql, params = conn.execute.call_args.args[0], conn.execute.call_args.args[1]
        assert "INSERT INTO raw.bot_heartbeat" in str(sql)
        assert params == {"signals": 3, "actions": 1, "errors": 0}

    def test_record_failure_is_swallowed_not_raised(self):
        engine, conn = _make_fake_engine()
        recorder = HeartbeatRecorder(engine)
        conn.execute.side_effect = Exception("db gone")

        recorder.record(signals=1, actions=0, errors=1)  # must not raise
