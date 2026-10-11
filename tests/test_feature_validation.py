"""Validation state must fail closed when a refresh is incomplete."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from execution.freshness import FreshnessMonitor

NOW = datetime(2026, 9, 23, 1, tzinfo=UTC)


def test_invalidation_precedes_new_validation_and_stores_feature_timestamp():
    from ingestion.feature_validation import mark_features_invalid, mark_features_valid

    engine = MagicMock()
    conn = engine.begin.return_value.__enter__.return_value
    conn.execute.return_value.rowcount = 1
    generation = mark_features_invalid(engine, ["BTCUSDT"])
    mark_features_valid(engine, generation, ["BTCUSDT"])
    statements = [str(call.args[0]) for call in conn.execute.call_args_list]
    assert "CREATE TABLE IF NOT EXISTS raw.feature_validation" in statements[1]
    assert "SET is_valid = FALSE" in statements[2]
    assert "MAX(timestamp)" in statements[-1]
    assert "features.feat_technical" in statements[-1]
    assert "is_valid = TRUE" in statements[-1]
    assert conn.execute.call_args.kwargs == {}  # token is a SQL bind, not interpolation
    assert conn.execute.call_args.args[1]["generation"] == generation
    assert engine.begin.call_count == 2


def test_older_refresh_cannot_publish_after_new_invalidation():
    from ingestion.feature_validation import mark_features_valid

    engine = MagicMock()
    engine.begin.return_value.__enter__.return_value.execute.return_value.rowcount = 0
    with pytest.raises(RuntimeError, match="superseded"):
        mark_features_valid(engine, "old-generation", ["BTCUSDT"])


@pytest.mark.parametrize(
    "row,expected",
    [
        (None, "missing"),
        ((False, NOW), "invalid"),
        ((True, NOW - timedelta(hours=1)), "unvalidated"),
        ((True, NOW), "fresh"),
    ],
)
def test_entry_gate_requires_current_validated_feature_version(row, expected):
    engine = MagicMock()
    result = engine.connect.return_value.__enter__.return_value.execute.return_value
    result.scalar.side_effect = [NOW, NOW, None]
    result.first.return_value = row
    monitor = FreshnessMonitor(engine)

    statuses = monitor.check("BTCUSDT", now=NOW)
    assert statuses[-1].source == "feature_validation"
    assert statuses[-1].status == expected
    allowed = not [status for status in statuses if status.required and status.status != "fresh"]
    assert allowed is (expected == "fresh")


def test_missing_validation_table_blocks_entry():
    engine = MagicMock()
    connection = engine.connect.return_value.__enter__.return_value
    result = connection.execute.return_value
    result.scalar.side_effect = [NOW, NOW, None]
    result.first.side_effect = RuntimeError("relation raw.feature_validation does not exist")
    statuses = FreshnessMonitor(engine).check("BTCUSDT", now=NOW)
    assert statuses[-1].status == "unavailable"
    assert statuses[-1].required


def test_refresh_lock_rejects_concurrent_writer():
    from ingestion.feature_validation import feature_refresh_lock

    engine = MagicMock()
    conn = engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.scalar.return_value = False
    with pytest.raises(RuntimeError, match="already running"):
        with feature_refresh_lock(engine):
            pytest.fail("lock should reject the writer")


# ---------------------------------------------------------------------------
# feature_refresh_lock's session-connection invariant: lock and unlock must
# run on the SAME pooled connection, held open across the whole critical
# section. A plain MagicMock can't catch a regression here -- every
# engine.connect() call returns the identical mock object regardless of how
# many times it's invoked. These fakes hand out a DISTINCT connection object
# per connect() call, so a bug that opens a second connection (e.g. to
# unlock) is visible.
# ---------------------------------------------------------------------------

class _FakeLockConnection:
    """One fake pooled connection; logs enter/exec/exit onto a shared list
    so ordering across connections is observable."""

    def __init__(self, log, conn_id, try_lock_result):
        self._log = log
        self.id = conn_id
        self._try_lock_result = try_lock_result

    def __enter__(self):
        self._log.append(("enter", self.id))
        return self

    def __exit__(self, exc_type, exc, tb):
        self._log.append(("exit", self.id, exc_type))
        return False  # never swallow

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self._log.append(("exec", self.id, sql, params))
        result = MagicMock()
        if "pg_try_advisory_lock" in sql:
            result.scalar.return_value = self._try_lock_result
        elif "pg_advisory_unlock" in sql:
            result.scalar.return_value = True
        return result


class _FakeLockEngine:
    """engine.connect() returns a NEW _FakeLockConnection every call."""

    def __init__(self, try_lock_result=True):
        self.log: list = []
        self._next_id = 0
        self._try_lock_result = try_lock_result

    def connect(self):
        self._next_id += 1
        return _FakeLockConnection(self.log, self._next_id, self._try_lock_result)


def _connect_calls(log):
    return [entry for entry in log if entry[0] == "enter"]


def _execs_matching(log, needle):
    return [entry for entry in log if entry[0] == "exec" and needle in entry[2]]


def test_lock_and_unlock_execute_on_the_identical_connection():
    from ingestion.feature_validation import feature_refresh_lock

    engine = _FakeLockEngine(try_lock_result=True)
    with feature_refresh_lock(engine):
        pass

    assert len(_connect_calls(engine.log)) == 1, "lock and unlock must share one connection"
    lock_calls = _execs_matching(engine.log, "pg_try_advisory_lock")
    unlock_calls = _execs_matching(engine.log, "pg_advisory_unlock")
    assert len(lock_calls) == 1 and len(unlock_calls) == 1
    assert lock_calls[0][1] == unlock_calls[0][1], "lock/unlock ran on different connections"


def test_unlock_runs_when_the_critical_section_raises():
    from ingestion.feature_validation import feature_refresh_lock

    engine = _FakeLockEngine(try_lock_result=True)
    with pytest.raises(ValueError, match="boom"):
        with feature_refresh_lock(engine):
            raise ValueError("boom")

    assert _execs_matching(engine.log, "pg_advisory_unlock"), "unlock must still run"
    assert len(_connect_calls(engine.log)) == 1


def test_connection_stays_open_until_after_unlock_executes():
    """The unlock must run BEFORE the connection is returned to the pool
    (context exit), not after -- unlocking a closed/returned connection
    would be a no-op or hit a different session."""
    from ingestion.feature_validation import feature_refresh_lock

    engine = _FakeLockEngine(try_lock_result=True)
    with feature_refresh_lock(engine):
        pass

    unlock_index = next(i for i, e in enumerate(engine.log)
                        if e[0] == "exec" and "pg_advisory_unlock" in e[2])
    exit_index = next(i for i, e in enumerate(engine.log) if e[0] == "exit")
    assert unlock_index < exit_index


def test_failed_try_lock_raises_and_never_unlocks_another_writers_lock():
    from ingestion.feature_validation import feature_refresh_lock

    engine = _FakeLockEngine(try_lock_result=False)
    entered_body = False
    with pytest.raises(RuntimeError, match="already running"):
        with feature_refresh_lock(engine):
            entered_body = True

    assert not entered_body, "critical section must not run without the lock"
    assert not _execs_matching(engine.log, "pg_advisory_unlock"), (
        "a failed try_lock must never release someone else's lock"
    )
    # The connection that made the failed attempt is still cleanly closed.
    assert len(_connect_calls(engine.log)) == 1
    assert any(e[0] == "exit" for e in engine.log)
