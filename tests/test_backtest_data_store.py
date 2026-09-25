"""Unit tests for backtest.data_store — parquet snapshots of raw.ohlcv and
features.feat_technical that auto-refresh when the live table's per-symbol
fingerprint (row count + max timestamp) no longer matches what's cached.

No real DB — the DB-touching bits (_fingerprint_symbol, _query_ohlcv,
_query_features) are tested with a mocked engine; the snapshot read/write
logic uses a real parquet round-trip against a tmp_path so serialization
edge cases (Decimal, tz-aware timestamps, NaN) are genuinely exercised.

Run:
    poetry run pytest tests/test_backtest_data_store.py -v
"""
from __future__ import annotations

import decimal
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd

from backtest import data_store


class TestFingerprintSymbol:

    def test_queries_count_and_max_timestamp(self):
        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchone.return_value = (26280, "2026-07-04 00:00:00+00", "digest")

        result = data_store._fingerprint_symbol(engine, "raw.ohlcv", "BTCUSDT")

        assert result == (26280, "2026-07-04 00:00:00+00", "digest")
        sql = str(conn.execute.call_args[0][0])
        assert "count" in sql.lower()
        assert "max" in sql.lower()
        assert "raw.ohlcv" in sql
        assert "row_to_json" in sql.lower()
        assert "order by timestamp" in sql.lower()

    def test_empty_table_returns_zero_none(self):
        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchone.return_value = (0, None, "empty")

        assert data_store._fingerprint_symbol(engine, "raw.ohlcv", "BTCUSDT") == (0, None, "empty")

    def test_correction_changes_fingerprint_without_new_timestamp(self):
        engine = MagicMock()
        conn = engine.connect.return_value.__enter__.return_value
        conn.execute.return_value.fetchone.side_effect = [
            (24, "2026-09-22", "before"), (24, "2026-09-22", "corrected"),
        ]
        before = data_store._fingerprint_symbol(engine, "raw.ohlcv", "BTCUSDT")
        after = data_store._fingerprint_symbol(engine, "raw.ohlcv", "BTCUSDT")
        assert before != after


class TestLoadOrRefresh:
    """The core freshness logic: skip the DB when the on-disk snapshot's
    fingerprint still matches; rebuild when it doesn't."""

    def test_correction_refreshes_snapshot_at_same_count_and_timestamp(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data_store, "SNAPSHOT_DIR", str(tmp_path))
        query = MagicMock(side_effect=[
            pd.DataFrame({"close": [100.0]}), pd.DataFrame({"close": [102.0]}),
        ])
        fingerprint = MagicMock(side_effect=[(1, "same", "old"), (1, "same", "new")])
        data_store._load_or_refresh("corrected", "BTCUSDT", query, fingerprint)
        result = data_store._load_or_refresh("corrected", "BTCUSDT", query, fingerprint)
        assert result["close"].tolist() == [102.0]
        assert query.call_count == 2

    def test_first_call_queries_and_writes_snapshot(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data_store, "SNAPSHOT_DIR", str(tmp_path))
        query_fn = MagicMock(return_value=pd.DataFrame({"a": [1, 2, 3]}))
        fp_fn = MagicMock(return_value=(3, "2026-01-01"))

        df = data_store._load_or_refresh("BTCUSDT_test", "BTCUSDT", query_fn, fp_fn)

        query_fn.assert_called_once()
        pd.testing.assert_frame_equal(df, pd.DataFrame({"a": [1, 2, 3]}))

    def test_second_call_with_unchanged_fingerprint_skips_query(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data_store, "SNAPSHOT_DIR", str(tmp_path))
        query_fn = MagicMock(return_value=pd.DataFrame({"a": [1, 2, 3]}))
        fp_fn = MagicMock(return_value=(3, "2026-01-01"))

        data_store._load_or_refresh("BTCUSDT_test", "BTCUSDT", query_fn, fp_fn)
        df2 = data_store._load_or_refresh("BTCUSDT_test", "BTCUSDT", query_fn, fp_fn)

        assert query_fn.call_count == 1  # not called again
        pd.testing.assert_frame_equal(df2, pd.DataFrame({"a": [1, 2, 3]}))

    def test_changed_fingerprint_triggers_requery(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data_store, "SNAPSHOT_DIR", str(tmp_path))
        query_fn = MagicMock(side_effect=[
            pd.DataFrame({"a": [1, 2, 3]}),
            pd.DataFrame({"a": [1, 2, 3, 4]}),
        ])
        fp_fn = MagicMock(side_effect=[(3, "2026-01-01"), (4, "2026-01-02")])

        df1 = data_store._load_or_refresh("BTCUSDT_test", "BTCUSDT", query_fn, fp_fn)
        df2 = data_store._load_or_refresh("BTCUSDT_test", "BTCUSDT", query_fn, fp_fn)

        assert query_fn.call_count == 2
        assert len(df1) == 3
        assert len(df2) == 4

    def test_tz_aware_timestamps_round_trip_through_parquet(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data_store, "SNAPSHOT_DIR", str(tmp_path))
        ts = [datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 2, tzinfo=timezone.utc)]
        query_fn = MagicMock(return_value=pd.DataFrame({"timestamp": ts, "close": [1.0, 2.0]}))
        fp_fn = MagicMock(return_value=(2, "2026-01-02"))

        data_store._load_or_refresh("BTCUSDT_test", "BTCUSDT", query_fn, fp_fn)
        df2 = data_store._load_or_refresh("BTCUSDT_test", "BTCUSDT", query_fn, fp_fn)  # from disk

        assert df2["timestamp"].dt.tz is not None
        assert list(df2["timestamp"]) == ts


class TestQueryOhlcv:

    def test_coerces_decimal_to_float_and_localizes_naive_timestamps(self):
        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchall.return_value = [
            (datetime(2026, 1, 1), decimal.Decimal("100.5"), decimal.Decimal("101.0"),
             decimal.Decimal("99.5"), decimal.Decimal("100.8")),
        ]
        conn.execute.return_value.keys.return_value = ["timestamp", "open", "high", "low", "close"]

        df = data_store._query_ohlcv(engine, "BTCUSDT")

        assert df["open"].dtype == float
        assert df["timestamp"].dt.tz is not None

    def test_empty_result_returns_empty_frame(self):
        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchall.return_value = []
        conn.execute.return_value.keys.return_value = ["timestamp", "open", "high", "low", "close"]

        df = data_store._query_ohlcv(engine, "BTCUSDT")

        assert df.empty


class TestQueryFeatures:

    def test_coerces_feature_columns_to_float(self):
        from signals.ml_model import MLSignalGenerator

        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__.return_value = conn
        cols = ["symbol", "timestamp", "close"] + MLSignalGenerator.FEATURES
        row = ["BTCUSDT", datetime(2026, 1, 1), decimal.Decimal("100.0")] + [
            decimal.Decimal("1.0") for _ in MLSignalGenerator.FEATURES
        ]
        conn.execute.return_value.fetchall.return_value = [tuple(row)]
        conn.execute.return_value.keys.return_value = cols

        df = data_store._query_features(engine, "BTCUSDT")

        assert df["close"].dtype == float
        for col in MLSignalGenerator.FEATURES:
            assert df[col].dtype == float
        assert df["timestamp"].dt.tz is not None
