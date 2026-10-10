"""Unit tests for the ingestion package.

No real API calls or DB connections — everything is mocked.

Run:
    poetry run pytest tests/test_ingestion.py -v
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    """Isolate every test from local .env / shell environment."""
    monkeypatch.setenv("BINANCE_API_KEY", "test_key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
    monkeypatch.setenv("BINANCE_TESTNET", "true")
    monkeypatch.setenv("TRADING_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT")
    monkeypatch.setenv("POSTGRES_USER", "test_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test_pass")
    monkeypatch.setenv("POSTGRES_DB", "test_db")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")


# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

_MOCK_KLINES = [
    [
        1_609_459_200_000,  # 2021-01-01 00:00:00 UTC
        "29000.0", "29500.0", "28800.0", "29300.0",
        "100.5", 1_609_462_799_999,
        "2930000.0", 1500, "50.25", "1465000.0", "0",
    ],
    [
        1_609_462_800_000,  # 2021-01-01 01:00:00 UTC
        "29300.0", "29800.0", "29100.0", "29600.0",
        "120.3", 1_609_466_399_999,
        "3552000.0", 1800, "60.15", "1776000.0", "0",
    ],
]


# ===========================================================================
# 1. BinanceClientWrapper.ping
# ===========================================================================

class TestBinanceClientPing:
    """Mock the Binance client and assert ping() returns the right bool."""

    @patch("ingestion.binance_client.Client")
    def test_ping_returns_true_when_server_responds(self, MockClient):
        MockClient.return_value.ping.return_value = {}

        from ingestion.binance_client import BinanceClientWrapper
        assert BinanceClientWrapper().ping() is True

    @patch("ingestion.binance_client.Client")
    def test_ping_returns_false_when_connection_raises(self, MockClient):
        MockClient.return_value.ping.side_effect = Exception("timeout")

        from ingestion.binance_client import BinanceClientWrapper
        assert BinanceClientWrapper().ping() is False


# ===========================================================================
# 1a. BinanceClientWrapper testnet/mainnet source selection
# ===========================================================================

class TestClientSourceSelection:
    """Explicit testnet=False must build a keyless mainnet client — the env
    keys are testnet credentials and must never reach the live exchange."""

    @patch("ingestion.binance_client.Client")
    def test_default_follows_env_testnet_true(self, MockClient):
        from ingestion.binance_client import BinanceClientWrapper
        BinanceClientWrapper()
        MockClient.assert_called_once_with("test_key", "test_secret", testnet=True)

    @patch("ingestion.binance_client.Client")
    def test_explicit_mainnet_uses_empty_keys(self, MockClient):
        from ingestion.binance_client import BinanceClientWrapper
        BinanceClientWrapper(testnet=False)
        MockClient.assert_called_once_with("", "", testnet=False)

    @patch("ingestion.binance_client.Client")
    def test_explicit_testnet_keeps_env_keys(self, MockClient):
        from ingestion.binance_client import BinanceClientWrapper
        BinanceClientWrapper(testnet=True)
        MockClient.assert_called_once_with("test_key", "test_secret", testnet=True)


# ===========================================================================
# 1b. BinanceClientWrapper.get_account_balance — must never fabricate a value
# ===========================================================================

class TestGetAccountBalance:
    @patch("ingestion.binance_client.Client")
    def test_returns_free_balance_on_success(self, MockClient):
        MockClient.return_value.get_asset_balance.return_value = {"free": "123.45"}

        from ingestion.binance_client import BinanceClientWrapper
        assert BinanceClientWrapper().get_account_balance("USDT") == 123.45

    @patch("ingestion.binance_client.Client")
    def test_raises_when_api_call_fails(self, MockClient):
        MockClient.return_value.get_asset_balance.side_effect = Exception("timeout")

        from ingestion.binance_client import BinanceClientWrapper
        with pytest.raises(Exception, match="timeout"):
            BinanceClientWrapper().get_account_balance("USDT")

    @patch("ingestion.binance_client.Client")
    def test_raises_when_no_balance_entry_returned(self, MockClient):
        MockClient.return_value.get_asset_balance.return_value = None

        from ingestion.binance_client import BinanceClientWrapper
        with pytest.raises(ValueError):
            BinanceClientWrapper().get_account_balance("USDT")


# ===========================================================================
# 1c. BinanceClientWrapper.get_symbol_info_cached — fetch once, cache after
# ===========================================================================

class TestGetSymbolInfoCached:
    @patch("ingestion.binance_client.Client")
    def test_returns_info_from_api(self, MockClient):
        MockClient.return_value.get_symbol_info.return_value = {"symbol": "BTCUSDT", "filters": []}

        from ingestion.binance_client import BinanceClientWrapper
        wrapper = BinanceClientWrapper()
        assert wrapper.get_symbol_info_cached("BTCUSDT") == {"symbol": "BTCUSDT", "filters": []}

    @patch("ingestion.binance_client.Client")
    def test_only_calls_api_once_per_symbol(self, MockClient):
        MockClient.return_value.get_symbol_info.return_value = {"symbol": "BTCUSDT", "filters": []}

        from ingestion.binance_client import BinanceClientWrapper
        wrapper = BinanceClientWrapper()
        wrapper.get_symbol_info_cached("BTCUSDT")
        wrapper.get_symbol_info_cached("BTCUSDT")

        assert MockClient.return_value.get_symbol_info.call_count == 1

    @patch("ingestion.binance_client.Client")
    def test_raises_when_api_returns_none(self, MockClient):
        MockClient.return_value.get_symbol_info.return_value = None

        from ingestion.binance_client import BinanceClientWrapper
        wrapper = BinanceClientWrapper()
        with pytest.raises(ValueError):
            wrapper.get_symbol_info_cached("BTCUSDT")


# ===========================================================================
# 2. BinanceClientWrapper.fetch_ohlcv — DataFrame shape and dtypes
# ===========================================================================

class TestFetchOHLCV:
    """Mock API response; assert DataFrame columns and dtypes."""

    @patch("ingestion.binance_client.Client")
    def test_returns_dataframe_with_all_required_columns(self, MockClient):
        MockClient.return_value.get_historical_klines.return_value = _MOCK_KLINES

        from ingestion.binance_client import BinanceClientWrapper
        df = BinanceClientWrapper().fetch_ohlcv("BTCUSDT")

        required = {
            "symbol", "timestamp", "open", "high", "low",
            "close", "volume", "quote_volume", "trades", "taker_buy_vol",
        }
        assert required.issubset(set(df.columns))

    @patch("ingestion.binance_client.Client")
    def test_timestamp_is_utc_aware(self, MockClient):
        MockClient.return_value.get_historical_klines.return_value = _MOCK_KLINES

        from ingestion.binance_client import BinanceClientWrapper
        df = BinanceClientWrapper().fetch_ohlcv("BTCUSDT")

        assert df["timestamp"].dt.tz is not None, "timestamp must be timezone-aware"
        assert str(df["timestamp"].dt.tz) == "UTC"

    @patch("ingestion.binance_client.Client")
    def test_price_and_volume_columns_are_float(self, MockClient):
        MockClient.return_value.get_historical_klines.return_value = _MOCK_KLINES

        from ingestion.binance_client import BinanceClientWrapper
        df = BinanceClientWrapper().fetch_ohlcv("BTCUSDT")

        for col in ("open", "high", "low", "close", "volume"):
            assert pd.api.types.is_float_dtype(df[col]), f"'{col}' must be float"

    @patch("ingestion.binance_client.Client")
    def test_symbol_column_matches_requested_symbol(self, MockClient):
        MockClient.return_value.get_historical_klines.return_value = _MOCK_KLINES

        from ingestion.binance_client import BinanceClientWrapper
        df = BinanceClientWrapper().fetch_ohlcv("ETHUSDT")

        assert (df["symbol"] == "ETHUSDT").all()

    @patch("ingestion.binance_client.Client")
    def test_empty_api_response_returns_empty_dataframe(self, MockClient):
        MockClient.return_value.get_historical_klines.return_value = []

        from ingestion.binance_client import BinanceClientWrapper
        df = BinanceClientWrapper().fetch_ohlcv("BTCUSDT")

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    @patch("ingestion.binance_client.Client")
    def test_only_completed_candles_crossing_hour_boundary(self, MockClient):
        from ingestion.binance_client import BinanceClientWrapper

        MockClient.return_value.get_historical_klines.return_value = _MOCK_KLINES
        wrapper = BinanceClientWrapper()
        at_boundary = datetime(2021, 1, 1, 1, tzinfo=UTC)
        df = wrapper.fetch_ohlcv("BTCUSDT", as_of=at_boundary)
        assert list(df["close"]) == [29300.0]
        assert wrapper.fetch_ohlcv(
            "BTCUSDT", as_of=at_boundary + timedelta(minutes=59, seconds=59)
        )["close"].tolist() == [29300.0]
        assert wrapper.fetch_ohlcv(
            "BTCUSDT", as_of=at_boundary + timedelta(hours=1)
        )["close"].tolist() == [29300.0, 29600.0]


# ===========================================================================
# 3. upsert_ohlcv_rows — idempotency and provisional-row repair
# ===========================================================================

class TestBackfillIdempotency:
    """Run insert twice with same data; assert row count doesn't grow."""

    _ROW = {
        "symbol": "BTCUSDT",
        "timestamp": datetime(2024, 1, 1, tzinfo=UTC),
        "open": 42_000.0, "high": 43_000.0, "low": 41_000.0,
        "close": 42_500.0, "volume": 10.5,
        "quote_volume": 441_000.0, "trades": 500, "taker_buy_vol": 5.25,
    }

    def test_insert_sql_repairs_conflicting_candle_values(self):
        from ingestion.rest_backfill import OHLCV_INSERT_SQL

        assert "ON CONFLICT" in OHLCV_INSERT_SQL
        assert "DO UPDATE SET" in OHLCV_INSERT_SQL
        for column in ("open", "high", "low", "close", "volume",
                       "quote_volume", "trades", "taker_buy_vol"):
            assert f"{column} = EXCLUDED.{column}" in OHLCV_INSERT_SQL

    def test_calling_upsert_twice_invokes_execute_values_twice(self):
        from ingestion.rest_backfill import upsert_ohlcv_rows

        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value.rowcount = 1

        with patch("ingestion.rest_backfill.execute_values") as mock_ev:
            upsert_ohlcv_rows(mock_conn, [self._ROW])
            upsert_ohlcv_rows(mock_conn, [self._ROW])

            assert mock_ev.call_count == 2

    def test_empty_row_list_skips_db_call_and_returns_zero(self):
        from ingestion.rest_backfill import upsert_ohlcv_rows

        mock_conn = MagicMock()
        with patch("ingestion.rest_backfill.execute_values") as mock_ev:
            result = upsert_ohlcv_rows(mock_conn, [])

            mock_ev.assert_not_called()
            assert result == 0


class TestIncrementalRepair:
    def test_overlap_refetches_latest_and_gaps_and_stops_at_fixed_snapshot(self):
        from ingestion.rest_backfill import backfill_symbol

        now = datetime(2024, 1, 3, 3, tzinfo=UTC)
        latest = datetime(2024, 1, 3, 1, tzinfo=UTC)
        client = MagicMock()
        client.fetch_ohlcv.return_value = pd.DataFrame({
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime([
                "2024-01-03T00:00:00Z", "2024-01-03T01:00:00Z"
            ]),
            "close": [42.0, 43.0],
        })
        with patch("ingestion.rest_backfill._get_latest_timestamp", return_value=latest), patch(
            "ingestion.rest_backfill.upsert_ohlcv_rows", return_value=2
        ) as upsert:
            assert backfill_symbol(MagicMock(), client, "BTCUSDT", now=now) == 2
        assert client.fetch_ohlcv.call_args.kwargs["start_date"] <= now - timedelta(hours=48)
        assert client.fetch_ohlcv.call_args.kwargs["end_date"] == now
        assert [r["close"] for r in upsert.call_args.args[1]] == [42.0, 43.0]

    def test_delayed_exchange_candle_is_picked_up_on_next_run(self):
        from ingestion.rest_backfill import backfill_symbol

        now = datetime(2024, 1, 3, 3, tzinfo=UTC)
        latest = datetime(2024, 1, 3, 1, tzinfo=UTC)
        client = MagicMock()
        client.fetch_ohlcv.side_effect = [
            pd.DataFrame(),
            pd.DataFrame({
                "symbol": ["BTCUSDT"],
                "timestamp": pd.to_datetime(["2024-01-03T02:00:00Z"]),
                "close": [44.0],
            }),
        ]
        with patch("ingestion.rest_backfill._get_latest_timestamp", return_value=latest), patch(
            "ingestion.rest_backfill.upsert_ohlcv_rows", return_value=1
        ) as upsert:
            assert backfill_symbol(MagicMock(), client, "BTCUSDT", now=now) == 0
            assert backfill_symbol(MagicMock(), client, "BTCUSDT", now=now) == 1
        assert client.fetch_ohlcv.call_args_list[0].kwargs["start_date"] == (
            client.fetch_ohlcv.call_args_list[1].kwargs["start_date"]
        )
        assert upsert.call_args.args[1][0]["close"] == 44.0

    def test_nonadvancing_full_batch_fails_instead_of_looping(self):
        from ingestion.rest_backfill import backfill_symbol

        now = datetime(2024, 1, 3, 3, tzinfo=UTC)
        latest = datetime(2024, 1, 3, 1, tzinfo=UTC)
        client = MagicMock()
        client.fetch_ohlcv.return_value = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01T00:00:00Z"] * 1000),
        })
        with patch("ingestion.rest_backfill._get_latest_timestamp", return_value=latest), patch(
            "ingestion.rest_backfill.upsert_ohlcv_rows", return_value=0
        ):
            with pytest.raises(RuntimeError, match="advance"):
                backfill_symbol(MagicMock(), client, "BTCUSDT", now=now)


class TestBackfillFailure:
    @pytest.fixture(autouse=True)
    def _block_real_db_connections(self):
        """run_backfill's default invalidate_features=True path builds a
        real engine via ingestion.feature_validation.build_validation_engine
        and locks/writes through it before any of the mocks below patch it
        out. Every test in this class must mock that path explicitly; this
        is a backstop so a hole that misses one of those patches fails
        loudly and immediately (AssertionError) instead of silently trying
        a real network connection to Postgres and hanging/erroring on
        auth — see the module docstring's "no real API calls or DB
        connections" guarantee."""
        with patch("psycopg2.connect", side_effect=AssertionError(
            "TestBackfillFailure must not open a real DB connection"
        )):
            yield

    def test_manual_backfill_invalidates_under_refresh_lock(self):
        from contextlib import contextmanager

        from ingestion.rest_backfill import run_backfill

        events = []
        engine = MagicMock()
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value.fetchone.return_value = (1,)

        @contextmanager
        def locked(_engine):
            events.append("lock")
            try:
                yield
            finally:
                events.append("unlock")

        def invalid(_engine, symbols):
            assert symbols == ["BTCUSDT"]
            events.append("invalid")

        def backfill(*_args, **_kwargs):
            events.append("backfill")
            return 0

        with patch("ingestion.feature_validation.build_validation_engine", return_value=engine), \
             patch("ingestion.feature_validation.feature_refresh_lock", locked), \
             patch("ingestion.feature_validation.mark_features_invalid", invalid), \
             patch("ingestion.models.create_all_tables"), \
             patch("sqlalchemy.create_engine", return_value=MagicMock()), \
             patch("ingestion.binance_client.BinanceClientWrapper"), \
             patch("ingestion.rest_backfill._get_db_connection", return_value=conn), \
             patch("ingestion.rest_backfill.backfill_symbol", backfill):
            run_backfill(["BTCUSDT"])

        assert events == ["lock", "invalid", "backfill", "unlock"]
        engine.dispose.assert_called_once()

    def test_empty_symbol_list_cannot_report_success(self):
        from ingestion.rest_backfill import run_backfill

        with patch("ingestion.models.create_all_tables"), patch(
            "sqlalchemy.create_engine"
        ), patch("ingestion.binance_client.BinanceClientWrapper"), patch(
            "ingestion.rest_backfill._get_db_connection"
        ):
            with pytest.raises(ValueError, match="symbol"):
                run_backfill([])

    def test_symbol_failure_propagates_after_summary(self):
        from ingestion.rest_backfill import run_backfill

        # run_backfill's default invalidate_features=True path goes through
        # ingestion.feature_validation.build_validation_engine ->
        # feature_refresh_lock -> mark_features_invalid before ever reaching
        # backfill_symbol. Those three must be mocked here too, or this
        # test opens a real engine/connection against whatever POSTGRES_*
        # the _test_env fixture set, same hole as the test above.
        with patch("ingestion.feature_validation.build_validation_engine",
                   return_value=MagicMock()), \
             patch("ingestion.feature_validation.feature_refresh_lock"), \
             patch("ingestion.feature_validation.mark_features_invalid"), \
             patch("ingestion.models.create_all_tables"), patch(
            "sqlalchemy.create_engine"
        ), patch("ingestion.binance_client.BinanceClientWrapper"), patch(
            "ingestion.rest_backfill._get_db_connection"
        ), patch("ingestion.rest_backfill.backfill_symbol", side_effect=RuntimeError("API failed")):
            with pytest.raises(RuntimeError, match="BTCUSDT"):
                run_backfill(["BTCUSDT"])


# ===========================================================================
# 4. ORM model — column presence, schema, and primary key
# ===========================================================================

class TestOHLCVModelColumns:
    """Assert all required columns exist on the SQLAlchemy model."""

    def test_ohlcv_has_exactly_the_required_columns(self):
        from ingestion.models import OHLCV

        actual = {col.name for col in OHLCV.__table__.columns}
        required = {
            "symbol", "timestamp", "open", "high", "low",
            "close", "volume", "quote_volume", "trades", "taker_buy_vol",
        }
        assert required == actual

    def test_ohlcv_is_in_raw_schema(self):
        from ingestion.models import OHLCV

        assert OHLCV.__table__.schema == "raw"

    def test_ohlcv_composite_primary_key_is_symbol_and_timestamp(self):
        from ingestion.models import OHLCV

        pk_cols = {col.name for col in OHLCV.__table__.primary_key}
        assert pk_cols == {"symbol", "timestamp"}

    def test_symbols_table_is_in_raw_schema(self):
        from ingestion.models import Symbol

        assert Symbol.__table__.schema == "raw"

    def test_symbols_primary_key_is_symbol(self):
        from ingestion.models import Symbol

        pk_cols = {col.name for col in Symbol.__table__.primary_key}
        assert "symbol" in pk_cols
