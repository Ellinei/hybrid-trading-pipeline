"""Unit tests for the ingestion package.

No real API calls or DB connections — everything is mocked.

Run:
    poetry run pytest tests/test_ingestion.py -v
"""
from __future__ import annotations

from datetime import datetime, timezone
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
        with pytest.raises(Exception):
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


# ===========================================================================
# 3. upsert_ohlcv_rows — idempotency via ON CONFLICT DO NOTHING
# ===========================================================================

class TestBackfillIdempotency:
    """Run insert twice with same data; assert row count doesn't grow."""

    _ROW = {
        "symbol": "BTCUSDT",
        "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "open": 42_000.0, "high": 43_000.0, "low": 41_000.0,
        "close": 42_500.0, "volume": 10.5,
        "quote_volume": 441_000.0, "trades": 500, "taker_buy_vol": 5.25,
    }

    def test_insert_sql_contains_on_conflict_do_nothing(self):
        from ingestion.rest_backfill import OHLCV_INSERT_SQL

        assert "ON CONFLICT" in OHLCV_INSERT_SQL
        assert "DO NOTHING" in OHLCV_INSERT_SQL

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
