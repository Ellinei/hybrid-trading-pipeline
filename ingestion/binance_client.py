"""Thin wrapper around the python-binance Client.

Provides:
- Exponential-backoff retry decorator
- fetch_ohlcv() → UTC-aware DataFrame
- get_current_price() / ping() helpers
- Symbols loaded from TRADING_SYMBOLS env var
"""
from __future__ import annotations

import functools
import os
import time
from datetime import datetime

import pandas as pd
import structlog
from binance.client import Client
from dotenv import load_dotenv

log = structlog.get_logger()

# Column names returned by Binance get_historical_klines
_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_vol",
    "taker_buy_quote_vol", "ignore",
]

_OHLCV_COLS = [
    "symbol", "timestamp", "open", "high", "low", "close",
    "volume", "quote_volume", "trades", "taker_buy_vol",
]


def _retry(max_retries: int = 3, backoff_base: float = 2.0):
    """Decorator: retry on any exception with exponential back-off."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_retries - 1:
                        raise
                    wait = backoff_base ** attempt
                    log.warning(
                        "api_retry",
                        func=func.__name__,
                        attempt=attempt + 1,
                        wait_s=wait,
                        error=str(exc),
                    )
                    time.sleep(wait)

        return wrapper

    return decorator


class BinanceClientWrapper:
    """Binance REST API wrapper with testnet support and structured logging."""

    def __init__(self, testnet: bool | None = None) -> None:
        load_dotenv()
        if testnet is None:
            testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
        if testnet:
            api_key = os.getenv("BINANCE_API_KEY", "")
            api_secret = os.getenv("BINANCE_API_SECRET", "")
        else:
            # Explicit mainnet is data-only: market-data endpoints are
            # public, and the env keys are testnet credentials that must
            # never be sent to the live exchange.
            api_key = ""
            api_secret = ""
        symbols_raw = os.getenv("TRADING_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT")

        self.symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]
        self.client = Client(api_key, api_secret, testnet=testnet)
        self._log = structlog.get_logger().bind(testnet=testnet)
        self._symbol_info_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_symbols(self) -> list[str]:
        return self.symbols

    @_retry(max_retries=3)
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles and return a UTC-aware DataFrame.

        Returns an empty DataFrame (with correct columns) when the API
        returns no data.
        """
        start_ms = (
            str(int(start_date.timestamp() * 1000)) if start_date else None
        )
        end_ms = (
            str(int(end_date.timestamp() * 1000)) if end_date else None
        )

        klines = self.client.get_historical_klines(
            symbol, interval, start_ms, end_ms, limit=limit
        )

        if not klines:
            return pd.DataFrame(columns=_OHLCV_COLS)

        df = pd.DataFrame(klines, columns=_KLINE_COLS)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["symbol"] = symbol

        for col in ("open", "high", "low", "close", "volume",
                    "quote_volume", "taker_buy_vol"):
            df[col] = df[col].astype(float)
        df["trades"] = df["trades"].astype(int)

        return df[_OHLCV_COLS]

    @_retry(max_retries=3)
    def get_current_price(self, symbol: str) -> float:
        """Return the latest mark price for a symbol."""
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker["price"])

    def get_account_balance(self, asset: str = "USDT") -> float:
        """Return the free balance for *asset* from the testnet account.

        Raises on failure — callers that size trades off this number must
        not proceed with a guessed balance.
        """
        balance = self.client.get_asset_balance(asset=asset)
        if not balance:
            raise ValueError(f"no balance entry returned for asset {asset!r}")
        return float(balance["free"])

    def get_symbol_info_cached(self, symbol: str) -> dict:
        """Fetch and cache get_symbol_info(symbol) for the process lifetime.

        Exchange filters (stepSize, tickSize, minNotional) change rarely; one
        fetch per symbol per process avoids hammering /api/v3/exchangeInfo on
        every trade.
        """
        if symbol not in self._symbol_info_cache:
            info = self.client.get_symbol_info(symbol)
            if info is None:
                raise ValueError(f"get_symbol_info returned None for {symbol!r}")
            self._symbol_info_cache[symbol] = info
        return self._symbol_info_cache[symbol]

    def ping(self) -> bool:
        """Return True if the Binance server is reachable, False otherwise."""
        try:
            self.client.ping()
            return True
        except Exception as exc:
            self._log.warning("ping_failed", error=str(exc))
            return False
