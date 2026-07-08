"""Local parquet snapshots of raw.ohlcv and features.feat_technical.

Every backtest run re-queried Postgres for the same multi-year dataset
(four near-duplicate loaders across run_backtest.py/engine.py/
signal_cache.py each did their own SELECT). This module is the shared
DB-access layer underneath all of them: a snapshot is read from disk when
its per-symbol fingerprint (row count + max timestamp) still matches the
live table, and rebuilt automatically the moment it doesn't — so repeated
experiment runs skip the DB entirely once warm, without ever silently
serving stale data.

Decimal->float coercion and UTC tz-localization happen here, once, before
anything is written to disk or handed to a caller — the ReplayAggregator's
NaN-vs-NULL parity (pinned by tests/test_backtest_signal_cache.py) depends
on every caller seeing the same float/NaN shape regardless of whether the
data came from a fresh query or a cached snapshot.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Callable

import pandas as pd
from sqlalchemy import text

SNAPSHOT_DIR = "cache/snapshots"


def _fingerprint_symbol(engine, table: str, symbol: str) -> tuple[int, str | None]:
    """Cheap (row_count, max_timestamp) aggregate — the staleness signal,
    far cheaper than re-reading the whole table to check if it changed."""
    query = text(f"SELECT COUNT(*), MAX(timestamp) FROM {table} WHERE symbol = :symbol")
    with engine.connect() as conn:
        count, max_ts = conn.execute(query, {"symbol": symbol}).fetchone()
    return (count, str(max_ts) if max_ts is not None else None)


def _snapshot_paths(name: str) -> tuple[str, str]:
    return (
        os.path.join(SNAPSHOT_DIR, f"{name}.parquet"),
        os.path.join(SNAPSHOT_DIR, f"{name}.meta.json"),
    )


def _load_or_refresh(
    name: str,
    symbol: str,
    query_fn: Callable[[], pd.DataFrame],
    fingerprint_fn: Callable[[], tuple[int, str | None]],
) -> pd.DataFrame:
    """Read the *name* snapshot from disk if its stored fingerprint matches
    *fingerprint_fn()*; otherwise call *query_fn()*, cache the result, and
    return it. *query_fn*/*fingerprint_fn* take no arguments — callers bind
    engine/symbol via closures/partials so this function has no DB
    dependency of its own and is trivially testable.
    """
    data_path, meta_path = _snapshot_paths(name)
    current_fp = fingerprint_fn()

    if os.path.exists(data_path) and os.path.exists(meta_path):
        with open(meta_path) as f:
            cached_fp = json.load(f).get("fingerprint")
        if cached_fp == list(current_fp):
            return pd.read_parquet(data_path)

    df = query_fn()
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    df.to_parquet(data_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(
            {"symbol": symbol, "fingerprint": list(current_fp),
             "built_at": datetime.now(timezone.utc).isoformat()},
            f,
        )
    return df


def _query_ohlcv(engine, symbol: str) -> pd.DataFrame:
    query = text(
        "SELECT timestamp, open, high, low, close FROM raw.ohlcv"
        " WHERE symbol = :symbol ORDER BY timestamp ASC"
    )
    with engine.connect() as conn:
        result = conn.execute(query, {"symbol": symbol})
        df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
    if df.empty:
        return df
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df


def _query_features(engine, symbol: str) -> pd.DataFrame:
    from signals.ml_model import MLSignalGenerator

    feat_cols = MLSignalGenerator.FEATURES
    select_cols = ", ".join(["symbol", "timestamp", "close"] + feat_cols)
    query = text(
        f"SELECT {select_cols} FROM features.feat_technical"
        " WHERE symbol = :symbol ORDER BY timestamp ASC"
    )
    with engine.connect() as conn:
        result = conn.execute(query, {"symbol": symbol})
        df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
    if df.empty:
        return df
    for col in feat_cols + ["close"]:
        df[col] = df[col].astype(float)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df


def load_ohlcv(engine, symbol: str) -> pd.DataFrame:
    """All raw.ohlcv rows for one symbol (timestamp, open, high, low,
    close; floats, UTC-aware timestamps), from a local parquet snapshot
    when fresh, else a live query that rebuilds the snapshot.
    """
    return _load_or_refresh(
        f"ohlcv_{symbol}", symbol,
        query_fn=lambda: _query_ohlcv(engine, symbol),
        fingerprint_fn=lambda: _fingerprint_symbol(engine, "raw.ohlcv", symbol),
    )


def load_features(engine, symbol: str) -> pd.DataFrame:
    """All features.feat_technical rows for one symbol (symbol, timestamp,
    close, + the 18 ML feature columns; floats, UTC-aware timestamps),
    from a local parquet snapshot when fresh, else a live query that
    rebuilds the snapshot. Does NOT compute target/future_close — callers
    add that themselves, since whether trailing NaN-label rows should be
    dropped varies by caller (engine._load_all_features drops them,
    signal_cache.load_feature_matrix keeps them for near-end-of-history
    signal generation).
    """
    return _load_or_refresh(
        f"features_{symbol}", symbol,
        query_fn=lambda: _query_features(engine, symbol),
        fingerprint_fn=lambda: _fingerprint_symbol(engine, "features.feat_technical", symbol),
    )
