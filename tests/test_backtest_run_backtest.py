"""Unit tests for backtest.run_backtest's cache-key mechanism.

The content-addressed cache key replaces a soft label-only warning with a
hard mismatch check: a --load-cache run recomputes the key from current
CLI args + a live feature-data fingerprint and refuses to silently trade
on a stale/mismatched cache. No real DB — the fingerprint fetch is a thin
query wrapper tested separately with a mocked engine.

Run:
    poetry run pytest tests/test_backtest_run_backtest.py -v
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from backtest.run_backtest import (
    PROB_CACHE_ALGORITHM_VERSION,
    _compute_cache_key,
    _feature_fingerprint,
    _load_atr,
    _load_bars,
    _validate_cache_key,
)


def _key_kwargs(**overrides):
    base = dict(
        label_horizon=8,
        label_min_return=0.003,
        min_train_days=60,
        walk_forward_months=1,
        train_window_days=540,
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        fingerprint={"BTCUSDT": (26280, "2026-07-04T00:00:00+00:00"),
                     "ETHUSDT": (26280, "2026-07-04T00:00:00+00:00"),
                     "SOLUSDT": (26280, "2026-07-04T00:00:00+00:00")},
    )
    base.update(overrides)
    return base


class TestComputeCacheKey:

    def test_deterministic_for_same_inputs(self):
        a = _compute_cache_key(**_key_kwargs())
        b = _compute_cache_key(**_key_kwargs())
        assert a == b

    def test_symbol_order_does_not_matter(self):
        a = _compute_cache_key(**_key_kwargs(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"]))
        b = _compute_cache_key(**_key_kwargs(symbols=["SOLUSDT", "BTCUSDT", "ETHUSDT"]))
        assert a == b

    def test_differs_when_label_horizon_changes(self):
        a = _compute_cache_key(**_key_kwargs(label_horizon=8))
        b = _compute_cache_key(**_key_kwargs(label_horizon=1))
        assert a != b

    def test_differs_when_symbol_set_changes(self):
        a = _compute_cache_key(**_key_kwargs(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"]))
        b = _compute_cache_key(**_key_kwargs(symbols=["BTCUSDT", "ETHUSDT"]))
        assert a != b

    def test_differs_when_fingerprint_changes(self):
        """New bars landing since the cache was built (row count grows) must
        change the key — this is the actual staleness signal."""
        a = _compute_cache_key(**_key_kwargs())
        b = _compute_cache_key(**_key_kwargs(fingerprint={
            "BTCUSDT": (26281, "2026-07-04T01:00:00+00:00"),
            "ETHUSDT": (26280, "2026-07-04T00:00:00+00:00"),
            "SOLUSDT": (26280, "2026-07-04T00:00:00+00:00"),
        }))
        assert a != b

    def test_does_not_take_frozen_params(self):
        """frozen_params is an OUTPUT of building the cache (the tuned
        hyperparameters), not an input that determines what the cache
        should contain — it must not be a parameter of the key function."""
        import inspect

        sig = inspect.signature(_compute_cache_key)
        assert "frozen_params" not in sig.parameters


class TestFeatureFingerprint:

    def test_queries_count_and_max_timestamp_per_symbol(self):
        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchone.return_value = (
            26280, "2026-07-04T00:00:00+00:00", "digest",
        )

        result = _feature_fingerprint(engine, ["BTCUSDT", "ETHUSDT"])

        assert result == {
            "BTCUSDT": (26280, "2026-07-04T00:00:00+00:00", "digest"),
            "ETHUSDT": (26280, "2026-07-04T00:00:00+00:00", "digest"),
        }
        assert conn.execute.call_count == 2
        sql = str(conn.execute.call_args_list[0][0][0])
        assert "count" in sql.lower()
        assert "max" in sql.lower()

    def test_missing_symbol_data_returns_zero_count(self):
        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchone.return_value = (0, None, "empty")

        result = _feature_fingerprint(engine, ["BTCUSDT"])

        assert result == {"BTCUSDT": (0, None, "empty")}


class TestLoadBars:
    """_load_bars now delegates to data_store.load_ohlcv (a snapshot-backed
    full-history load) and applies start/end filtering itself in pandas,
    instead of pushing the date range into SQL."""

    def _ohlcv_df(self):
        return pd.DataFrame({
            "timestamp": [
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 1, 2, tzinfo=UTC),
                datetime(2026, 1, 3, tzinfo=UTC),
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
        })

    def test_no_filter_returns_all_bars(self):
        with patch("backtest.run_backtest.data_store.load_ohlcv", return_value=self._ohlcv_df()):
            bars = _load_bars(MagicMock(), "BTCUSDT", None, None)

        assert len(bars) == 3
        assert bars[0].close == 100.5

    def test_start_end_filter_bars_in_range(self):
        with patch("backtest.run_backtest.data_store.load_ohlcv", return_value=self._ohlcv_df()):
            bars = _load_bars(
                MagicMock(), "BTCUSDT",
                datetime(2026, 1, 2, tzinfo=UTC),
                datetime(2026, 1, 2, tzinfo=UTC),
            )

        assert len(bars) == 1
        assert bars[0].close == 101.5

    def test_empty_snapshot_returns_empty_list(self):
        with patch("backtest.run_backtest.data_store.load_ohlcv", return_value=pd.DataFrame()):
            bars = _load_bars(MagicMock(), "BTCUSDT", None, None)

        assert bars == []


class TestLoadAtr:

    def _features_df(self):
        return pd.DataFrame({
            "timestamp": [
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 1, 2, tzinfo=UTC),
                datetime(2026, 1, 3, tzinfo=UTC),
            ],
            "range_14": [1.5, float("nan"), 2.5],
        })

    def test_drops_nan_range_14(self):
        with patch("backtest.run_backtest.data_store.load_features",
                   return_value=self._features_df()):
            atr = _load_atr(MagicMock(), "BTCUSDT", None, None)

        assert len(atr) == 2
        assert datetime(2026, 1, 2, tzinfo=UTC) not in atr

    def test_start_end_filter(self):
        with patch("backtest.run_backtest.data_store.load_features",
                   return_value=self._features_df()):
            atr = _load_atr(
                MagicMock(), "BTCUSDT",
                datetime(2026, 1, 3, tzinfo=UTC),
                datetime(2026, 1, 3, tzinfo=UTC),
            )

        assert atr == {datetime(2026, 1, 3, tzinfo=UTC): 2.5}


class TestValidateCacheKey:
    """Intentional tightening (Phase 13 review fix): a cache with no
    cache_key, no algorithm_version, or a stale algorithm_version used to
    proceed with only a soft warning — that let legacy pre-fix caches
    (built before the label-availability and causal-CV fixes) silently
    produce results that look like a fresh, correct run. All three cases
    are now hard refusals unless --force is passed."""

    def test_matching_key_and_current_version_proceeds_silently(self):
        ok, message = _validate_cache_key(
            meta={"cache_key": "abc123", "algorithm_version": PROB_CACHE_ALGORITHM_VERSION},
            expected_key="abc123", force=False,
        )
        assert ok is True
        assert message == ""

    def test_no_cache_key_is_refused(self):
        """A cache built before the content-key mechanism existed has no
        cache_key at all — this now hard-fails instead of warning-and-
        proceeding."""
        ok, message = _validate_cache_key(
            meta={}, expected_key="abc123", force=False,
        )
        assert ok is False
        assert "no content key" in message.lower()

    def test_no_cache_key_with_force_proceeds_with_warning(self):
        ok, message = _validate_cache_key(
            meta={}, expected_key="abc123", force=True,
        )
        assert ok is True
        assert "no content key" in message.lower()
        assert "not comparable" in message.lower()

    def test_key_without_algorithm_version_is_refused(self):
        """A cache with a cache_key but no algorithm_version predates the
        label-availability and CV fixes even though it has a content key —
        must not be silently trusted just because the key happens to
        recompute to a match."""
        ok, message = _validate_cache_key(
            meta={"cache_key": "abc123"}, expected_key="abc123", force=False,
        )
        assert ok is False
        assert "algorithm_version" in message.lower()

    def test_key_without_algorithm_version_with_force_proceeds_with_warning(self):
        ok, message = _validate_cache_key(
            meta={"cache_key": "abc123"}, expected_key="abc123", force=True,
        )
        assert ok is True
        assert "algorithm_version" in message.lower()
        assert "not comparable" in message.lower()

    def test_old_algorithm_version_is_refused(self):
        """A cache built under a since-superseded algorithm version (e.g.
        v2, before the causal-CV fix) must be refused even if its content
        key otherwise matches."""
        ok, message = _validate_cache_key(
            meta={"cache_key": "abc123", "algorithm_version": PROB_CACHE_ALGORITHM_VERSION - 1},
            expected_key="abc123", force=False,
        )
        assert ok is False
        assert "algorithm_version mismatch" in message.lower()

    def test_old_algorithm_version_with_force_proceeds_with_warning(self):
        ok, message = _validate_cache_key(
            meta={"cache_key": "abc123", "algorithm_version": PROB_CACHE_ALGORITHM_VERSION - 1},
            expected_key="abc123", force=True,
        )
        assert ok is True
        assert "algorithm_version mismatch" in message.lower()
        assert "not comparable" in message.lower()

    def test_mismatched_key_without_force_blocks(self):
        ok, message = _validate_cache_key(
            meta={"cache_key": "abc123", "algorithm_version": PROB_CACHE_ALGORITHM_VERSION},
            expected_key="different", force=False,
        )
        assert ok is False
        assert "key mismatch" in message.lower()

    def test_mismatched_key_with_force_proceeds_with_warning(self):
        ok, message = _validate_cache_key(
            meta={"cache_key": "abc123", "algorithm_version": PROB_CACHE_ALGORITHM_VERSION},
            expected_key="different", force=True,
        )
        assert ok is True
        assert "key mismatch" in message.lower()
