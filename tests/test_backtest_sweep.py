"""Tests for backtest.sweep — grid generation and result ranking.

No DB or network: verifies the combinatorial logic and table-formatting in
isolation from the data loading and engine.
"""
from __future__ import annotations

from datetime import UTC, datetime

from backtest.exit_simulation import OHLCVBar
from backtest.sweep import PARAM_GRID, _CachingAggregator, _grid_configs, _row
from execution.risk_manager import RiskConfig
from signals.base import AggregatedSignal, Signal, SignalDirection


class TestGridConfigs:
    def test_returns_correct_number_of_combinations(self):
        expected = 1
        for values in PARAM_GRID.values():
            expected *= len(values)
        assert len(_grid_configs()) == expected

    def test_all_values_appear_in_grid(self):
        configs = _grid_configs()
        for key, values in PARAM_GRID.items():
            seen = {getattr(c, key) for c in configs}
            assert seen == set(values), f"{key}: expected {set(values)}, got {seen}"

    def test_each_config_is_risk_config_instance(self):
        for cfg in _grid_configs():
            assert isinstance(cfg, RiskConfig)

    def test_no_duplicate_configs(self):
        configs = _grid_configs()
        # represent as frozenset of items for dedup check
        seen = set()
        for cfg in configs:
            key = (
                cfg.stop_loss_atr_mult, cfg.take_profit_atr_mult,
                cfg.min_confidence, cfg.max_position_pct,
            )
            assert key not in seen, f"Duplicate config: {key}"
            seen.add(key)


class TestRowBuilding:
    def _make_metrics(self, pf=0.5, ret=-0.1, sharpe=-2.0, wr=0.3, dd=0.1, tc=10):
        return {
            "profit_factor": pf, "total_return": ret, "sharpe_ratio": sharpe,
            "win_rate": wr, "max_drawdown": dd, "trade_count": tc,
        }

    def test_row_keys_match_expected_columns(self):
        rc = RiskConfig()
        row = _row(1, rc, self._make_metrics())
        expected_keys = {
            "rank", "profit_factor", "total_return", "sharpe_ratio",
            "win_rate", "max_drawdown", "trade_count",
            "stop_loss_atr_mult", "take_profit_atr_mult",
            "min_confidence", "max_position_pct",
        }
        assert set(row.keys()) == expected_keys

    def test_row_copies_risk_config_fields(self):
        rc = RiskConfig(
            stop_loss_atr_mult=2.5, take_profit_atr_mult=5.0,
            min_confidence=0.70, max_position_pct=0.05,
        )
        row = _row(3, rc, self._make_metrics())
        assert row["stop_loss_atr_mult"]   == 2.5
        assert row["take_profit_atr_mult"] == 5.0
        assert row["min_confidence"]       == 0.70
        assert row["max_position_pct"]     == 0.05
        assert row["rank"] == 3

    def test_row_rounds_float_metrics(self):
        rc = RiskConfig()
        row = _row(1, rc, self._make_metrics(pf=1.23456789))
        assert row["profit_factor"] == round(1.23456789, 4)


class TestSweepRanking:
    """Validate that sorting rows by (profit_factor desc, sharpe desc) works."""

    def test_higher_profit_factor_ranks_first(self):
        rc = RiskConfig()
        rows = [
            _row(1, rc, {"profit_factor": 0.5, "total_return": -0.1, "sharpe_ratio": -1.0,
                         "win_rate": 0.3, "max_drawdown": 0.1, "trade_count": 5}),
            _row(2, rc, {"profit_factor": 1.5, "total_return": 0.2, "sharpe_ratio": 1.0,
                         "win_rate": 0.6, "max_drawdown": 0.05, "trade_count": 8}),
            _row(3, rc, {"profit_factor": 0.8, "total_return": -0.05, "sharpe_ratio": 0.1,
                         "win_rate": 0.4, "max_drawdown": 0.08, "trade_count": 6}),
        ]
        rows.sort(key=lambda r: (r["profit_factor"], r["sharpe_ratio"]), reverse=True)
        assert rows[0]["profit_factor"] == 1.5
        assert rows[1]["profit_factor"] == 0.8
        assert rows[2]["profit_factor"] == 0.5

    def test_sharpe_breaks_ties_on_equal_profit_factor(self):
        rc = RiskConfig()
        rows = [
            _row(1, rc, {"profit_factor": 1.0, "total_return": 0.0, "sharpe_ratio": -1.0,
                         "win_rate": 0.5, "max_drawdown": 0.0, "trade_count": 2}),
            _row(2, rc, {"profit_factor": 1.0, "total_return": 0.0, "sharpe_ratio": 2.0,
                         "win_rate": 0.5, "max_drawdown": 0.0, "trade_count": 2}),
        ]
        rows.sort(key=lambda r: (r["profit_factor"], r["sharpe_ratio"]), reverse=True)
        assert rows[0]["sharpe_ratio"] == 2.0


class TestCachingAggregator:
    """Verify that _CachingAggregator calls the real aggregator at most once
    per (symbol, timestamp) and returns identical results on repeat calls."""

    def _make_signal(self, symbol: str, ts: datetime) -> AggregatedSignal:
        s = Signal(SignalDirection.BUY, 0.8, "technical", symbol, ts)
        return AggregatedSignal(
            direction=SignalDirection.BUY, confidence=0.8,
            symbol=symbol, timestamp=ts, signals=[s],
            reasoning="test", weights_used={},
        )

    def test_populate_calls_real_aggregator_once_per_key(self):
        from datetime import timedelta
        from unittest.mock import MagicMock

        ts1 = datetime(2026, 1, 1, tzinfo=UTC)
        ts2 = ts1 + timedelta(hours=1)
        bars = {
            "BTCUSDT": [
                OHLCVBar(timestamp=ts1, open=100, high=101, low=99, close=100),
                OHLCVBar(timestamp=ts2, open=101, high=102, low=100, close=101),
            ]
        }

        real = MagicMock()
        real.aggregate.side_effect = lambda sym, as_of: self._make_signal(sym, as_of)

        ca = _CachingAggregator(real)
        ca.populate(bars)

        assert real.aggregate.call_count == 2

    def test_aggregate_returns_cached_result_without_calling_real(self):
        from unittest.mock import MagicMock

        ts = datetime(2026, 1, 1, tzinfo=UTC)
        expected_signal = self._make_signal("BTCUSDT", ts)

        real = MagicMock()
        real.aggregate.return_value = expected_signal

        ca = _CachingAggregator(real)
        # prime the cache
        ca._cache[("BTCUSDT", ts)] = expected_signal

        result = ca.aggregate("BTCUSDT", as_of=ts)
        assert result is expected_signal
        real.aggregate.assert_not_called()

    def test_fallback_when_key_not_cached(self):
        from unittest.mock import MagicMock

        ts = datetime(2026, 6, 1, tzinfo=UTC)
        expected = self._make_signal("ETHUSDT", ts)

        real = MagicMock()
        real.aggregate.return_value = expected

        ca = _CachingAggregator(real)
        result = ca.aggregate("ETHUSDT", as_of=ts)

        assert result is expected
        real.aggregate.assert_called_once_with("ETHUSDT", as_of=ts)
