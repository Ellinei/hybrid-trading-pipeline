"""Unit tests for backtest.experiment — the `cli experiment` orchestration:
resolve/build a content-addressed prob cache, then write a self-contained
results/<exp_id>/ artifact set (trades, equity curve, metrics, config).

No real DB — cache resolution is tested with mocked fingerprint/precompute
calls; artifact writing uses tmp_path with a fake BacktestResult-shaped
object (real file I/O, so CSV/JSON serialization is genuinely exercised).

Run:
    poetry run pytest tests/test_backtest_experiment.py -v
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backtest.experiment import (
    ExperimentConfig,
    _build_ledger_row,
    _cache_path_for_key,
    _compute_metrics,
    _write_artifacts,
    resolve_prob_cache,
)


@dataclass
class _FakeResult:
    trades: list
    equity_curve: list
    metrics: dict
    warnings: list = field(default_factory=list)
    retrain_events: list = field(default_factory=list)


def _sample_result():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trades = [
        {"symbol": "BTCUSDT", "entry_price": 100.0, "exit_price": 105.0,
         "quantity": 1.0, "entry_time": t0, "exit_time": t0, "pnl": 5.0, "reason": "signal_close"},
        {"symbol": "ETHUSDT", "entry_price": 50.0, "exit_price": 48.0,
         "quantity": 1.0, "entry_time": t0, "exit_time": t0, "pnl": -2.0, "reason": "stop_loss"},
    ]
    return _FakeResult(
        trades=trades,
        equity_curve=[(t0, 10000.0), (t0, 10003.0)],
        metrics={
            "trade_count": 2, "win_rate": 0.5, "profit_factor": 2.5,
            "max_drawdown": 0.01, "sharpe_ratio": 0.3, "total_return": 0.0003,
        },
    )


class TestCachePathForKey:
    def test_derives_stable_short_filename(self):
        path = _cache_path_for_key("abcdef1234567890" * 4)
        assert path.startswith(os.path.join("cache", "probs_"))
        assert path.endswith(".pkl")

    def test_different_keys_different_paths(self):
        assert _cache_path_for_key("a" * 64) != _cache_path_for_key("b" * 64)


class TestBuildLedgerRow:

    def test_row_contains_key_metrics_as_markdown_table_row(self):
        cfg = ExperimentConfig(exp_id="exp008", symbols=["BTCUSDT", "ETHUSDT"])
        metrics = {
            "trade_count": 170, "win_rate": 0.6118, "profit_factor": 1.23,
            "expectancy": 4.44, "max_drawdown": 0.0675, "sharpe_ratio": 0.61,
        }

        row = _build_ledger_row(cfg, metrics, git_sha="abc1234",
                                 data_range="2023-07-05..2026-07-04")

        assert row.startswith("|") and row.rstrip().endswith("|")
        assert "exp008" in row
        assert "abc1234" in row
        assert "170" in row
        assert "1.23" in row
        assert "4.44" in row
        assert "61.2%" in row or "61.18%" in row
        assert "2023-07-05..2026-07-04" in row

    def test_infinite_pf_renders_as_inf_not_a_crash(self):
        cfg = ExperimentConfig(exp_id="exp008")
        metrics = {
            "trade_count": 5, "win_rate": 1.0, "profit_factor": float("inf"),
            "expectancy": 4.44, "max_drawdown": 0.0, "sharpe_ratio": 0.61,
        }

        row = _build_ledger_row(cfg, metrics, git_sha="abc1234", data_range="—")

        assert "inf" in row


class TestWriteArtifacts:

    def test_writes_all_expected_files(self, tmp_path):
        cfg = ExperimentConfig(exp_id="exp999", symbols=["BTCUSDT", "ETHUSDT"],
                                output_dir=str(tmp_path))
        result = _sample_result()

        out_dir = _write_artifacts(cfg, result, cache_path="cache/probs_deadbeef.pkl")

        assert os.path.isdir(out_dir)
        for name in ("trades.csv", "equity_curve.csv", "metrics.json",
                     "config.json", "report.html", "ledger_row.md"):
            assert os.path.exists(os.path.join(out_dir, name)), name

    def test_trades_csv_matches_trade_rows(self, tmp_path):
        cfg = ExperimentConfig(exp_id="exp999", output_dir=str(tmp_path))
        result = _sample_result()

        out_dir = _write_artifacts(cfg, result, cache_path="x.pkl")

        df = pd.read_csv(os.path.join(out_dir, "trades.csv"))
        assert len(df) == 2
        assert set(df["symbol"]) == {"BTCUSDT", "ETHUSDT"}

    def test_equity_curve_csv_has_timestamp_and_equity_columns(self, tmp_path):
        cfg = ExperimentConfig(exp_id="exp999", output_dir=str(tmp_path))
        result = _sample_result()

        out_dir = _write_artifacts(cfg, result, cache_path="x.pkl")

        df = pd.read_csv(os.path.join(out_dir, "equity_curve.csv"))
        assert list(df.columns) == ["timestamp", "equity"]
        assert len(df) == 2

    def test_metrics_json_includes_per_symbol_and_concentration_check(self, tmp_path):
        cfg = ExperimentConfig(exp_id="exp999", output_dir=str(tmp_path))
        result = _sample_result()

        out_dir = _write_artifacts(cfg, result, cache_path="x.pkl")

        with open(os.path.join(out_dir, "metrics.json")) as f:
            metrics = json.load(f)
        assert metrics["trade_count"] == 2
        assert "BTCUSDT" in metrics["per_symbol"]
        assert "max_single_trade_profit_share" in metrics

    def test_config_json_captures_resolved_config_and_cache_path(self, tmp_path):
        cfg = ExperimentConfig(exp_id="exp999", ml_buy_threshold=0.55,
                                max_hold_bars=24, output_dir=str(tmp_path))
        result = _sample_result()

        with patch("backtest.experiment._git_sha", return_value="deadbee (uncommitted WT)"):
            out_dir = _write_artifacts(cfg, result, cache_path="cache/probs_abc.pkl")

        with open(os.path.join(out_dir, "config.json")) as f:
            config = json.load(f)
        assert config["exp_id"] == "exp999"
        assert config["ml_buy_threshold"] == 0.55
        assert config["max_hold_bars"] == 24
        assert config["cache_path"] == "cache/probs_abc.pkl"
        assert config["git_sha"] == "deadbee (uncommitted WT)"

    def test_no_trades_still_writes_empty_artifacts(self, tmp_path):
        cfg = ExperimentConfig(exp_id="exp999", output_dir=str(tmp_path))
        result = _FakeResult(trades=[], equity_curve=[], metrics={
            "trade_count": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "max_drawdown": 0.0, "sharpe_ratio": 0.0, "total_return": 0.0,
        })

        out_dir = _write_artifacts(cfg, result, cache_path="x.pkl")

        assert not os.path.exists(os.path.join(out_dir, "trades.csv"))
        with open(os.path.join(out_dir, "metrics.json")) as f:
            metrics = json.load(f)
        assert metrics["trade_count"] == 0


class TestComputeMetrics:
    def test_includes_all_expected_keys(self):
        result = _sample_result()
        metrics = _compute_metrics(result)

        for key in ("trade_count", "win_rate", "profit_factor", "expectancy",
                    "max_drawdown", "sharpe_ratio", "total_return",
                    "exit_reason_breakdown", "holding_hours", "per_symbol",
                    "max_single_trade_profit_share"):
            assert key in metrics


class TestResolvePreCache:

    def _feature_df(self):
        return pd.DataFrame({
            "symbol": ["BTCUSDT"], "timestamp": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            "close": [100.0], "future_close": [101.0], "target": [0],
        })

    def test_reuses_existing_matching_cache_without_rebuilding(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        os.makedirs("cache", exist_ok=True)
        cfg = ExperimentConfig(exp_id="exp999", symbols=["BTCUSDT"])

        with patch("backtest.experiment._feature_fingerprint", return_value={"BTCUSDT": (1, "t")}), \
             patch("backtest.experiment._compute_cache_key", return_value="fixedkey" * 8):
            cache_path = _cache_path_for_key("fixedkey" * 8)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({"prob_map": {"cached": True}, "feature_df": self._feature_df()}, f)

            with patch("backtest.experiment.precompute_walk_forward_probs") as mock_precompute:
                prob_map, feature_df, resolved_path = resolve_prob_cache(
                    MagicMock(), cfg, bars_by_symbol={},
                )

        mock_precompute.assert_not_called()
        assert prob_map == {"cached": True}
        assert resolved_path == cache_path

    def test_builds_and_saves_when_no_cache_exists(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = ExperimentConfig(exp_id="exp999", symbols=["BTCUSDT"])

        with patch("backtest.experiment._feature_fingerprint", return_value={"BTCUSDT": (1, "t")}), \
             patch("backtest.experiment._compute_cache_key", return_value="newkey1" * 8), \
             patch("backtest.experiment._load_all_features", return_value=self._feature_df()), \
             patch("backtest.experiment.precompute_walk_forward_probs",
                   return_value={"built": True}) as mock_precompute, \
             patch("backtest.experiment._build_cache_meta", return_value={"cache_key": "newkey1" * 8}):
            prob_map, feature_df, resolved_path = resolve_prob_cache(
                MagicMock(), cfg, bars_by_symbol={},
            )

        mock_precompute.assert_called_once()
        assert prob_map == {"built": True}
        assert os.path.exists(resolved_path)
