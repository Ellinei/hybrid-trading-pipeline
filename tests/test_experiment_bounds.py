"""Evaluation bounds apply to scoring, while cache and regime retain history."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backtest.exit_simulation import OHLCVBar
from backtest.experiment import ExperimentConfig, run_experiment
from backtest.signal_cache import precompute_walk_forward_probs
from signals.regime import RegimeState


UTC = timezone.utc


def _at(hour: int) -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=hour)


def _bar(hour: int) -> OHLCVBar:
    return OHLCVBar(timestamp=_at(hour), open=100, high=101, low=99, close=100)


def _empty_result():
    return SimpleNamespace(
        trades=[], equity_curve=[],
        metrics={"trade_count": 0, "win_rate": 0.0, "profit_factor": 0.0,
                 "max_drawdown": 0.0, "sharpe_ratio": 0.0, "total_return": 0.0},
    )


@pytest.mark.parametrize("start,end", [
    (_at(2), _at(2)),
    (_at(3), _at(2)),
    (_at(0).replace(tzinfo=None), _at(2)),
    (_at(0), _at(2).replace(tzinfo=None)),
    (_at(0), None),
])
def test_config_rejects_invalid_window(start, end):
    with pytest.raises(ValueError):
        ExperimentConfig(exp_id="bounded", start=start, end=end)


def test_replay_is_start_inclusive_end_exclusive_and_warmup_is_preserved(tmp_path):
    history = [_bar(hour) for hour in range(5)]
    cfg = ExperimentConfig(exp_id="bounded", symbols=["BTCUSDT"],
                           start=_at(2), end=_at(4), output_dir=str(tmp_path))
    engine = MagicMock()

    with patch("backtest.experiment.BinanceClientWrapper"), \
         patch("backtest.experiment._load_bars", return_value=history), \
         patch("backtest.experiment._load_atr", return_value={}), \
         patch("backtest.experiment.resolve_prob_cache", return_value=({}, None, "cache.pkl")) as cache, \
         patch("backtest.experiment.run_backtest", return_value=_empty_result()) as replay:
        metrics = run_experiment(engine, cfg)

    assert [bar.timestamp for bar in cache.call_args.args[2]["BTCUSDT"]] == [
        _at(0), _at(1), _at(2), _at(3), _at(4)]
    assert [bar.timestamp for bar in replay.call_args.kwargs["bars_by_symbol"]["BTCUSDT"]] == [
        _at(2), _at(3)]
    assert replay.call_args.kwargs["config"].start == _at(2)
    assert replay.call_args.kwargs["config"].end == _at(4)
    assert metrics["trade_count"] == 0
    assert not (tmp_path / "bounded" / "trades.csv").exists()
    assert json.loads((tmp_path / "bounded" / "config.json").read_text())["start"] == "2026-01-01T02:00:00+00:00"
    assert json.loads((tmp_path / "bounded" / "config.json").read_text())["end"] == "2026-01-01T04:00:00+00:00"
    ledger = (tmp_path / "bounded" / "ledger_row.md").read_text()
    assert "2026-01-01T02:00:00+00:00..2026-01-01T04:00:00+00:00" in ledger


def test_empty_evaluation_window_produces_no_replay_bars_or_trades(tmp_path):
    cfg = ExperimentConfig(exp_id="empty", symbols=["BTCUSDT"],
                           start=_at(6), end=_at(8), output_dir=str(tmp_path))
    with patch("backtest.experiment.BinanceClientWrapper"), \
         patch("backtest.experiment._load_bars", return_value=[_bar(0), _bar(1)]), \
         patch("backtest.experiment._load_atr", return_value={}), \
         patch("backtest.experiment.resolve_prob_cache", return_value=({}, None, "cache.pkl")), \
         patch("backtest.experiment.run_backtest", return_value=_empty_result()) as replay:
        metrics = run_experiment(MagicMock(), cfg)

    assert replay.call_args.kwargs["bars_by_symbol"] == {"BTCUSDT": []}
    assert metrics["trade_count"] == 0


def test_regime_block_rate_scores_only_days_with_replayed_bars(tmp_path):
    cfg = ExperimentConfig(exp_id="regime", symbols=["BTCUSDT"],
                           start=_at(24), end=_at(48), output_dir=str(tmp_path),
                           regime_candidate="r1_sma200")
    regime = {_at(0).date(): RegimeState.RISK_OFF,
              _at(24).date(): RegimeState.RISK_ON}
    with patch("backtest.experiment.BinanceClientWrapper"), \
         patch("backtest.experiment._load_bars", return_value=[_bar(0), _bar(24)]), \
         patch("backtest.experiment._load_atr", return_value={}), \
         patch("backtest.experiment.resolve_prob_cache", return_value=({}, None, "cache.pkl")), \
         patch("backtest.experiment._resolve_regime", return_value=regime), \
         patch("backtest.experiment.run_backtest", return_value=_empty_result()):
        metrics = run_experiment(MagicMock(), cfg)

    assert metrics["regime_days_blocked_pct"] == 0.0
    artifact = json.loads((tmp_path / "regime" / "metrics.json").read_text())
    assert artifact["regime_days_blocked_pct"] == 0.0


def test_utc_window_normalizes_equivalent_offset():
    plus_two = timezone(timedelta(hours=2))
    cfg = ExperimentConfig(
        exp_id="offset", start=datetime(2026, 1, 1, 2, tzinfo=plus_two),
        end=datetime(2026, 1, 1, 4, tzinfo=plus_two),
    )
    assert cfg.start == _at(0)
    assert cfg.end == _at(2)
    assert cfg.start.tzinfo == UTC


def test_future_feature_rows_do_not_change_earlier_cached_prediction(monkeypatch):
    """Full-history cache reuse must preserve the prefix of walk-forward output."""
    class FakeML:
        FEATURES = ["feature"]
        LABEL_HORIZON_BARS = 1

        def __init__(self, db_engine):
            self.scaler = SimpleNamespace(transform=lambda x: x)
            self.model = None

        def train_walk_forward(self, frame, frozen_params):
            assert frame["timestamp"].max() < pd.Timestamp(_at(24))
            probability = float(frame["target"].mean())
            self.model = SimpleNamespace(
                predict_proba=lambda x: np.tile([1 - probability, probability], (len(x), 1)))
            return True, {"frozen": True}

    monkeypatch.setattr("backtest.signal_cache.MLSignalGenerator", FakeML)
    times = [_at(24 * d) for d in range(5)]
    base = pd.DataFrame({
        "symbol": ["BTCUSDT"] * 5, "timestamp": times,
        "feature": [1.0, 2.0, 3.0, 4.0, 5.0],
        "future_close": [101.0] * 5, "target": [1, 1, 0, 0, 0],
    })
    altered = base.copy()
    altered.loc[altered["timestamp"] >= pd.Timestamp(_at(24 * 3)),
                ["feature", "future_close", "target"]] = [999.0, 999.0, 1]
    stream = [(timestamp, "BTCUSDT") for timestamp in times]

    # Each label is the next daily row, revealed 24h later. The first retrain
    # is on day 2, when only day 0's label has been revealed.
    before = precompute_walk_forward_probs(
        base, stream, min_train_days=2, walk_forward_months=1)
    after = precompute_walk_forward_probs(
        altered, stream, min_train_days=2, walk_forward_months=1)

    assert before[("BTCUSDT", _at(24 * 2))] == after[("BTCUSDT", _at(24 * 2))]
