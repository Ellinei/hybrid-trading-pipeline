"""Unit tests for config/strategy.py — the Phase 13 validated-strategy
single source of truth (exp008: exp007r base + Phase 12 R1 regime gate).

Run:
    poetry run pytest tests/test_config_strategy.py -v
"""
from __future__ import annotations

from config.strategy import StrategyConfig


class TestStrategyConfigDefaults:
    """Defaults must match exp008 — the adopted strategy in
    docs/experiments.md — not the class defaults scattered across
    RiskManager/MLSignalGenerator/SignalAggregator."""

    def test_defaults_match_exp008(self):
        cfg = StrategyConfig()

        assert cfg.weights == {"technical": 0.40, "ml": 0.40, "sentiment": 0.20}
        assert cfg.buy_gate == 0.15
        assert cfg.sell_gate == -0.15
        assert cfg.ml_buy_threshold == 0.55
        assert cfg.ml_sell_threshold == 0.35
        assert cfg.max_hold_bars == 24
        assert cfg.regime_candidate == "r1_sma200"
        assert cfg.sentiment_enabled is False

    def test_symbols_default_matches_bot_runner_fallback(self):
        cfg = StrategyConfig()
        assert cfg.symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    def test_fields_are_overridable(self):
        cfg = StrategyConfig(ml_buy_threshold=0.65, regime_candidate=None)
        assert cfg.ml_buy_threshold == 0.65
        assert cfg.regime_candidate is None
