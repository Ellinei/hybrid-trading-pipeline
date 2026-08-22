"""Live strategy configuration — the validated exp008 knobs in one place.

Phase 13's config-wiring half. exp008 (docs/experiments.md) is the
adopted strategy (exp007r base + Phase 12 R1 regime gate) — before this,
execution/bot_runner.py constructed RiskManager/SignalAggregator/
MLSignalGenerator with every field at its class default (ml_buy_threshold
0.65, max_hold_bars disabled, no regime gate, live VADER sentiment),
none of which matches what was actually backtested. This dataclass is
the single source of truth bot_runner builds its components from, and
prints at startup so a human can see exactly what's live before it
trades.

Mirrors backtest.experiment.ExperimentConfig's decision knobs, not its
backtest-only scheduling fields (train_window_days/walk_forward_months/
min_train_days govern how the model is *trained*, which stays a separate
`cli train` step — Phase 13 doesn't touch training). Adds the two fields
ExperimentConfig has no reason to hold: sentiment_enabled and
max_hold_bars (RiskConfig's live time-stop knob, not a replay/FusionParams
field).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StrategyConfig:
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    weights: dict = field(default_factory=lambda: {"technical": 0.40, "ml": 0.40, "sentiment": 0.20})
    buy_gate: float = 0.15
    sell_gate: float = -0.15
    ml_buy_threshold: float = 0.55
    ml_sell_threshold: float = 0.35
    max_hold_bars: int = 24
    regime_candidate: str | None = "r1_sma200"
    sentiment_enabled: bool = False
