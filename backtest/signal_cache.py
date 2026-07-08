"""Component-level signal cache for fast sweep replays.

The v1 sweep (`_CachingAggregator`) caches FUSED AggregatedSignals, so any
change to fusion weights, gates, rule toggles, or ML thresholds invalidates
the whole cache — and walk-forward runs can't be cached at all. This module
caches one level lower:

- the feature matrix (one DB read via ``load_feature_matrix``),
- per-bar walk-forward ML probabilities (one train+predict pass per label
  config via ``precompute_walk_forward_probs``),

and fuses them on demand through ``ReplayAggregator`` — pure in-memory
arithmetic, so sweeping fusion/rule/risk knobs replays in seconds instead
of re-running a full walk-forward. Rule logic is shared with live code via
``TechnicalSignalGenerator.evaluate_rules``; the label definition is shared
via ``MLSignalGenerator`` class constants.

NaN vs None: live generators see SQL NULL as None and substitute neutral
defaults; the replay sees NaN. Every rule comparison evaluates identically
under both (NaN comparisons are False, matching the neutral defaults), and
the parity tests pin that equivalence.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import structlog

from signals.base import AggregatedSignal, SignalDirection
from signals.ml_model import MLSignalGenerator
from signals.technical import TechnicalSignalGenerator

log = structlog.get_logger()

_DIR_SCORE = {
    SignalDirection.BUY: 1,
    SignalDirection.HOLD: 0,
    SignalDirection.SELL: -1,
}

DEFAULT_WEIGHTS = {"technical": 0.40, "ml": 0.40, "sentiment": 0.20}


def load_feature_matrix(db_engine, symbols: list[str]):
    """Load ALL feat_technical rows + label columns, via backtest.data_store
    (snapshot-backed, one load per symbol).

    Unlike ``engine._load_all_features`` this keeps the final label-horizon
    rows (future_close = NaN) — they carry no training label but ARE the
    latest feature rows for signal generation near the end of history.
    """
    import pandas as pd

    from backtest import data_store

    frames = [data_store.load_features(db_engine, symbol) for symbol in symbols]
    frames = [f for f in frames if not f.empty]

    if not frames:
        return pd.DataFrame()

    full = pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"])
    horizon = MLSignalGenerator.LABEL_HORIZON_BARS
    full["future_close"] = full.groupby("symbol")["close"].shift(-horizon)
    full["target"] = MLSignalGenerator.compute_target(full["close"], full["future_close"])
    return full.reset_index(drop=True)


def precompute_walk_forward_probs(
    feature_df,
    bar_stream: list[tuple[datetime, str]],
    min_train_days: int = 60,
    walk_forward_months: int = 1,
    train_window_days: int = 0,
) -> dict[tuple[str, datetime], float]:
    """Replicate the engine's walk-forward schedule; return {(symbol, bar_ts): prob_up}.

    *bar_stream* is the chronologically sorted [(timestamp, symbol), ...] the
    replay will walk — the retrain schedule is driven by BAR timestamps
    exactly like ``engine.run_backtest`` (first bar + min_train_days, then
    every 30*months days), so boundaries land on the same bars.

    Bars before the first successful retrain get no entry (warm-up → HOLD).
    Bars whose latest feature row has any NaN feature get no entry (the live
    generator returns HOLD on null features).
    """
    import numpy as np

    if feature_df is None or not len(feature_df) or not bar_stream:
        return {}

    feat_cols = MLSignalGenerator.FEATURES
    horizon = MLSignalGenerator.LABEL_HORIZON_BARS

    # Per-symbol sorted timestamp index for latest-row-at-or-before lookup
    ts_index: dict[str, list] = {}
    row_frames: dict[str, object] = {}
    for symbol, g in feature_df.groupby("symbol"):
        g = g.reset_index(drop=True)
        ts_index[symbol] = list(g["timestamp"])
        row_frames[symbol] = g

    ml = MLSignalGenerator(db_engine=None)
    probs: dict[tuple[str, datetime], float] = {}
    pending: list[tuple[str, datetime]] = []
    next_retrain_at: datetime | None = None
    model_ready = False
    frozen_params: dict | None = None

    def _flush_pending() -> None:
        """Batch-predict prob_up for all pending bars with the current model."""
        if not model_ready or not pending:
            pending.clear()
            return
        rows_by_symbol: dict[str, list[int]] = {}
        keys_by_symbol: dict[str, list[tuple[str, datetime]]] = {}
        for symbol, ts in pending:
            idx_list = ts_index.get(symbol)
            if not idx_list:
                continue
            i = bisect_right(idx_list, ts) - 1
            if i < 0:
                continue
            rows_by_symbol.setdefault(symbol, []).append(i)
            keys_by_symbol.setdefault(symbol, []).append((symbol, ts))
        for symbol, idxs in rows_by_symbol.items():
            X = row_frames[symbol].iloc[idxs][feat_cols].to_numpy(dtype=float)
            valid = ~np.isnan(X).any(axis=1)
            if not valid.any():
                continue
            X_scaled = ml.scaler.transform(X[valid])
            p = ml.model.predict_proba(X_scaled)[:, 1]
            keys = [k for k, ok in zip(keys_by_symbol[symbol], valid) if ok]
            for key, prob in zip(keys, p.tolist()):
                probs[key] = float(prob)
        pending.clear()

    import pandas as pd

    for ts, symbol in bar_stream:
        if next_retrain_at is None:
            next_retrain_at = ts + timedelta(days=min_train_days)
        if ts >= next_retrain_at:
            _flush_pending()
            label_cutoff = pd.Timestamp(ts) - timedelta(hours=horizon)
            mask = (feature_df["timestamp"] < label_cutoff) & feature_df["future_close"].notna()
            if train_window_days > 0:
                mask &= feature_df["timestamp"] >= label_cutoff - timedelta(days=train_window_days)
            slice_df = feature_df[mask].copy()
            # Tune once (first boundary, full GridSearchCV) then freeze —
            # ~35 windows/run previously each ran an 8x5-fold search
            # (~1,400 fits/run); every later boundary now does one fit.
            trained, frozen_params = ml.train_walk_forward(slice_df, frozen_params)
            if trained:
                model_ready = True
                log.info("cache_walk_forward_retrained", at=str(ts), rows=len(slice_df))
            else:
                log.warning("cache_walk_forward_retrain_skipped", at=str(ts))
            next_retrain_at = ts + timedelta(days=30 * walk_forward_months)
        if model_ready:
            pending.append((symbol, ts))
    _flush_pending()

    return probs


@dataclass
class FusionParams:
    """Every signal-layer knob the sweep can vary without re-precomputing."""
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    buy_gate: float = 0.15
    sell_gate: float = -0.15
    ml_buy_threshold: float = 0.65
    ml_sell_threshold: float = 0.35
    require_uptrend_for_buys: bool = False
    guard_bb_dip_buy: bool = False


class ReplayAggregator:
    """Drop-in for SignalAggregator in backtests: pure in-memory fusion.

    Fuses cached technical rules (via evaluate_rules) + cached walk-forward
    ML probabilities + neutral sentiment with the given FusionParams.
    Parity with the live SignalAggregator at default params is pinned by
    tests/test_backtest_signal_cache.py.
    """

    def __init__(self, feature_df, prob_map: dict, params: FusionParams | None = None):
        self.params = params or FusionParams()
        self._probs = prob_map
        self._ts_index: dict[str, list] = {}
        self._rows: dict[str, list[dict]] = {}
        if feature_df is not None and len(feature_df):
            rule_cols = ["rsi_14_proxy", "macd_line", "bb_zscore",
                         "pct_change_1h", "pct_change_24h", "close_vs_sma50"]
            for symbol, g in feature_df.groupby("symbol"):
                g = g.reset_index(drop=True)
                self._ts_index[symbol] = list(g["timestamp"])
                self._rows[symbol] = g[rule_cols].to_dict("records")

    def aggregate(self, symbol: str, as_of: datetime) -> AggregatedSignal:
        p = self.params

        # Technical: latest feature row at or before as_of
        idx_list = self._ts_index.get(symbol)
        if idx_list:
            i = bisect_right(idx_list, as_of) - 1
        else:
            i = -1
        if i < 0:
            tech_dir, tech_conf = SignalDirection.HOLD, 0.5
        else:
            tech_dir, tech_conf, _ = TechnicalSignalGenerator.evaluate_rules(
                self._rows[symbol][i],
                require_uptrend_for_buys=p.require_uptrend_for_buys,
                guard_bb_dip_buy=p.guard_bb_dip_buy,
            )

        # ML: cached prob under the walk-forward model active at as_of
        prob_up = self._probs.get((symbol, as_of))
        if prob_up is None:
            ml_dir, ml_conf = SignalDirection.HOLD, 0.5
        elif prob_up > p.ml_buy_threshold:
            ml_dir, ml_conf = SignalDirection.BUY, prob_up
        elif prob_up < p.ml_sell_threshold:
            ml_dir, ml_conf = SignalDirection.SELL, 1.0 - prob_up
        else:
            ml_dir, ml_conf = SignalDirection.HOLD, 0.5

        # Sentiment: neutral in every backtest (HOLD → contributes 0)
        score = (
            _DIR_SCORE[tech_dir] * tech_conf * p.weights.get("technical", 0.0)
            + _DIR_SCORE[ml_dir] * ml_conf * p.weights.get("ml", 0.0)
        )

        if score > p.buy_gate:
            direction = SignalDirection.BUY
        elif score < p.sell_gate:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.HOLD

        return AggregatedSignal(
            direction=direction,
            confidence=min(abs(score), 1.0),
            symbol=symbol,
            timestamp=as_of,
            signals=[],
            reasoning="replay",
            weights_used=p.weights,
        )
