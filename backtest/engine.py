"""Backtest orchestration: replay signals + risk gating + fill simulation
over historical bars and report performance metrics.

Stop-loss/take-profit are computed from the bar's raw close price (the
"decision price"), matching live TradeExecutor — actual fills (entry and
exit) are simulated separately with fee + slippage applied on top. Realized
equity is tracked from closed trades only (no mark-to-market of open
positions), matching RiskManager.check_drawdown's realized-pnl-only
circuit breaker.
"""
from __future__ import annotations

import structlog
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from backtest.exit_simulation import (
    OHLCVBar,
    check_exit,
    simulate_entry_fill,
    simulate_exit_fill,
)
from backtest.metrics import max_drawdown, profit_factor, sharpe_ratio, total_return, win_rate
from backtest.sim_risk_manager import SimulatedRiskManager
from backtest.state import BacktestState, OpenPosition
from execution.exchange_filters import FilterViolation, SymbolFilters, round_qty_down, validate_notional
from execution.risk_manager import RiskConfig
from signals.base import SignalDirection

log = structlog.get_logger()

_ML_LOOKAHEAD_WARNING = (
    "ML signal model was trained on an 80/20 split of this SAME history — "
    "most backtest bars are in-sample (training-on-test), a strong "
    "optimistic bias, not a mild one. True walk-forward retraining "
    "(Phase 8.5) would likely show worse performance than reported here."
)
_DEFAULT_PERIODS_PER_YEAR = 24 * 365  # hourly bars, matches the live pipeline's 1h interval


def _load_all_features(db_engine, symbols: list[str]):
    """Pre-load all feat_technical rows + computed target column for walk-forward.

    Loads once per symbol via backtest.data_store (snapshot-backed, avoids
    N×retraining-boundary queries). Returns a pandas DataFrame sorted by
    (symbol, timestamp) with an added 'target' column using the SAME label
    definition as MLSignalGenerator (k-bar forward net return above cost).
    Rows without a known future close are dropped.
    """
    import pandas as pd

    from backtest import data_store
    from signals.ml_model import MLSignalGenerator

    frames = [data_store.load_features(db_engine, symbol) for symbol in symbols]
    frames = [f for f in frames if not f.empty]

    if not frames:
        return pd.DataFrame()

    full = pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"])
    horizon = MLSignalGenerator.LABEL_HORIZON_BARS
    full["future_close"] = full.groupby("symbol")["close"].shift(-horizon)
    full["target"] = MLSignalGenerator.compute_target(full["close"], full["future_close"])
    full = full.dropna(subset=["future_close"])
    return full.reset_index(drop=True)


@dataclass
class BacktestConfig:
    symbols: list[str]
    start: datetime
    end: datetime
    initial_balance: float = 10_000.0
    fee_rate: float = 0.001
    slippage_bps: float = 5.0
    risk_config: RiskConfig = field(default_factory=RiskConfig)


@dataclass
class BacktestResult:
    trades: list[dict]
    equity_curve: list[tuple[datetime, float]]
    metrics: dict[str, float]
    warnings: list[str]
    retrain_events: list[dict] = field(default_factory=list)


def _infer_periods_per_year_from_equity_curve(
    equity_curve: list[tuple[datetime, float]]
) -> float:
    """Sharpe must be annualized using the equity curve's OWN sampling
    frequency, not the underlying bar frequency — the curve is sampled at
    trade closes (sparse, irregular), not once per bar. Using bar frequency
    here previously inflated Sharpe's magnitude by sqrt(bars_per_year /
    closes_per_year)."""
    if len(equity_curve) < 2:
        return _DEFAULT_PERIODS_PER_YEAR
    total_seconds = (equity_curve[-1][0] - equity_curve[0][0]).total_seconds()
    if total_seconds <= 0:
        return _DEFAULT_PERIODS_PER_YEAR
    avg_seconds_between_points = total_seconds / (len(equity_curve) - 1)
    return (365 * 24 * 3600) / avg_seconds_between_points


def _open_position(
    state: BacktestState,
    risk: SimulatedRiskManager,
    symbol: str,
    bar: OHLCVBar,
    filters: SymbolFilters | None,
    config: BacktestConfig,
) -> None:
    atr = state.current_atr.get(symbol, 0.0)
    balance = risk.get_account_balance()
    if not balance or balance <= 0:
        return

    qty = risk.calculate_position_size(bar.close, atr, balance)
    if filters is not None:
        qty = round_qty_down(qty, filters)
    if qty <= 0:
        return
    if filters is not None:
        try:
            validate_notional(qty, bar.close, filters)
        except FilterViolation:
            return

    stop_loss, take_profit = risk.calculate_stops(bar.close, atr)
    fill_price, fee = simulate_entry_fill(bar.close, qty, config.fee_rate, config.slippage_bps)

    state.balance -= (fill_price * qty + fee)
    state.positions[symbol] = OpenPosition(
        symbol=symbol, quantity=qty, entry_price=fill_price, entry_time=bar.timestamp,
        stop_loss=stop_loss, take_profit=take_profit, entry_fee=fee,
    )


def _close_position(
    state: BacktestState,
    config: BacktestConfig,
    symbol: str,
    position: OpenPosition,
    exit_price_raw: float,
    timestamp: datetime,
    reason: str,
) -> None:
    fill_price, fee = simulate_exit_fill(
        exit_price_raw, position.quantity, config.fee_rate, config.slippage_bps
    )
    pnl = (fill_price - position.entry_price) * position.quantity - position.entry_fee - fee

    state.balance += (fill_price * position.quantity - fee)
    state.closed_trades.append({
        "symbol": symbol,
        "entry_price": position.entry_price,
        "exit_price": fill_price,
        "quantity": position.quantity,
        "entry_time": position.entry_time,
        "exit_time": timestamp,
        "pnl": pnl,
        "reason": reason,
    })
    realized_equity = config.initial_balance + sum(t["pnl"] for t in state.closed_trades)
    state.equity_curve.append((timestamp, realized_equity))
    del state.positions[symbol]


def run_backtest(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    atr_by_symbol: dict[str, dict[datetime, float]],
    filters_by_symbol: dict[str, SymbolFilters],
    aggregator,
    config: BacktestConfig,
    walk_forward_months: int = 0,
    min_train_days: int = 60,
    feature_df=None,
    db_engine=None,
    train_window_days: int = 0,
) -> BacktestResult:
    """Replay historical bars through the live signal/risk/fill logic.

    *aggregator* must expose aggregate(symbol, as_of) -> AggregatedSignal —
    in production this is the real SignalAggregator(db_engine); in tests it
    can be any fake with that interface.

    When *walk_forward_months* > 0, the ML model is retrained in-memory at
    each monthly boundary using only data that existed up to that point,
    making all post-warm-up ML signals genuinely out-of-sample.

    *train_window_days* > 0 caps each retrain to a sliding window of that
    many days instead of an expanding all-history window — bounds retrain
    cost on multi-year datasets.
    """
    import pandas as pd

    state = BacktestState(balance=config.initial_balance)
    risk = SimulatedRiskManager(
        state, initial_balance=config.initial_balance, config=config.risk_config
    )
    retrain_events: list[dict] = []

    # Walk-forward setup
    ml_gen = getattr(aggregator, "ml", None)
    next_retrain_at: datetime | None = None
    wf_active = walk_forward_months > 0 and ml_gen is not None
    frozen_params: dict | None = None

    if wf_active:
        if feature_df is None and db_engine is not None:
            feature_df = _load_all_features(db_engine, config.symbols)
        ml_gen.set_walk_forward_mode(True)

    # All symbols share one balance/position-count/drawdown state, so bars
    # MUST be walked in one global chronological stream — never all of one
    # symbol's bars before moving to the next, or balance, open-position
    # count, and the equity curve all reflect a scrambled timeline instead
    # of what was actually true at each point in time.
    merged = sorted(
        (
            (bar.timestamp, symbol, bar)
            for symbol in config.symbols
            for bar in bars_by_symbol.get(symbol, [])
        ),
        key=lambda item: item[0],
    )

    for timestamp, symbol, bar in merged:
        # Walk-forward: trigger retraining at each monthly boundary
        if wf_active and feature_df is not None and len(feature_df):
            if next_retrain_at is None:
                next_retrain_at = timestamp + timedelta(days=min_train_days)
            if timestamp >= next_retrain_at:
                # Exclude the final label-horizon hours: a row's target uses
                # the close LABEL_HORIZON_BARS after it, so rows newer than
                # (now - horizon) have labels that peek past "now" — training
                # on them leaks future prices into the model.
                from signals.ml_model import MLSignalGenerator
                label_cutoff = pd.Timestamp(timestamp) - timedelta(
                    hours=MLSignalGenerator.LABEL_HORIZON_BARS
                )
                mask = feature_df["timestamp"] < label_cutoff
                if train_window_days > 0:
                    window_start = label_cutoff - timedelta(days=train_window_days)
                    mask &= feature_df["timestamp"] >= window_start
                slice_df = feature_df[mask].copy()
                # Tune once (first boundary, full GridSearchCV) then freeze
                # — mirrors backtest.signal_cache.precompute_walk_forward_probs
                # so both walk-forward code paths behave consistently and
                # pay the ~40-fit search once per run, not once per window.
                trained, frozen_params = ml_gen.train_walk_forward(slice_df, frozen_params)
                retrain_events.append(
                    {"at": timestamp, "rows": len(slice_df), "trained": trained}
                )
                if trained:
                    log.info("walk_forward_retrained",
                             at=str(timestamp), rows=len(slice_df))
                else:
                    log.warning("walk_forward_retrain_skipped", at=str(timestamp))
                next_retrain_at = timestamp + timedelta(days=30 * walk_forward_months)
        atr_lookup = atr_by_symbol.get(symbol, {})
        filters = filters_by_symbol.get(symbol)

        position = state.positions.get(symbol)
        if position is not None:
            hit = check_exit(bar, position)
            if hit is not None:
                reason, trigger_price = hit
                _close_position(state, config, symbol, position, trigger_price,
                                 timestamp, reason)
                position = None
            elif (
                config.risk_config.max_hold_bars > 0
                and timestamp - position.entry_time
                    >= timedelta(hours=config.risk_config.max_hold_bars)
            ):
                # Time-stop: the ML edge is defined over LABEL_HORIZON_BARS;
                # holding past max_hold_bars is unmodeled exposure.
                _close_position(state, config, symbol, position, bar.close,
                                 timestamp, "time_stop")
                position = None

        state.current_atr[symbol] = atr_lookup.get(timestamp, 0.0)

        signal = aggregator.aggregate(symbol, as_of=timestamp)
        approved, _reason = risk.approve_trade(signal, bar.close)
        if not approved:
            continue

        if signal.direction == SignalDirection.BUY:
            if symbol in state.positions:
                continue  # already holding — one position per symbol in this model
            _open_position(state, risk, symbol, bar, filters, config)
        else:  # SELL — approve_trade already confirmed an open position exists
            position = state.positions.get(symbol)
            if position is not None:
                _close_position(state, config, symbol, position, bar.close,
                                 timestamp, "signal_close")

    periods_per_year = _infer_periods_per_year_from_equity_curve(state.equity_curve)
    metrics = {
        "win_rate": win_rate(state.closed_trades),
        "profit_factor": profit_factor(state.closed_trades),
        "max_drawdown": max_drawdown(state.equity_curve),
        "sharpe_ratio": sharpe_ratio(state.equity_curve, periods_per_year),
        "total_return": total_return(state.equity_curve),
        "trade_count": len(state.closed_trades),
    }

    if wf_active:
        warning = (
            f"Walk-forward mode: ML model retrained every {walk_forward_months} "
            f"month(s) with {min_train_days}-day warm-up. "
            "All post-warm-up ML signals are out-of-sample."
        )
    else:
        warning = _ML_LOOKAHEAD_WARNING

    return BacktestResult(
        trades=state.closed_trades,
        equity_curve=state.equity_curve,
        metrics=metrics,
        warnings=[warning],
        retrain_events=retrain_events,
    )
