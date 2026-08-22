"""Main bot loop: generate signals → validate → execute every N seconds.

Usage:
    python -m execution.bot_runner --dry-run --once        # single cycle, no orders
    python -m execution.bot_runner --dry-run               # loop every hour
    python -m execution.bot_runner --once                  # single cycle, live testnet
    python -m execution.bot_runner --interval 1800         # loop every 30 min
"""
from __future__ import annotations

import os
import time
from urllib.parse import quote_plus

import structlog
from dotenv import load_dotenv
from sqlalchemy import create_engine

from config.strategy import StrategyConfig
from execution.heartbeat import HeartbeatRecorder
from execution.risk_manager import RiskConfig, RiskManager
from execution.trade_executor import TradeExecutor
from ingestion.binance_client import BinanceClientWrapper
from signals.aggregator import SignalAggregator
from signals.ml_model import MLSignalGenerator

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Component wiring — StrategyConfig (the validated exp008 knobs) -> the
# live objects that actually gate/execute trades. Kept as free functions
# (not inlined in __main__) so each knob's live wiring is independently
# testable — see tests/test_bot_runner.py.
# ---------------------------------------------------------------------------

def resolve_regime(engine, cfg: StrategyConfig) -> dict | None:
    """Today's BTC macro regime dict for cfg.regime_candidate, or None if
    the gate is disabled. Reuses backtest.run_backtest._load_bars — the
    exact same BTCUSDT-bars -> daily-close -> REGIME_FUNCTIONS path
    backtest.experiment._resolve_regime uses — so live and backtest can
    never classify the same day differently due to divergent loading."""
    if cfg.regime_candidate is None:
        return None
    from backtest.run_backtest import _load_bars
    from signals.regime import REGIME_FUNCTIONS, resample_daily_close

    if cfg.regime_candidate not in REGIME_FUNCTIONS:
        raise ValueError(
            f"Unknown regime candidate: {cfg.regime_candidate!r}. "
            f"Choices: {sorted(REGIME_FUNCTIONS)}"
        )
    btc_bars = _load_bars(engine, "BTCUSDT", None, None)
    daily_close = resample_daily_close(btc_bars)
    return REGIME_FUNCTIONS[cfg.regime_candidate](daily_close)


def build_aggregator(engine, cfg: StrategyConfig, regime: dict | None) -> SignalAggregator:
    ml = MLSignalGenerator(
        engine,
        buy_prob_threshold=cfg.ml_buy_threshold,
        sell_prob_threshold=cfg.ml_sell_threshold,
    )
    return SignalAggregator(
        engine,
        weights=cfg.weights,
        buy_gate=cfg.buy_gate,
        sell_gate=cfg.sell_gate,
        ml=ml,
        regime=regime,
        sentiment_enabled=cfg.sentiment_enabled,
    )


def build_risk_manager(client, engine, cfg: StrategyConfig) -> RiskManager:
    return RiskManager(client, engine, config=RiskConfig(max_hold_bars=cfg.max_hold_bars))


# ---------------------------------------------------------------------------
# Cycle logic
# ---------------------------------------------------------------------------

def run_once(
    symbols:        list[str],
    executor:       TradeExecutor,
    aggregator:     SignalAggregator,
    dry_run:        bool = False,
    max_hold_bars:  int  = 0,
    engine=None,
    strategy_cfg:   StrategyConfig | None = None,
    heartbeat=None,
) -> None:
    """Close any time-stop-expired positions, refresh the regime gate,
    then generate signals for every symbol and attempt execution.

    Expired positions are closed FIRST — mirrors backtest.engine checking
    time-stop ahead of a new entry signal for the same bar
    (engine.py:273-282).

    Regime is RECOMPUTED every cycle (when *engine*/*strategy_cfg* are
    given), not read once at process start: resolve_regime returns a
    dict[date, RegimeState] with entries only up to whenever it was
    built, so a dict frozen at startup silently stops matching
    regime.get(now.date()) on every day after — the gate fails OPEN
    (allows BUYs it should block), the unsafe direction, exactly the
    failure mode a multi-day paper soak would hit. On refresh failure the
    PREVIOUS regime is kept (fail-safe, not fail-open) and the cycle
    still runs — resolve_regime reads through data_store's parquet
    snapshot, so the steady-state cost is a cheap fingerprint check, not
    a full requery.

    When *heartbeat* is given, one raw.bot_heartbeat row is written per
    cycle (signals evaluated, actions taken, errors caught) — ops/
    dashboard's only way to distinguish a hung bot from a quiet market.
    """
    errors = 0
    try:
        executor.close_expired_positions(max_hold_bars, dry_run=dry_run)
    except Exception:
        log.exception("time_stop_cycle_error")
        errors += 1

    if engine is not None and strategy_cfg is not None:
        try:
            aggregator._regime = resolve_regime(engine, strategy_cfg)
        except Exception:
            log.exception("regime_refresh_error")
            errors += 1

    signals = 0
    actions = 0
    for symbol in symbols:
        try:
            signal = aggregator.aggregate(symbol)
            signals += 1
            if executor.execute_signal(signal, dry_run=dry_run) is not None:
                actions += 1
        except Exception:
            log.exception("cycle_error", symbol=symbol)
            errors += 1

    if heartbeat is not None:
        heartbeat.record(signals=signals, actions=actions, errors=errors)


def run_loop(
    symbols:          list[str],
    executor:         TradeExecutor,
    aggregator:       SignalAggregator,
    interval_seconds: int  = 3600,
    dry_run:          bool = False,
    max_hold_bars:    int  = 0,
    engine=None,
    strategy_cfg:     StrategyConfig | None = None,
    heartbeat=None,
) -> None:
    """Run indefinitely, once per *interval_seconds*. Ctrl-C to stop.

    *engine*/*strategy_cfg*, when given, are forwarded to every run_once
    call so the regime gate is recomputed each cycle instead of frozen at
    the first one — see run_once's docstring. *heartbeat*, when given, is
    likewise forwarded so every cycle records a raw.bot_heartbeat row."""
    log.info("bot_starting", interval_s=interval_seconds,
             symbols=symbols, dry_run=dry_run)
    try:
        while True:
            run_once(symbols, executor, aggregator, dry_run=dry_run,
                      max_hold_bars=max_hold_bars, engine=engine,
                      strategy_cfg=strategy_cfg, heartbeat=heartbeat)
            log.info("cycle_complete_sleeping", seconds=interval_seconds)
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        log.info("bot_stopped")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid trading bot — testnet only"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single cycle then exit",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log trades without placing real orders",
    )
    parser.add_argument(
        "--interval", type=int, default=3600,
        help="Seconds between cycles in loop mode (default: 3600)",
    )
    args = parser.parse_args()

    load_dotenv()

    engine = create_engine(
        "postgresql+psycopg2://{u}:{p}@{h}:{port}/{db}".format(
            u=quote_plus(os.getenv("POSTGRES_USER",     "trader")),
            p=quote_plus(os.getenv("POSTGRES_PASSWORD", "")),
            h=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            db=os.getenv("POSTGRES_DB",   "trading"),
        )
    )

    symbols = [
        s.strip()
        for s in os.getenv("TRADING_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
        if s.strip()
    ]
    strategy_cfg = StrategyConfig(symbols=symbols)
    log.info("strategy_config_resolved", **{
        k: v for k, v in strategy_cfg.__dict__.items()
    })

    client     = BinanceClientWrapper()
    regime     = resolve_regime(engine, strategy_cfg)
    risk       = build_risk_manager(client, engine, strategy_cfg)
    aggregator = build_aggregator(engine, strategy_cfg, regime)
    executor   = TradeExecutor(client, risk, engine)
    heartbeat  = HeartbeatRecorder(engine)

    try:
        if args.once:
            run_once(symbols, executor, aggregator, dry_run=args.dry_run,
                      max_hold_bars=strategy_cfg.max_hold_bars,
                      engine=engine, strategy_cfg=strategy_cfg, heartbeat=heartbeat)
        else:
            run_loop(symbols, executor, aggregator,
                     interval_seconds=args.interval, dry_run=args.dry_run,
                     max_hold_bars=strategy_cfg.max_hold_bars,
                     engine=engine, strategy_cfg=strategy_cfg, heartbeat=heartbeat)
    finally:
        engine.dispose()
