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
from dataclasses import asdict, is_dataclass
from urllib.parse import quote_plus

import structlog
from dotenv import load_dotenv
from sqlalchemy import create_engine

from config.strategy import StrategyConfig
from execution.freshness import FreshnessMonitor
from execution.heartbeat import HeartbeatRecorder
from execution.journal import DecisionJournal
from execution.risk_manager import RiskConfig, RiskManager
from execution.trade_executor import TradeExecutor
from ingestion.binance_client import BinanceClientWrapper
from signals.aggregator import SignalAggregator
from signals.base import SignalDirection
from signals.ml_model import MLSignalGenerator

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Component wiring — StrategyConfig (the validated exp008 knobs) -> the
# live objects that actually gate/execute trades. Kept as free functions
# (not inlined in __main__) so each knob's live wiring is independently
# testable — see tests/test_bot_runner.py.
# ---------------------------------------------------------------------------

def resolve_regime(engine, cfg: StrategyConfig) -> dict | None:
    """Entry-date regime map from complete prior UTC-day BTC candles.

    None means the gate is deliberately disabled; an empty map means it is
    enabled but BTC history cannot yet support a classification.
    """
    if cfg.regime_candidate is None:
        return None
    from backtest.run_backtest import _load_bars
    from signals.regime import build_causal_regime

    btc_bars = _load_bars(engine, "BTCUSDT", None, None)
    return build_causal_regime(btc_bars, cfg.regime_candidate)


def initial_regime(engine, cfg: StrategyConfig) -> tuple[dict | None, str | None]:
    """Startup regime resolution — a DB failure here must not abort startup.

    Unlike the per-cycle refresh in run_once, this runs before the main
    loop exists, so there is no earlier successful state to fall back on.
    An unknown regime_candidate is a config bug, not a data failure: it is
    validated BEFORE the try block so it always raises rather than being
    swallowed into a silent refresh_failed. A disabled gate resolves
    immediately. A recoverable query failure returns ({}, "refresh_failed")
    — {} (not None) means enabled-but-unavailable, which still blocks BUY
    — so main() can build the aggregator and reach run_once, where SELLs
    and close_expired_positions run and the first cycle retries the
    refresh (see run_once's regime handling above).
    """
    if cfg.regime_candidate is None:
        return None, None
    from signals.regime import REGIME_FUNCTIONS
    if cfg.regime_candidate not in REGIME_FUNCTIONS:
        raise ValueError(
            f"Unknown regime candidate: {cfg.regime_candidate!r}. "
            f"Choices: {sorted(REGIME_FUNCTIONS)}"
        )
    try:
        return resolve_regime(engine, cfg), None
    except Exception:
        log.exception("startup_regime_query_failed")
        return {}, "refresh_failed"


def build_market_data_client(*, paper: bool) -> BinanceClientWrapper:
    """Paper quotes use public mainnet, matching the ingested mainnet candles."""
    return BinanceClientWrapper(testnet=False) if paper else BinanceClientWrapper()


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
    freshness=None,
    journal=None,
) -> None:
    """Close any time-stop-expired positions, refresh the regime gate,
    then generate signals for every symbol and attempt execution.

    Expired positions are closed FIRST — mirrors backtest.engine checking
    time-stop ahead of a new entry signal for the same bar
    (engine.py:273-282).

    Regime is recomputed every cycle when the engine and strategy are
    given. A failed refresh blocks BUYs even if an older state is cached.
    SELL signals and time-stop exits continue through the cycle.

    When *heartbeat* is given, one raw.bot_heartbeat row is written per
    cycle (signals evaluated, actions taken, errors caught) — ops/
    dashboard's only way to distinguish a hung bot from a quiet market.
    """
    errors = 0
    mode = "dry_run" if dry_run else getattr(executor, "mode", "live")

    def record(symbol, outcome, reason, signal=None, **details):
        nonlocal errors
        if journal is not None:
            risk_config = getattr(getattr(executor, "risk", None), "config", None)
            paper_config = getattr(executor, "config", None)
            if not journal.record(
                symbol=symbol, mode=mode, outcome=outcome, reason=reason,
                signal=signal, config=strategy_cfg,
                risk_config=asdict(risk_config) if is_dataclass(risk_config) else None,
                paper_config=asdict(paper_config) if is_dataclass(paper_config) else None,
                **details,
            ):
                errors += 1

    try:
        exits = executor.close_expired_positions(max_hold_bars, dry_run=dry_run)
        if journal is not None:
            for result in exits or []:
                record(result.get("symbol", "unknown"), "closed",
                       result.get("close_reason", "position exit"), trade=result)
    except Exception:
        log.exception("time_stop_cycle_error")
        errors += 1
        record("*", "error", "position exit evaluation failed")

    if engine is not None and strategy_cfg is not None:
        try:
            aggregator._regime = resolve_regime(engine, strategy_cfg)
            aggregator._regime_refresh_error = None
        except Exception:
            log.exception("regime_refresh_error")
            if strategy_cfg.regime_candidate is not None:
                aggregator._regime_refresh_error = "refresh_failed"
            errors += 1

    signals = 0
    actions = 0
    for symbol in symbols:
        signal = None
        try:
            signal = aggregator.aggregate(symbol)
            signals += 1
            health = []
            regime_metadata = next((s.metadata for s in getattr(signal, "signals", [])
                                    if getattr(s, "source", None) == "regime"), None)
            if regime_metadata and regime_metadata.get("entry_blocked"):
                record(symbol, "blocked", regime_metadata["blocking_reason"], signal)
                continue
            if freshness is not None and signal.direction == SignalDirection.BUY:
                allowed, reason, health = freshness.entry_check(symbol)
                if not allowed:
                    record(symbol, "blocked", reason, signal, freshness=health)
                    log.warning("entry_blocked_stale_data", symbol=symbol, reason=reason)
                    continue
            result = executor.execute_signal(signal, dry_run=dry_run)
            if result is not None:
                actions += 1
                record(symbol, "executed",
                       result.get("close_reason") or result.get("status", "executed"),
                       signal, trade=result, freshness=health)
            else:
                decision = getattr(executor, "last_decision", {})
                record(symbol, decision.get("outcome", "rejected"),
                       decision.get("reason", "execution declined"), signal, freshness=health)
        except Exception:
            log.exception("cycle_error", symbol=symbol)
            errors += 1
            record(symbol, "error", "signal evaluation or execution failed", signal)

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
    freshness=None,
    journal=None,
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
                      strategy_cfg=strategy_cfg, heartbeat=heartbeat,
                      freshness=freshness, journal=journal)
            log.info("cycle_complete_sleeping", seconds=interval_seconds)
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        log.info("bot_stopped")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    """CLI entry point — wrapped as a function (rather than left inline
    under ``__main__``) so tests can drive it with mocked builders/engine,
    e.g. to prove a startup regime DB failure still reaches run_once."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid trading bot — testnet only"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single cycle then exit",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run", action="store_true",
        help="Log trades without placing real orders",
    )
    mode_group.add_argument("--paper", action="store_true", help="Use persistent simulated funds")
    parser.add_argument("--paper-balance", type=float, default=10000.0,
                        help="Initial USDT for a NEW paper account (existing accounts are resumed)")
    parser.add_argument("--paper-fee", type=float, default=0.001)
    parser.add_argument("--paper-slippage-bps", type=float, default=5.0)
    parser.add_argument("--candle-max-age", type=int, default=10800,
                        help="Seconds; candle open time")
    parser.add_argument("--feature-max-age", type=int, default=10800,
                        help="Seconds; feature source time")
    parser.add_argument(
        "--interval", type=int, default=3600,
        help="Seconds between cycles in loop mode (default: 3600)",
    )
    args = parser.parse_args(argv)
    if args.interval <= 0 or args.candle_max_age <= 0 or args.feature_max_age <= 0:
        parser.error("Interval and freshness limits must be positive")
    from execution.paper import PaperConfig, PaperExecutor, PaperRiskManager, PaperStore

    try:
        paper_config = PaperConfig(args.paper_balance, args.paper_fee, args.paper_slippage_bps)
    except ValueError as exc:
        parser.error(str(exc))

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

    # Everything below allocates a resource keyed to *engine* (or, for
    # paper_store, its own sqlite file) before the loop ever runs. Any of
    # them can raise (e.g. PaperStore finding its sqlite file locked by
    # another bot) — the try/finally starts here, not after construction,
    # so a failure partway through still disposes whatever was already
    # built rather than leaking the engine.
    paper_store = None
    try:
        symbols = [
            s.strip()
            for s in os.getenv("TRADING_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
            if s.strip()
        ]
        strategy_cfg = StrategyConfig(symbols=symbols, candle_max_age_seconds=args.candle_max_age,
                                      feature_max_age_seconds=args.feature_max_age)
        log.info("strategy_config_resolved", **{
            k: v for k, v in strategy_cfg.__dict__.items()
        })

        client     = build_market_data_client(paper=args.paper)
        regime, regime_refresh_error = initial_regime(engine, strategy_cfg)
        aggregator = build_aggregator(engine, strategy_cfg, regime)
        aggregator._regime_refresh_error = regime_refresh_error
        if args.paper:
            paper_store = PaperStore(initial_balance=paper_config.initial_balance)
            risk = PaperRiskManager(
                paper_store, engine, RiskConfig(max_hold_bars=strategy_cfg.max_hold_bars),
            )
            executor = PaperExecutor(client, paper_store, risk, paper_config)
        else:
            risk = build_risk_manager(client, engine, strategy_cfg)
            executor = TradeExecutor(client, risk, engine)
        heartbeat  = HeartbeatRecorder(engine)
        freshness = FreshnessMonitor(engine, strategy_cfg.candle_max_age_seconds,
                                     strategy_cfg.feature_max_age_seconds)
        journal = DecisionJournal()

        if args.once:
            run_once(symbols, executor, aggregator, dry_run=args.dry_run,
                      max_hold_bars=strategy_cfg.max_hold_bars,
                      engine=engine, strategy_cfg=strategy_cfg, heartbeat=heartbeat,
                      freshness=freshness, journal=journal)
        else:
            run_loop(symbols, executor, aggregator,
                     interval_seconds=args.interval, dry_run=args.dry_run,
                     max_hold_bars=strategy_cfg.max_hold_bars,
                     engine=engine, strategy_cfg=strategy_cfg, heartbeat=heartbeat,
                     freshness=freshness, journal=journal)
    finally:
        # Nested so a paper_store.close() failure can't skip engine.dispose().
        try:
            if paper_store is not None:
                paper_store.close()
        finally:
            engine.dispose()


if __name__ == "__main__":
    main()
