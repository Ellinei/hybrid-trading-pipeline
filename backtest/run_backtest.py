"""CLI entry point for the historical backtest harness.

Usage:
    python -m backtest.run_backtest
    python -m backtest.run_backtest --symbols BTCUSDT,ETHUSDT --start 2026-04-01 --end 2026-06-01
    python -m backtest.run_backtest --output trades.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone
from urllib.parse import quote_plus

import structlog
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from backtest.engine import BacktestConfig, BacktestResult, _load_all_features, run_backtest
from backtest.exit_simulation import OHLCVBar
from backtest.metrics import (
    exit_reason_breakdown,
    expectancy,
    holding_hours_stats,
    per_window_metrics,
)
from execution.exchange_filters import parse_symbol_filters
from execution.risk_manager import RiskConfig
from ingestion.binance_client import BinanceClientWrapper
from signals.aggregator import SignalAggregator
from signals.ml_model import MLSignalGenerator
from signals.technical import TechnicalSignalGenerator

log = structlog.get_logger()


def _load_bars(engine, symbol: str, start: datetime | None, end: datetime | None) -> list[OHLCVBar]:
    clauses = ["symbol = :symbol"]
    params: dict = {"symbol": symbol}
    if start is not None:
        clauses.append("timestamp >= :start")
        params["start"] = start
    if end is not None:
        clauses.append("timestamp <= :end")
        params["end"] = end

    query = text(f"""
        SELECT timestamp, open, high, low, close
        FROM raw.ohlcv
        WHERE {' AND '.join(clauses)}
        ORDER BY timestamp ASC
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [
        OHLCVBar(
            timestamp=row.timestamp, open=float(row.open), high=float(row.high),
            low=float(row.low), close=float(row.close),
        )
        for row in rows
    ]


def _load_atr(engine, symbol: str, start: datetime | None, end: datetime | None) -> dict[datetime, float]:
    clauses = ["symbol = :symbol"]
    params: dict = {"symbol": symbol}
    if start is not None:
        clauses.append("timestamp >= :start")
        params["start"] = start
    if end is not None:
        clauses.append("timestamp <= :end")
        params["end"] = end

    query = text(f"""
        SELECT timestamp, range_14
        FROM features.feat_technical
        WHERE {' AND '.join(clauses)}
        ORDER BY timestamp ASC
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return {row.timestamp: float(row.range_14) for row in rows if row.range_14 is not None}


def _print_report(symbols: list[str], result: BacktestResult) -> None:
    print(f"\n=== Backtest Report ({', '.join(symbols)}) ===")
    print(f"  Trades         : {result.metrics['trade_count']}")
    print(f"  Win rate       : {result.metrics['win_rate']:.2%}")
    print(f"  Profit factor  : {result.metrics['profit_factor']:.2f}")
    print(f"  Expectancy     : ${expectancy(result.trades):.2f}/trade")
    print(f"  Max drawdown   : {result.metrics['max_drawdown']:.2%}")
    print(f"  Sharpe ratio   : {result.metrics['sharpe_ratio']:.2f}")
    print(f"  Total return   : {result.metrics['total_return']:.2%}")

    if result.trades:
        reasons = exit_reason_breakdown(result.trades)
        mix = ", ".join(f"{r}={n}" for r, n in sorted(reasons.items()))
        print(f"  Exit reasons   : {mix}")
        hold = holding_hours_stats(result.trades)
        print(f"  Hold time (h)  : median={hold['median']:.1f}, "
              f"mean={hold['mean']:.1f}, p90={hold['p90']:.1f}")

    if result.retrain_events:
        boundaries = [e["at"] for e in result.retrain_events if e["trained"]]
        windows = per_window_metrics(result.trades, boundaries)
        traded_windows = [w for w in windows if w["trade_count"] > 0]
        profitable = sum(1 for w in traded_windows if w["profit_factor"] > 1.0)
        print(f"\n  Walk-forward windows: {len(windows)} "
              f"({len(traded_windows)} traded, {profitable} with PF>1)")
        print("  window_start         trades  PF     win%   pnl")
        for w in windows:
            pf = w["profit_factor"]
            pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
            print(f"  {w['window_start']:%Y-%m-%d %H:%M}     "
                  f"{w['trade_count']:>4}  {pf_str:>5}  "
                  f"{w['win_rate']:>5.1%}  {w['pnl']:>8.2f}")

    if result.warnings:
        print("\n  Warnings:")
        for w in result.warnings:
            print(f"   - {w}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical backtest harness")
    parser.add_argument("--symbols", type=str, default=None,
                         help="Comma-separated symbols (default: TRADING_SYMBOLS env)")
    parser.add_argument("--start", type=str, default=None,
                         help="ISO date (default: earliest available data per symbol)")
    parser.add_argument("--end", type=str, default=None,
                         help="ISO date (default: latest available data per symbol)")
    parser.add_argument("--output", type=str, default=None, help="Optional CSV path for the trade log")
    parser.add_argument("--walk-forward", type=int, default=0, metavar="MONTHS",
                         help="Enable walk-forward ML retraining every N months (0=disabled)")
    parser.add_argument("--min-train-days", type=int, default=60,
                         help="Warm-up days before first model fit in walk-forward mode (default: 60)")
    parser.add_argument("--train-window-days", type=int, default=0,
                         help="Cap the walk-forward training window to a sliding N days "
                              "(0=expanding window over all history)")
    parser.add_argument("--label-horizon", type=int, default=None, metavar="BARS",
                         help="Override MLSignalGenerator.LABEL_HORIZON_BARS for this run "
                              "(experiment knob; default: class constant)")
    parser.add_argument("--label-min-return", type=float, default=None,
                         help="Override MLSignalGenerator.LABEL_MIN_RETURN for this run "
                              "(experiment knob; default: class constant)")
    parser.add_argument("--bb-guard", action="store_true", default=False,
                         help="Require pct_1h > 0 on the bb_zscore<-2 BUY rule "
                              "(falling-knife guard, A/B toggle)")
    parser.add_argument("--require-uptrend", action="store_true", default=False,
                         help="Gate BUY rules on close_vs_sma50 > 0 (regime filter, A/B toggle)")
    parser.add_argument("--buy-gate", type=float, default=0.15,
                         help="Aggregator fused-score BUY threshold (default: 0.15)")
    parser.add_argument("--sell-gate", type=float, default=-0.15,
                         help="Aggregator fused-score SELL threshold — the de facto exit "
                              "trigger (default: -0.15)")
    parser.add_argument("--ml-buy-threshold", type=float, default=0.65,
                         help="ML prob_up above which the ML source says BUY (default: 0.65)")
    parser.add_argument("--ml-sell-threshold", type=float, default=0.35,
                         help="ML prob_up below which the ML source says SELL (default: 0.35)")
    parser.add_argument("--max-hold-bars", type=int, default=0,
                         help="Time-stop: close positions after N hourly bars (0=disabled)")
    args = parser.parse_args()

    if args.label_horizon is not None or args.label_min_return is not None:
        # Class-level override so BOTH target computations (SQL training path
        # and the walk-forward feature pre-loader) pick up the same label.
        from signals.ml_model import MLSignalGenerator
        if args.label_horizon is not None:
            MLSignalGenerator.LABEL_HORIZON_BARS = args.label_horizon
        if args.label_min_return is not None:
            MLSignalGenerator.LABEL_MIN_RETURN = args.label_min_return
        print(f"  Label override: horizon={MLSignalGenerator.LABEL_HORIZON_BARS} bars, "
              f"min_return={MLSignalGenerator.LABEL_MIN_RETURN}")

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

    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else
        [s.strip() for s in os.getenv("TRADING_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",") if s.strip()]
    )
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc) if args.start else None
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) if args.end else None

    client = BinanceClientWrapper()
    aggregator = SignalAggregator(
        engine,
        buy_gate=args.buy_gate,
        sell_gate=args.sell_gate,
        technical=TechnicalSignalGenerator(
            engine,
            require_uptrend_for_buys=args.require_uptrend,
            guard_bb_dip_buy=args.bb_guard,
        ),
        ml=MLSignalGenerator(
            engine,
            buy_prob_threshold=args.ml_buy_threshold,
            sell_prob_threshold=args.ml_sell_threshold,
        ),
    )

    bars_by_symbol, atr_by_symbol, filters_by_symbol = {}, {}, {}
    for symbol in symbols:
        bars_by_symbol[symbol] = _load_bars(engine, symbol, start, end)
        atr_by_symbol[symbol] = _load_atr(engine, symbol, start, end)
        try:
            filters_by_symbol[symbol] = parse_symbol_filters(client.get_symbol_info_cached(symbol))
        except Exception as exc:
            log.warning("symbol_filters_unavailable", symbol=symbol, error=str(exc))
            filters_by_symbol[symbol] = None
        if not bars_by_symbol[symbol]:
            log.warning("no_ohlcv_data_for_symbol", symbol=symbol, start=start, end=end)

    config = BacktestConfig(
        symbols=symbols,
        start=start or datetime(1970, 1, 1, tzinfo=timezone.utc),
        end=end or datetime.now(timezone.utc),
        risk_config=RiskConfig(max_hold_bars=args.max_hold_bars),
    )

    walk_forward = args.walk_forward
    min_train_days = args.min_train_days
    feature_df = None

    if walk_forward > 0:
        print(f"\n  Walk-forward mode: ML retrain every {walk_forward} month(s), "
              f"warm-up {min_train_days} day(s)")
        print("  Pre-loading feature data for walk-forward…", flush=True)
        feature_df = _load_all_features(engine, symbols)
        print(f"  Loaded {len(feature_df)} feature rows across {len(symbols)} symbol(s).")

    result = run_backtest(
        bars_by_symbol=bars_by_symbol,
        atr_by_symbol=atr_by_symbol,
        filters_by_symbol=filters_by_symbol,
        aggregator=aggregator,
        config=config,
        walk_forward_months=walk_forward,
        min_train_days=min_train_days,
        feature_df=feature_df,
        db_engine=engine,
        train_window_days=args.train_window_days,
    )

    _print_report(symbols, result)

    if args.output and result.trades:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result.trades[0].keys()))
            writer.writeheader()
            writer.writerows(result.trades)
        print(f"\n  Trade log written to {args.output}")

    engine.dispose()


if __name__ == "__main__":
    main()
