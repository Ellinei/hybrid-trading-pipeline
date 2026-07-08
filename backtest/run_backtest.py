"""CLI entry point for the historical backtest harness.

Usage:
    python -m backtest.run_backtest
    python -m backtest.run_backtest --symbols BTCUSDT,ETHUSDT --start 2026-04-01 --end 2026-06-01
    python -m backtest.run_backtest --output trades.csv
    python -m backtest.run_backtest --prob-stats --walk-forward 1 --train-window-days 540
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from urllib.parse import quote_plus

import pandas as pd
import structlog
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from backtest import data_store
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
    df = data_store.load_ohlcv(engine, symbol)
    if df.empty:
        return []
    if start is not None:
        df = df[df["timestamp"] >= start]
    if end is not None:
        df = df[df["timestamp"] <= end]
    return [
        OHLCVBar(
            timestamp=row.timestamp, open=row.open, high=row.high,
            low=row.low, close=row.close,
        )
        for row in df.itertuples()
    ]


def _load_atr(engine, symbol: str, start: datetime | None, end: datetime | None) -> dict[datetime, float]:
    df = data_store.load_features(engine, symbol)
    if df.empty:
        return {}
    if start is not None:
        df = df[df["timestamp"] >= start]
    if end is not None:
        df = df[df["timestamp"] <= end]
    return {
        row.timestamp: row.range_14
        for row in df.itertuples()
        if pd.notna(row.range_14)
    }


def _feature_fingerprint(engine, symbols: list[str]) -> dict[str, tuple[int, str | None]]:
    """Cheap per-symbol (row_count, max_timestamp) snapshot of feat_technical.

    Used as a proxy for "has the underlying feature data changed since a
    prob cache was built" without re-reading the whole table — a count(*)
    and max(timestamp) aggregate query rather than the full feature load.
    """
    fingerprint: dict[str, tuple[int, str | None]] = {}
    for symbol in symbols:
        query = text(
            "SELECT COUNT(*), MAX(timestamp) FROM features.feat_technical WHERE symbol = :symbol"
        )
        with engine.connect() as conn:
            count, max_ts = conn.execute(query, {"symbol": symbol}).fetchone()
        fingerprint[symbol] = (count, str(max_ts) if max_ts is not None else None)
    return fingerprint


def _compute_cache_key(
    *,
    label_horizon: int,
    label_min_return: float,
    min_train_days: int,
    walk_forward_months: int,
    train_window_days: int,
    symbols: list[str],
    fingerprint: dict[str, tuple[int, str | None]],
) -> str:
    """Content-addressed key over everything that determines what a prob
    cache SHOULD contain. Deliberately excludes tuned hyperparameters
    (frozen_params) — those are an output of building the cache from these
    inputs, not an input themselves, and are already fully determined by
    them (tuning is deterministic given the same feature data and schedule).
    """
    payload = {
        "label_horizon": label_horizon,
        "label_min_return": label_min_return,
        "min_train_days": min_train_days,
        "walk_forward_months": walk_forward_months,
        "train_window_days": train_window_days,
        "symbols": sorted(symbols),
        "fingerprint": {sym: list(fingerprint[sym]) for sym in sorted(fingerprint)},
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_cache_key(meta: dict, expected_key: str, force: bool) -> tuple[bool, str]:
    """Decide whether a --load-cache run may proceed.

    Returns (ok, message). ok=False means the caller should print message
    and exit without running — a genuine mismatch (stale cache: symbols,
    labels, schedule, or the underlying feature data changed) unless
    *force* is set. A cache with no "cache_key" in its meta predates this
    mechanism and is treated as unverifiable, not mismatched — it must not
    hard-fail, or every already-built cache breaks on first use.
    """
    cached_key = meta.get("cache_key")
    if cached_key is None:
        return True, (
            "WARNING: cache has no content key (built by an older run) — "
            "cannot verify it matches current symbols/labels/data; proceeding"
        )
    if cached_key == expected_key:
        return True, ""
    if force:
        return True, (
            f"WARNING: cache key mismatch (cached={cached_key[:12]}, "
            f"expected={expected_key[:12]}) — proceeding anyway (--force)"
        )
    return False, (
        f"ERROR: cache key mismatch (cached={cached_key[:12]}, "
        f"expected={expected_key[:12]}) — symbols, labels, min_train_days, "
        f"train_window_days, or the underlying feature data changed since "
        f"this cache was built. Rebuild it (--cache-only) or pass --force "
        f"to proceed anyway."
    )


def _build_cache_meta(
    engine, symbols: list[str], min_train_days: int,
    walk_forward_months: int, train_window_days: int,
) -> dict:
    """Meta dict stored alongside a saved prob cache, including its
    content-addressed key (see _compute_cache_key)."""
    meta = {
        "label_horizon": MLSignalGenerator.LABEL_HORIZON_BARS,
        "label_min_return": MLSignalGenerator.LABEL_MIN_RETURN,
        "min_train_days": min_train_days,
        "walk_forward_months": walk_forward_months,
        "train_window_days": train_window_days,
        "symbols": symbols,
    }
    meta["cache_key"] = _compute_cache_key(
        label_horizon=meta["label_horizon"],
        label_min_return=meta["label_min_return"],
        min_train_days=min_train_days,
        walk_forward_months=walk_forward_months,
        train_window_days=train_window_days,
        symbols=symbols,
        fingerprint=_feature_fingerprint(engine, symbols),
    )
    return meta


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


def _print_prob_stats(
    feature_df,
    bars_by_symbol: dict[str, list[OHLCVBar]],
    symbols: list[str],
    min_train_days: int,
    walk_forward_months: int,
    train_window_days: int,
    thresholds: tuple[float, ...] = (0.65, 0.60, 0.55, 0.50),
) -> None:
    """Diagnose ML prob_up distributions per month across the walk-forward run.

    Prints for each calendar month: how many bars have a prob_up computed
    (post-warm-up only), the maximum prob, and what fraction exceed each
    threshold.  Used to diagnose the 'strategy self-disables' symptom where
    prob_up > 0.65 never fires on mature training data (exp002).
    """
    from backtest.signal_cache import precompute_walk_forward_probs

    # Build the chronological (timestamp, symbol) bar stream, same merge
    # order the engine uses — retrain boundaries must land on the same bars.
    bar_stream: list[tuple[datetime, str]] = sorted(
        (bar.timestamp, symbol)
        for symbol in symbols
        for bar in bars_by_symbol.get(symbol, [])
    )
    if not bar_stream:
        print("  (no bars loaded — nothing to analyse)")
        return

    print(f"\n  Precomputing walk-forward probs "
          f"(min_train_days={min_train_days}, "
          f"walk_forward_months={walk_forward_months}, "
          f"train_window_days={train_window_days or 'expanding'})…", flush=True)

    prob_map = precompute_walk_forward_probs(
        feature_df, bar_stream,
        min_train_days=min_train_days,
        walk_forward_months=walk_forward_months,
        train_window_days=train_window_days,
    )

    if not prob_map:
        print("  (no probs computed — warm-up covers the entire dataset)")
        return

    # Group by year-month for each symbol
    from collections import defaultdict
    monthly: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (symbol, ts), prob in prob_map.items():
        month_key = (symbol, ts.strftime("%Y-%m"))
        monthly[month_key].append(prob)

    thr_header = "  ".join(f">={t:.2f}" for t in thresholds)
    print(f"\n=== ML prob_up distribution by month ===")
    print(f"  {'symbol':<10} {'month':<8}  {'n':>5}  {'max':>5}  {thr_header}")
    print(f"  {'-'*10} {'-'*8}  {'-'*5}  {'-'*5}  " + "  ".join("-"*6 for _ in thresholds))

    for (symbol, month), probs in sorted(monthly.items()):
        n = len(probs)
        max_p = max(probs)
        pcts = "  ".join(f"{sum(p >= t for p in probs)/n:>5.1%}" for t in thresholds)
        print(f"  {symbol:<10} {month:<8}  {n:>5}  {max_p:.3f}  {pcts}")

    all_probs = list(prob_map.values())
    print(f"\n  ALL: n={len(all_probs)}, "
          f"max={max(all_probs):.3f}, "
          f"mean={sum(all_probs)/len(all_probs):.3f}, "
          + "  ".join(f">={t:.2f}: {sum(p>=t for p in all_probs)/len(all_probs):.1%}"
                      for t in thresholds))


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
    parser.add_argument("--prob-stats", action="store_true", default=False,
                         help="Diagnostic mode: precompute walk-forward ML prob_up distributions "
                              "per month and exit without running the full backtest simulation. "
                              "Use with --walk-forward to match the retraining schedule of a real run.")
    parser.add_argument("--save-cache", type=str, default=None, metavar="PATH",
                         help="After the walk-forward run, precompute prob_map+feature_df and "
                              "pickle them to PATH so subsequent sweeps can skip retraining")
    parser.add_argument("--load-cache", type=str, default=None, metavar="PATH",
                         help="Load precomputed prob_map+feature_df from PATH (no GridSearchCV); "
                              "combine with --bb-guard/--buy-gate/etc. for fast parameter sweeps")
    parser.add_argument("--cache-only", action="store_true", default=False,
                         help="Precompute and save the walk-forward prob cache (--save-cache required) "
                              "then exit — skips the full backtest simulation (~15 min vs ~50 min). "
                              "Use once to build the cache; then sweep variants with --load-cache.")
    parser.add_argument("--force", action="store_true", default=False,
                         help="With --load-cache: proceed even if the cache's content key doesn't "
                              "match current symbols/labels/schedule/feature-data (normally a hard "
                              "error — the cache may no longer reflect what it's replaying).")
    args = parser.parse_args()

    if args.label_horizon is not None or args.label_min_return is not None:
        # Class-level override so BOTH target computations (SQL training path
        # and the walk-forward feature pre-loader) pick up the same label.
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

    if (walk_forward > 0 or args.prob_stats or args.save_cache or args.cache_only) and not args.load_cache:
        wf_months = walk_forward if walk_forward > 0 else 1
        print(f"\n  Walk-forward mode: ML retrain every {wf_months} month(s), "
              f"warm-up {min_train_days} day(s)")
        print("  Pre-loading feature data for walk-forward…", flush=True)
        feature_df = _load_all_features(engine, symbols)
        print(f"  Loaded {len(feature_df)} feature rows across {len(symbols)} symbol(s).")

    if args.prob_stats:
        _print_prob_stats(
            feature_df, bars_by_symbol, symbols,
            min_train_days=min_train_days,
            walk_forward_months=walk_forward if walk_forward > 0 else 1,
            train_window_days=args.train_window_days,
        )
        engine.dispose()
        return

    if args.cache_only:
        if not args.save_cache:
            print("  ERROR: --cache-only requires --save-cache PATH")
            engine.dispose()
            return
        from backtest.signal_cache import precompute_walk_forward_probs
        _wf_co = walk_forward if walk_forward > 0 else 1
        _bar_stream_co: list[tuple[datetime, str]] = sorted(
            (bar.timestamp, sym)
            for sym in symbols
            for bar in bars_by_symbol.get(sym, [])
        )
        print(f"\n  --cache-only: precomputing walk-forward probs "
              f"(wf={_wf_co}mo, min_train={min_train_days}d, "
              f"window={args.train_window_days or 'expanding'})…", flush=True)
        _prob_map_co = precompute_walk_forward_probs(
            feature_df, _bar_stream_co,
            min_train_days=min_train_days,
            walk_forward_months=_wf_co,
            train_window_days=args.train_window_days,
        )
        _meta_co = _build_cache_meta(
            engine, symbols, min_train_days, _wf_co, args.train_window_days,
        )
        _cache_dir_co = os.path.dirname(args.save_cache)
        if _cache_dir_co:
            os.makedirs(_cache_dir_co, exist_ok=True)
        with open(args.save_cache, "wb") as _f:
            pickle.dump({"prob_map": _prob_map_co, "feature_df": feature_df,
                         "bar_stream": _bar_stream_co, "meta": _meta_co}, _f)
        print(f"  Cache saved -> {args.save_cache} ({len(_prob_map_co)} entries)")
        engine.dispose()
        return

    if args.load_cache:
        from backtest.signal_cache import FusionParams, ReplayAggregator
        print(f"\n  Loading cache from {args.load_cache}…", flush=True)
        with open(args.load_cache, "rb") as _f:
            _cache = pickle.load(_f)
        _prob_map = _cache["prob_map"]
        feature_df = _cache["feature_df"]
        _meta = _cache.get("meta", {})
        _cached_h = _meta.get("label_horizon", MLSignalGenerator.LABEL_HORIZON_BARS)
        _cached_r = _meta.get("label_min_return", MLSignalGenerator.LABEL_MIN_RETURN)
        # The walk-forward schedule (min_train_days/walk_forward_months/
        # train_window_days) is baked into the cache at build time and isn't
        # meant to be overridden at load time (--load-cache mode is for
        # sweeping fusion params like --ml-buy-threshold, not the schedule)
        # — so re-derive the expected key from what the cache SAYS it was
        # built with, not from this invocation's (likely default) args.
        # Symbols and label constants, by contrast, legitimately vary
        # between the build and a later replay and are checked against the
        # CURRENT values — a genuine divergence there is exactly the class
        # of bug this key is meant to catch.
        _expected_key = _compute_cache_key(
            label_horizon=MLSignalGenerator.LABEL_HORIZON_BARS,
            label_min_return=MLSignalGenerator.LABEL_MIN_RETURN,
            min_train_days=_meta.get("min_train_days", min_train_days),
            walk_forward_months=_meta.get("walk_forward_months", 1),
            train_window_days=_meta.get("train_window_days", args.train_window_days),
            symbols=symbols,
            fingerprint=_feature_fingerprint(engine, symbols),
        )
        _ok, _msg = _validate_cache_key(_meta, _expected_key, args.force)
        if _msg:
            print(f"  {_msg}")
        if not _ok:
            engine.dispose()
            sys.exit(1)
        print(f"  Cache: {len(_prob_map)} prob entries, label h={_cached_h}/r={_cached_r}")
        _fusion = FusionParams(
            buy_gate=args.buy_gate,
            sell_gate=args.sell_gate,
            ml_buy_threshold=args.ml_buy_threshold,
            ml_sell_threshold=args.ml_sell_threshold,
            require_uptrend_for_buys=args.require_uptrend,
            guard_bb_dip_buy=args.bb_guard,
        )
        aggregator = ReplayAggregator(feature_df, _prob_map, _fusion)
        walk_forward = 0  # probs already embedded; engine must not retrain

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

    if args.save_cache:
        from backtest.signal_cache import precompute_walk_forward_probs
        _wf_cs = walk_forward if walk_forward > 0 else 1
        _bar_stream_cs: list[tuple[datetime, str]] = sorted(
            (bar.timestamp, sym)
            for sym in symbols
            for bar in bars_by_symbol.get(sym, [])
        )
        print(f"\n  Precomputing probs for cache "
              f"(wf={_wf_cs}mo, min_train={min_train_days}d, "
              f"window={args.train_window_days or 'expanding'})…", flush=True)
        _prob_map_cs = precompute_walk_forward_probs(
            feature_df, _bar_stream_cs,
            min_train_days=min_train_days,
            walk_forward_months=_wf_cs,
            train_window_days=args.train_window_days,
        )
        _meta_cs = _build_cache_meta(
            engine, symbols, min_train_days, _wf_cs, args.train_window_days,
        )
        _cache_dir = os.path.dirname(args.save_cache)
        if _cache_dir:
            os.makedirs(_cache_dir, exist_ok=True)
        with open(args.save_cache, "wb") as _f:
            pickle.dump(
                {"prob_map": _prob_map_cs, "feature_df": feature_df,
                 "bar_stream": _bar_stream_cs, "meta": _meta_cs},
                _f,
            )
        print(f"  Cache saved -> {args.save_cache} ({len(_prob_map_cs)} entries)")

    engine.dispose()


if __name__ == "__main__":
    main()
