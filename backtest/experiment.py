"""`cli experiment`: one command from args -> results/<exp_id>/ artifacts.

Resolves (building if needed) a content-addressed walk-forward prob cache,
replays it via ReplayAggregator, and writes a self-contained results/
<exp_id>/ artifact set (trades, equity curve, metrics, resolved config) —
the "one command, minutes" experiment flow instead of run_backtest's
build-a-cache-then-manually-inspect-stdout workflow.

This is intentionally a NEW, additive entry point: `python -m cli backtest`
(= run_backtest.py, unchanged) keeps its existing explicit --load-cache/
--save-cache/--cache-only behavior for backward compatibility and
fine-grained control; `cli experiment` is the higher-level convenience
layer with auto-caching as its default, since it has no prior behavior it
needs to stay compatible with.
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from backtest.engine import BacktestConfig, _load_all_features, run_backtest
from backtest.metrics import (
    exit_reason_breakdown,
    expectancy,
    holding_hours_stats,
    max_single_trade_profit_share,
    per_symbol_metrics,
)
from backtest.run_backtest import (
    _build_cache_meta,
    _compute_cache_key,
    _feature_fingerprint,
    _load_atr,
    _load_bars,
)
from backtest.signal_cache import FusionParams, ReplayAggregator, precompute_walk_forward_probs
from execution.exchange_filters import parse_symbol_filters
from execution.risk_manager import RiskConfig
from ingestion.binance_client import BinanceClientWrapper
from signals.ml_model import MLSignalGenerator


@dataclass
class ExperimentConfig:
    exp_id: str
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    buy_gate: float = 0.15
    sell_gate: float = -0.15
    ml_buy_threshold: float = 0.65
    ml_sell_threshold: float = 0.35
    bb_guard: bool = False
    require_uptrend: bool = False
    max_hold_bars: int = 0
    min_train_days: int = 60
    walk_forward_months: int = 1
    train_window_days: int = 540
    output_dir: str = "results"


def _git_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--short"], capture_output=True, text=True,
        ).stdout.strip()
        return f"{sha} (uncommitted WT)" if dirty else sha
    except Exception:
        return "unknown"


def _cache_path_for_key(key: str) -> str:
    return os.path.join("cache", f"probs_{key[:16]}.pkl")


def resolve_prob_cache(
    engine, cfg: ExperimentConfig, bars_by_symbol: dict,
) -> tuple[dict, object, str]:
    """Return (prob_map, feature_df, cache_path) for *cfg* — reusing an
    existing content-addressed cache file when one already matches this
    exact (symbols, labels, schedule, feature-data) combination, else
    building and saving a new one. Every distinct experiment config gets
    its own auto-named cache; no manual --save-cache/--load-cache
    bookkeeping required.
    """
    fingerprint = _feature_fingerprint(engine, cfg.symbols)
    key = _compute_cache_key(
        label_horizon=MLSignalGenerator.LABEL_HORIZON_BARS,
        label_min_return=MLSignalGenerator.LABEL_MIN_RETURN,
        min_train_days=cfg.min_train_days,
        walk_forward_months=cfg.walk_forward_months,
        train_window_days=cfg.train_window_days,
        symbols=cfg.symbols,
        fingerprint=fingerprint,
    )
    cache_path = _cache_path_for_key(key)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        return cache["prob_map"], cache["feature_df"], cache_path

    feature_df = _load_all_features(engine, cfg.symbols)
    bar_stream = sorted(
        (bar.timestamp, sym)
        for sym in cfg.symbols
        for bar in bars_by_symbol.get(sym, [])
    )
    prob_map = precompute_walk_forward_probs(
        feature_df, bar_stream,
        min_train_days=cfg.min_train_days,
        walk_forward_months=cfg.walk_forward_months,
        train_window_days=cfg.train_window_days,
    )
    meta = _build_cache_meta(
        engine, cfg.symbols, cfg.min_train_days, cfg.walk_forward_months, cfg.train_window_days,
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(
            {"prob_map": prob_map, "feature_df": feature_df, "bar_stream": bar_stream, "meta": meta},
            f,
        )
    return prob_map, feature_df, cache_path


def _compute_metrics(result) -> dict:
    return {
        "trade_count": result.metrics["trade_count"],
        "win_rate": result.metrics["win_rate"],
        "profit_factor": result.metrics["profit_factor"],
        "expectancy": expectancy(result.trades),
        "max_drawdown": result.metrics["max_drawdown"],
        "sharpe_ratio": result.metrics["sharpe_ratio"],
        "total_return": result.metrics["total_return"],
        "exit_reason_breakdown": exit_reason_breakdown(result.trades),
        "holding_hours": holding_hours_stats(result.trades),
        "per_symbol": per_symbol_metrics(result.trades),
        "max_single_trade_profit_share": max_single_trade_profit_share(result.trades),
    }


def _build_ledger_row(cfg: ExperimentConfig, metrics: dict, git_sha: str, data_range: str) -> str:
    """A single markdown table row matching docs/experiments.md's exact
    column format, ready to paste — but never auto-appended: the ledger's
    value is the human verdict/notes column, and acceptance (LOSO,
    holdout, concentration) needs a judgment call this function can't make.
    """
    pf = metrics["profit_factor"]
    pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (
        f"| {cfg.exp_id} | {date_str} | {git_sha} | <!-- config delta vs baseline --> "
        f"| {data_range} | {metrics['trade_count']} | {pf_str} | "
        f"{metrics['expectancy']:+.2f} | {metrics['win_rate']:.1%} | "
        f"{metrics['max_drawdown']:.2%} | {metrics['sharpe_ratio']:.2f} | "
        f"— (cache replay, see report.html for monthly breakdown) | <!-- notes --> |"
    )


def _data_range(equity_curve) -> str:
    if not equity_curve:
        return "—"
    start = equity_curve[0][0].strftime("%Y-%m-%d")
    end = equity_curve[-1][0].strftime("%Y-%m-%d")
    return f"{start}..{end}"


def _write_artifacts(cfg: ExperimentConfig, result, cache_path: str) -> str:
    out_dir = os.path.join(cfg.output_dir, cfg.exp_id)
    os.makedirs(out_dir, exist_ok=True)

    if result.trades:
        with open(os.path.join(out_dir, "trades.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result.trades[0].keys()))
            writer.writeheader()
            writer.writerows(result.trades)

    with open(os.path.join(out_dir, "equity_curve.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "equity"])
        writer.writerows(result.equity_curve)

    metrics = _compute_metrics(result)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    git_sha = _git_sha()
    config_dump = asdict(cfg)
    config_dump["git_sha"] = git_sha
    config_dump["cache_path"] = cache_path
    config_dump["generated_at"] = datetime.now(timezone.utc).isoformat()
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_dump, f, indent=2)

    from backtest.report import build_report_html
    html = build_report_html(cfg.exp_id, metrics, result.trades, result.equity_curve)
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)

    ledger_row = _build_ledger_row(cfg, metrics, git_sha, _data_range(result.equity_curve))
    with open(os.path.join(out_dir, "ledger_row.md"), "w", encoding="utf-8") as f:
        f.write(ledger_row + "\n")

    return out_dir


def run_experiment(engine, cfg: ExperimentConfig) -> dict:
    """Resolve/build the prob cache, replay it, write results/<exp_id>/,
    and return the metrics dict (with an extra "_output_dir" key)."""
    client = BinanceClientWrapper()
    bars_by_symbol, atr_by_symbol, filters_by_symbol = {}, {}, {}
    for symbol in cfg.symbols:
        bars_by_symbol[symbol] = _load_bars(engine, symbol, None, None)
        atr_by_symbol[symbol] = _load_atr(engine, symbol, None, None)
        try:
            filters_by_symbol[symbol] = parse_symbol_filters(client.get_symbol_info_cached(symbol))
        except Exception:
            filters_by_symbol[symbol] = None

    prob_map, feature_df, cache_path = resolve_prob_cache(engine, cfg, bars_by_symbol)

    fusion = FusionParams(
        buy_gate=cfg.buy_gate,
        sell_gate=cfg.sell_gate,
        ml_buy_threshold=cfg.ml_buy_threshold,
        ml_sell_threshold=cfg.ml_sell_threshold,
        require_uptrend_for_buys=cfg.require_uptrend,
        guard_bb_dip_buy=cfg.bb_guard,
    )
    aggregator = ReplayAggregator(feature_df, prob_map, fusion)

    backtest_config = BacktestConfig(
        symbols=cfg.symbols,
        start=datetime(1970, 1, 1, tzinfo=timezone.utc),
        end=datetime.now(timezone.utc),
        risk_config=RiskConfig(max_hold_bars=cfg.max_hold_bars),
    )
    result = run_backtest(
        bars_by_symbol=bars_by_symbol,
        atr_by_symbol=atr_by_symbol,
        filters_by_symbol=filters_by_symbol,
        aggregator=aggregator,
        config=backtest_config,
        walk_forward_months=0,
    )

    out_dir = _write_artifacts(cfg, result, cache_path)
    metrics = _compute_metrics(result)
    metrics["_output_dir"] = out_dir
    return metrics
