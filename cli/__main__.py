"""Unified CLI: python -m cli <subcommand> [args...]

Every subcommand except `features` is a thin subprocess forward to an
existing module's own `python -m` entry point — there is exactly one
implementation of each command's argument parsing and behavior; this
dispatcher adds no logic of its own beyond routing and (for `features`)
mapping env vars for the dbt invocation documented in the README.

Usage:
    python -m cli backfill
    python -m cli backtest --walk-forward 1 --load-cache cache/wf_540d_h8.pkl
    python -m cli sweep --top 15 --output sweep_results.csv
    python -m cli train
    python -m cli trade --dry-run --once
    python -m cli features
    python -m cli dashboard
    python -m cli verify-parity
    python -m cli experiment --id exp008 --ml-buy-threshold 0.55 --max-hold-bars 24
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_SUBCOMMANDS = {
    "backfill": "ingestion.rest_backfill",
    "validate": "ingestion.validate",
    "train": "signals.ml_model",
    "backtest": "backtest.run_backtest",
    "sweep": "backtest.sweep",
    "trade": "execution.bot_runner",
}

_SPECIAL = ("features", "dashboard", "verify-parity", "experiment")


def _print_help() -> None:
    print(__doc__)
    print("Subcommands:")
    for name in sorted(list(_SUBCOMMANDS) + list(_SPECIAL)):
        print(f"  {name}")


def _dbt_env() -> dict:
    """Map POSTGRES_* env vars to the DBT_POSTGRES_* names dbt/profiles.yml
    expects (see README's dbt invocation)."""
    env = os.environ.copy()
    env["DBT_POSTGRES_HOST"] = os.getenv("POSTGRES_HOST", "localhost")
    env["DBT_POSTGRES_PORT"] = os.getenv("POSTGRES_PORT", "5433")
    env["DBT_POSTGRES_USER"] = os.getenv("POSTGRES_USER", "trader")
    env["DBT_POSTGRES_PASSWORD"] = os.getenv("POSTGRES_PASSWORD", "")
    env["DBT_POSTGRES_DB"] = os.getenv("POSTGRES_DB", "trading")
    return env


def _run_features(rest: list[str]) -> int:
    """dbt deps -> dbt run -> dbt test, in dbt/, stopping at the first
    failure — the same three-step chain the README documents by hand."""
    if rest:
        print(f"  NOTE: 'features' takes no extra args; ignoring: {rest}")
    env = _dbt_env()
    for step in ("deps", "run", "test"):
        result = subprocess.run(["dbt", step, "--profiles-dir", "."], cwd="dbt", env=env)
        if result.returncode != 0:
            return result.returncode
    return 0


def _parse_experiment_args(rest: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cli experiment",
        description="Resolve/build the prob cache, replay, write results/<id>/ artifacts.",
    )
    parser.add_argument("--id", required=True, dest="exp_id")
    parser.add_argument("--symbols", type=str, default=None,
                         help="Comma-separated symbols (default: TRADING_SYMBOLS env)")
    parser.add_argument("--buy-gate", type=float, default=0.15)
    parser.add_argument("--sell-gate", type=float, default=-0.15)
    parser.add_argument("--ml-buy-threshold", type=float, default=0.65)
    parser.add_argument("--ml-sell-threshold", type=float, default=0.35)
    parser.add_argument("--bb-guard", action="store_true", default=False)
    parser.add_argument("--require-uptrend", action="store_true", default=False)
    parser.add_argument("--max-hold-bars", type=int, default=0)
    parser.add_argument("--min-train-days", type=int, default=60)
    parser.add_argument("--walk-forward", type=int, default=1, dest="walk_forward_months")
    parser.add_argument("--train-window-days", type=int, default=540)
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args(rest)


def _run_experiment(rest: list[str]) -> int:
    from backtest.experiment import ExperimentConfig, run_experiment
    from backtest.run_backtest import build_engine

    args = _parse_experiment_args(rest)
    engine = build_engine()
    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else
        [s.strip() for s in os.getenv("TRADING_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",") if s.strip()]
    )
    cfg = ExperimentConfig(
        exp_id=args.exp_id, symbols=symbols,
        buy_gate=args.buy_gate, sell_gate=args.sell_gate,
        ml_buy_threshold=args.ml_buy_threshold, ml_sell_threshold=args.ml_sell_threshold,
        bb_guard=args.bb_guard, require_uptrend=args.require_uptrend,
        max_hold_bars=args.max_hold_bars, min_train_days=args.min_train_days,
        walk_forward_months=args.walk_forward_months, train_window_days=args.train_window_days,
        output_dir=args.output_dir,
    )

    metrics = run_experiment(engine, cfg)
    engine.dispose()

    print(f"\n=== {args.exp_id} ===")
    print(f"  Trades         : {metrics['trade_count']}")
    print(f"  Win rate       : {metrics['win_rate']:.2%}")
    print(f"  Profit factor  : {metrics['profit_factor']:.2f}")
    print(f"  Expectancy     : ${metrics['expectancy']:.2f}/trade")
    print(f"  Max drawdown   : {metrics['max_drawdown']:.2%}")
    print(f"  Sharpe ratio   : {metrics['sharpe_ratio']:.2f}")
    print(f"  Artifacts      : {metrics['_output_dir']}/")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if not argv or argv[0] in ("-h", "--help"):
        _print_help()
        return 0 if argv else 1

    subcommand, rest = argv[0], argv[1:]

    if subcommand == "features":
        return _run_features(rest)
    if subcommand == "experiment":
        return _run_experiment(rest)
    if subcommand == "dashboard":
        result = subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])
        return result.returncode
    if subcommand == "verify-parity":
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/test_backtest_golden_parity.py", "-v", "--no-cov", *rest,
        ])
        return result.returncode

    module = _SUBCOMMANDS.get(subcommand)
    if module is None:
        print(f"Unknown subcommand: {subcommand}")
        _print_help()
        return 1

    result = subprocess.run([sys.executable, "-m", module, *rest])
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
