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
"""
from __future__ import annotations

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

_SPECIAL = ("features", "dashboard", "verify-parity")


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


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if not argv or argv[0] in ("-h", "--help"):
        _print_help()
        return 0 if argv else 1

    subcommand, rest = argv[0], argv[1:]

    if subcommand == "features":
        return _run_features(rest)
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
