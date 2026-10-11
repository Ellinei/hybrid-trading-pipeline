"""Read-only operations and experiment data, independent of Streamlit."""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path


def read_operations(path, symbol=None, mode=None, limit=200):
    """Missing stores/tables are empty states; never create or initialize an account."""
    path = Path(path)
    result = {"decisions": [], "account": [], "positions": [], "closed": []}
    if not path.is_file():
        return result
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "decisions" in tables:
            clauses, params = [], []
            for column, value in (("symbol", symbol), ("mode", mode)):
                if value:
                    clauses.append(f"{column} = ?")
                    params.append(value)
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            result["decisions"] = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM decisions" + where + " ORDER BY id DESC LIMIT ?",
                    [*params, min(max(int(limit), 1), 1000)],
                )
            ]
        if "paper_account" in tables:
            result["account"] = [dict(r) for r in conn.execute("SELECT * FROM paper_account")]
        if "paper_trades" in tables:
            result["positions"] = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM paper_trades WHERE status = 'open' ORDER BY id DESC"
                )
            ]
            result["closed"] = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM paper_trades WHERE status = 'closed' ORDER BY id DESC LIMIT 200"
                )
            ]
        return result
    finally:
        conn.close()


def load_experiments(root):
    """Load artifacts produced by backtest.experiment; report corrupt runs separately."""
    runs, warnings = [], []
    for metrics_path in sorted(Path(root).glob("*/metrics.json")):
        folder = metrics_path.parent
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            if not isinstance(metrics, dict) or not isinstance(config, dict):
                raise ValueError("metrics/config must be objects")
            symbols = config.get("symbols", [])
            per_symbol = metrics.get("per_symbol", {})
            if not isinstance(symbols, list) or not all(isinstance(s, str) for s in symbols):
                raise ValueError("symbols must be a list of strings")
            if not isinstance(per_symbol, dict) or not all(
                isinstance(value, dict) for value in per_symbol.values()
            ):
                raise ValueError("per_symbol must contain metric objects")
            for metric_set in (metrics, *per_symbol.values()):
                for key in (
                    "trade_count",
                    "total_return",
                    "max_drawdown",
                    "sharpe_ratio",
                    "profit_factor",
                    "win_rate",
                    "pnl",
                ):
                    value = metric_set.get(key)
                    if value is not None and (
                        isinstance(value, bool) or not isinstance(value, int | float)
                    ):
                        raise ValueError("metrics must be numeric")
            start = end = None
            with (folder / "equity_curve.csv").open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    timestamp = datetime.fromisoformat(row["timestamp"])
                    start = min(start, timestamp) if start else timestamp
                    end = max(end, timestamp) if end else timestamp
            # Declared scoring bounds remain meaningful for sparse/no-trade runs.
            # Older artifacts lack them, so retain their curve-derived fallback.
            if config.get("start") is not None:
                start = datetime.fromisoformat(config["start"])
            if config.get("end") is not None:
                end = datetime.fromisoformat(config["end"])
            runs.append(
                {
                    "id": folder.name,
                    "metrics": metrics,
                    "config": config,
                    "start": start.isoformat() if start else None,
                    "end": end.isoformat() if end else None,
                }
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            warnings.append(
                f"{folder.name}: incomplete or invalid artifacts ({type(exc).__name__})"
            )
    return runs, warnings


def comparison_rows(runs, symbol=None):
    rows = []
    for run in runs:
        metrics = run["metrics"]
        if symbol:
            metrics = metrics.get("per_symbol", {}).get(symbol)
            if metrics is None:
                continue
        keys = (
            ("trade_count", "win_rate", "profit_factor", "pnl")
            if symbol
            else ("trade_count", "total_return", "max_drawdown", "sharpe_ratio", "profit_factor")
        )
        rows.append(
            {
                "experiment": run["id"],
                "start": run["start"],
                "end": run["end"],
                "symbols": ", ".join(run["config"].get("symbols", [])),
                **{key: metrics.get(key) for key in keys},
            }
        )
    return rows
