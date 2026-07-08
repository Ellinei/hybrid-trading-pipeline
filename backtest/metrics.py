"""Pure performance metrics over a trade log + equity curve."""
from __future__ import annotations

import math
from datetime import datetime


def win_rate(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t["pnl"] > 0)
    return wins / len(trades)


def profit_factor(trades: list[dict]) -> float:
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = -sum(t["pnl"] for t in trades if t["pnl"] < 0)
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def max_drawdown(equity_curve: list[tuple[datetime, float]]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0][1]
    worst = 0.0
    for _, equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            worst = max(worst, (peak - equity) / peak)
    return worst


def total_return(equity_curve: list[tuple[datetime, float]]) -> float:
    if len(equity_curve) < 2:
        return 0.0
    start = equity_curve[0][1]
    end = equity_curve[-1][1]
    if start == 0:
        return 0.0
    return (end - start) / start


def expectancy(trades: list[dict]) -> float:
    """Mean net PnL per trade (the per-trade edge after fees/slippage)."""
    if not trades:
        return 0.0
    return sum(t["pnl"] for t in trades) / len(trades)


def exit_reason_breakdown(trades: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in trades:
        reason = t.get("reason", "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def holding_hours_stats(trades: list[dict]) -> dict[str, float]:
    """Median/mean/p90 of holding time in hours (nearest-rank percentile)."""
    hours = sorted(
        (t["exit_time"] - t["entry_time"]).total_seconds() / 3600.0
        for t in trades
        if t.get("exit_time") is not None and t.get("entry_time") is not None
    )
    if not hours:
        return {"median": 0.0, "mean": 0.0, "p90": 0.0}
    n = len(hours)
    return {
        "median": hours[(n - 1) // 2],
        "mean": sum(hours) / n,
        "p90": hours[max(0, math.ceil(0.9 * n) - 1)],
    }


def per_window_metrics(
    trades: list[dict], boundaries: list[datetime]
) -> list[dict]:
    """Bucket trades into retrain windows and score each independently.

    *boundaries* are the walk-forward retrain timestamps; window i spans
    [boundaries[i], boundaries[i+1]) and the last window is open-ended.
    Trades are attributed by entry_time (the model that opened them).
    Trades entered before the first boundary (warm-up) are excluded.
    """
    if not boundaries:
        return []
    ordered = sorted(boundaries)
    windows = []
    for i, start in enumerate(ordered):
        end = ordered[i + 1] if i + 1 < len(ordered) else None
        bucket = [
            t for t in trades
            if t["entry_time"] >= start and (end is None or t["entry_time"] < end)
        ]
        windows.append({
            "window_start": start,
            "window_end": end,
            "trade_count": len(bucket),
            "pnl": sum(t["pnl"] for t in bucket),
            "profit_factor": profit_factor(bucket),
            "win_rate": win_rate(bucket),
        })
    return windows


def per_symbol_metrics(trades: list[dict]) -> dict[str, dict]:
    """PF/win_rate/trade_count/pnl per symbol — the LOSO breakdown used
    throughout the experiment ledger to check a winner isn't propped up by
    a single symbol."""
    by_symbol: dict[str, list[dict]] = {}
    for t in trades:
        by_symbol.setdefault(t["symbol"], []).append(t)
    return {
        symbol: {
            "trade_count": len(ts),
            "profit_factor": profit_factor(ts),
            "win_rate": win_rate(ts),
            "pnl": sum(t["pnl"] for t in ts),
        }
        for symbol, ts in by_symbol.items()
    }


def max_single_trade_profit_share(trades: list[dict]) -> float:
    """Largest single winning trade's share of gross profit — the
    acceptance rule's '>40% of gross profit from one trade' concentration
    guard, i.e. is a passing PF actually one lucky trade carrying the rest."""
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    gross_profit = sum(wins)
    if gross_profit <= 0:
        return 0.0
    return max(wins) / gross_profit


def sharpe_ratio(
    equity_curve: list[tuple[datetime, float]], periods_per_year: float
) -> float:
    """Annualized Sharpe ratio (zero risk-free rate) over per-period returns."""
    if len(equity_curve) < 2:
        return 0.0

    returns = []
    for (_, prev), (_, curr) in zip(equity_curve, equity_curve[1:]):
        if prev != 0:
            returns.append((curr - prev) / prev)

    if len(returns) < 2:
        return 0.0

    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0

    return (mean / std) * math.sqrt(periods_per_year)
