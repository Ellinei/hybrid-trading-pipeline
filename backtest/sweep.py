"""Grid-search over RiskConfig parameters using the existing backtest engine.

Performance design:
  Signals for a given (symbol, timestamp) are deterministic — they don't
  depend on RiskConfig. A naive loop of 192 run_backtest() calls would query
  the DB ~1.44M times (192 × 7,488 unique bars). Instead:
  1. PRE-COMPUTE: one pass through all bars to generate and cache every signal.
  2. SWEEP: 192 pure in-memory replays through a CachingAggregator that
     returns cached signals without touching the DB.
  Net queries: ~7,500 (1×) instead of ~1.44M (192×).

Usage:
    python -m backtest.sweep
    python -m backtest.sweep --symbols BTCUSDT,ETHUSDT --start 2026-04-01 --end 2026-06-01
    python -m backtest.sweep --top 20 --output sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
from datetime import UTC, datetime
from urllib.parse import quote_plus

import structlog
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from backtest.engine import BacktestConfig, run_backtest
from backtest.exit_simulation import OHLCVBar
from execution.exchange_filters import parse_symbol_filters
from execution.risk_manager import RiskConfig
from ingestion.binance_client import BinanceClientWrapper
from signals.aggregator import SignalAggregator
from signals.base import AggregatedSignal

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Parameter grid — each key maps to a list of values to sweep.
# 4 × 4 × 4 × 3 = 192 combinations; each replay is pure in-memory.
# ---------------------------------------------------------------------------
PARAM_GRID: dict[str, list] = {
    "stop_loss_atr_mult":   [1.0, 1.5, 2.0, 2.5],
    "take_profit_atr_mult": [2.0, 3.0, 4.0, 5.0],
    "min_confidence":       [0.55, 0.60, 0.65, 0.70],
    "max_position_pct":     [0.01, 0.02, 0.05],
}

# Fine-grained grid for --fine mode.
# 4 × 2 × 3 × 3 = 72 combinations — explores below the current best SL=1.0×ATR
# and tightens the range around best-performing conf>=0.60 and smaller position sizes.
FINE_PARAM_GRID: dict[str, list] = {
    "stop_loss_atr_mult":   [0.5, 0.75, 1.0, 1.25],
    "take_profit_atr_mult": [2.0, 3.0],
    "min_confidence":       [0.58, 0.60, 0.62],
    "max_position_pct":     [0.005, 0.01, 0.015],
}


def _grid_configs(grid: dict[str, list] | None = None) -> list[RiskConfig]:
    """Return one RiskConfig per combination in the given grid (default PARAM_GRID)."""
    g = grid if grid is not None else PARAM_GRID
    keys = list(g.keys())
    combos = itertools.product(*[g[k] for k in keys])
    return [RiskConfig(**dict(zip(keys, combo, strict=False))) for combo in combos]


# ---------------------------------------------------------------------------
# Caching aggregator — DB queries happen once during pre-compute phase only
# ---------------------------------------------------------------------------

class _CachingAggregator:
    """Wraps SignalAggregator; returns cached signals on repeated lookups.

    Phase 1 (pre-compute): call populate() once to prime the cache.
    Phase 2 (sweep):       aggregate() returns from the cache, zero DB I/O.
    """

    def __init__(self, real_aggregator: SignalAggregator) -> None:
        self._real = real_aggregator
        self._cache: dict[tuple, AggregatedSignal] = {}

    def populate(
        self,
        bars_by_symbol: dict[str, list[OHLCVBar]],
    ) -> None:
        """Generate and cache every signal needed by the sweep."""
        all_bars = sorted(
            ((bar.timestamp, symbol, bar)
             for symbol, bars in bars_by_symbol.items()
             for bar in bars),
            key=lambda x: x[0],
        )
        total = len(all_bars)
        print(f"\nPre-computing signals for {total} bars…", flush=True)
        for idx, (ts, symbol, _bar) in enumerate(all_bars, 1):
            key = (symbol, ts)
            if key not in self._cache:
                self._cache[key] = self._real.aggregate(symbol, as_of=ts)
            if idx % 500 == 0 or idx == total:
                pct = idx / total * 100
                print(f"  {idx}/{total} ({pct:.0f}%)", end="\r", flush=True)
        print(flush=True)
        print(f"  Cached {len(self._cache)} unique (symbol, timestamp) signals.", flush=True)

    def aggregate(self, symbol: str, as_of: datetime | None = None) -> AggregatedSignal:
        key = (symbol, as_of)
        if key in self._cache:
            return self._cache[key]
        # fallback for any key not in cache (shouldn't happen in sweep)
        return self._real.aggregate(symbol, as_of=as_of)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_bars(engine, symbol: str, start, end) -> list[OHLCVBar]:
    clauses = ["symbol = :symbol"]
    params: dict = {"symbol": symbol}
    if start is not None:
        clauses.append("timestamp >= :start")
        params["start"] = start
    if end is not None:
        clauses.append("timestamp <= :end")
        params["end"] = end
    query = text(
        f"SELECT timestamp, open, high, low, close FROM raw.ohlcv"
        f" WHERE {' AND '.join(clauses)} ORDER BY timestamp ASC"
    )
    with engine.connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [
        OHLCVBar(
            timestamp=row.timestamp, open=float(row.open), high=float(row.high),
            low=float(row.low), close=float(row.close),
        )
        for row in rows
    ]


def _load_atr(engine, symbol: str, start, end) -> dict[datetime, float]:
    clauses = ["symbol = :symbol"]
    params: dict = {"symbol": symbol}
    if start is not None:
        clauses.append("timestamp >= :start")
        params["start"] = start
    if end is not None:
        clauses.append("timestamp <= :end")
        params["end"] = end
    query = text(
        f"SELECT timestamp, range_14 FROM features.feat_technical"
        f" WHERE {' AND '.join(clauses)} ORDER BY timestamp ASC"
    )
    with engine.connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return {row.timestamp: float(row.range_14) for row in rows if row.range_14 is not None}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _row(rank: int, rc: RiskConfig, m: dict) -> dict:
    return {
        "rank":                rank,
        "profit_factor":       round(m["profit_factor"], 4),
        "total_return":        round(m["total_return"],  4),
        "sharpe_ratio":        round(m["sharpe_ratio"],  4),
        "win_rate":            round(m["win_rate"],       4),
        "max_drawdown":        round(m["max_drawdown"],   4),
        "trade_count":         m["trade_count"],
        "stop_loss_atr_mult":  rc.stop_loss_atr_mult,
        "take_profit_atr_mult": rc.take_profit_atr_mult,
        "min_confidence":      rc.min_confidence,
        "max_position_pct":    rc.max_position_pct,
    }


def _print_table(rows: list[dict], top: int) -> None:
    shown = rows[:top]
    if not shown:
        print("No results to display.")
        return
    headers = list(shown[0].keys())
    widths = {h: max(len(h), max(len(str(r[h])) for r in shown)) for h in headers}
    sep = "  ".join("-" * widths[h] for h in headers)
    header_line = "  ".join(h.ljust(widths[h]) for h in headers)

    print(f"\n=== Sweep Results (top {len(shown)} of {len(rows)} configs) ===")
    print(header_line)
    print(sep)
    for r in shown:
        print("  ".join(str(r[h]).ljust(widths[h]) for h in headers))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-sweep RiskConfig parameters")
    parser.add_argument("--symbols", type=str, default=None)
    parser.add_argument("--start",   type=str, default=None)
    parser.add_argument("--end",     type=str, default=None)
    parser.add_argument("--top",     type=int, default=10,
                        help="Number of top configs to display (default: 10)")
    parser.add_argument("--output",  type=str, default=None,
                        help="Optional CSV path for full sweep results")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Show per-bar signal logs (very noisy)")
    parser.add_argument("--fine", action="store_true", default=False,
                        help="Use fine-grained 72-combo grid (vs 192 default); "
                             "outputs to sweep_fine.csv by default")
    args = parser.parse_args()

    import logging
    import warnings
    warnings.filterwarnings("ignore")   # sklearn/xgboost pickle version noise
    if not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)
        import structlog as _sl
        _sl.configure(wrapper_class=_sl.make_filtering_bound_logger(logging.WARNING))

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
        [
            s.strip()
            for s in os.getenv("TRADING_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
            if s.strip()
        ]
    )
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC) if args.start else None
    end   = datetime.fromisoformat(args.end  ).replace(tzinfo=UTC) if args.end   else None

    # 1. Load raw market data (bars, ATR, exchange filters) — one trip to DB per symbol
    print(f"Loading market data for {symbols}…", flush=True)
    client = BinanceClientWrapper()
    bars_by_symbol: dict = {}
    atr_by_symbol:  dict = {}
    filters_by_symbol: dict = {}
    for symbol in symbols:
        bars_by_symbol[symbol] = _load_bars(engine, symbol, start, end)
        atr_by_symbol[symbol]  = _load_atr(engine,  symbol, start, end)
        try:
            filters_by_symbol[symbol] = parse_symbol_filters(
                client.get_symbol_info_cached(symbol)
            )
        except Exception as exc:
            log.warning("symbol_filters_unavailable", symbol=symbol, error=str(exc))
            filters_by_symbol[symbol] = None
        n = len(bars_by_symbol[symbol])
        print(f"  {symbol}: {n} bars", flush=True)
        if not n:
            print(f"  WARNING: no OHLCV data for {symbol} — it will be skipped.", flush=True)

    # 2. Pre-compute signals ONCE (the expensive DB work — done here, not 192×)
    caching_aggregator = _CachingAggregator(SignalAggregator(engine))
    caching_aggregator.populate(bars_by_symbol)

    # 3. Sweep configs in pure memory (no DB I/O per iteration)
    configs = _grid_configs(FINE_PARAM_GRID if args.fine else None)
    total = len(configs)
    print(f"\nRunning {total} in-memory replays…", flush=True)

    rows: list[dict] = []
    for i, rc in enumerate(configs, 1):
        cfg = BacktestConfig(
            symbols=symbols,
            start=start or datetime(1970, 1, 1, tzinfo=UTC),
            end=end   or datetime.now(UTC),
            risk_config=rc,
        )
        result = run_backtest(
            bars_by_symbol=bars_by_symbol,
            atr_by_symbol=atr_by_symbol,
            filters_by_symbol=filters_by_symbol,
            aggregator=caching_aggregator,
            config=cfg,
        )
        rows.append(_row(i, rc, result.metrics))
        if i % 20 == 0 or i == total:
            pct = i / total * 100
            print(f"  {i}/{total} ({pct:.0f}%)", end="\r", flush=True)

    print(flush=True)

    # 4. Rank and report
    rows.sort(key=lambda r: (r["profit_factor"], r["sharpe_ratio"]), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    _print_table(rows, args.top)

    baseline_pf = next(
        (r["profit_factor"] for r in rows
         if r["stop_loss_atr_mult"] == 1.5
         and r["take_profit_atr_mult"] == 3.0
         and r["min_confidence"] == 0.55
         and r["max_position_pct"] == 0.02),
        None,
    )
    if baseline_pf is not None:
        print(f"  Baseline (current defaults) profit_factor: {baseline_pf:.4f}")

    profitable = [r for r in rows if r["profit_factor"] > 1.0]
    print(f"  Configs with profit_factor > 1.0: {len(profitable)} / {total}")
    if profitable:
        best = rows[0]
        print(
            f"  Best config: SL={best['stop_loss_atr_mult']}*ATR  "
            f"TP={best['take_profit_atr_mult']}*ATR  "
            f"conf>={best['min_confidence']}  "
            f"size={best['max_position_pct']:.0%}  "
            f"-> PF {best['profit_factor']:.2f}, "
            f"return {best['total_return']:.2%}, "
            f"Sharpe {best['sharpe_ratio']:.2f}"
        )

    output_path = args.output or ("sweep_fine.csv" if args.fine else None)
    if output_path:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Full results written to {output_path}")

    engine.dispose()


if __name__ == "__main__":
    main()
