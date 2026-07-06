# Experiment Ledger

Canonical run (the only numbers that count):

```
python -m backtest.run_backtest --walk-forward 1 --min-train-days 60 [--start/--end] --output results/trades_<exp_id>.csv
```

Acceptance rules:
- Quoted results need >=30 OOS trades (target >=100 on multi-year data).
- "Improvement" = PF >= +0.05 vs current baseline AND no single trade >40% of gross profit.
- Final 6 months of data = holdout — excluded from all tuning via `--end`, spent once at the final gate.
- Winners must pass leave-one-symbol-out: PF > 1.0 on >=2 of 3 symbols.
- Win rate, exit-reason mix, hold-time stats are diagnostics, never optimization targets.

| exp_id | date | git SHA | config delta | data range | OOS trades | PF | expectancy | win rate | max DD | Sharpe | % windows PF>1 | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp000 | 2026-07-05 | 8a1cb18 | none (baseline, defaults) | 2026-04-01..2026-07-03 testnet 1h, 3 symbols (~1,319 bars/sym, gappy) | 60 | 0.51 | -$18.21 | 55.0% | 10.87% | -4.51 | 0/1 traded windows | Exit mix: signal_close=48, stop_loss=11, take_profit=1. Hold: med 7.0h / mean 8.2h / p90 15.0h — supports k=8 label horizon. Strategy loses money OOS. |
| exp001 | 2026-07-06 | 8a1cb18 (uncommitted WT) | legacy strategy on 3y mainnet: --label-horizon 1 --label-min-return 0.0 --train-window-days 540 | 2023-07-05..2026-07-04 mainnet 1h, 3 symbols (26,280 bars/sym, gapless) | 238 | 0.73 | -$4.80 | 47.5% | 12.06% | -1.29 | 4/13 traded (35 windows) | NEW BASELINE TO BEAT. Exit mix: signal_close=230, stop_loss=8, TP never. Hold: med 5.0h / mean 5.4h / p90 9.0h. **ZERO trades after 2024-08 — 22 consecutive empty windows**: once the training window matures, ML prob_up>0.65 apparently never fires, so no entry clears the stacked gates; strategy self-disables for 2 years of data. |
| exp002 | — | a9db0ca | prob-stats diagnostic: same label as exp001, walk-forward schedule to match. cmd: `python -m backtest.run_backtest --prob-stats --walk-forward 1 --min-train-days 60 --train-window-days 540 --label-horizon 1 --label-min-return 0.0` | same 3y mainnet | — | — | — | — | — | — | — | **Diagnostic only** — no P&L. Shows per-month max/pct_above_0.65/0.55/0.50 for ML prob_up. Goal: confirm threshold 0.65 explains dead zone; pick threshold for exp003. Fill in results after running. |
