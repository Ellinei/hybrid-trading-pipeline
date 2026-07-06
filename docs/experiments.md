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
| exp002 | 2026-07-07 | 33276fe | prob-stats diagnostic: `--prob-stats --walk-forward 1 --min-train-days 60 --train-window-days 540 --label-horizon 1 --label-min-return 0.0` | same 3y mainnet | — | — | — | — | — | — | — | **Diagnostic only — no P&L.** ALL: n=74520, max=0.839, mean=0.507. >=0.65: 1.1%, >=0.60: 4.7%, >=0.55: 20.6%, >=0.50: 56.1%. Root cause confirmed: horizon=1/min_return=0.0 is ~coin-flip; model gets better-calibrated with more data so probs compress toward 0.5. Max prob_up declines from ~0.84 (2023-09) to ~0.54-0.64 (2026-04+). Dead zone after 2024-08 is threshold=0.65 issue, not model failure. Fix: lower ml_buy_threshold to 0.55 (exp003). |
| exp003 | 2026-07-07 | 33276fe | same as exp001 + --ml-buy-threshold 0.55. cmd: `python -m backtest.run_backtest --walk-forward 1 --min-train-days 60 --train-window-days 540 --label-horizon 1 --label-min-return 0.0 --ml-buy-threshold 0.55 --output results/trades_exp003.csv` | same 3y mainnet | 305 | 0.82 | -$3.39 | 58.4% | 11.61% | -1.10 | 6/11 traded (35 windows) | **Not an improvement.** Exit mix: signal_close=289, stop_loss=16. Threshold fix generates trades in first 11 windows only — circuit-breaker fires 2024-07-25 (-$193 ETHUSDT stop-loss, DD=10.3%) and kills the remaining 24 windows. Root cause: **label mismatch** — horizon=1/min_return=0.0 is near-coinflip so model confidence is low; stop-losses are sized for 8h holds but signals fire at 1h cadence. Fix: revert to default label (h=8, r=0.003) which matches the actual 7h hold time (exp004). |
| exp004 | — | 33276fe | default label (h=8, r=0.003) + --ml-buy-threshold 0.55. cmd: `python -m backtest.run_backtest --walk-forward 1 --min-train-days 60 --train-window-days 540 --ml-buy-threshold 0.55 --output results/trades_exp004.csv` | same 3y mainnet | — | — | — | — | — | — | — | Hypothesis: default label is calibrated for 7h hold time so risk params (1.5×ATR SL) are consistent; threshold=0.55 activates ML across all windows. Run --prob-stats first to confirm prob distribution has enough mass above 0.55 with h=8 label. |
