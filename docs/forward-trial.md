# Forward-validation and paper-trial protocol

Status: registered on 2026-09-23; observation has not started. Software checks
and historical replay are prerequisites, not evidence of a completed trial.

## Frozen candidate

- Symbols: BTCUSDT, ETHUSDT, SOLUSDT; public mainnet candles and paper quotes.
- Technical/ML/sentiment weights: 0.40/0.40/0.20; sentiment neutral/disabled.
- Fused BUY/SELL gates: 0.15 / -0.15; ML thresholds: 0.55 / 0.35.
- Time stop: 24 hourly bars. Other risk settings are the resolved `RiskConfig`
  written into the decision journal; archive them at activation.
- Regime: R1 BTC close versus 200-day SMA, using only the prior completed UTC
  day, with missing or invalid classifications blocking BUY.
- Initial paper cash: 10,000 USDT; fee: 0.001 per side; adverse slippage: 5 bps
  per side. Use a new account file; never reset a running trial to erase losses.

The historical exp008 files remain unchanged. A corrected causal baseline is
stored under a new experiment ID. Any change to candidate rules, training
policy, or cost assumptions starts a separately identified trial; it must not
silently replace this candidate's results.

## Information boundaries

The research scoring interval for new forward observations is
**2026-09-24T00:00:00Z inclusive through 2026-12-24T00:00:00Z exclusive**.
No data from this interval was inspected for this protocol. Training may use
only labels whose outcome candles have completed before its declared cutoff.
Use a 540-day trailing window and preserve model metadata and parameters.
Do not tune the candidate using forward outcomes. January 2026 was already
inspected and remains a regression interval, not an unused holdout.

If activation occurs after September 24, report the actual later paper start;
do not invent observations for missed days. Historical reconstruction and
observed paper fills are distinct datasets and must be reported separately.

## Activation prerequisites

1. Complete the combined unit/regression suite and review. Preserve the source
   revision and an exact snapshot of any uncommitted source used by the run.
2. Review the corrected historical result and its limitations. If insufficient,
   record that finding and limit paper observation to operational validation.
3. Refresh completed candles, including at least 200 consecutive complete BTC
   days. Build and test features through the validation-aware CLI or hourly DAG.
4. Train a new model using an explicit UTC cutoff; archive the model, scaler,
   metadata, training parameters, and checksums. Existing model files have no
   automatic provenance guarantee.
5. Verify required timestamps, validation state, current regime classification,
   database connectivity, and a single running bot process. Record any blocked
   entry status rather than loosening gates to make the trial trade.

## Run a separate account

Use PowerShell from the repository root after prerequisites pass:

```powershell
$env:BOT_OPERATIONS_DB = Join-Path (Get-Location) 'results/forward_20260924.sqlite3'
.venv/Scripts/python.exe -m cli trade --paper --once
.venv/Scripts/python.exe -m cli trade --paper --interval 3600
```

The hourly entry cadence matches the historical decision frequency. Protective
paper stops are checked at cycle quotes and cannot reconstruct intracycle moves.
A shorter interval changes observed execution behavior and should be logged as
a separate operational configuration, not presented as exact replay parity.

Run the dashboard with the same `BOT_OPERATIONS_DB`. Preserve SQLite files and
WAL sidecars correctly when backing up a running account. Keep logs and record
the actual start/end times in UTC. No real exchange orders are required.

## Operational acceptance

Observe at least seven consecutive days after activation. Report heartbeat
gaps, restart recovery, midnight rollover, data-refresh failures/recovery,
regime/freshness/validation blocks, evaluation errors, open positions and closed
trades. Deterministic tests cover failure and exit paths even if the market does
not produce trades. A zero-trade soak establishes no profitability evidence.

Open positions currently use entry-cost valuation for paper drawdown. Report
that limitation; do not call it a mark-to-market equity curve. Paper fills also
omit order-book liquidity and partial fills.

## Performance decision

Report costs, net P&L, profit factor, drawdown, sample size, per-symbol results,
exit reasons, and the fraction of time entries were blocked. Preserve all runs,
including failures and no-trade intervals.

Apply the ledger's minimum of 30 out-of-sample trades, targeting at least 100.
Below that threshold the performance result is inconclusive. For any claim of
improvement use the existing ledger rule: PF improves by at least 0.05 over the
comparable baseline, no single trade supplies over 40% of gross profit, and
per-symbol PF exceeds 1 on at least two of three symbols. Do not count known
historical data as fresh forward evidence or optimize win rate as the target.

At the end, publish an operational verdict separately from the performance
verdict. A subsequent testnet execution milestone requires its own readiness
decision; this protocol never authorizes mainnet orders.
