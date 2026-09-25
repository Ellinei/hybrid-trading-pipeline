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

## Activation record — 2026-09-25

**Scope: operational validation only.** The corrected historical baseline
`exp008_corrected_20260925` fails the Phase 12 gate (75 < 100 trades), and
its PF depends on one hyperparameter (see `docs/experiments.md`). Following
prerequisite 2, this trial reports operational behaviour. Any performance
figures it produces are inconclusive unless the minimum-sample rule is met,
and must not be read as validation.

- **Actual paper start:** first cycle `2026-09-25T12:37:51Z`. This is one day
  after the registered interval opened. The missed day is not reconstructed.
- **Source revision:** `8969efb` (clean tree, `source_dirty: false`).
- **Account:** `results/forward_20260924.sqlite3` (new file, 10,000 USDT,
  fee 0.001 per side, 5 bps slippage per side, defaults).
- **Data at activation:** 28,264 hourly candles per symbol,
  2023-07-05 20:00 to 2026-09-25 11:00 UTC, with zero gaps. Features were
  published valid at 11:00 (generation `ebcce6ac-…`) by the
  `ohlcv_feature_refresh` DAG run `scheduled__2026-09-25T11:05:00+00:00`. The
  R1 regime is RISK_ON for 2026-09-25; the first classified date is
  2024-01-22.
- **Model:** trained with `cli train --as-of 2026-09-25T00:00:00Z
  --train-window-days 540 --frozen-params results/forward_20260924/frozen_params.json`.
  The frozen parameters are the corrected baseline's: lr 0.05, depth 3,
  100 trees. The disk path fits on 31,059 rows (labels revealed by
  2026-06-08 17:00) and evaluates on the last 7,773. The live model's
  newest training label is therefore about 3.5 months older than the cutoff,
  whereas walk-forward fits on everything available. This is a known parity
  difference; it is not a leak.
  - `xgboost_signal.pkl` sha256 `56eb6952ae2a037ba3d03dd4cb6b169a2a94a1d8e87e097fafe74f3dd8f28477`
  - `scaler.pkl` sha256 `bcfb20661a3e5713428d43beeb3e0ce796a7cfe4200358e62cc7f957ef302f1c`
  - Copies and metadata are in `results/forward_20260924/model/`. The
    previous (July, no provenance) model is archived at
    `backups/models_pre_forward_20260925/`.
  - Holdout at a 0.5 threshold: precision 0.46, recall 0.009. The model
    rarely signals "up", so zero-trade stretches are expected.
- **Strategy (resolved):** buy/sell gate 0.15 / -0.15, ML 0.55 / 0.35,
  max hold 24, R1 regime, sentiment off, 3 h candle and feature freshness.
- **First cycle:** three SELL signals, each rejected for lack of a position;
  0 errors. Required freshness sources are fresh; streaming is stale but
  informational only.
- **Runtime:** Docker `postgres` plus `airflow-scheduler` with the hourly
  `ohlcv_feature_refresh` DAG unpaused. The bot runs as a detached host
  process (see `results/forward_20260924/bot.pid`), logging to
  `results/forward_20260924/bot.log`. The machine and Docker Desktop must
  stay up for the 7-day observation. Report any downtime as heartbeat gaps.
