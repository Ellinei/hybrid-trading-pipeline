# Forward-validation and paper-trial protocol

Status: registered on 2026-09-23; ran from 2026-09-25 to 2026-10-21 as an
operations-only trial and has ended. See the verdicts at the end.

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

## Amendment — 2026-09-25: intermittent operation on a PC

**Why:** the host PC cannot stay on for seven consecutive days. The user made
this decision about three hours after activation, with 0 trades and no
performance observed, so it was not made after seeing any outcome.

**Operation:** each session starts with `scripts\trial_start.ps1` and ends
with `scripts\trial_stop.ps1`. The start script brings up Postgres and the
Airflow scheduler and runs one catch-up refresh, which backfills every candle
missed while the PC was off. It then launches the bot detached, on the same
account. It never creates a new account and never starts a second bot. The
stop script waits for the current cycle to finish, then stops the bot and the
services. Sessions are logged in `results/forward_20260924/sessions.log`, and
`scripts\trial_status.ps1` reports status (read-only).

**Operational acceptance** replaces "seven consecutive days" with:
- at least **168 distinct UTC hours with a heartbeat** since activation,
  spread over at least 7 calendar days (`scripts/trial_report.py` counts
  them);
- at least one UTC midnight rollover while running;
- at least one restart in which the account resumes (done: see below);
- one deliberate stale-data test: pause `ohlcv_feature_refresh` for more than
  3 hours while the bot runs, confirm that BUYs are blocked while SELLs and
  exits continue, then unpause and confirm that entries recover;
- every downtime gap reported with its duration.

**Consequences:** no decisions are made while the PC is off, and missed hours
are not reconstructed. Open paper positions are not monitored during
downtime. Their time stops and price stops are evaluated in the first cycle
after restart, at the quote current then, so exits can be later and outcomes
worse than in continuous operation. Any trade that spans a downtime gap must
be flagged in the report. The performance sample shrinks in proportion to
downtime; below 30 trades the performance verdict stays inconclusive.

**Evidence so far:** two restarts on 2026-09-25 (bot down 15:28–15:44 and
15:45–15:47 UTC). In both, the account and journal resumed, the gates were
evaluated, and there were 0 errors. The first test run of
`trial_start.ps1` exposed a silent failure: the refresh trigger's output was
discarded, and its first attempt, made right after the scheduler started,
never registered, so the script waited out its timeout. That run's bot still
started on published data. The script now retries the trigger until the run
is registered and prints the error otherwise; the second test passed.

## Host move to an AWS VM (2026-10-14)

To reach 168 heartbeat-hours without leaving the PC on, the trial moved for
about one week to an AWS EC2 VM: `c7i-flex.large`, Ubuntu 24.04, x86_64,
`ap-southeast-1`, instance `i-06bbeead16b68a793`. Teardown is planned around
2026-10-21, after the account and database are copied back. Model, strategy
and account are unchanged, so this is the same trial on a different host,
from Windows to Linux.

- **Downtime:** the bot was down from the last Windows heartbeat
  (2026-10-13 19:33 UTC) to the first VM heartbeat (2026-10-14 13:17 UTC).
  The gap is reported, not reconstructed.
- **State moved:**
  - The Postgres volume moved as a tarball of the stopped volume, restored
    with the same image digest. Row counts and the heartbeat count matched
    before and after.
  - The account moved as `results/forward_20260924.sqlite3`. Its WAL was
    checkpointed first, because the Windows bot had stopped without a clean
    close and 60 decisions were still in the `-wal` file. The decision
    counts matched afterwards: hold 23, rejected 40.
- **Code and model:**
  - Code is at `75a5102`.
  - Python is 3.13.2, with packages from the Windows venv's `pip freeze`
    (scikit-learn 1.9.1, xgboost 2.1.4).
  - The model `.pkl` files match `SHA256SUMS`. `metadata.json` matches in
    its CRLF form.
  - Probabilities on fixed inputs are bit-identical on both hosts.
- **Linux runtime:**
  - systemd unit `trial-bot.service` runs the same command and appends to
    the same `bot.log` and `sessions.log`. It restarts on failure after
    60 s.
  - A VM-only compose override binds Postgres to `127.0.0.1` and restarts
    the services on reboot.
  - The security group allows only SSH from the owner's IP.
- **Linux-only fix:** `dbt/` needed a write ACL for the container user
  (uid 50000). Without it, `dbt deps` exited 2 and the first catch-up run
  `scheduled__2026-10-14T11:05` failed.
- **First VM cycle:** 20/168 hours, 22 cycles, 0 errors, cash 10,000.00.
- **No second bot:** the Windows account file was renamed to
  `forward_20260924.sqlite3.MOVED-TO-AWS-2026-10-14`, so
  `scripts\trial_start.ps1` refuses to start on the PC.
- From this point the evidence validates the VM deployment, not the Windows
  launcher.
- **Log timestamps:** `bot.log` timestamps are local time. Windows lines are
  UTC+7, and VM lines (from 2026-10-14 13:17) are UTC. Heartbeats in
  Postgres are always UTC.

## Stale-data test and teardown (2026-10-21)

**Stale-data test.** This ran on the VM. A one-shot job paused
`ohlcv_feature_refresh` at 10:05:26 UTC, right after
`scheduled__2026-10-21T09:05` succeeded. A systemd timer unpaused it at
13:21:13 UTC, so the pause lasted 3 h 16 min. Candles and features stayed at
the 09:00 candle for the whole pause. Nothing else writes those tables: the
refresh is the only DAG, and the VM has no cron jobs.
- **Fresh control:** the 10:19 and 11:19 cycles were identical. SOLUSDT's BUY
  at 0.24 was rejected on confidence (`confidence 0.24 < 0.55`), BTCUSDT
  signalled SELL and ETHUSDT HOLD. Candles and features were fresh, at
  1.33 h and 2.33 h old.
- **Blocked:** at 12:19 and 13:19 the same SOLUSDT BUY was journalled
  `blocked` with `candles: stale; technical: stale` (3.33 h and 4.33 h old;
  feature validation still fresh). The bot logged `entry_blocked_stale_data`.
  These two are the trial's only `blocked` decisions.
- **SELLs continued:** BTCUSDT's SELL was evaluated in both stale cycles and
  reached the executor, which rejected it only because there was no position
  to close. Cycle errors stayed at 0 and heartbeats continued every hour.
- **Exits:** not exercised live, because the account had no open position.
  Exit evaluation runs before signal evaluation in every cycle and is not
  gated on freshness. The deterministic test
  `tests/test_operations.py::test_stale_buy_blocked_then_recovers_without_blocking_exits`
  covers it.
- **Data recovery:** the unpause triggered the catch-up run
  `scheduled__2026-10-21T12:05` (13:21:13–13:21:25), which succeeded. Every
  later hourly run succeeded, and candles 08:00–16:00 are complete for all
  three symbols.
- **Entry recovery:** no BUY signal occurred after the unpause; every symbol
  signalled SELL from 14:19 to 17:19. Recovery was therefore not observed
  through a live BUY. Instead, the bot's own `FreshnessMonitor.entry_check`,
  built the way `cli trade` builds it, was run read-only at 17:11 UTC. It
  allowed entries for all three symbols, with candles and features fresh at
  1.20 h and validation fresh. The probe's first run reported every source
  `unavailable` because it loaded `.env` from the wrong directory; that was
  a probe error, and the rerun used the repository's `.env`.

**Final state when the bot stopped (2026-10-21 17:23:33 UTC):**
- **Hours:** 192 / 168 heartbeat-hours and 194 cycles, with 0 cycle errors.
  Decisions span 11 UTC calendar days.
- **Account:** cash 10,000.00, 0 trades, no open positions. 582 decisions:
  blocked 2, hold 158, rejected 422.
- **systemd:** active from 2026-10-14 13:16:59 until the stop, with
  NRestarts 0 and result `success`.
- **Refresh runs on the VM:** 172 succeeded and 1 failed. The failure is
  `scheduled__2026-10-14T11:05`, the setup failure described above.
  `dag_runs.txt` also holds 35 earlier successful runs from the PC sessions,
  which arrived with the migrated Postgres volume.
- **Downtime gaps:** the six that `trial_report.py` lists all fall before or
  during the move. There were none on the VM.

**Teardown:**
- **Bot stop:** the bot was stopped between cycles. Its clean close emptied
  the WAL: `wal_checkpoint` returned `(0, 0, 0)` and `integrity_check`
  returned ok.
- **Services:** the refresh was paused and the services stopped. The stopped
  volume was archived with alpine `tar czf`, the same method used for the
  move.
- **Archive:** everything is in `backups/vm_teardown_2026-10-21/`
  (untracked):
  - the account file and `results/forward_20260924/`;
  - the volume tarball (`pgdata_final.tgz`);
  - the final report and counts, the DAG run list and the systemd journal;
  - host state (code `75a5102`, image digests);
  - the stale-test scripts and event log.
- **Verification:** after download, all 27 files matched the VM's
  `SHA256SUMS`, and the tarball passes `gzip -t`. The downloaded account
  passes the integrity check, and its decision counts match the VM's.
- **Local restore:**
  - The stale empty `-wal`/`-shm` were deleted first.
  - `results/forward_20260924.sqlite3` is now the final account.
  - The VM's `bot.log` and `sessions.log` replace the local copies, which
    they extend byte for byte.
  - The renamed pre-move file is kept.
  - The local Postgres volume was not touched, because it also holds
    research data. The final trial database exists only as the tarball.
- **Restarting the trial locally:** with the account file restored,
  `scripts\trial_start.ps1` no longer refuses to start. The trial is not
  running. A local restart would run a different source revision and must be
  recorded here.
- **AWS cleanup:** instance `i-06bbeead16b68a793` was terminated at
  17:51:43 UTC, and its root volume was deleted with it. Security group
  `sg-0d53a70e2b0378a31` and key pair `trial-vm-2026-10` were deleted. No
  Elastic IPs or snapshots existed.
- **Stale move checksum:** `backups/vm_move_2026-10-14/SHA256SUMS` (written
  19:42 +07) lists `4182c086…` for the account file. The file in that folder
  and the renamed local file are both `7c7d7e83…` and were last modified at
  20:00:24 +07. The sums file predates the last write to that database; the
  two copies match each other.

## Verdicts (2026-10-21)

**Operational verdict: pass; exits and entry recovery not observed live.**
Against the amended acceptance list:
- 168 distinct heartbeat-hours over at least 7 calendar days: met, with 192
  hours over 11 days.
- At least one UTC midnight rollover while running: met. The VM ran
  continuously from 2026-10-14 to 2026-10-21 with 0 restarts.
- At least one restart in which the account resumes: met, twice on
  2026-09-25.
- Stale-data test: BUYs were blocked and SELLs were still evaluated. Exits and
  entry recovery were not observed live, because no position was open and no
  BUY occurred after the unpause. A deterministic test covers exits, and a
  read-only probe confirmed that entries were allowed again.
- Every downtime gap reported with its duration: met. The final
  `trial_report.py` output lists six gaps over 70 minutes, all before or
  during the move:
  - 2026-09-25 16:47 to 2026-09-26 02:13 UTC (9.4 h)
  - 2026-09-26 02:13 to 06:14 UTC (4.0 h)
  - 2026-09-26 09:14 to 10:51 UTC (1.6 h)
  - 2026-09-26 10:51 to 15:21 UTC (4.5 h)
  - 2026-09-26 19:21 to 2026-10-13 17:33 UTC (406.2 h, PC off)
  - 2026-10-13 19:33 to 2026-10-14 13:17 UTC (17.7 h, the move)

  The two shorter restarts on 2026-09-25 are described in the amendment.

**Performance verdict: inconclusive.** The trial made 0 trades, below the
30-trade minimum, so it gives no evidence about profitability.
