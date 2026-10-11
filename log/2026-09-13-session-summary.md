# Session summary — 2026-09-13

Project: `hybrid-trading-pipeline`

## Requests and scope

1. Find and fix a project bug.
2. Improve the code.
3. Brainstorm new features. The requested brainstorming skill was not installed;
   the user authorized proceeding without it.
4. Implement the proposed features in the order chosen by the assistant.
5. Create a `log` folder and export this session summary.

## Streaming bug fix

The streaming processor kept only its latest 100 trades. This discarded trades
that were still inside the one- and five-minute feature windows, undercounting
trade activity and distorting VWAP and other window-based calculations.

A deterministic regression reproduced 250 trades within a minute being reported
as 100. The 100-trade control passed. Debugger inspection confirmed that appending
the 101st trade immediately removed the oldest buffered trade. Increasing only
the cap restored the expected count, isolating the cause.

Replaced the count cap with five-minute retention at feature flush time. Idle
symbols retain one expired tick for their last known price. Added high-volume,
expiry-boundary, and idle-symbol regressions. The streaming suite passed all
35 tests after this fix.

## Streaming improvements

- Validate messages before modifying buffers, including symbol and timestamp.
- Reject malformed payloads, booleans used as amounts, non-finite numbers, and
  non-positive trade prices or quantities.
- Insert delayed trades in event-time order so they cannot replace the latest
  price or reverse the calculated price change.
- Keep the append path for trades arriving in order.

The streaming suite then passed all 59 tests, including 24 additional cases.

Primary files:

- `streaming/feature_processor.py`
- `tests/test_streaming.py`

## Features implemented

### 1. Data freshness monitor

- Checks source timestamps in `raw.ohlcv` and `features.feat_technical` before
  new BUY entries in bot CLI modes.
- Missing, stale, future-dated, or unavailable required sources block entries.
- SELL signals and timed/protective exit processing continue.
- Default limits are three hours, configurable with `--candle-max-age` and
  `--feature-max-age` in seconds.
- Dashboard status also includes streaming snapshot freshness as informational
  data; the current technical/ML path does not require it for entry approval.
- Entries automatically become eligible again when required data is fresh.

### 2. Decision journal

- Persists decisions in a local SQLite store.
- Records signal details, strategy/risk configuration, execution mode, outcomes,
  rejection reasons, entry freshness checks, and applicable paper cost settings.
- Records aggregation/execution failures and automatic closes.
- Provides dashboard filters, detailed inspection, and CSV export.
- Journal write failures are logged and counted without interrupting exits.

### 3. Persistent paper trading

- Added `--paper`, mutually exclusive with `--dry-run`.
- Simulates a spot account using market quotes and exchange symbol filters,
  without submitting orders or reading exchange balances.
- Reuses live risk approval and sizing rules with simulated account state.
- Persists cash and positions across restarts, separately from live trade tables.
- Defaults to 10,000 USDT, a 0.1% fee per side, and five basis points of adverse
  slippage per side.
- Includes entry fees in cash debits and both entry/exit fees in closed P&L.
- Stores cost assumptions per position.
- Uses atomic transactions for approvals and ledger updates, with protection
  against concurrent duplicate positions and spending the same cash twice.
- Supports signal closes, stop-loss, take-profit, and timed exits.

### 4. Experiment comparison dashboard

- Added Operations and Experiments pages to the existing Streamlit dashboard.
- Reads saved experiment `metrics.json`, `config.json`, and `equity_curve.csv`
  without retraining or querying PostgreSQL.
- Compares portfolio return, drawdown, Sharpe ratio, profit factor, and trade count.
- Supports per-symbol breakdowns, configuration inspection, and CSV export.
- Shows evaluation windows and warns when selected runs use different windows.
- Reports and skips incomplete or malformed experiment artifacts.

## Main new files

- `execution/freshness.py`
- `execution/journal.py`
- `execution/paper.py`
- `dashboard/data.py`
- `dashboard/operations_view.py`
- `dashboard/experiments_view.py`
- `dashboard/pages/1_Operations.py`
- `dashboard/pages/2_Experiments.py`
- `tests/test_operations.py`
- `tests/test_operations_dashboard.py`
- `docs/operations.md`

Integration changes also touched `execution/bot_runner.py`,
`execution/trade_executor.py`, `config/strategy.py`, `tests/test_bot_runner.py`,
`README.md`, and `.gitignore`.

## Verification

The default Python installation lacked pytest, so an isolated `.venv` was created
and dependencies installed. Network installation and pytest temporary-directory
access required execution outside the sandbox.

Final non-golden suite:

```powershell
.venv/Scripts/python -m pytest -o addopts='' -p no:cacheprovider -m 'not golden' -q --tb=short --disable-warnings
```

Result: **445 passed, 1 deselected, 189 warnings**. The live-database golden test
was excluded. Dashboard interaction tests passed using Streamlit AppTest.
New-module lint checks and `git diff --check` passed. CLI help exposed the new
paper and freshness arguments.

Tests covered freshness boundaries and recovery, journal persistence/filtering,
paper restart recovery, fee accounting, protective/timed exits, transaction
rollback, concurrent entries, malformed artifacts, and dashboard interactions.

## Usage

With the existing market-data, feature database, and trained model prepared:

```powershell
python -m cli trade --paper --once
python -m cli trade --paper --interval 60
python -m cli dashboard
```

The default operational store is `results/operations.sqlite3`. Set
`BOT_OPERATIONS_DB` to use a different path, and use the same path for the bot and
dashboard. `--paper-balance` initializes a new account only; it does not reset an
existing account.

See [the operations guide](../docs/operations.md) for full setup and configuration.

## Limitations and final workspace state

- Paper stops are evaluated at bot-cycle quotes; moves between cycles are not
  reconstructed. The simulator does not model partial fills or order-book liquidity.
- Paper equity/drawdown value open positions at entry cost, including entry fees,
  rather than marking them to current market prices.
- Data refresh frequency must match freshness limits. Daily feature refreshes
  will become stale under the default three-hour limit.
- Dashboard threshold controls preview status; bot arguments control entry checks.
- Journal storage has no automatic retention policy and is not a transactional
  exchange audit log.
- No live orders were placed and no existing paper account was modified by tests.
- Changes remain uncommitted. The pre-existing untracked `backups/` and `results/`
  directories were preserved.
- This summary records validation completed during the session; the export itself
  did not rerun the test suite.
