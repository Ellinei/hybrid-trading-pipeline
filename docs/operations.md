# Operations and strategy evaluation

## Start a persistent paper account

```powershell
python -m cli trade --paper --once
python -m cli trade --paper --interval 60
python -m cli dashboard
```

Paper mode uses public mainnet quotes and symbol filters, matching the default
mainnet candle source, plus the feature database and signal model. Prepare the
data and trained model as for normal trading. It reads quotes
and exchange symbol filters but never submits orders or reads exchange balances.
It reuses the live risk approval and sizing rules with a simulated account.

The default account starts with 10,000 USDT. `--paper-balance` only sets the initial
balance when the account is first created; restarting resumes existing cash and
positions. Paper data is separate from `raw.trades` and `raw.bot_state`.
`--paper` and `--dry-run` are mutually exclusive. Existing dry-run behavior remains
available for checking order construction without maintaining simulated positions.

Costs default to `--paper-fee 0.001` (0.1% per side) and
`--paper-slippage-bps 5` (0.05% adverse slippage per side). Each position stores its
cost assumptions, so a restart or changed arguments do not rewrite existing
positions. Cash debits include entry fees; closing P&L deducts both sides' fees.
Approvals and ledger changes use one SQLite write transaction, preventing
concurrent cycles from spending the same available cash or opening duplicate
positions on the same symbol.

Stop-loss and take-profit checks run at each cycle's observed quote. Time stops
use the strategy's `max_hold_bars` in hours. A quote gap fills at the observed
quote with adverse slippage, not the stop price. Moves between polling cycles
are not reconstructed, and the simulator does not model partial fills or order
book liquidity. Use a shorter interval for closer monitoring; this also evaluates
entry signals more often. Paper equity and drawdown value open positions at entry
cost, not current market prices, and include entry fees.

## Storage

The decision journal and paper account share `results/operations.sqlite3` by
default. Set `BOT_OPERATIONS_DB` to an absolute path to select a separate account
or a persistent container volume. Use the same path for the bot and dashboard.
SQLite WAL sidecar files may exist while the store is in use. Do not share this
file over a network filesystem. No automatic account reset is provided: select
a new file when starting a separate trial.

## Freshness checks

Every bot CLI mode checks `raw.ohlcv` and `features.feat_technical` before a BUY.
The checks use the stored source timestamps, not the time a signal was generated.
Missing, stale, future-dated, or unavailable required sources block new entries.
SELL signals and timed/protective exit processing still run. A healthy reading
automatically permits the next entry evaluation; there is no manual reset.

Defaults are three hours for hourly candle **open times** and technical feature
source times. This allows for the completed hourly candle and pipeline delay.
Override with positive second counts:

```powershell
python -m cli trade --paper --candle-max-age 9000 --feature-max-age 9000
```

The `ohlcv_feature_refresh` Airflow DAG runs at minute 5 of every UTC hour:
completed-candle backfill → dbt dependencies → dbt build and tests. It permits
one active run at a time and propagates per-symbol ingestion failures. The old
independent daily `ohlcv_backfill` and `dbt_features` DAGs are replaced; pause
their old registrations during deployment. Incremental backfill re-fetches an
overlap window and updates corrected candles. Older repairs require an explicit
full-range backfill. Content digests invalidate snapshots and probability caches
when existing values change without new timestamps.

The macro regime guard is separate from candle freshness. A completed UTC day's
BTC classification becomes usable on the next UTC date. R1 requires 200
consecutive complete days, with all 24 hourly candles present per day. Missing
history, unavailable classification, and refresh failures block new BUYs while
exits continue. Explicitly disabling the regime guard is a different configuration.

Feature validation is also required. Refresh writers invalidate the published
state before changing data and publish a new generation only after dbt tests
succeed. A PostgreSQL advisory lock serializes the manual command and the
scheduled refresh. A failed build/test leaves the state invalid, even if a
newly built table has recent timestamps. Missing state on an existing
installation blocks entries until the first successful validation-aware refresh:

```powershell
.venv/Scripts/python.exe -m cli features
```

Use this command or the hourly DAG for publication. Running `dbt` directly does
not maintain the publication protocol and is not an operational refresh path.
After standalone backfill, run the command above before expecting new entries.

The Operations page shows status for each symbol and source, including an
informational three-minute limit for streaming snapshot publication. Streaming
snapshots are not required by the current technical/ML signal path and do not
gate entries. Snapshot freshness alone does not prove recent market trades have
arrived. Dashboard threshold controls only preview status; bot command-line
arguments control execution.

## Decision journal

The Operations page filters the latest decisions by symbol and mode (live,
dry_run, paper). Every evaluated signal records its direction, component signals,
confidence, reasoning, outcome, strategy/risk configuration, and paper costs when
applicable. Entry records include source-health checks. Rejections retain the
executor's reason; aggregation/execution failures and automatic closes are also
recorded. Expand a decision to inspect details or export the filtered rows as CSV.

Journal write failures are logged and counted as cycle errors; they do not
interrupt exits. This is an operational history, not a transactional exchange
audit log. The store currently has no automatic retention policy.

## Compare saved experiments

Use paired, timezone-aware bounds to pin an experiment's scored interval:

```powershell
.venv/Scripts/python.exe -m cli experiment --id causal_baseline --ml-buy-threshold 0.55 --max-hold-bars 24 --regime r1_sma200 --start 2023-07-05T00:00:00Z --end 2026-01-01T00:00:00Z
```

Start is inclusive and end is exclusive. Training and regime warm-up history
remain available before start. Saved configuration records the interval; the
dashboard uses it even when no trades occur. Preserve historical result folders
by choosing a new experiment ID for every distinct run.

Open **Experiments** in the dashboard sidebar and select a results directory.
The page reads `metrics.json`, `config.json`, and `equity_curve.csv` from existing
experiment subdirectories without rerunning training or querying PostgreSQL.

Select runs to compare portfolio return, drawdown, Sharpe ratio, profit factor,
and trade count. Select a symbol for its trade count, win rate, profit factor,
and P&L. Evaluation start/end timestamps and traded symbols remain visible;
different evaluation windows trigger a warning. Full configurations and metrics
are expandable, and the comparison can be downloaded as CSV. Incomplete or
malformed runs are reported and skipped. No winning strategy is selected
automatically.

## Verification

```powershell
python -m pytest tests/test_operations.py tests/test_operations_dashboard.py -o addopts='' -p no:cacheprovider
```

Tests use temporary SQLite files, synthetic artifacts, and mocked quotes/databases.
They do not place exchange orders or alter an existing paper account.
