# Phase 13 evaluation-window implementation

`cli experiment` now accepts paired `--start` and `--end` ISO 8601 timestamps with a UTC offset. The interval is **[start, end)** in UTC: a bar exactly at start is scored and one exactly at end is excluded. Missing one bound, a naive timestamp, or `start >= end` is rejected before the CLI creates a database engine. `ExperimentConfig` enforces the same contract for direct callers and normalizes equivalent offsets to UTC.

`run_experiment` still loads the full historical bar and feature data for walk-forward training and BTC regime warmup. It passes only interval bars into `run_backtest`, because the engine currently does not apply its `BacktestConfig.start/end` values. The probability cache retains the original full-history training schedule; the feature lookup uses rows at or before each bar, and each retrain uses rows strictly before the boundary minus the label horizon, with non-null future closes. Pending predictions are flushed before a later retrain. A regression changes later feature rows and confirms an earlier probability remains unchanged. Regime decisions use `build_causal_regime`, which assigns a completed BTC UTC day's result to the following decision day; a test confirms a future BTC day cannot alter an earlier decision.

The saved `config.json` contains normalized start/end and `evaluation_interval`. The ledger row records the declared interval even if no trades occur. `regime_days_blocked_pct` is calculated over dates with replayed bars in the interval. A run with no replay bars can still produce an empty, auditable result.

Example command (documentation copy):

```powershell
.venv/Scripts/python.exe -m cli experiment --id corrected-candidate --regime r1_sma200 --start 2026-01-01T00:00:00Z --end 2026-02-01T00:00:00Z
```

Verification: `.venv/Scripts/python.exe -m pytest tests/test_cli.py tests/test_experiment_bounds.py tests/test_backtest_experiment.py -o addopts='' -p no:cacheprovider --tb=short --basetemp <unique-TEMP>/pytest` passed **52** tests. The non-golden repository suite with `-k 'not golden'` passed **478**, with **2 deselected**. Existing dependency and NumPy deprecation warnings remain. No database, exchange, or actual experiment call was made for this implementation.

Corrected exp008 metrics have not been produced here. The stricter regime helper requires 200 consecutive complete BTC UTC days for R1; gaps or incomplete days yield no classification and may reduce the eligible evaluation sample. Preserve and compare the original artifacts when the corrected baseline is run.
