# Phase 13 review fixes — 2026-09-24

Continues `log/2026-09-23-handover.md`. The user resumed the work and asked
for Sonnet agents. Two rounds of implementation ran with separate file
ownership in the shared working tree. Worktrees were not used because all
Phase 13 work is uncommitted. Two independent read-only reviews ran between
the rounds. Nothing was committed, and no Docker, database, exchange,
experiment or model-training operation was run. `backups/`, `results/`,
`cache/` and `models/` were not modified.

## Starting state (re-measured, not taken from the handover)

- Non-golden suite: **504 passed, 5 failed, 1 deselected**.
- Docker daemon not running; nothing listening on 5433 (the project
  TimescaleDB). A separate `postgres` process listens on 5432; it is not the
  project database.
- `cli experiment` would have reused `cache/probs_773a23f6009e7083.pkl`,
  built for `exp008_causal_20260923` before the label fix.
  `resolve_prob_cache` keyed on config and feature fingerprint only.
  No test caught this.

## Round 1 — handover findings

| Issue | Fix | Files |
|---|---|---|
| Probability cache identity in `cli experiment` | `resolve_prob_cache` builds the full-history bar stream before keying. It passes the algorithm version and bar stream to the key and meta. It reuses a file only when the stored `cache_key` matches and a version is present, otherwise it rebuilds. Existing cache files were left in place. | `backtest/experiment.py`, `tests/test_backtest_experiment.py` |
| Blocked-days metric reported 0% with no classifications | `_compute_metrics(..., replay_days=)` counts RISK_OFF and unclassified replay days as blocked. It also emits `regime_days_risk_off_pct` (old definition, comparable with exp008) and `regime_days_unavailable_pct`. | `backtest/experiment.py` |
| Startup regime failure exited before exits ran | `initial_regime()`: an unknown candidate raises `ValueError` before the try. A data failure returns `({}, "refresh_failed")`: the gate stays enabled but unavailable and blocks BUY, and the first cycle retries. The script body became `main(argv=None)`; `python -m` and `cli trade` behave as before. | `execution/bot_runner.py`, `tests/test_bot_runner.py` |
| Live-training label reveal time | Confirmed `feat_technical.timestamp` is the candle open time (`ingestion/binance_client.py:130` through the dbt models). A label counts as known at open + 1h (`BAR_DURATION`, `label_reveal_time`). The rule is applied to the loader cutoff, the `train()` cutoff, the holdout purge and the CV purge. Metadata is now `training_last_label_open_timestamp` plus `training_last_label_reveal_timestamp`. | `signals/ml_model.py`, `tests/test_training_provenance.py` |
| Over-strict fixture | Test data kept its true derived reveal time (next daily row). The first retrain moved to day 2. An agent's fabricated `+1h` reveal time was rejected. | `tests/test_experiment_bounds.py` |
| Ingestion unit test opened a real connection | Patched `build_validation_engine`, the lock and invalidation. A class-level autouse fixture makes any `psycopg2.connect` fail loudly. | `tests/test_ingestion.py` |

## Independent reviews (read-only, after round 1)

- **Leakage and look-ahead review.** No active leak was found in the
  `cli experiment` path. Important findings: (1) `run_backtest --load-cache`
  accepted legacy caches with only a warning, and `cache/wf_540d_h8*.pkl`
  are on disk; (2) the walk-forward GridSearchCV used `TimeSeriesSplit` over
  symbol-grouped row order with no purge, which biases the frozen
  hyperparameters (from Phase 11a, commit 7b73f06). It also noted a latent
  UTC-normalization gap at three regime lookup sites.
- **Fail-safe and operations review.** No blocking findings. Entries fail
  closed and exits keep running on every path traced. Important: no test
  pinned lock and unlock to the same advisory-lock connection. Minor: setup
  failures skipped cleanup, and `days_blocked_pct` still carried the old 0%
  defect.

## Round 2 — review findings

| Issue | Fix | Files |
|---|---|---|
| Non-causal walk-forward CV | `_time_grouped_purged_cv` is shared with `partition_training_rows`. It sorts globally by (timestamp, symbol), uses time-grouped folds, and purges labels revealed after each fold's validation start. X and y are built from the re-sorted frame. Frames without timestamps (synthetic tests only) fall back to the old split with a warning. `PROB_CACHE_ALGORITHM_VERSION` 2 → 3. | `signals/ml_model.py`, `backtest/run_backtest.py`, `tests/test_walk_forward_cv.py` |
| `--load-cache` accepted legacy caches | Refuses a cache with no key, no version, or a stale version unless `--force` is given; `--force` prints a warning that results are not comparable. This intentionally tightens the old contract. | `backtest/run_backtest.py`, `tests/test_backtest_run_backtest.py` |
| Advisory-lock connection invariant | Tests use a fake engine that returns a distinct connection per `connect()`. They check that lock and unlock use the same connection, that unlock runs when the body raises, that unlock runs before the connection exits, and that a failed try-lock never unlocks. No bug found in the module. | `tests/test_feature_validation.py` |
| Setup failure skipped cleanup | Construction now happens inside `try/finally`; the paper-store close is nested so that engine disposal always runs. | `execution/bot_runner.py`, `tests/test_bot_runner.py` |
| Dead helper with the 0% defect | `days_blocked_pct(regime, days)` counts unclassified days as blocked. | `signals/regime.py`, `tests/test_signals_regime.py` |
| UTC regime lookups | `utc_date(ts)` is used by the live aggregator, `ReplayAggregator`, and the experiment's `replay_days`. There is a regression test with a UTC+10 timestamp. | `signals/regime.py`, `signals/aggregator.py`, `backtest/signal_cache.py`, `backtest/experiment.py`, `tests/test_regime_causal.py` |

Root also removed unused mock bindings and unused imports flagged by ruff.
`_compute_metrics` now calls the shared `days_blocked_pct`, so the blocked
metric has a single definition.

## Re-audit of round 2 (the same leakage reviewer, read-only)

No new leak or regression was found. The reviewer checked four things:
- Every production walk-forward caller builds its features with
  `_load_all_features`, which always carries `future_timestamp`, so the
  timestamp-less fallback is reachable only from synthetic or mocked tests.
- Fold indices align with X and y, which are built from the re-sorted frame.
- The `reveal <= validation_start` purge is consistent with
  `available_training_rows` on the hourly grid.
- No v2 or legacy cache can pass either entry point.

Runtime checks for the baseline run:
- The run log shows no `train_in_memory_legacy_frame_lacks_timestamps` or
  `train_in_memory_causal_cv_unavailable` warnings.
- `train_in_memory_complete` (with `best_params`) appears at the first
  successful retrain boundary.
- `cache_walk_forward_retrained` lines appear, showing the probabilities
  were rebuilt rather than reused.
- `config.json` `cache_path` is a new hash, not one of the three existing
  `probs_*.pkl` files.
- Frozen hyperparameters and PF are expected to differ from
  `exp008_causal_20260923`; that is the fix taking effect, not a regression.

## Verification at completion

- Non-golden suite: **545 passed, 0 failed, 1 deselected** (golden).
  Command: `.venv/Scripts/python.exe -m pytest -o addopts='' -p no:cacheprovider -m 'not golden' -q --tb=short --disable-warnings --basetemp <new TEMP dir>`.
- `ruff --select E9,F` clean on all changed and new Python files. Style-only
  warnings (E501, UP017, SIM117) remain, partly from before this session.
- `git diff --check` clean (line-ending warnings only).
- `--help` works for `execution.bot_runner`, `backtest.run_backtest` and
  `cli experiment`.
- **Not run:** the golden replay, Airflow image import and any experiment.
  All three need the Docker database, which is offline.

## Known limits and follow-ups (not fixed)

- Live and backtest training are **not** identical. The backtest also
  requires `row_time < boundary - horizon hours`. The live reveal rule is
  at least as strict on hourly data, but parity is not claimed.
- `cli experiment` loads features through `engine._load_all_features`, which
  drops each symbol's last 8 rows. Replay therefore holds for about the
  last 8 hours of loaded history. This is conservative, has no effect on
  windows that end before the data does, and matters only for runs that end
  at "now".
- `result.warnings` always carries the in-sample look-ahead warning for
  `cli experiment`, whose probabilities are in fact precomputed causally.
  The artifacts don't surface it today.
- `cli features` runs `dbt run` then `dbt test`; the hourly DAG runs
  `dbt build`. Both publish only on success, but partial-failure states
  differ.
- `run_once` can count one event twice in `errors` when the exit
  evaluation and the journal write both fail. This affects diagnostics only.
- `feature_refresh_lock` stays idle in a transaction during the refresh. A
  server `idle_in_transaction_session_timeout` would drop the lock; none is
  configured.

## 2026-09-25 — checkpoint and corrected baseline

- Checkpoint commit `0add8fd` (60 files, local only, not pushed).
  `backups/` and `results/` are still untracked. A secret scan was clean.
- Docker started; only the `postgres` service was brought up (no bot,
  scheduler or Airflow). The container was recreated; the named-volume data
  is intact (26,405 candles per symbol, 2023-07-05 20:00 to
  2026-07-10 00:00, zero hourly gaps).
- **Golden replay:** first run failed only because the new `--load-cache`
  refusal blocked the legacy fixture cache. The harness now passes
  `--force` with a comment; the reference CSV is unchanged. Result:
  **1 passed, byte-identical to the preserved exp007 trade log.**
- **Corrected baseline** `exp008_corrected_20260925`, window
  [2023-07-05, 2026-01-01), same knobs, no retuning: **75 trades, PF 1.89,
  max DD 4.06%, Sharpe 1.40.** It **fails** the Phase 12 gate on trade
  count (75 < 100). Days blocked is 40.7% (corrected definition) or 23.9%
  (registration-time definition). Every runtime check passed. Log:
  `results/exp008_corrected_20260925.run.log`. The ledger row is in
  `docs/experiments.md`.
- **Attribution:** rebuilding the probabilities per grid point reproduces
  the preliminary run exactly with lr 0.1 and the corrected run exactly
  with lr 0.05 (depth 3, 100 trees). The whole PF 1.18 → 1.89 change
  comes from the frozen learning rate the (causal vs leaky) CV selected.
  That is fragility evidence, not validation.

## Status of results (as of 2026-09-24; superseded by the section above)

`results/exp008_causal_20260923` (95 trades, PF 1.181) remains a
**preliminary diagnostic**. It predates the sparse-label fix, the cache
identity fix and the causal CV fix. Every existing cache predates algorithm
version 3 and will be rebuilt. The corrected baseline has **not** been run.

## Next steps

1. Start Docker and confirm the project database on 5433.
2. Run the golden replay against the final code. Preserve its reference and
   report any difference rather than rebaselining silently.
3. Run the frozen candidate under a **new** experiment ID with the same
   window `[2023-07-05T00:00:00Z, 2026-01-01T00:00:00Z)`, without retuning.
   Record the result in `docs/experiments.md`, including a failure of
   acceptance if it fails.
4. Only then do the forward-trial prerequisites in `docs/forward-trial.md`.
   The scoring window opened 2026-09-24; report the actual activation date.
