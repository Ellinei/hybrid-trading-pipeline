# Implementation ledger — plan: docs/next-move-2026-09-23.md

**STOPPED by user request.** See `log/2026-09-23-handover.md` for the authoritative
stop state, partial implementations, latest failing checks, and resume steps.

## Authorization and baseline

- User requested Sol agents to implement the proposed next move.
- Branch: `codex/phase13-forward-validation`; existing tracked and untracked work preserved. No automatic staging, commits, orders, or changes to existing paper accounts.
- Baseline: 445 passed, 1 deselected, 189 warnings (22.79 seconds).
- Command: `.venv/Scripts/python.exe -m pytest -o addopts='' -p no:cacheprovider -m 'not golden' -q --tb=short --disable-warnings --basetemp <new writable temporary directory>/pytest`.
- Earlier permission failures came from inaccessible existing pytest temp directories. A new directory created under `$env:TEMP` works.

## Task ownership and interface review

| Task pair / task | Shared interface or consistency check | Ruling |
|---|---|---|
| Regime / evaluation | Both previously constructed daily regime dictionaries | Regime agent owns a shared `build_causal_regime(bars, candidate)` helper; evaluation agent calls it. Only evaluation agent edits `backtest/experiment.py`. |
| Regime / ingestion | Completed hourly bars feed completed daily classifications | Ingestion rejects open candles; regime validates full UTC-day history independently. |
| Regime / paper quotes | Runner constructs exchange client | Regime agent owns runner and selects public mainnet quotes for paper; configured exchange remains for live. |
| Ingestion | Daily schedule cannot satisfy three-hour freshness | Make hourly backfill and transformations a successful dependency chain; keep freshness guard unchanged. |
| Evaluation | Scored interval must not discard training/regime warm-up | Bound scoring separately from causal history; save bounds in artifacts. |
| Training | Current per-symbol concatenation is not globally temporal | Separate follow-up task for timestamp ordering, label purge, and model provenance. |
| Paper trial | Seven-day observation cannot be completed in a coding session | Complete prerequisites and runbook, report real runtime blockers; never represent synthetic tests as a completed trial. |

Ruling: use this user-approved assessment as the implementation brief, with the explicit interface decisions above. It has no separate spec. Parallel work has disjoint file ownership; broad integration review follows implementation.

## Active work

- Sol `regime_correctness`: causal regime builder, entry failure behavior, paper quote source, regressions.
- Sol `hourly_ingestion`: completed candles, correction/gap overlap, hourly dependent DAG, regressions.
- Sol `evaluation_bounds`: explicit experiment bounds, saved metadata, warm-up preservation, regressions.
- Root: baseline verification, documentation, coordination and integration review.

## Pending

- Timestamp-purged live training and artifact provenance.
- Independent review and combined regression suite.
- Corrected baseline experiment when required services/data are available.
- Forward trial registration, operational runbook, and actual observation period.

## Resume and verification updates

- Work resumed after an automatic approval-review service failed due to its usage limit. The user resumed with unrestricted local permissions. No denied action was bypassed while the restriction was active.
- Regime/evaluation implementation reports are saved. Fresh Sol agents resumed the incomplete validation gate, implemented training provenance, and independently reviewed causal evaluation.
- Cache correction regression: three failures reproduced the old two-field fingerprint limitation; content-digest implementation passed all 28 snapshot/cache CLI tests. The PostgreSQL fingerprint query was also executed successfully against existing data.
- Existing PostgreSQL container was started for historical integration. No bot or scheduler was started. Latest stored candles are 2026-07-10 for all three symbols.
- Golden replay passed: 1 test, using existing database/cache/reference data and public exchange metadata.
- Dashboard bound regression failed on an empty trial; after the fix the declared evaluation window is displayed even with no equity events (1 test passed).
- Preliminary corrected regime replay: `exp008_causal_20260923`, interval `[2023-07-05T00:00:00Z, 2026-01-01T00:00:00Z)`, 95 trades, PF 1.1810, max DD 3.9335%. This does not meet the historical Phase 12 thresholds of approximately PF 1.45 and 100 trades. No strategy retuning was performed.
- Independent review reproduced an additional gappy-data label-horizon leak in walk-forward caching; this must be fixed and cache-versioned before treating the corrected baseline as final.
- Prospective trial protocol saved in `docs/forward-trial.md`. No observation period has started.
