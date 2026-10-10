"""Unit tests for execution/bot_runner.py's component-builder functions —
Phase 13 wiring that makes the live bot construct RiskManager/
SignalAggregator/MLSignalGenerator from the validated StrategyConfig
(config/strategy.py, exp008) instead of every class's own defaults.

No real DB, Binance, or exchange connections — all I/O is mocked.

Run:
    poetry run pytest tests/test_bot_runner.py -v
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from config.strategy import StrategyConfig
from execution import bot_runner
from execution.bot_runner import (
    build_aggregator,
    build_market_data_client,
    build_risk_manager,
    initial_regime,
    resolve_regime,
    run_loop,
    run_once,
)


class TestMarketDataClient:
    @patch("execution.bot_runner.BinanceClientWrapper")
    def test_paper_uses_public_mainnet_quotes(self, wrapper):
        build_market_data_client(paper=True)
        wrapper.assert_called_once_with(testnet=False)

    @patch("execution.bot_runner.BinanceClientWrapper")
    def test_live_uses_configured_exchange(self, wrapper):
        build_market_data_client(paper=False)
        wrapper.assert_called_once_with()


# ===========================================================================
# build_aggregator — every StrategyConfig knob must reach the live decision
# point, not just exist as an unused dataclass field.
# ===========================================================================

class TestBuildAggregator:
    def test_wires_ml_buy_threshold_into_ml_generator(self):
        cfg = StrategyConfig(ml_buy_threshold=0.55, ml_sell_threshold=0.35)
        agg = build_aggregator(MagicMock(), cfg, regime=None)
        assert agg.ml.buy_prob_threshold == 0.55
        assert agg.ml.sell_prob_threshold == 0.35

    def test_wires_weights_and_gates(self):
        cfg = StrategyConfig(
            weights={"technical": 0.5, "ml": 0.3, "sentiment": 0.2},
            buy_gate=0.20, sell_gate=-0.20,
        )
        agg = build_aggregator(MagicMock(), cfg, regime=None)
        assert agg.weights == {"technical": 0.5, "ml": 0.3, "sentiment": 0.2}
        assert agg.buy_gate == 0.20
        assert agg.sell_gate == -0.20

    def test_wires_sentiment_enabled(self):
        cfg = StrategyConfig(sentiment_enabled=False)
        agg = build_aggregator(MagicMock(), cfg, regime=None)
        assert agg.sentiment_enabled is False

    def test_passes_regime_dict_through(self):
        from signals.regime import RegimeState

        regime = {datetime(2026, 1, 1).date(): RegimeState.RISK_OFF}
        cfg = StrategyConfig()
        agg = build_aggregator(MagicMock(), cfg, regime=regime)
        assert agg._regime == regime


# ===========================================================================
# build_risk_manager — max_hold_bars must reach RiskConfig
# ===========================================================================

class TestBuildRiskManager:
    def test_wires_max_hold_bars(self):
        cfg = StrategyConfig(max_hold_bars=24)
        risk = build_risk_manager(MagicMock(), MagicMock(), cfg)
        assert risk.config.max_hold_bars == 24

    def test_default_config_disables_time_stop(self):
        cfg = StrategyConfig(max_hold_bars=0)
        risk = build_risk_manager(MagicMock(), MagicMock(), cfg)
        assert risk.config.max_hold_bars == 0


# ===========================================================================
# resolve_regime — same BTCUSDT-bars -> daily-close -> REGIME_FUNCTIONS
# path as backtest.experiment._resolve_regime, for live/backtest parity.
# ===========================================================================

class TestResolveRegime:
    def test_none_candidate_returns_none_without_touching_db(self):
        cfg = StrategyConfig(regime_candidate=None)
        engine = MagicMock()

        result = resolve_regime(engine, cfg)

        assert result is None
        engine.connect.assert_not_called()

    def test_r1_candidate_computes_regime_from_btc_bars(self):
        from backtest.exit_simulation import OHLCVBar
        from signals.regime import RegimeState

        bars = [
            OHLCVBar(timestamp=datetime(2026, 1, d, tzinfo=UTC),
                     open=100.0, high=101.0, low=99.0, close=100.0)
            for d in range(1, 5)
        ]
        cfg = StrategyConfig(regime_candidate="r1_sma200")

        with patch("backtest.run_backtest._load_bars", return_value=bars) as mock_load:
            result = resolve_regime(MagicMock(), cfg)

        mock_load.assert_called_once()
        assert mock_load.call_args[0][1] == "BTCUSDT"
        # Fewer than 200 days of history -> R1's documented warm-up default.
        assert all(s == RegimeState.RISK_ON for s in result.values())

    def test_unknown_candidate_raises(self):
        cfg = StrategyConfig(regime_candidate="not_a_real_candidate")
        with patch("backtest.run_backtest._load_bars", return_value=[]):
            with pytest.raises(ValueError):
                resolve_regime(MagicMock(), cfg)


# ===========================================================================
# initial_regime — startup-only resolution. Unlike the per-cycle refresh in
# run_once, there is no earlier successful state to fall back to here, so a
# DB failure must degrade to enabled-but-unavailable ({}) rather than
# raising and aborting the process before close_expired_positions ever runs.
# ===========================================================================

class TestInitialRegime:
    def test_disabled_gate_returns_none_without_touching_db(self):
        cfg = StrategyConfig(regime_candidate=None)
        engine = MagicMock()

        regime, error = initial_regime(engine, cfg)

        assert regime is None
        assert error is None
        engine.connect.assert_not_called()

    def test_success_returns_regime_map_and_no_error(self):
        cfg = StrategyConfig(regime_candidate="r1_sma200")
        fresh_regime = {"some": "map"}

        with patch("execution.bot_runner.resolve_regime", return_value=fresh_regime):
            regime, error = initial_regime(MagicMock(), cfg)

        assert regime == fresh_regime
        assert error is None

    def test_db_failure_returns_empty_dict_not_none(self):
        """{} (enabled-but-unavailable) still blocks BUY; None would
        deliberately disable the gate and open it — the unsafe direction."""
        cfg = StrategyConfig(regime_candidate="r1_sma200")

        with patch("execution.bot_runner.resolve_regime", side_effect=RuntimeError("db down")):
            regime, error = initial_regime(MagicMock(), cfg)

        assert regime == {}
        assert regime is not None
        assert error == "refresh_failed"

    def test_unknown_candidate_raises_before_any_query(self):
        engine = MagicMock()
        with pytest.raises(ValueError, match="Unknown regime candidate"):
            initial_regime(engine, StrategyConfig(regime_candidate="not_a_real_candidate"))
        engine.connect.assert_not_called()


# ===========================================================================
# run_once — the live time-stop wiring: expired positions must be closed
# BEFORE new entry signals are generated each cycle (mirrors backtest.
# engine.py checking time-stop ahead of evaluating a new signal per bar).
# ===========================================================================

class TestRunOnceTimeStopOrdering:
    def test_closes_expired_positions_before_generating_new_signals(self):
        executor = MagicMock()
        aggregator = MagicMock()
        call_order = []
        executor.close_expired_positions.side_effect = (
            lambda *a, **k: call_order.append("close_expired") or []
        )
        aggregator.aggregate.side_effect = (
            lambda *a, **k: call_order.append("aggregate") or MagicMock()
        )

        run_once(["BTCUSDT"], executor, aggregator, dry_run=False, max_hold_bars=24)

        assert call_order == ["close_expired", "aggregate"]

    def test_passes_max_hold_bars_and_dry_run_through(self):
        executor = MagicMock()
        aggregator = MagicMock()

        run_once(["BTCUSDT"], executor, aggregator, dry_run=True, max_hold_bars=24)

        executor.close_expired_positions.assert_called_once_with(24, dry_run=True)

    def test_zero_max_hold_bars_still_calls_close_expired_positions(self):
        """close_expired_positions itself no-ops on max_hold_bars<=0 —
        run_once doesn't need its own separate skip branch."""
        executor = MagicMock()
        aggregator = MagicMock()

        run_once(["BTCUSDT"], executor, aggregator, dry_run=False, max_hold_bars=0)

        executor.close_expired_positions.assert_called_once_with(0, dry_run=False)

    def test_close_expired_positions_error_does_not_block_the_cycle(self):
        executor = MagicMock()
        executor.close_expired_positions.side_effect = RuntimeError("db gone")
        aggregator = MagicMock()

        run_once(["BTCUSDT"], executor, aggregator, dry_run=False, max_hold_bars=24)

        executor.execute_signal.assert_called_once()


# ===========================================================================
# run_once — regime must be RECOMPUTED every cycle, not frozen at startup.
# A dict[date, RegimeState] built once at bot start only has entries up
# to that day; in a long-running loop, aggregate()'s regime.get(now.date())
# silently misses on every later day and the gate fails OPEN (allows BUYs
# it should block) — the unsafe direction. Phase 13's own acceptance
# milestone is a >=2-week paper soak, so this must hold across days.
# ===========================================================================

class TestRunOnceRegimeRefresh:
    def test_refreshes_regime_before_generating_signals(self):
        from signals.regime import RegimeState

        executor = MagicMock()
        aggregator = MagicMock()
        aggregator._regime = {"stale": "value"}
        engine = MagicMock()
        cfg = StrategyConfig(regime_candidate="r1_sma200")
        fresh_regime = {datetime(2026, 7, 11).date(): RegimeState.RISK_OFF}

        call_order = []
        aggregator.aggregate.side_effect = (
            lambda *a, **k: call_order.append(("aggregate", aggregator._regime)) or MagicMock()
        )

        with patch("execution.bot_runner.resolve_regime",
                   return_value=fresh_regime) as mock_resolve:
            run_once(["BTCUSDT"], executor, aggregator, engine=engine, strategy_cfg=cfg)

        mock_resolve.assert_called_once_with(engine, cfg)
        assert call_order == [("aggregate", fresh_regime)]

    def test_regime_refresh_failure_blocks_entries_and_does_not_block_cycle(self):
        executor = MagicMock()
        aggregator = MagicMock()
        aggregator._regime = {"previous": "regime"}
        engine = MagicMock()
        cfg = StrategyConfig(regime_candidate="r1_sma200")

        with patch("execution.bot_runner.resolve_regime", side_effect=RuntimeError("db gone")):
            run_once(["BTCUSDT"], executor, aggregator, engine=engine, strategy_cfg=cfg)

        assert aggregator._regime == {"previous": "regime"}
        assert aggregator._regime_refresh_error == "refresh_failed"
        aggregator.aggregate.assert_called_once()

    def test_regime_refresh_recovery_clears_failure(self):
        from signals.regime import RegimeState

        executor = MagicMock()
        aggregator = MagicMock()
        aggregator._regime_refresh_error = "refresh_failed"
        engine = MagicMock()
        cfg = StrategyConfig(regime_candidate="r1_sma200")
        fresh = {datetime(2026, 7, 11).date(): RegimeState.RISK_ON}

        with patch("execution.bot_runner.resolve_regime", return_value=fresh):
            run_once(["ETHUSDT"], executor, aggregator, engine=engine, strategy_cfg=cfg)

        assert aggregator._regime is fresh
        assert aggregator._regime_refresh_error is None

    def test_refresh_failure_still_processes_time_stop_and_sell(self):
        from signals.base import AggregatedSignal, SignalDirection

        executor = MagicMock()
        executor.close_expired_positions.return_value = [
            {"symbol": "BTCUSDT", "close_reason": "time_stop"},
        ]
        aggregator = MagicMock()
        aggregator.aggregate.return_value = AggregatedSignal(
            SignalDirection.SELL, 0.4, "ETHUSDT", datetime.now(UTC), [], "sell", {},
        )
        journal = MagicMock()
        journal.record.return_value = True
        cfg = StrategyConfig(regime_candidate="r1_sma200")

        with patch("execution.bot_runner.resolve_regime", side_effect=RuntimeError("db gone")):
            run_once(["ETHUSDT"], executor, aggregator, engine=MagicMock(),
                     strategy_cfg=cfg, journal=journal, max_hold_bars=24)

        executor.close_expired_positions.assert_called_once()
        executor.execute_signal.assert_called_once()
        assert aggregator._regime_refresh_error == "refresh_failed"

    def test_no_engine_or_cfg_skips_regime_refresh(self):
        """Backward compatible: callers that don't pass engine/strategy_cfg
        (e.g. no regime gate configured at all) get the pre-Phase-13
        behavior — no DB touch, no crash."""
        executor = MagicMock()
        aggregator = MagicMock()

        with patch("execution.bot_runner.resolve_regime") as mock_resolve:
            run_once(["BTCUSDT"], executor, aggregator)

        mock_resolve.assert_not_called()
        aggregator.aggregate.assert_called_once()


class TestRunOnceHeartbeat:
    def test_records_signal_action_error_counts_for_a_clean_cycle(self):
        executor = MagicMock()
        executor.execute_signal.return_value = {"id": 1}  # every symbol takes an action
        aggregator = MagicMock()
        heartbeat = MagicMock()

        run_once(["BTCUSDT", "ETHUSDT"], executor, aggregator, heartbeat=heartbeat)

        heartbeat.record.assert_called_once_with(signals=2, actions=2, errors=0)

    def test_hold_signals_count_as_signals_not_actions(self):
        executor = MagicMock()
        executor.execute_signal.return_value = None  # rejected/HOLD -> no action taken
        aggregator = MagicMock()
        heartbeat = MagicMock()

        run_once(["BTCUSDT"], executor, aggregator, heartbeat=heartbeat)

        heartbeat.record.assert_called_once_with(signals=1, actions=0, errors=0)

    def test_per_symbol_cycle_error_counted_and_does_not_stop_other_symbols(self):
        executor = MagicMock()
        aggregator = MagicMock()
        aggregator.aggregate.side_effect = [RuntimeError("boom"), MagicMock()]
        heartbeat = MagicMock()

        run_once(["BTCUSDT", "ETHUSDT"], executor, aggregator, heartbeat=heartbeat)

        heartbeat.record.assert_called_once_with(signals=1, actions=1, errors=1)

    def test_time_stop_and_regime_refresh_errors_are_counted(self):
        executor = MagicMock()
        executor.close_expired_positions.side_effect = RuntimeError("db gone")
        aggregator = MagicMock()
        heartbeat = MagicMock()
        engine = MagicMock()
        cfg = StrategyConfig(regime_candidate="r1_sma200")

        with patch("execution.bot_runner.resolve_regime", side_effect=RuntimeError("db gone")):
            run_once(["BTCUSDT"], executor, aggregator, heartbeat=heartbeat,
                      engine=engine, strategy_cfg=cfg, max_hold_bars=24)

        heartbeat.record.assert_called_once_with(signals=1, actions=1, errors=2)

    def test_no_heartbeat_param_skips_recording(self):
        """Backward compatible: callers without a heartbeat recorder
        (e.g. no DB configured) get pre-Phase-13 behavior — no crash."""
        executor = MagicMock()
        aggregator = MagicMock()

        run_once(["BTCUSDT"], executor, aggregator)  # must not raise


class TestRunLoopPassesMaxHoldBarsThrough:
    def test_run_loop_forwards_max_hold_bars_to_run_once(self):
        executor = MagicMock()
        aggregator = MagicMock()

        with patch("execution.bot_runner.run_once") as mock_run_once, \
             patch("execution.bot_runner.time.sleep", side_effect=KeyboardInterrupt):
            run_loop(["BTCUSDT"], executor, aggregator, max_hold_bars=24)

        mock_run_once.assert_called_once_with(
            ["BTCUSDT"], executor, aggregator, dry_run=False, max_hold_bars=24,
            engine=None, strategy_cfg=None, heartbeat=None, freshness=None, journal=None,
        )

    def test_run_loop_forwards_engine_and_strategy_cfg_for_per_cycle_regime_refresh(self):
        executor = MagicMock()
        aggregator = MagicMock()
        engine = MagicMock()
        cfg = StrategyConfig()

        with patch("execution.bot_runner.run_once") as mock_run_once, \
             patch("execution.bot_runner.time.sleep", side_effect=KeyboardInterrupt):
            run_loop(["BTCUSDT"], executor, aggregator, engine=engine, strategy_cfg=cfg)

        _, kwargs = mock_run_once.call_args
        assert kwargs["engine"] is engine
        assert kwargs["strategy_cfg"] is cfg

    def test_run_loop_forwards_heartbeat_to_run_once(self):
        executor = MagicMock()
        aggregator = MagicMock()
        heartbeat = MagicMock()

        with patch("execution.bot_runner.run_once") as mock_run_once, \
             patch("execution.bot_runner.time.sleep", side_effect=KeyboardInterrupt):
            run_loop(["BTCUSDT"], executor, aggregator, heartbeat=heartbeat)

        _, kwargs = mock_run_once.call_args
        assert kwargs["heartbeat"] is heartbeat


# ===========================================================================
# main() — end-to-end wiring, everything mocked (no DB/network). Proves a
# startup regime DB failure (Task 1's bug: resolve_regime raising before
# run_once existed used to abort the process before close_expired_positions
# could run) now still reaches run_once with an enabled-but-unavailable
# regime, instead of crashing before exit processing has a chance to run.
# ===========================================================================

class TestMainStartupRegimeFailure:
    def test_startup_db_failure_still_reaches_run_once(self):
        with patch("execution.bot_runner.create_engine") as mock_create_engine, \
             patch("execution.bot_runner.build_market_data_client"), \
             patch("execution.bot_runner.resolve_regime", side_effect=RuntimeError("db down")), \
             patch("execution.bot_runner.build_aggregator") as mock_build_aggregator, \
             patch("execution.bot_runner.build_risk_manager"), \
             patch("execution.bot_runner.TradeExecutor"), \
             patch("execution.bot_runner.HeartbeatRecorder"), \
             patch("execution.bot_runner.FreshnessMonitor"), \
             patch("execution.bot_runner.DecisionJournal"), \
             patch("execution.bot_runner.run_once") as mock_run_once:
            aggregator = mock_build_aggregator.return_value

            bot_runner.main(["--once", "--dry-run"])

        mock_run_once.assert_called_once()
        _, kwargs = mock_run_once.call_args
        assert kwargs["dry_run"] is True
        assert kwargs["engine"] is mock_create_engine.return_value
        # build_aggregator is called with the enabled-but-unavailable ({})
        # regime, not None (which would have opened the gate instead).
        assert mock_build_aggregator.call_args[0][2] == {}
        assert aggregator._regime_refresh_error == "refresh_failed"
        mock_create_engine.return_value.dispose.assert_called_once()


# ===========================================================================
# main() — Task 2: every setup step used to run BEFORE the try/finally that
# disposes the engine, so a construction failure partway through (e.g.
# PaperStore finding its sqlite file locked by another bot) leaked the
# engine. The try/finally now starts right after engine construction.
# ===========================================================================

class TestMainSetupFailureStillDisposesEngine:
    def test_paper_store_construction_failure_still_disposes_engine_and_reraises(self):
        import sqlite3

        with patch("execution.bot_runner.create_engine") as mock_create_engine, \
             patch("execution.bot_runner.build_market_data_client"), \
             patch("execution.bot_runner.resolve_regime", return_value={}), \
             patch("execution.bot_runner.build_aggregator"), \
             patch("execution.paper.PaperStore",
                   side_effect=sqlite3.OperationalError("database is locked")), \
             patch("execution.bot_runner.run_once") as mock_run_once:
            with pytest.raises(sqlite3.OperationalError):
                bot_runner.main(["--once", "--paper"])

        mock_create_engine.return_value.dispose.assert_called_once()
        mock_run_once.assert_not_called()
