from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier
from unittest.mock import MagicMock

import pytest

from config.strategy import StrategyConfig
from dashboard.data import comparison_rows, load_experiments, read_operations
from execution.bot_runner import run_once
from execution.freshness import FreshnessMonitor, classify_source
from execution.journal import DecisionJournal
from execution.paper import PaperConfig, PaperExecutor, PaperRiskManager, PaperStore
from signals.base import AggregatedSignal, SignalDirection

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def signal(direction=SignalDirection.BUY, symbol="BTCUSDT"):
    return AggregatedSignal(direction, 0.9, symbol, NOW, [], "test signal", {"technical": 1.0})


@pytest.mark.parametrize(
    "age,status",
    [(None, "missing"), (-1, "future"), (0, "fresh"), (10800, "fresh"), (10801, "stale")],
)
def test_source_freshness_boundaries(age, status):
    timestamp = NOW - timedelta(seconds=age) if age is not None else None
    assert classify_source("candles", timestamp, 10800, NOW).status == status


@pytest.mark.parametrize("limit", [0, -1, float("nan"), float("inf")])
def test_invalid_freshness_threshold(limit):
    with pytest.raises(ValueError):
        FreshnessMonitor(MagicMock(), limit)


def test_monitor_handles_missing_stream_without_blocking_entry():
    engine = MagicMock()
    now = datetime.now(UTC)
    engine.connect.return_value.__enter__.return_value.execute.return_value.scalar.side_effect = [
        now,
        now,
        None,
    ]
    engine.connect.return_value.__enter__.return_value.execute.return_value.first.return_value = (
        True, now,
    )
    allowed, reason, rows = FreshnessMonitor(engine).entry_check("BTCUSDT")
    assert allowed
    assert reason == "data fresh"
    assert rows[-2]["status"] == "missing"
    assert rows[-1]["status"] == "fresh"


def test_monitor_unavailable_required_source_blocks_entry():
    engine = MagicMock()
    engine.connect.side_effect = RuntimeError("offline")
    allowed, reason, _ = FreshnessMonitor(engine).entry_check("BTCUSDT")
    assert not allowed
    assert "unavailable" in reason


def test_stale_buy_blocked_then_recovers_without_blocking_exits(tmp_path):
    journal = DecisionJournal(tmp_path / "ops.sqlite3")
    executor = MagicMock(mode="paper")
    executor.close_expired_positions.return_value = []
    executor.execute_signal.return_value = {"status": "open"}
    aggregator = MagicMock()
    aggregator.aggregate.return_value = signal()
    freshness = MagicMock()
    freshness.entry_check.side_effect = [(False, "technical: stale", []), (True, "data fresh", [])]

    run_once(["BTCUSDT"], executor, aggregator, freshness=freshness, journal=journal)
    executor.execute_signal.assert_not_called()
    executor.close_expired_positions.assert_called_once()
    run_once(["BTCUSDT"], executor, aggregator, freshness=freshness, journal=journal)
    executor.execute_signal.assert_called_once()
    assert [r["outcome"] for r in read_operations(journal.path)["decisions"]] == [
        "executed",
        "blocked",
    ]


def test_sell_still_executes_when_freshness_unavailable(tmp_path):
    executor = MagicMock(mode="paper")
    executor.close_expired_positions.return_value = []
    executor.execute_signal.return_value = {"status": "closed"}
    aggregator = MagicMock()
    aggregator.aggregate.return_value = signal(SignalDirection.SELL)
    freshness = MagicMock()
    freshness.entry_check.side_effect = RuntimeError("must not be used for exits")
    run_once(["BTCUSDT"], executor, aggregator, freshness=freshness)
    freshness.entry_check.assert_not_called()
    executor.execute_signal.assert_called_once()


def test_journal_survives_restart_and_preserves_config_and_signal(tmp_path):
    path = tmp_path / "ops.sqlite3"
    assert DecisionJournal(path).record(
        symbol="BTCUSDT",
        mode="paper",
        outcome="hold",
        reason="signal is HOLD",
        signal=signal(SignalDirection.HOLD),
        config=StrategyConfig(),
    )
    DecisionJournal(path)
    rows = read_operations(path, symbol="BTCUSDT", mode="paper")["decisions"]
    assert len(rows) == 1
    details = json.loads(rows[0]["details"])
    assert details["signal"]["direction"] == "HOLD"
    assert details["config"]["feature_max_age_seconds"] == 10800
    assert not read_operations(path, mode="live")["decisions"]


def test_journal_records_execution_rejection_and_aggregation_error(tmp_path):
    path = tmp_path / "ops.sqlite3"
    executor = MagicMock(mode="live")
    executor.close_expired_positions.return_value = []
    executor.execute_signal.return_value = None
    executor.last_decision = {"outcome": "rejected", "reason": "max open positions reached"}
    aggregator = MagicMock()
    aggregator.aggregate.side_effect = [signal(), RuntimeError("no model")]
    run_once(["BTCUSDT", "ETHUSDT"], executor, aggregator, journal=DecisionJournal(path))
    rows = read_operations(path)["decisions"]
    assert rows[0]["outcome"] == "error"
    assert rows[1]["reason"] == "max open positions reached"


def paper_components(path, *, initial=1000.0, config=None):
    store = PaperStore(path, initial)
    client = MagicMock()
    client.get_current_price.return_value = 100.0
    client.get_symbol_info_cached.return_value = {
        "filters": [
            {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "100000", "stepSize": "0.001"},
            {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
            {"filterType": "MIN_NOTIONAL", "minNotional": "10"},
        ]
    }
    risk = PaperRiskManager(store, MagicMock())
    risk.get_atr = lambda symbol: 2.0
    executor = PaperExecutor(client, store, risk, config, clock=lambda: NOW)
    return store, client, executor


def test_paper_restart_balances_fees_and_signal_close(tmp_path):
    path = tmp_path / "ops.sqlite3"
    store, client, executor = paper_components(path)
    trade = executor.execute_signal(signal())
    assert trade["entry_price"] == pytest.approx(100.05)
    assert executor.risk.get_account_balance() == pytest.approx(
        1000 - trade["quantity"] * trade["entry_price"] - trade["entry_fee"]
    )
    client.get_account_balance.assert_not_called()
    client.client.order_market_buy.assert_not_called()
    store.close()

    store, client, executor = paper_components(path, initial=99999)
    try:
        assert len(store.positions()) == 1
        client.get_current_price.return_value = 110
        closed = executor.execute_signal(signal(SignalDirection.SELL))
        assert closed["exit_price"] == pytest.approx(109.945)
        expected_pnl = (
            (closed["exit_price"] - trade["entry_price"]) * trade["quantity"]
            - trade["entry_fee"]
            - closed["exit_fee"]
        )
        assert closed["pnl"] == pytest.approx(expected_pnl)
        assert executor.risk.get_account_balance() == pytest.approx(1000 + expected_pnl)
        assert not store.positions()
        assert executor.execute_signal(signal(SignalDirection.SELL)) is None
        client.client.order_market_sell.assert_not_called()
    finally:
        store.close()


@pytest.mark.parametrize(
    "price,hours,reason", [(90, 0, "stop_loss"), (110, 0, "take_profit"), (100, 24, "time_stop")]
)
def test_paper_protective_and_timed_exits(tmp_path, price, hours, reason):
    store, client, executor = paper_components(tmp_path / "ops.sqlite3")
    try:
        executor.execute_signal(signal())
        executor.clock = lambda: NOW + timedelta(hours=hours)
        client.get_current_price.return_value = price
        closed = executor.close_expired_positions(24)
        assert len(closed) == 1
        assert closed[0]["close_reason"] == reason
        assert not executor.close_expired_positions(24)
    finally:
        store.close()


def test_paper_transaction_rolls_back_trade_on_cash_update_failure(tmp_path):
    store, _, executor = paper_components(tmp_path / "ops.sqlite3")
    try:
        store.conn.execute("""CREATE TRIGGER fail_cash BEFORE UPDATE OF cash ON paper_account
                            BEGIN SELECT RAISE(ABORT, 'test failure'); END""")
        with pytest.raises(Exception, match="test failure"):
            executor.execute_signal(signal())
        assert not store.positions()
        assert executor.risk.get_account_balance() == 1000
    finally:
        store.close()


def test_concurrent_paper_entries_cannot_duplicate_position(tmp_path):
    path = tmp_path / "ops.sqlite3"
    PaperStore(path, 1000).close()
    barrier = Barrier(2)

    def enter():
        store, _, executor = paper_components(path)
        try:
            barrier.wait(timeout=10)
            return executor.execute_signal(signal())
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: enter(), range(2)))
    assert sum(result is not None for result in results) == 1
    assert len(read_operations(path)["positions"]) == 1


def test_paper_spending_includes_fees(tmp_path):
    store, _, executor = paper_components(
        tmp_path / "ops.sqlite3", config=PaperConfig(fee_rate=0.1)
    )
    try:
        executor.risk.get_atr = lambda symbol: 0.01
        trade = executor.execute_signal(signal())
        assert trade is not None
        assert trade["quantity"] * trade["entry_price"] + trade["entry_fee"] <= 1000
        assert executor.risk.get_account_balance() >= 0
    finally:
        store.close()


@pytest.mark.parametrize(
    "config",
    [{"initial_balance": 0}, {"fee_rate": -1}, {"slippage_bps": 10000}, {"fee_rate": float("nan")}],
)
def test_paper_config_rejects_invalid_numbers(config):
    with pytest.raises(ValueError):
        PaperConfig(**config)


def write_experiment(root, name="exp001", start="2026-01-01T00:00:00+00:00"):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_text(
        json.dumps(
            {
                "trade_count": 4,
                "total_return": 0.1,
                "max_drawdown": 0.02,
                "profit_factor": 1.5,
                "per_symbol": {"BTCUSDT": {"trade_count": 2, "pnl": 10, "win_rate": 0.5}},
            }
        )
    )
    (folder / "config.json").write_text(json.dumps({"symbols": ["BTCUSDT"]}))
    (folder / "equity_curve.csv").write_text(f"timestamp,equity\n{start},1000\n")


def test_experiment_loading_comparison_and_invalid_artifact(tmp_path):
    write_experiment(tmp_path)
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "metrics.json").write_text("{")
    runs, warnings = load_experiments(tmp_path)
    assert len(runs) == 1
    assert len(warnings) == 1
    assert comparison_rows(runs)[0]["total_return"] == 0.1
    assert comparison_rows(runs, "BTCUSDT")[0]["pnl"] == 10
    assert comparison_rows(runs, "ETHUSDT") == []


def test_reading_missing_operations_store_does_not_create_it(tmp_path):
    path = tmp_path / "missing.sqlite3"
    assert read_operations(path)["account"] == []
    assert not path.exists()


@pytest.mark.parametrize(
    "bad_metrics", [{"per_symbol": None}, {"total_return": "bad"}, {"per_symbol": {"BTCUSDT": []}}]
)
def test_experiment_invalid_metric_shapes_are_skipped(tmp_path, bad_metrics):
    write_experiment(tmp_path)
    (tmp_path / "exp001" / "metrics.json").write_text(json.dumps(bad_metrics))
    runs, warnings = load_experiments(tmp_path)
    assert runs == []
    assert len(warnings) == 1


def test_paper_automatic_exit_is_journaled_before_hold(tmp_path):
    path = tmp_path / "ops.sqlite3"
    store, client, executor = paper_components(path)
    try:
        executor.execute_signal(signal())
        client.get_current_price.return_value = 90
        aggregator = MagicMock()
        aggregator.aggregate.return_value = signal(SignalDirection.HOLD)
        run_once(["BTCUSDT"], executor, aggregator, journal=DecisionJournal(path))
        rows = read_operations(path)["decisions"]
        assert rows[0]["outcome"] == "hold"
        assert rows[1]["symbol"] == "BTCUSDT"
        assert rows[1]["reason"] == "stop_loss"
        assert json.loads(rows[0]["details"])["paper_config"]["fee_rate"] == 0.001
    finally:
        store.close()
