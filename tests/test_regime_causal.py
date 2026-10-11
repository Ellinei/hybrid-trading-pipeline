"""Decision-time BTC regime contract shared by live and replay."""
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd

from backtest.exit_simulation import OHLCVBar
from backtest.signal_cache import ReplayAggregator
from config.strategy import StrategyConfig
from execution.bot_runner import run_once
from signals.aggregator import SignalAggregator
from signals.base import Signal, SignalDirection
from signals.regime import RegimeState, build_causal_regime

START = datetime(2025, 1, 1, tzinfo=UTC)


def btc_days(closes):
    return [
        OHLCVBar(START + timedelta(days=day, hours=hour), close, close, close, close)
        for day, close in enumerate(closes)
        for hour in range(24)
    ]


def buy_row(at):
    return pd.DataFrame([{
        "symbol": "ETHUSDT", "timestamp": at,
        "rsi_14_proxy": 25.0, "macd_line": 0.5, "bb_zscore": 0.0,
        "pct_change_1h": 0.5, "pct_change_24h": 0.0,
        "close_vs_sma50": 0.0,
    }])


def live_aggregator(regime, direction=SignalDirection.BUY):
    class Generator:
        def generate_signal(self, symbol, as_of=None):
            return Signal(direction, 0.8, "technical", symbol, as_of, {})

    class Neutral:
        def generate_signal(self, symbol, as_of=None):
            return Signal(SignalDirection.HOLD, 0.5, "ml", symbol, as_of, {})

    agg = SignalAggregator.__new__(SignalAggregator)
    agg.weights = {"technical": 0.4, "ml": 0.4, "sentiment": 0.2}
    agg.buy_gate = 0.15
    agg.sell_gate = -0.15
    agg.technical = Generator()
    agg.ml = Neutral()
    agg.sentiment = Neutral()
    agg._neutral_sentiment = Neutral()
    agg._regime = regime
    return agg


def test_future_btc_close_cannot_change_earlier_intraday_entry():
    bars = btc_days([100.0] * 199 + [200.0, 200.0])
    changed = list(bars)
    last = changed[-1]
    changed[-1] = OHLCVBar(last.timestamp, 1.0, 1.0, 1.0, 1.0)
    decision = START + timedelta(days=200, hours=1)

    first = build_causal_regime(bars, "r1_sma200")
    second = build_causal_regime(changed, "r1_sma200")

    assert first[decision.date()] == second[decision.date()] == RegimeState.RISK_ON
    first_result = ReplayAggregator(buy_row(decision), {}, regime=first).aggregate(
        "ETHUSDT", decision,
    )
    second_result = ReplayAggregator(buy_row(decision), {}, regime=second).aggregate(
        "ETHUSDT", decision,
    )
    assert first_result.direction == SignalDirection.BUY
    assert second_result.direction == SignalDirection.BUY


def test_utc_rollover_requires_completed_previous_btc_day():
    bars = btc_days([100.0] * 199 + [50.0])
    decision = START + timedelta(days=200)
    regime = build_causal_regime(bars, "r1_sma200")
    assert regime[decision.date()] == RegimeState.RISK_OFF
    assert (decision - timedelta(days=1)).date() not in regime
    result = ReplayAggregator(buy_row(decision), {}, regime=regime).aggregate(
        "ETHUSDT", decision,
    )
    assert result.direction == SignalDirection.HOLD
    assert live_aggregator(regime).aggregate("ETHUSDT", decision).direction == SignalDirection.HOLD


def test_incomplete_day_and_history_gap_leave_entry_unclassified():
    bars = btc_days([100.0] * 201)
    final_hour = START + timedelta(days=199, hours=23)
    missing_final_hour = [bar for bar in bars if bar.timestamp != final_hour]
    regime = build_causal_regime(missing_final_hour, "r1_sma200")
    assert (START + timedelta(days=200)).date() not in regime
    assert (START + timedelta(days=201)).date() not in regime


def test_enabled_gate_blocks_missing_or_invalid_state_on_fresh_eth():
    at = START + timedelta(days=200)
    for regime in ({}, {at.date(): "risk_on"}):
        live = live_aggregator(regime).aggregate("ETHUSDT", at)
        replay = ReplayAggregator(buy_row(at), {}, regime=regime).aggregate("ETHUSDT", at)
        assert live.direction == replay.direction == SignalDirection.HOLD
        assert "regime" in live.reasoning
        assert "regime" in replay.reasoning
    assert "classification_invalid" in live.reasoning


def test_disabled_gate_allows_buy_and_missing_gate_keeps_sell():
    at = START + timedelta(days=200)
    assert live_aggregator(None).aggregate("ETHUSDT", at).direction == SignalDirection.BUY
    replay = ReplayAggregator(buy_row(at), {}, regime=None).aggregate("ETHUSDT", at)
    live = live_aggregator({}, SignalDirection.SELL).aggregate("ETHUSDT", at)
    assert replay.direction == SignalDirection.BUY
    assert live.direction == SignalDirection.SELL


def test_missing_btc_blocks_eth_buy_even_when_eth_is_fresh_and_journals_reason():
    aggregator = live_aggregator({})
    executor = MagicMock()
    journal = MagicMock()
    journal.record.return_value = True
    freshness = MagicMock()
    freshness.entry_check.return_value = (True, "fresh", [])
    cfg = StrategyConfig(regime_candidate="r1_sma200")

    with patch("execution.bot_runner.resolve_regime", return_value={}):
        run_once(["ETHUSDT"], executor, aggregator, engine=MagicMock(),
                 strategy_cfg=cfg, freshness=freshness, journal=journal)

    executor.execute_signal.assert_not_called()
    assert journal.record.call_args.kwargs["reason"] == "classification_unavailable"
    meta = journal.record.call_args.kwargs["signal"].signals[-1].metadata
    assert meta["state"] is None
    assert meta["source_date"]
    assert meta["entry_blocked"] is True


def test_non_utc_timestamp_uses_utc_calendar_date_for_regime_lookup():
    """Defensive normalization (Task 4): a tz-aware as_of in a non-UTC zone
    must be converted to its UTC calendar date before the regime lookup,
    not have .date() taken on its local wall-clock fields. UTC+10 at
    08:00 local on day D+1 is 22:00 UTC on day D -- a naive `.date()`
    would read D+1 and hit the wrong regime entry."""
    local_tz = timezone(timedelta(hours=10))
    utc_instant = START + timedelta(days=200, hours=22)  # 22:00 UTC on day D
    correct_utc_day = utc_instant.date()
    wrong_local_day = utc_instant.astimezone(local_tz).date()
    assert wrong_local_day == correct_utc_day + timedelta(days=1)

    regime = {
        correct_utc_day: RegimeState.RISK_ON,   # correct lookup -> allowed
        wrong_local_day: RegimeState.RISK_OFF,  # bug would land here -> blocked
    }
    as_of_local = utc_instant.astimezone(local_tz)

    live = live_aggregator(regime).aggregate("ETHUSDT", as_of_local)
    replay = ReplayAggregator(buy_row(utc_instant), {}, regime=regime).aggregate(
        "ETHUSDT", as_of_local
    )

    assert live.direction == SignalDirection.BUY
    assert replay.direction == SignalDirection.BUY


def test_refresh_failure_blocks_buy_then_recovery_allows_it():
    today = datetime.now(UTC).date()
    aggregator = live_aggregator({today: RegimeState.RISK_ON})
    executor = MagicMock()
    cfg = StrategyConfig(regime_candidate="r1_sma200")
    engine = MagicMock()

    with patch("execution.bot_runner.resolve_regime", side_effect=[RuntimeError("db gone"),
                                                               {today: RegimeState.RISK_ON}]):
        run_once(["ETHUSDT"], executor, aggregator, engine=engine, strategy_cfg=cfg)
        executor.execute_signal.assert_not_called()
        run_once(["ETHUSDT"], executor, aggregator, engine=engine, strategy_cfg=cfg)

    assert aggregator._regime_refresh_error is None
    assert executor.execute_signal.call_count == 1
