"""Unit tests for signals.regime — daily BTC-level macro regime detection,
the Phase 12 candidate set pre-registered in docs/experiments.md before
this module existed (R1 SMA200, R2 hysteresis cross, R3 vol percentile,
R4 composite).

All synthetic, no DB — regime functions take pre-loaded OHLCVBar lists.

Run:
    poetry run pytest tests/test_signals_regime.py -v
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from backtest.exit_simulation import OHLCVBar
from signals.regime import (
    RegimeState,
    compute_regime_r1_sma200,
    compute_regime_r2_hysteresis_cross,
    compute_regime_r3_vol_percentile,
    compute_regime_r4_composite,
    days_blocked_pct,
    resample_daily_close,
    utc_date,
)

_START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _hourly_bars(daily_closes: list[float]) -> list[OHLCVBar]:
    """3 hourly bars/day, last one carrying the day's close."""
    bars = []
    for day_idx, close in enumerate(daily_closes):
        day_start = _START + timedelta(days=day_idx)
        for h in range(3):
            bars.append(OHLCVBar(
                timestamp=day_start + timedelta(hours=h),
                open=close, high=close, low=close,
                close=close if h == 2 else close - 1.0,
            ))
    return bars


class TestResampleDailyClose:

    def test_takes_last_close_per_day(self):
        bars = _hourly_bars([100.0, 101.0])
        daily = resample_daily_close(bars)

        assert len(daily) == 2
        assert daily[0] == (_START.date(), 100.0)
        assert daily[1] == ((_START + timedelta(days=1)).date(), 101.0)

    def test_sorted_ascending_regardless_of_input_order(self):
        bars = _hourly_bars([100.0, 101.0, 102.0])
        daily = resample_daily_close(list(reversed(bars)))
        assert [d for d, _ in daily] == sorted(d for d, _ in daily)

    def test_empty_input_returns_empty(self):
        assert resample_daily_close([]) == []


class TestRegimeR1Sma200:

    def test_uptrend_above_sma_is_risk_on(self):
        # 250 days flat at 100 (warm-up), then a clean uptrend
        closes = [100.0] * 210 + [100.0 + i for i in range(1, 20)]
        daily = resample_daily_close(_hourly_bars(closes))

        regime = compute_regime_r1_sma200(daily)

        last_date = daily[-1][0]
        assert regime[last_date] == RegimeState.RISK_ON

    def test_downtrend_below_sma_is_risk_off(self):
        closes = [100.0] * 210 + [100.0 - i for i in range(1, 20)]
        daily = resample_daily_close(_hourly_bars(closes))

        regime = compute_regime_r1_sma200(daily)

        last_date = daily[-1][0]
        assert regime[last_date] == RegimeState.RISK_OFF

    def test_warmup_before_200_days_defaults_risk_on(self):
        closes = [100.0 - i for i in range(50)]  # sharp downtrend, but <200 days
        daily = resample_daily_close(_hourly_bars(closes))

        regime = compute_regime_r1_sma200(daily)

        assert all(s == RegimeState.RISK_ON for s in regime.values())


class TestRegimeR2HysteresisCross:

    def test_no_flap_within_band_around_crossover(self):
        # sma50/sma200 hover within +-2% of each other -- must not flap
        # bar-to-bar; small parallel oscillation shouldn't cross the band.
        closes = [100.0] * 200 + [100.0 + (0.5 if i % 2 == 0 else -0.5) for i in range(60)]
        daily = resample_daily_close(_hourly_bars(closes))

        regime = compute_regime_r2_hysteresis_cross(daily, band=0.02)
        tail_states = list(regime.values())[-20:]

        assert len(set(tail_states)) == 1, "oscillating within the band must not flip state"

    def test_switches_to_risk_off_when_band_cleared(self):
        closes = [100.0] * 210 + [100.0 - i * 2 for i in range(1, 30)]  # clear, sustained drop
        daily = resample_daily_close(_hourly_bars(closes))

        regime = compute_regime_r2_hysteresis_cross(daily, band=0.02)

        assert list(regime.values())[-1] == RegimeState.RISK_OFF


class TestRegimeR3VolPercentile:

    def test_high_vol_period_is_risk_off(self):
        import random
        rng = random.Random(7)
        calm = [100.0 + rng.uniform(-0.1, 0.1) for _ in range(60)]
        volatile = [100.0]
        for _ in range(40):
            volatile.append(volatile[-1] * (1 + rng.uniform(-0.08, 0.08)))
        closes = calm + volatile
        daily = resample_daily_close(_hourly_bars(closes))

        regime = compute_regime_r3_vol_percentile(daily, window=30, threshold_pct=80)

        last_date = daily[-1][0]
        assert regime[last_date] == RegimeState.RISK_OFF

    def test_no_lookahead_uses_only_past_and_current_data(self):
        """The percentile rank at day i must not change if future days
        (after i) are altered -- verifies no future-data leakage."""
        closes_a = [100.0 + (i % 5) * 0.1 for i in range(80)]
        closes_b = list(closes_a)
        closes_b[-1] = 500.0  # a wild change on the LAST day only

        daily_a = resample_daily_close(_hourly_bars(closes_a))
        daily_b = resample_daily_close(_hourly_bars(closes_b))

        regime_a = compute_regime_r3_vol_percentile(daily_a)
        regime_b = compute_regime_r3_vol_percentile(daily_b)

        second_to_last_date = daily_a[-2][0]
        assert regime_a[second_to_last_date] == regime_b[second_to_last_date]


class TestRegimeR4Composite:

    def test_risk_off_when_either_subsignal_is_risk_off(self):
        # Uptrend (R1 risk-on) but volatile (R3 likely risk-off) -> composite risk-off
        import random
        rng = random.Random(3)
        closes = [100.0] * 200
        price = 100.0
        for _ in range(60):
            price *= (1 + rng.uniform(-0.06, 0.09))  # net-up but choppy/volatile
            closes.append(price)
        daily = resample_daily_close(_hourly_bars(closes))

        r1 = compute_regime_r1_sma200(daily)
        r3 = compute_regime_r3_vol_percentile(daily)
        composite = compute_regime_r4_composite(daily)

        last_date = daily[-1][0]
        if r1[last_date] == RegimeState.RISK_OFF or r3[last_date] == RegimeState.RISK_OFF:
            assert composite[last_date] == RegimeState.RISK_OFF

    def test_risk_on_only_when_both_subsignals_risk_on(self):
        closes = [100.0] * 210 + [100.0 + i * 0.1 for i in range(1, 40)]  # calm uptrend
        daily = resample_daily_close(_hourly_bars(closes))

        r1 = compute_regime_r1_sma200(daily)
        r3 = compute_regime_r3_vol_percentile(daily)
        composite = compute_regime_r4_composite(daily)

        last_date = daily[-1][0]
        expected = (RegimeState.RISK_ON if r1[last_date] == RegimeState.RISK_ON
                    and r3[last_date] == RegimeState.RISK_ON else RegimeState.RISK_OFF)
        assert composite[last_date] == expected


class TestDaysBlockedPct:

    def test_computes_risk_off_percentage(self):
        regime = {
            date(2026, 1, 1): RegimeState.RISK_ON,
            date(2026, 1, 2): RegimeState.RISK_OFF,
            date(2026, 1, 3): RegimeState.RISK_OFF,
            date(2026, 1, 4): RegimeState.RISK_ON,
        }
        days = set(regime)
        assert days_blocked_pct(regime, days) == pytest.approx(50.0)

    def test_empty_days_returns_zero(self):
        assert days_blocked_pct({}, set()) == 0.0

    def test_unclassified_days_count_as_blocked_not_zero(self):
        """The exact defect this module used to have: an empty (or
        gapped) regime dict over a non-empty scored-days set is 100%
        blocked, not 0% — matches
        backtest.experiment._compute_metrics semantics."""
        days = {date(2026, 1, 1), date(2026, 1, 2)}
        assert days_blocked_pct({}, days) == pytest.approx(100.0)

    def test_missing_days_mixed_with_classified_days(self):
        regime = {date(2026, 1, 1): RegimeState.RISK_ON}
        days = {date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)}
        # day 1 RISK_ON (not blocked), days 2 and 3 unclassified (blocked)
        assert days_blocked_pct(regime, days) == pytest.approx(200 / 3)

    def test_regime_keys_outside_days_are_ignored(self):
        regime = {
            date(2026, 1, 1): RegimeState.RISK_OFF,
            date(2026, 1, 9): RegimeState.RISK_OFF,  # not in days
        }
        days = {date(2026, 1, 1)}
        assert days_blocked_pct(regime, days) == pytest.approx(100.0)


class TestUtcDate:

    def test_naive_datetime_treated_as_already_utc(self):
        naive = datetime(2026, 1, 1, 23, 30)
        assert utc_date(naive) == date(2026, 1, 1)

    def test_non_utc_aware_datetime_normalized_to_utc_calendar_day(self):
        # UTC+10 at 05:30 local == Dec 31 19:30 UTC -- the previous day.
        plus10 = timezone(timedelta(hours=10))
        local = datetime(2026, 1, 1, 5, 30, tzinfo=plus10)
        assert utc_date(local) == date(2025, 12, 31)

    def test_utc_aware_datetime_unchanged(self):
        aware = datetime(2026, 1, 1, 23, 30, tzinfo=timezone.utc)
        assert utc_date(aware) == date(2026, 1, 1)
