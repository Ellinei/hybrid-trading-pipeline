"""Macro regime detection: daily BTC-level RISK_ON/RISK_OFF classification,
used as a post-fusion entry gate (blocks BUY, never SELL — exits always
work) so the strategy survives correlated BTC/ETH/SOL downtrends.

Daily BTC-level, not per-symbol/hourly — exp005's --require-uptrend
(hourly close_vs_sma50, gated per-symbol) blocked 28 of 35 walk-forward
windows and was rejected as too blunt. Four candidates were pre-registered
in docs/experiments.md BEFORE this module existed, specifically to avoid
picking a filter because it happens to fix the already-spent Jan-2026
holdout in hindsight. Test in that order; stop at the first that clears
the acceptance gate.

Every function operates on a single symbol's already-loaded hourly OHLCV
bars (BTCUSDT) — no new ingestion, no DB access here.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from datetime import UTC, date, timedelta

from backtest.exit_simulation import OHLCVBar


class RegimeState(enum.Enum):
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"


def resample_daily_close(bars: list[OHLCVBar]) -> list[tuple[date, float]]:
    """Last hourly close per UTC calendar day, sorted ascending by date."""
    daily: dict[date, float] = {}
    for bar in sorted(bars, key=lambda b: b.timestamp):
        daily[bar.timestamp.date()] = bar.close
    return sorted(daily.items())


def _sma(values: list[float], window: int) -> list[float | None]:
    """Simple moving average; None until the window is fully populated."""
    out: list[float | None] = []
    for i in range(len(values)):
        if i + 1 < window:
            out.append(None)
        else:
            out.append(sum(values[i + 1 - window:i + 1]) / window)
    return out


def _daily_log_returns(closes: list[float]) -> list[float | None]:
    out: list[float | None] = [None]
    for i in range(1, len(closes)):
        out.append(math.log(closes[i] / closes[i - 1]) if closes[i - 1] > 0 else None)
    return out


def compute_regime_r1_sma200(daily_close: list[tuple[date, float]]) -> dict[date, RegimeState]:
    """R1: BTC close vs 200-day SMA — classic trend gate. Warm-up (fewer
    than 200 days of history) defaults to RISK_ON: don't block trades
    before there's enough data to judge the trend."""
    dates = [d for d, _ in daily_close]
    closes = [c for _, c in daily_close]
    sma200 = _sma(closes, 200)
    return {
        d: RegimeState.RISK_ON if sma is None or close > sma else RegimeState.RISK_OFF
        for d, close, sma in zip(dates, closes, sma200, strict=False)
    }


def compute_regime_r2_hysteresis_cross(
    daily_close: list[tuple[date, float]], band: float = 0.02,
) -> dict[date, RegimeState]:
    """R2: BTC 50d/200d SMA cross with a ±2% hysteresis band around
    sma200, to avoid flapping when the two averages sit close together.
    Once RISK_ON, only flips to RISK_OFF when sma50 drops more than
    *band* below sma200 (and symmetrically for the reverse) — a plain
    crossover with no band would flip state on every small oscillation
    right at the crossing point."""
    dates = [d for d, _ in daily_close]
    closes = [c for _, c in daily_close]
    sma50 = _sma(closes, 50)
    sma200 = _sma(closes, 200)

    regime: dict[date, RegimeState] = {}
    state = RegimeState.RISK_ON
    for d, s50, s200 in zip(dates, sma50, sma200, strict=False):
        if s50 is None or s200 is None:
            regime[d] = state
            continue
        lower = s200 * (1 - band)
        upper = s200 * (1 + band)
        if state == RegimeState.RISK_ON and s50 < lower:
            state = RegimeState.RISK_OFF
        elif state == RegimeState.RISK_OFF and s50 > upper:
            state = RegimeState.RISK_ON
        regime[d] = state
    return regime


def compute_regime_r3_vol_percentile(
    daily_close: list[tuple[date, float]], window: int = 30, threshold_pct: float = 80.0,
) -> dict[date, RegimeState]:
    """R3: 30-day realized volatility (stdev of daily log returns),
    percentile-ranked against its own trailing (expanding, past-only)
    history — RISK_OFF above the 80th percentile. The rank at day i uses
    only vol observations up to and including day i, so it can't leak
    future data into today's classification."""
    dates = [d for d, _ in daily_close]
    closes = [c for _, c in daily_close]
    log_returns = _daily_log_returns(closes)

    vol: list[float | None] = []
    for i in range(len(closes)):
        window_returns = [r for r in log_returns[max(0, i - window + 1):i + 1] if r is not None]
        if len(window_returns) < window:
            vol.append(None)
        else:
            mean = sum(window_returns) / len(window_returns)
            variance = sum((r - mean) ** 2 for r in window_returns) / len(window_returns)
            vol.append(math.sqrt(variance))

    regime: dict[date, RegimeState] = {}
    vol_history: list[float] = []
    for d, v in zip(dates, vol, strict=False):
        if v is None:
            regime[d] = RegimeState.RISK_ON
            continue
        vol_history.append(v)
        if len(vol_history) < window:
            regime[d] = RegimeState.RISK_ON
            continue
        rank_pct = sum(1 for x in vol_history if x <= v) / len(vol_history) * 100
        regime[d] = RegimeState.RISK_OFF if rank_pct > threshold_pct else RegimeState.RISK_ON
    return regime


def compute_regime_r4_composite(daily_close: list[tuple[date, float]]) -> dict[date, RegimeState]:
    """R4: trend AND vol — RISK_ON only when R1 (trend) AND R3 (vol) both
    say RISK_ON; RISK_OFF if either alone would block."""
    r1 = compute_regime_r1_sma200(daily_close)
    r3 = compute_regime_r3_vol_percentile(daily_close)
    return {
        d: RegimeState.RISK_ON if r1[d] == RegimeState.RISK_ON and r3[d] == RegimeState.RISK_ON
        else RegimeState.RISK_OFF
        for d in r1
    }


REGIME_FUNCTIONS = {
    "r1_sma200": compute_regime_r1_sma200,
    "r2_hysteresis_cross": compute_regime_r2_hysteresis_cross,
    "r3_vol_percentile": compute_regime_r3_vol_percentile,
    "r4_composite": compute_regime_r4_composite,
}


@dataclass(frozen=True)
class RegimeDecision:
    state: RegimeState | None
    source_date: date | None
    blocking_reason: str | None


def regime_decision(
    regime: dict[date, RegimeState] | None,
    decision_date: date,
    refresh_error: str | None = None,
) -> RegimeDecision:
    """Entry gate result for a UTC decision date. None deliberately disables it."""
    if regime is None and refresh_error is None:
        return RegimeDecision(None, None, None)
    source_date = decision_date - timedelta(days=1)
    present = isinstance(regime, dict) and decision_date in regime
    state = regime.get(decision_date) if isinstance(regime, dict) else None
    if not isinstance(state, RegimeState):
        state = None
    if refresh_error:
        return RegimeDecision(state, source_date, refresh_error)
    if state is None:
        return RegimeDecision(None, source_date,
                              "classification_invalid" if present else "classification_unavailable")
    if state == RegimeState.RISK_OFF:
        return RegimeDecision(state, source_date, "risk_off")
    return RegimeDecision(state, source_date, None)


def utc_date(ts):
    """Calendar date of *ts* in UTC. Naive datetimes are treated as already
    UTC (matching build_causal_regime's own bars, whose timestamps carry no
    tzinfo but are UTC by construction) rather than being interpreted as
    local time, which .astimezone() would otherwise silently do."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC).date()


def build_causal_regime(
    bars: list[OHLCVBar], candidate: str,
) -> dict[date, RegimeState]:
    """Classify complete BTC UTC days for entries on the *following* day.

    Hourly candle timestamps denote candle open. A day is complete only
    after all 24 hourly candles exist with finite positive closes. Missing
    or invalid days reset the consecutive-history warm-up; no decision is
    emitted until the candidate's full history is available again.
    """
    if candidate not in REGIME_FUNCTIONS:
        raise ValueError(
            f"Unknown regime candidate: {candidate!r}. Choices: {sorted(REGIME_FUNCTIONS)}"
        )
    required_days = {"r1_sma200": 200, "r2_hysteresis_cross": 200,
                     "r3_vol_percentile": 60, "r4_composite": 200}[candidate]
    daily_hours: dict[date, dict[int, float]] = {}
    invalid_days: set[date] = set()
    for bar in bars:
        ts = bar.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        ts = ts.astimezone(UTC)
        day = utc_date(ts)
        if (ts.minute or ts.second or ts.microsecond
                or not math.isfinite(bar.close) or bar.close <= 0):
            invalid_days.add(day)
            continue
        hours = daily_hours.setdefault(day, {})
        if ts.hour in hours:
            invalid_days.add(day)
        hours[ts.hour] = bar.close

    decisions: dict[date, RegimeState] = {}
    streak: list[tuple[date, float]] = []

    def flush() -> None:
        if len(streak) < required_days:
            return
        states = REGIME_FUNCTIONS[candidate](streak)
        for index in range(required_days - 1, len(streak)):
            day = streak[index][0]
            decisions[day + timedelta(days=1)] = states[day]

    previous: date | None = None
    for day in sorted(set(daily_hours) | invalid_days):
        hours = daily_hours.get(day, {})
        if (day in invalid_days or len(hours) != 24
                or (previous is not None and day != previous + timedelta(days=1))):
            flush()
            streak = []
        previous = day
        if day in invalid_days or len(hours) != 24:
            continue
        streak.append((day, hours[23]))
    flush()
    return decisions


def days_blocked_pct(regime: dict[date, RegimeState], days: set[date]) -> float:
    """% of *days* RISK_OFF or unclassified — the diagnostic that would
    have caught exp005's ~80%-blocked over-filtering before it reached a
    full run. *days* must be the actual days scored (not regime's own
    keys): a day missing from *regime* still blocks entries exactly like
    RISK_OFF (matches backtest.experiment._compute_metrics), so an empty
    or gapped regime over a non-empty *days* set is NOT 0% blocked."""
    if not days:
        return 0.0
    blocked = sum(1 for d in days if regime.get(d) != RegimeState.RISK_ON)
    return blocked / len(days) * 100
