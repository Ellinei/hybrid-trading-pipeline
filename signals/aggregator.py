"""Multi-signal fusion: technical + ML + sentiment → single trade decision.

Usage:
    python -m signals.aggregator   # prints a signal report for all symbols
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import structlog
from dotenv import load_dotenv

from signals.base import AggregatedSignal, Signal, SignalDirection
from signals.ml_model import MLSignalGenerator
from signals.sentiment import NeutralSentimentSignalGenerator, SentimentSignalGenerator
from signals.technical import TechnicalSignalGenerator

log = structlog.get_logger()

DEFAULT_WEIGHTS: dict[str, float] = {
    "technical": 0.40,
    "ml":        0.40,
    "sentiment": 0.20,
}

_DIR_SCORE: dict[SignalDirection, int] = {
    SignalDirection.BUY:  1,
    SignalDirection.HOLD: 0,
    SignalDirection.SELL: -1,
}

_DISPLAY_NAME: dict[str, str] = {
    "technical": "Technical",
    "ml":        "ML",
    "sentiment": "Sentiment",
}


class SignalAggregator:
    """Fuses three signal sources into a single ``AggregatedSignal``."""

    def __init__(
        self,
        db_engine,
        weights: dict | None = None,
        buy_gate: float = 0.15,
        sell_gate: float = -0.15,
        technical: TechnicalSignalGenerator | None = None,
        ml: MLSignalGenerator | None = None,
        regime: dict | None = None,
    ) -> None:
        """*buy_gate*/*sell_gate* are the fused-score direction thresholds.
        The sell gate is the strategy's de facto exit threshold: SELL
        bypasses the risk manager's confidence check, so any fused score
        below *sell_gate* closes an open position.

        Pass pre-built *technical*/*ml* generators to inject non-default
        rule/threshold parameters (backtest sweeps, A/B toggles).

        *regime* is an optional dict[date, RegimeState] (signals.regime,
        Phase 12) — a daily BTC macro regime gate that blocks new BUYs
        when RISK_OFF. Mirrors ReplayAggregator's identical gate so
        backtest and live share the same semantics. None (the default)
        is a no-op — not yet enabled by default; Phase 13 wires it on via
        StrategyConfig."""
        self.weights   = weights or DEFAULT_WEIGHTS
        self.buy_gate  = buy_gate
        self.sell_gate = sell_gate
        self.technical = technical or TechnicalSignalGenerator(db_engine)
        self.ml        = ml or MLSignalGenerator(db_engine)
        self.sentiment = SentimentSignalGenerator()
        self._neutral_sentiment = NeutralSentimentSignalGenerator()
        self._regime = regime

    # ------------------------------------------------------------------
    # Core fusion
    # ------------------------------------------------------------------

    def aggregate(self, symbol: str, as_of: datetime | None = None) -> AggregatedSignal:
        now     = as_of if as_of is not None else datetime.now(timezone.utc)
        signals: list[Signal] = []

        # Live sentiment has no historical archive — when replaying a past
        # bar (as_of set), swap in the no-op neutral generator instead of
        # hitting live RSS feeds for "current" sentiment.
        sentiment_gen = self._neutral_sentiment if as_of is not None else self.sentiment

        for name, gen in (
            ("technical", self.technical),
            ("ml",        self.ml),
            ("sentiment", sentiment_gen),
        ):
            try:
                if as_of is not None:
                    signals.append(gen.generate_signal(symbol, as_of=as_of))
                else:
                    signals.append(gen.generate_signal(symbol))
            except Exception:
                log.exception("signal_generation_failed", source=name, symbol=symbol)
                signals.append(Signal(
                    direction=SignalDirection.HOLD, confidence=0.5,
                    source=name, symbol=symbol, timestamp=now,
                    metadata={"error": "generation_failed"},
                ))

        # Weighted score
        score = sum(
            _DIR_SCORE[s.direction] * s.confidence * self.weights.get(s.source, 0.0)
            for s in signals
        )

        if score > self.buy_gate:
            direction = SignalDirection.BUY
        elif score < self.sell_gate:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.HOLD

        # Regime gate: blocks new BUYs in a RISK_OFF macro regime; never
        # blocks SELL (exits must always work) or an already-HOLD signal.
        # getattr, not self._regime: existing tests construct this class
        # via __new__ (bypassing __init__) and don't set the attribute.
        regime = getattr(self, "_regime", None)
        if direction == SignalDirection.BUY and regime is not None:
            from signals.regime import RegimeState
            if regime.get(now.date()) == RegimeState.RISK_OFF:
                direction = SignalDirection.HOLD

        confidence = min(abs(score), 1.0)

        # Human-readable reasoning
        parts = []
        for s in signals:
            m    = s.metadata
            name = _DISPLAY_NAME.get(s.source, s.source)
            if s.source == "technical":
                rsi = m.get("rsi")
                hint = f"RSI={rsi:.1f}" if isinstance(rsi, float) else ""
            elif s.source == "ml":
                prob = m.get("probability_up")
                hint = f"prob_up={prob:.2f}" if prob is not None else "no_model"
            else:  # sentiment
                hint = f"score={m.get('sentiment_score', 0.0):.2f} articles={m.get('article_count', 0)}"
            parts.append(f"{name}: {s.direction.value}({s.confidence:.2f}) {hint}".strip())

        reasoning = (
            f"{direction.value} signal (confidence: {confidence:.2f}) | "
            + " | ".join(parts)
        )

        result = AggregatedSignal(
            direction=direction, confidence=confidence,
            symbol=symbol, timestamp=now,
            signals=signals, reasoning=reasoning,
            weights_used=self.weights,
        )
        log.info("signal_aggregated", symbol=symbol,
                 direction=direction.value, confidence=round(confidence, 3))
        return result

    def run_all_symbols(self, symbols: list[str]) -> dict[str, AggregatedSignal]:
        return {sym: self.aggregate(sym) for sym in symbols}


# ---------------------------------------------------------------------------
# Entry point — print signal report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from urllib.parse import quote_plus

    from sqlalchemy import create_engine

    load_dotenv()

    engine = create_engine(
        "postgresql+psycopg2://{u}:{p}@{h}:{port}/{db}".format(
            u=quote_plus(os.getenv("POSTGRES_USER",     "trader")),
            p=quote_plus(os.getenv("POSTGRES_PASSWORD", "")),
            h=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            db=os.getenv("POSTGRES_DB",   "trading"),
        )
    )

    syms = [s.strip() for s in os.getenv("TRADING_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")]
    agg  = SignalAggregator(engine)
    results = agg.run_all_symbols(syms)

    print("\n=== Signal Report ===")
    for sym, sig in results.items():
        print(f"\n  {sym}:")
        print(f"    Direction  : {sig.direction.value}")
        print(f"    Confidence : {sig.confidence:.3f}")
        print(f"    Reasoning  : {sig.reasoning}")

    engine.dispose()
