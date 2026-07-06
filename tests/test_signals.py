"""Unit tests for the multi-signal engine.

No real DB, Binance, or Reddit connections — all I/O is mocked.

Run:
    poetry run pytest tests/test_signals.py -v
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from signals.base import AggregatedSignal, Signal, SignalDirection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    """Predictable env for all signal tests — no real external connections."""
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_USER", "test_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test_pass")
    monkeypatch.setenv("POSTGRES_DB", "test_db")
    monkeypatch.setenv("TRADING_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT")


def _features(**overrides) -> dict:
    """Build a minimal features dict with sensible defaults (all 18 ML features)."""
    base = {
        # Original 8
        "rsi_14_proxy":  50.0,
        "macd_line":      0.0,
        "bb_zscore":      0.0,
        "pct_change_1h":  0.0,
        "pct_change_24h": 0.0,
        "log_return_1h":  0.001,
        "range_14":    1000.0,
        "volume":       100.0,
        # Extended 10
        "log_return_7d":   0.01,
        "pct_change_4h":   0.0,
        "close_vs_sma20":  0.0,
        "close_vs_sma50":  0.0,
        "bb_width":        2.0,
        "atr_normalized":  1.5,
        "vol_ma_ratio":    1.0,
        "candle_body_pct": 0.3,
        "upper_wick_pct":  0.1,
        "stoch_k":        50.0,
    }
    base.update(overrides)
    return base


def _signal(direction: SignalDirection, confidence: float, source: str) -> Signal:
    return Signal(
        direction=direction,
        confidence=confidence,
        source=source,
        symbol="BTCUSDT",
        timestamp=datetime.now(timezone.utc),
        metadata={},
    )


# ===========================================================================
# 1 + 2 + 3.  Technical signal rules
# ===========================================================================

class TestTechnicalSignals:

    @patch("signals.technical.TechnicalSignalGenerator.get_latest_features")
    def test_technical_buy_signal(self, mock_feat):
        """RSI < 30 AND MACD > 0 AND pct_1h > 0 → strong BUY, confidence 0.8."""
        from signals.technical import TechnicalSignalGenerator

        mock_feat.return_value = _features(rsi_14_proxy=25.0, macd_line=0.5, pct_change_1h=0.01)
        signal = TechnicalSignalGenerator(MagicMock()).generate_signal("BTCUSDT")

        assert signal.direction  == SignalDirection.BUY
        assert signal.confidence == pytest.approx(0.8)
        assert signal.source     == "technical"

    @patch("signals.technical.TechnicalSignalGenerator.get_latest_features")
    def test_technical_sell_signal(self, mock_feat):
        """RSI > 70 AND MACD < 0 → strong SELL, confidence 0.8."""
        from signals.technical import TechnicalSignalGenerator

        mock_feat.return_value = _features(rsi_14_proxy=75.0, macd_line=-0.5)
        signal = TechnicalSignalGenerator(MagicMock()).generate_signal("BTCUSDT")

        assert signal.direction  == SignalDirection.SELL
        assert signal.confidence == pytest.approx(0.8)

    @patch("signals.technical.TechnicalSignalGenerator.get_latest_features")
    def test_technical_hold_signal(self, mock_feat):
        """RSI=50, MACD=0 → no rule matches → HOLD."""
        from signals.technical import TechnicalSignalGenerator

        mock_feat.return_value = _features(rsi_14_proxy=50.0, macd_line=0.0)
        signal = TechnicalSignalGenerator(MagicMock()).generate_signal("BTCUSDT")

        assert signal.direction  == SignalDirection.HOLD
        assert signal.confidence == pytest.approx(0.5)

    @patch("signals.technical.TechnicalSignalGenerator.get_latest_features")
    def test_buy_signal_bb_below_lower_band(self, mock_feat):
        """BB zscore < -2.0 → BUY, confidence 0.75."""
        from signals.technical import TechnicalSignalGenerator

        mock_feat.return_value = _features(rsi_14_proxy=50.0, macd_line=0.0, bb_zscore=-2.5)
        signal = TechnicalSignalGenerator(MagicMock()).generate_signal("BTCUSDT")

        assert signal.direction  == SignalDirection.BUY
        assert signal.confidence == pytest.approx(0.75)

    @patch("signals.technical.TechnicalSignalGenerator.get_latest_features")
    def test_hold_when_no_features_in_db(self, mock_feat):
        """No rows in DB → HOLD fallback."""
        from signals.technical import TechnicalSignalGenerator

        mock_feat.return_value = None
        signal = TechnicalSignalGenerator(MagicMock()).generate_signal("BTCUSDT")

        assert signal.direction == SignalDirection.HOLD

    @patch("signals.technical.TechnicalSignalGenerator.get_latest_features")
    def test_rsi_zero_is_not_replaced_by_default_and_triggers_buy(self, mock_feat):
        """RSI=0.0 is extreme oversold (valid value). Must not be replaced by 50.0
        via a falsy `or` default. With positive 1h momentum confirming the bounce,
        it must fire BUY (falling-knife guard requires pct_1h > 0)."""
        from signals.technical import TechnicalSignalGenerator

        mock_feat.return_value = _features(rsi_14_proxy=0.0, macd_line=0.1, pct_change_1h=0.01)
        signal = TechnicalSignalGenerator(MagicMock()).generate_signal("BTCUSDT")

        assert signal.direction  == SignalDirection.BUY
        assert signal.confidence == pytest.approx(0.8)

    @patch("signals.technical.TechnicalSignalGenerator.get_latest_features")
    def test_rsi_oversold_without_momentum_is_hold(self, mock_feat):
        """RSI<30 + MACD positive but pct_1h <= 0 → HOLD.
        Falling-knife guard: crypto RSI<30 without price already ticking up
        historically precedes consolidation, not a V-reversal."""
        from signals.technical import TechnicalSignalGenerator

        mock_feat.return_value = _features(rsi_14_proxy=20.0, macd_line=0.1, pct_change_1h=0.0)
        signal = TechnicalSignalGenerator(MagicMock()).generate_signal("BTCUSDT")

        assert signal.direction == SignalDirection.HOLD


# ===========================================================================
# 3a2. Technical rule toggles (Phase 3 A/B knobs — legacy defaults off)
# ===========================================================================

class TestTechnicalRuleToggles:

    def test_bb_guard_off_buys_falling_knife(self):
        """Legacy default: bb_zscore<-2 buys even while price is falling."""
        from signals.technical import TechnicalSignalGenerator

        feats = _features(bb_zscore=-2.5, pct_change_1h=-0.5)
        direction, confidence, _ = TechnicalSignalGenerator.evaluate_rules(feats)

        assert direction == SignalDirection.BUY
        assert confidence == pytest.approx(0.75)

    def test_bb_guard_on_blocks_falling_knife(self):
        """guard_bb_dip_buy=True requires pct_1h > 0 before the dip-buy."""
        from signals.technical import TechnicalSignalGenerator

        feats = _features(bb_zscore=-2.5, pct_change_1h=-0.5)
        direction, _, _ = TechnicalSignalGenerator.evaluate_rules(
            feats, guard_bb_dip_buy=True
        )

        assert direction == SignalDirection.HOLD

    def test_bb_guard_on_allows_bouncing_dip(self):
        from signals.technical import TechnicalSignalGenerator

        feats = _features(bb_zscore=-2.5, pct_change_1h=0.3)
        direction, confidence, _ = TechnicalSignalGenerator.evaluate_rules(
            feats, guard_bb_dip_buy=True
        )

        assert direction == SignalDirection.BUY
        assert confidence == pytest.approx(0.75)

    def test_regime_filter_blocks_buys_below_sma50(self):
        """require_uptrend_for_buys=True blocks BUY rules when close_vs_sma50 <= 0."""
        from signals.technical import TechnicalSignalGenerator

        feats = _features(rsi_14_proxy=25.0, macd_line=0.5,
                          pct_change_1h=0.5, close_vs_sma50=-3.0)
        direction, _, _ = TechnicalSignalGenerator.evaluate_rules(
            feats, require_uptrend_for_buys=True
        )

        assert direction == SignalDirection.HOLD

    def test_regime_filter_allows_buys_above_sma50(self):
        from signals.technical import TechnicalSignalGenerator

        feats = _features(rsi_14_proxy=25.0, macd_line=0.5,
                          pct_change_1h=0.5, close_vs_sma50=2.0)
        direction, confidence, _ = TechnicalSignalGenerator.evaluate_rules(
            feats, require_uptrend_for_buys=True
        )

        assert direction == SignalDirection.BUY
        assert confidence == pytest.approx(0.8)

    def test_regime_filter_never_blocks_sells(self):
        """SELL rules are exits — the regime filter must not gate them."""
        from signals.technical import TechnicalSignalGenerator

        feats = _features(rsi_14_proxy=75.0, macd_line=-0.5, close_vs_sma50=-3.0)
        direction, _, _ = TechnicalSignalGenerator.evaluate_rules(
            feats, require_uptrend_for_buys=True
        )

        assert direction == SignalDirection.SELL


# ===========================================================================
# 3b. Technical signal — as_of historical cutoff (for backtesting)
# ===========================================================================

class TestTechnicalAsOf:
    def test_as_of_none_queries_latest_row_unchanged(self):
        from signals.technical import TechnicalSignalGenerator

        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchone.return_value = None

        TechnicalSignalGenerator(engine).get_latest_features("BTCUSDT")

        sql, params = conn.execute.call_args[0]
        assert "as_of" not in params
        assert "<= :as_of" not in str(sql)

    def test_as_of_set_adds_timestamp_cutoff_to_query(self):
        from signals.technical import TechnicalSignalGenerator

        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchone.return_value = None
        cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)

        TechnicalSignalGenerator(engine).get_latest_features("BTCUSDT", as_of=cutoff)

        sql, params = conn.execute.call_args[0]
        assert params["as_of"] == cutoff
        assert "<= :as_of" in str(sql)

    @patch("signals.technical.TechnicalSignalGenerator.get_latest_features")
    def test_generate_signal_passes_as_of_through_and_uses_it_as_timestamp(self, mock_feat):
        from signals.technical import TechnicalSignalGenerator

        mock_feat.return_value = _features()
        cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)

        signal = TechnicalSignalGenerator(MagicMock()).generate_signal("BTCUSDT", as_of=cutoff)

        mock_feat.assert_called_once_with("BTCUSDT", as_of=cutoff)
        assert signal.timestamp == cutoff


# ===========================================================================
# 4.  ML signal — no model file on disk
# ===========================================================================

class TestMLSignal:

    @patch("signals.ml_model.MLSignalGenerator.is_trained", return_value=False)
    def test_ml_signal_no_model(self, _):
        """When model files don't exist → HOLD with confidence 0.5."""
        from signals.ml_model import MLSignalGenerator

        signal = MLSignalGenerator(MagicMock()).generate_signal("BTCUSDT")

        assert signal.direction  == SignalDirection.HOLD
        assert signal.confidence == pytest.approx(0.5)
        assert signal.source     == "ml"

    def test_walk_forward_mode_returns_hold_during_warmup(self):
        """set_walk_forward_mode(True) + no in-memory model → HOLD with warmup reason."""
        from signals.ml_model import MLSignalGenerator

        gen = MLSignalGenerator(MagicMock())
        gen.set_walk_forward_mode(True)

        signal = gen.generate_signal("BTCUSDT")

        assert signal.direction == SignalDirection.HOLD
        assert signal.metadata.get("reason") == "walk_forward_warmup"

    def test_train_in_memory_insufficient_data_returns_false(self):
        """3-row DataFrame (< n_splits+1=6) → False; model stays None."""
        import numpy as np
        import pandas as pd
        from signals.ml_model import MLSignalGenerator

        gen = MLSignalGenerator(MagicMock())
        df = pd.DataFrame(
            np.zeros((3, len(MLSignalGenerator.FEATURES) + 1)),
            columns=MLSignalGenerator.FEATURES + ["target"],
        )
        df["target"] = [0, 1, 0]

        assert gen.train_in_memory(df) is False
        assert gen.is_trained_in_memory() is False

    def test_compute_target_cost_hurdle_edges(self):
        """Label is 1 only when forward return STRICTLY exceeds LABEL_MIN_RETURN:
        exactly at the hurdle → 0, just above → 1, below/negative → 0."""
        import pandas as pd
        from signals.ml_model import MLSignalGenerator

        hurdle = MLSignalGenerator.LABEL_MIN_RETURN
        close = pd.Series([100.0, 100.0, 100.0, 100.0])
        future_close = pd.Series([
            100.0 * (1 + hurdle),        # exactly at hurdle → 0
            100.0 * (1 + hurdle) + 0.01, # just above → 1
            100.0,                        # flat → 0
            99.0,                         # down → 0
        ])

        target = MLSignalGenerator.compute_target(close, future_close)

        assert target.tolist() == [0, 1, 0, 0]

    def test_train_in_memory_smoke(self):
        """200 synthetic rows → True; in-memory model and scaler are set."""
        import numpy as np
        import pandas as pd
        from signals.ml_model import MLSignalGenerator

        gen = MLSignalGenerator(MagicMock())
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            rng.standard_normal((200, len(MLSignalGenerator.FEATURES))),
            columns=MLSignalGenerator.FEATURES,
        )
        df["target"] = (rng.random(200) > 0.5).astype(int)

        result = gen.train_in_memory(df)

        assert result is True
        assert gen.is_trained_in_memory() is True
        assert gen.model is not None
        assert gen.scaler is not None


# ===========================================================================
# 4b. ML signal — as_of historical cutoff (for backtesting)
# ===========================================================================

class TestMLAsOf:
    def test_as_of_none_queries_latest_row_unchanged(self):
        from signals.ml_model import MLSignalGenerator

        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchone.return_value = None

        MLSignalGenerator(engine).get_latest_features("BTCUSDT")

        sql, params = conn.execute.call_args[0]
        assert "as_of" not in params
        assert "<= :as_of" not in str(sql)

    def test_as_of_set_adds_timestamp_cutoff_to_query(self):
        from signals.ml_model import MLSignalGenerator

        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchone.return_value = None
        cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)

        MLSignalGenerator(engine).get_latest_features("BTCUSDT", as_of=cutoff)

        sql, params = conn.execute.call_args[0]
        assert params["as_of"] == cutoff
        assert "<= :as_of" in str(sql)

    @patch("signals.ml_model.MLSignalGenerator.is_trained", return_value=False)
    def test_generate_signal_passes_as_of_through_as_timestamp_even_without_model(self, _):
        from signals.ml_model import MLSignalGenerator

        cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
        signal = MLSignalGenerator(MagicMock()).generate_signal("BTCUSDT", as_of=cutoff)

        assert signal.timestamp == cutoff


# ===========================================================================
# 5.  Sentiment — graceful degradation without Reddit credentials
# ===========================================================================

class TestSentimentSignal:

    @patch("signals.sentiment.feedparser.parse")
    def test_sentiment_empty_feeds(self, mock_parse):
        """Empty RSS feeds (no articles) → HOLD with confidence 0.4."""
        from signals.sentiment import SentimentSignalGenerator

        empty_feed = MagicMock()
        empty_feed.entries = []
        mock_parse.return_value = empty_feed

        signal = SentimentSignalGenerator().generate_signal("BTCUSDT")

        assert signal.direction  == SignalDirection.HOLD
        assert signal.confidence == pytest.approx(0.4)
        assert signal.source     == "sentiment"

    def test_generate_signal_accepts_as_of_without_error(self, ):
        """Live sentiment has no historical archive — as_of is accepted but
        ignored (always scores current live articles)."""
        from signals.sentiment import SentimentSignalGenerator

        cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with patch("signals.sentiment.feedparser.parse") as mock_parse:
            empty_feed = MagicMock()
            empty_feed.entries = []
            mock_parse.return_value = empty_feed
            signal = SentimentSignalGenerator().generate_signal("BTCUSDT", as_of=cutoff)
        assert signal.direction == SignalDirection.HOLD

    def test_empty_articles_returns_neutral_score(self):
        from signals.sentiment import SentimentSignalGenerator

        assert SentimentSignalGenerator().score_articles([]) == pytest.approx(0.0)

    def test_vader_scores_positive_article(self):
        from signals.sentiment import SentimentSignalGenerator

        articles = [{
            "title":     "Bitcoin is incredible and amazing!",
            "summary":   "",
            "published": "",
        }]
        score = SentimentSignalGenerator().score_articles(articles)
        assert score > 0.2, f"Expected positive score, got {score}"


# ===========================================================================
# 5b. NeutralSentimentSignalGenerator — no-op stand-in for backtesting
# ===========================================================================

class TestNeutralSentimentSignalGenerator:
    def test_returns_hold_with_no_io(self):
        """No feedparser/network call — used when sentiment has no history."""
        from signals.sentiment import NeutralSentimentSignalGenerator

        cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
        signal = NeutralSentimentSignalGenerator().generate_signal("BTCUSDT", as_of=cutoff)

        assert signal.direction == SignalDirection.HOLD
        assert signal.confidence == pytest.approx(0.5)
        assert signal.source == "sentiment"
        assert signal.timestamp == cutoff


# ===========================================================================
# 6 + 7.  Aggregator — weights and exception fallback
# ===========================================================================

class TestSignalAggregator:

    def _agg(self, tech, ml, sent, tc=0.8, mc=0.7, sc=0.6,
             buy_gate=0.15, sell_gate=-0.15):
        """Build an aggregator with mocked generators. Bypasses __init__."""
        from signals.aggregator import SignalAggregator

        agg = SignalAggregator.__new__(SignalAggregator)
        agg.weights = {"technical": 0.40, "ml": 0.40, "sentiment": 0.20}
        agg.buy_gate  = buy_gate
        agg.sell_gate = sell_gate
        agg.technical = MagicMock()
        agg.ml        = MagicMock()
        agg.sentiment = MagicMock()
        agg.technical.generate_signal.return_value = _signal(tech, tc, "technical")
        agg.ml.generate_signal.return_value        = _signal(ml,   mc, "ml")
        agg.sentiment.generate_signal.return_value = _signal(sent, sc, "sentiment")
        return agg

    def test_aggregator_weights_all_buy_produces_buy(self):
        """test_aggregator_weights — three BUY signals → final BUY."""
        agg = self._agg(SignalDirection.BUY, SignalDirection.BUY, SignalDirection.BUY)
        result = agg.aggregate("BTCUSDT")
        assert result.direction == SignalDirection.BUY

    def test_aggregator_all_sell_produces_sell(self):
        agg = self._agg(SignalDirection.SELL, SignalDirection.SELL, SignalDirection.SELL)
        result = agg.aggregate("BTCUSDT")
        assert result.direction == SignalDirection.SELL

    def test_aggregator_symmetric_buy_sell_gives_hold(self):
        # BUY(0.5)*0.4 + SELL(0.5)*0.4 + HOLD(0.5)*0.2 = 0.0 → HOLD
        agg = self._agg(
            SignalDirection.BUY,  SignalDirection.SELL, SignalDirection.HOLD,
            tc=0.5, mc=0.5, sc=0.5,
        )
        result = agg.aggregate("BTCUSDT")
        assert result.direction == SignalDirection.HOLD

    def test_custom_sell_gate_changes_exit_threshold(self):
        """A lone moderate SELL scores -0.22; a stricter sell_gate=-0.30
        keeps it HOLD, while the default -0.15 lets it through."""
        # technical SELL(0.55)*0.40 = -0.22, others HOLD
        default_agg = self._agg(SignalDirection.SELL, SignalDirection.HOLD,
                                SignalDirection.HOLD, tc=0.55)
        strict_agg = self._agg(SignalDirection.SELL, SignalDirection.HOLD,
                               SignalDirection.HOLD, tc=0.55, sell_gate=-0.30)

        assert default_agg.aggregate("BTCUSDT").direction == SignalDirection.SELL
        assert strict_agg.aggregate("BTCUSDT").direction == SignalDirection.HOLD

    def test_custom_buy_gate_changes_entry_threshold(self):
        """technical BUY(0.55)*0.40 = 0.22: passes 0.15, fails 0.30."""
        default_agg = self._agg(SignalDirection.BUY, SignalDirection.HOLD,
                                SignalDirection.HOLD, tc=0.55)
        strict_agg = self._agg(SignalDirection.BUY, SignalDirection.HOLD,
                               SignalDirection.HOLD, tc=0.55, buy_gate=0.30)

        assert default_agg.aggregate("BTCUSDT").direction == SignalDirection.BUY
        assert strict_agg.aggregate("BTCUSDT").direction == SignalDirection.HOLD

    def test_aggregator_fallback_one_generator_throws(self):
        """test_aggregator_fallback — one generator raises, others still used."""
        from signals.aggregator import SignalAggregator

        agg = SignalAggregator.__new__(SignalAggregator)
        agg.weights   = {"technical": 0.40, "ml": 0.40, "sentiment": 0.20}
        agg.buy_gate  = 0.15
        agg.sell_gate = -0.15
        agg.technical = MagicMock()
        agg.ml        = MagicMock()
        agg.sentiment = MagicMock()

        agg.technical.generate_signal.side_effect = RuntimeError("DB gone")
        agg.ml.generate_signal.return_value        = _signal(SignalDirection.BUY, 0.8, "ml")
        agg.sentiment.generate_signal.return_value = _signal(SignalDirection.BUY, 0.6, "sentiment")

        result = agg.aggregate("BTCUSDT")

        assert result is not None
        assert isinstance(result, AggregatedSignal)
        # technical falls back to HOLD → ml+sentiment still push BUY
        # 0*0.4 + 0.8*0.4 + 0.6*0.2 = 0.32 + 0.12 = 0.44 > 0.15 → BUY
        assert result.direction == SignalDirection.BUY


# ===========================================================================
# 7b. Aggregator — as_of threading + neutral sentiment swap-in
# ===========================================================================

class TestAggregatorAsOf:
    def test_as_of_none_calls_generators_without_as_of_kwarg(self):
        agg = self._agg_for_as_of()
        agg.aggregate("BTCUSDT")

        _, kwargs = agg.technical.generate_signal.call_args
        assert "as_of" not in kwargs
        _, kwargs = agg.ml.generate_signal.call_args
        assert "as_of" not in kwargs
        # Live mode (as_of=None) uses the real/live sentiment generator.
        agg.sentiment.generate_signal.assert_called_once_with("BTCUSDT")

    def test_as_of_set_passes_to_technical_and_ml_and_uses_neutral_sentiment(self):
        agg = self._agg_for_as_of()
        cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)

        result = agg.aggregate("BTCUSDT", as_of=cutoff)

        _, kwargs = agg.technical.generate_signal.call_args
        assert kwargs["as_of"] == cutoff
        _, kwargs = agg.ml.generate_signal.call_args
        assert kwargs["as_of"] == cutoff
        # the mocked live sentiment generator must never be called when as_of is set
        agg.sentiment.generate_signal.assert_not_called()
        sentiment_sub_signal = next(s for s in result.signals if s.source == "sentiment")
        assert sentiment_sub_signal.direction == SignalDirection.HOLD
        assert result.timestamp == cutoff

    def _agg_for_as_of(self):
        from signals.aggregator import SignalAggregator
        from signals.sentiment import NeutralSentimentSignalGenerator

        agg = SignalAggregator.__new__(SignalAggregator)
        agg.weights   = {"technical": 0.40, "ml": 0.40, "sentiment": 0.20}
        agg.buy_gate  = 0.15
        agg.sell_gate = -0.15
        agg.technical = MagicMock()
        agg.ml        = MagicMock()
        agg.sentiment = MagicMock()
        agg._neutral_sentiment = NeutralSentimentSignalGenerator()
        agg.technical.generate_signal.return_value = _signal(SignalDirection.BUY, 0.8, "technical")
        agg.ml.generate_signal.return_value        = _signal(SignalDirection.BUY, 0.8, "ml")
        agg.sentiment.generate_signal.return_value = _signal(SignalDirection.HOLD, 0.4, "sentiment")
        return agg


# ===========================================================================
# 8.  Reasoning string contains all three source names
# ===========================================================================

class TestSignalReasoningString:

    def test_signal_reasoning_string(self):
        """test_signal_reasoning_string — all three sources appear in reasoning."""
        from signals.aggregator import SignalAggregator

        agg = SignalAggregator.__new__(SignalAggregator)
        agg.weights   = {"technical": 0.40, "ml": 0.40, "sentiment": 0.20}
        agg.buy_gate  = 0.15
        agg.sell_gate = -0.15
        agg.technical = MagicMock()
        agg.ml        = MagicMock()
        agg.sentiment = MagicMock()
        agg.technical.generate_signal.return_value = _signal(SignalDirection.BUY,  0.8, "technical")
        agg.ml.generate_signal.return_value        = _signal(SignalDirection.HOLD, 0.5, "ml")
        agg.sentiment.generate_signal.return_value = _signal(SignalDirection.SELL, 0.4, "sentiment")

        result = agg.aggregate("BTCUSDT")

        reasoning_lower = result.reasoning.lower()
        assert "technical"  in reasoning_lower
        assert "ml"         in reasoning_lower
        assert "sentiment"  in reasoning_lower
