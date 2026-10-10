"""Crypto news sentiment signal using public RSS feeds + VADER.

No API credentials required — RSS is fully public.  feedparser handles
malformed XML gracefully, and a single feed failure (HTTP error, parse
error) is logged and skipped without affecting other feeds.

Add more feed sources by appending to ``RSS_FEEDS`` — no other code changes.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from time import mktime
from typing import Any

import feedparser
import structlog
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from signals.base import Signal, SignalDirection

log = structlog.get_logger()


class SentimentSignalGenerator:
    """Score recent crypto news with VADER and emit a BUY/SELL/HOLD signal."""

    RSS_FEEDS: dict[str, list[str]] = {
        "general": [
            "https://feeds.feedburner.com/CoinDesk",
            "https://cointelegraph.com/rss",
            "https://cryptopanic.com/news/rss/",
        ],
    }

    SYMBOL_KEYWORDS: dict[str, list[str]] = {
        "BTCUSDT": ["bitcoin", "btc"],
        "ETHUSDT": ["ethereum", "eth"],
        "SOLUSDT": ["solana", "sol"],
    }

    def __init__(self) -> None:
        self.vader = SentimentIntensityAnalyzer()

    # ------------------------------------------------------------------
    # Article fetching
    # ------------------------------------------------------------------

    def fetch_articles(
        self,
        symbol: str,
        max_age_hours: int = 24,
    ) -> list[dict[str, Any]]:
        """Fetch + filter articles from all RSS feeds.

        Filters by:
        - Publish time within the last *max_age_hours* (when available)
        - Title or summary containing one of the symbol's keywords (case-insensitive)
        """
        keywords = self.SYMBOL_KEYWORDS.get(
            symbol, [symbol.lower().replace("USDT", "")]
        )
        cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
        articles: list[dict[str, Any]] = []

        for feed_url in self._all_feeds():
            try:
                parsed = feedparser.parse(feed_url)
            except Exception as exc:
                log.warning("rss_fetch_failed", url=feed_url, error=str(exc))
                continue

            for entry in getattr(parsed, "entries", []):
                # Skip articles older than the cutoff when we have a timestamp
                published_struct = getattr(entry, "published_parsed", None)
                if published_struct:
                    try:
                        published_dt = datetime.fromtimestamp(
                            mktime(published_struct), tz=UTC
                        )
                        if published_dt < cutoff:
                            continue
                    except (TypeError, ValueError):
                        pass  # bad timestamp — keep the article

                title   = (entry.get("title")   or "").lower()
                summary = (entry.get("summary") or "").lower()
                if any(kw in title or kw in summary for kw in keywords):
                    articles.append({
                        "title":     entry.get("title", ""),
                        "summary":   entry.get("summary", ""),
                        "published": entry.get("published", ""),
                    })

        return articles

    def _all_feeds(self) -> list[str]:
        return [url for urls in self.RSS_FEEDS.values() for url in urls]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_articles(self, articles: list[dict[str, Any]]) -> float:
        """Simple-average VADER compound score across articles.  0.0 if empty."""
        if not articles:
            return 0.0

        scores = [
            self.vader.polarity_scores(
                (a.get("title", "") + " " + a.get("summary", "")).strip()
            )["compound"]
            for a in articles
        ]
        return sum(scores) / len(scores)

    # ------------------------------------------------------------------
    # Signal
    # ------------------------------------------------------------------

    def generate_signal(self, symbol: str, as_of: datetime | None = None) -> Signal:
        """as_of is accepted for interface parity with the other generators
        but ignored — live RSS has no historical archive to query, so this
        always scores current articles regardless of as_of."""
        now       = datetime.now(UTC)
        articles  = self.fetch_articles(symbol)
        sentiment = self.score_articles(articles)

        if sentiment > 0.2:
            direction  = SignalDirection.BUY
            confidence = min(0.4 + sentiment * 0.3, 0.7)
        elif sentiment < -0.2:
            direction  = SignalDirection.SELL
            confidence = min(0.4 + abs(sentiment) * 0.3, 0.7)
        else:
            direction  = SignalDirection.HOLD
            confidence = 0.4

        return Signal(
            direction=direction, confidence=confidence,
            source="sentiment", symbol=symbol, timestamp=now,
            metadata={
                "sentiment_score": sentiment,
                "article_count":   len(articles),
                "feeds_checked":   self._all_feeds(),
            },
        )


class NeutralSentimentSignalGenerator:
    """No-op stand-in for backtesting.

    Live sentiment has no historical archive (RSS is fetch-on-read only),
    so a backtest can't replay it. This always returns a neutral HOLD with
    no I/O — equivalent to "sentiment abstains" in the weighted aggregation,
    since SignalDirection.HOLD always contributes a score of 0.
    """

    def generate_signal(self, symbol: str, as_of: datetime | None = None) -> Signal:
        now = as_of if as_of is not None else datetime.now(UTC)
        return Signal(
            direction=SignalDirection.HOLD, confidence=0.5,
            source="sentiment", symbol=symbol, timestamp=now,
            metadata={"reason": "no_historical_sentiment_data"},
        )
