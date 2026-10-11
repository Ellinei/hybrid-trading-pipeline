"""Source-time freshness checks; only required inputs gate new entries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import isfinite

from sqlalchemy import text


@dataclass(frozen=True)
class SourceStatus:
    source: str
    timestamp: datetime | None
    age_seconds: float | None
    status: str
    required: bool


def classify_source(source, timestamp, max_age_seconds, now, required=True):
    if not isfinite(max_age_seconds) or max_age_seconds <= 0:
        raise ValueError("Freshness limits must be finite and positive")
    if timestamp is None:
        return SourceStatus(source, None, None, "missing", required)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    age = (now - timestamp).total_seconds()
    status = "future" if age < 0 else "stale" if age > max_age_seconds else "fresh"
    return SourceStatus(source, timestamp, age, status, required)


class FreshnessMonitor:
    def __init__(self, engine, candle_max_age=10800, feature_max_age=10800):
        for age in (candle_max_age, feature_max_age):
            if not isfinite(age) or age <= 0:
                raise ValueError("Freshness limits must be finite and positive")
        self.engine = engine
        self.sources = (
            ("candles", "raw.ohlcv", candle_max_age, True),
            ("technical", "features.feat_technical", feature_max_age, True),
            ("streaming", "raw.realtime_features", 180, False),
        )

    def check(self, symbol, now=None):
        now = now or datetime.now(UTC)
        statuses = []
        feature_timestamp = None
        for source, table, max_age, required in self.sources:
            try:
                # Table names come exclusively from the fixed source list above.
                with self.engine.connect() as conn:
                    timestamp = conn.execute(
                        text(f"SELECT MAX(timestamp) FROM {table} WHERE symbol = :symbol"),
                        {"symbol": symbol},
                    ).scalar()
                statuses.append(classify_source(source, timestamp, max_age, now, required))
                if source == "technical":
                    feature_timestamp = timestamp
            except Exception:
                statuses.append(SourceStatus(source, None, None, "unavailable", required))
        try:
            with self.engine.connect() as conn:
                row = conn.execute(text("""
                    SELECT is_valid, feature_timestamp
                    FROM raw.feature_validation WHERE symbol = :symbol
                """), {"symbol": symbol}).first()
            if row is None:
                status = "missing"
                validated_timestamp = None
            else:
                valid, validated_timestamp = row[0], row[1]
                status = (
                    "invalid" if not valid else
                    "unvalidated" if feature_timestamp is None or
                    validated_timestamp != feature_timestamp else "fresh"
                )
            statuses.append(SourceStatus(
                "feature_validation", validated_timestamp, None, status, True
            ))
        except Exception:
            statuses.append(SourceStatus("feature_validation", None, None, "unavailable", True))
        return statuses

    def entry_check(self, symbol):
        statuses = self.check(symbol)
        blocked = [s for s in statuses if s.required and s.status != "fresh"]
        reason = "; ".join(f"{s.source}: {s.status}" for s in blocked) or "data fresh"
        return not blocked, reason, [asdict(s) for s in statuses]
