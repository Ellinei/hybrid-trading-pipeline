"""Publication state for technical features used by the entry gate.

The table is created only by refresh writers. Readers fail closed if it is
absent, so an uninitialised installation cannot publish a BUY signal.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from urllib.parse import quote_plus
from uuid import uuid4

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# A session lock covers the whole refresh, including dbt subprocesses. It is
# shared by the Airflow job and the manual features command.
_LOCK_KEY = 583_219_746_033


def build_validation_engine():
    """Build a small DB engine available in both the host and Airflow image."""
    load_dotenv()
    return create_engine(
        "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}".format(
            user=quote_plus(os.getenv("POSTGRES_USER", "trader")),
            password=quote_plus(os.getenv("POSTGRES_PASSWORD", "")),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            database=os.getenv("POSTGRES_DB", "trading"),
        )
    )


def configured_symbols() -> list[str]:
    symbols = [s.strip() for s in os.getenv(
        "TRADING_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT"
    ).split(",") if s.strip()]
    if not symbols:
        raise ValueError("At least one trading symbol is required")
    return list(dict.fromkeys(symbols))


@contextmanager
def feature_refresh_lock(engine):
    """Reject overlapping CLI/Airflow refreshes before either can write."""
    with engine.connect() as conn:
        acquired = conn.execute(
            text("SELECT pg_try_advisory_lock(:key)"), {"key": _LOCK_KEY}
        ).scalar()
        if not acquired:
            raise RuntimeError("Another feature refresh is already running")
        try:
            yield
        finally:
            conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": _LOCK_KEY})


def mark_features_invalid(engine, symbols: list[str] | None = None) -> str:
    """Invalidate every previous publication before any source/build write."""
    symbols = configured_symbols() if symbols is None else symbols
    if not symbols:
        raise ValueError("At least one trading symbol is required")
    generation = str(uuid4())
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw.feature_validation (
                symbol TEXT PRIMARY KEY,
                is_valid BOOLEAN NOT NULL DEFAULT FALSE,
                feature_timestamp TIMESTAMPTZ,
                generation TEXT NOT NULL,
                validated_at TIMESTAMPTZ
            )
        """))
        conn.execute(text("UPDATE raw.feature_validation SET is_valid = FALSE"))
        for symbol in symbols:
            conn.execute(text("""
                INSERT INTO raw.feature_validation
                    (symbol, is_valid, feature_timestamp, generation, validated_at)
                VALUES (:symbol, FALSE, NULL, :generation, NULL)
                ON CONFLICT (symbol) DO UPDATE SET
                    is_valid = FALSE, feature_timestamp = NULL,
                    generation = EXCLUDED.generation, validated_at = NULL
            """), {"symbol": symbol, "generation": generation})
    return generation


def mark_features_valid(
    engine, generation: str, symbols: list[str] | None = None
) -> None:
    """Publish only rows from this refresh after successful dbt tests.

    The generation check prevents an older run from validating a newer one's
    partially built table. A symbol with no feature rows remains invalid.
    """
    symbols = configured_symbols() if symbols is None else symbols
    if not symbols:
        raise ValueError("At least one trading symbol is required")
    with engine.begin() as conn:
        for symbol in symbols:
            result = conn.execute(text("""
                UPDATE raw.feature_validation AS v SET
                    feature_timestamp = (
                        SELECT MAX(timestamp) FROM features.feat_technical
                        WHERE symbol = :symbol
                    ),
                    is_valid = TRUE,
                    validated_at = CURRENT_TIMESTAMP
                WHERE v.symbol = :symbol AND v.generation = :generation
                  AND EXISTS (
                      SELECT 1 FROM features.feat_technical
                      WHERE symbol = :symbol
                  )
            """), {"symbol": symbol, "generation": generation})
            if result.rowcount != 1:
                raise RuntimeError(
                    f"Cannot validate {symbol}: refresh superseded or features missing"
                )
