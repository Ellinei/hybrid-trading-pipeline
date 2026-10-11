"""Hourly completed-candle backfill and validated technical features."""
from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def _refresh() -> None:
    """Keep one DB session lock for backfill, dbt build, and publication."""
    os.environ.setdefault("POSTGRES_HOST", "postgres")
    os.environ.setdefault("POSTGRES_PORT", "5432")

    from ingestion.feature_validation import (
        build_validation_engine,
        configured_symbols,
        feature_refresh_lock,
        mark_features_invalid,
        mark_features_valid,
    )
    from ingestion.rest_backfill import run_backfill

    symbols = configured_symbols()
    engine = build_validation_engine()
    try:
        with feature_refresh_lock(engine):
            generation = mark_features_invalid(engine, symbols)
            run_backfill(
                symbols=symbols, days=2, interval="1h", source="mainnet",
                invalidate_features=False,
            )
            dbt_dir = "/opt/airflow/dbt"
            subprocess.run(
                ["dbt", "deps", "--profiles-dir", dbt_dir],
                cwd=dbt_dir, check=True,
            )
            # dbt build executes model tests and raises on any test failure.
            subprocess.run(
                ["dbt", "build", "--profiles-dir", dbt_dir],
                cwd=dbt_dir, check=True,
            )
            mark_features_valid(engine, generation, symbols)
    finally:
        engine.dispose()


with DAG(
    dag_id="ohlcv_feature_refresh",
    default_args=default_args,
    description="Hourly completed Binance candles and validated dbt features",
    schedule="5 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ingestion", "binance", "dbt"],
) as dag:
    refresh_task = PythonOperator(
        task_id="refresh_ohlcv_and_features", python_callable=_refresh,
    )
