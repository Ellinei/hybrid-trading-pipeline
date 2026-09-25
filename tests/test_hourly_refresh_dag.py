"""Dependency-light contract tests for the single locked hourly refresh."""
from __future__ import annotations

import runpy
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _load_dag(monkeypatch):
    class DAG:
        def __init__(self, **kwargs):
            self.settings = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    class Operator:
        def __init__(self, task_id, **kwargs):
            self.task_id = task_id
            self.settings = kwargs

    airflow = types.ModuleType("airflow")
    airflow.DAG = DAG
    operators = types.ModuleType("airflow.operators")
    python = types.ModuleType("airflow.operators.python")
    python.PythonOperator = Operator
    for name, module in {
        "airflow": airflow,
        "airflow.operators": operators,
        "airflow.operators.python": python,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return runpy.run_path(str(
        Path(__file__).resolve().parents[1] / "airflow/dags/ohlcv_backfill_dag.py"
    ))


@pytest.mark.parametrize("failure", [None, "backfill", "deps", "build"])
def test_refresh_publishes_only_after_success(monkeypatch, failure):
    namespace = _load_dag(monkeypatch)
    dag = namespace["dag"]
    assert dag.settings["schedule"] == "5 * * * *"
    assert dag.settings["max_active_runs"] == 1
    assert dag.settings["catchup"] is False
    assert dag.settings["default_args"]["retries"] >= 1

    events = []
    engine = MagicMock()

    @contextmanager
    def locked(_engine):
        events.append("lock")
        try:
            yield
        finally:
            events.append("unlock")

    def invalid(_engine, _symbols):
        events.append("invalid")
        return "generation"

    def backfill(**kwargs):
        events.append("backfill")
        assert kwargs["invalidate_features"] is False
        if failure == "backfill":
            raise RuntimeError("backfill failed")

    def run(args, **kwargs):
        events.append(args[1])
        assert kwargs["check"] is True
        if failure == args[1]:
            raise RuntimeError(f"{failure} failed")

    def valid(_engine, generation, _symbols):
        assert generation == "generation"
        events.append("valid")

    with patch("ingestion.feature_validation.build_validation_engine", return_value=engine), \
         patch("ingestion.feature_validation.configured_symbols", return_value=["BTCUSDT"]), \
         patch("ingestion.feature_validation.feature_refresh_lock", locked), \
         patch("ingestion.feature_validation.mark_features_invalid", invalid), \
         patch("ingestion.feature_validation.mark_features_valid", valid), \
         patch("ingestion.rest_backfill.run_backfill", backfill), \
         patch("subprocess.run", run):
        if failure:
            with pytest.raises(RuntimeError, match="failed"):
                namespace["refresh_task"].settings["python_callable"]()
        else:
            namespace["refresh_task"].settings["python_callable"]()

    assert events[:2] == ["lock", "invalid"]
    assert ("valid" in events) is (failure is None)
    assert events[-1] == "unlock"
    engine.dispose.assert_called_once()
