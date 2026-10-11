from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from execution.freshness import SourceStatus
from execution.journal import DecisionJournal
from execution.paper import PaperStore
from tests.test_operations import NOW, signal, write_experiment

PAGES = Path(__file__).resolve().parents[1] / "dashboard" / "pages"


def test_operations_page_empty_and_unavailable_data(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_OPERATIONS_DB", str(tmp_path / "missing.sqlite3"))
    monkeypatch.setenv("TRADING_SYMBOLS", "BTCUSDT")
    with (
        patch("dashboard.operations_view.operations_engine"),
        patch("dashboard.operations_view.FreshnessMonitor") as monitor,
    ):
        monitor.return_value.check.return_value = [
            SourceStatus("technical", None, None, "unavailable", True)
        ]
        app = AppTest.from_file(PAGES / "1_Operations.py").run(timeout=15)
        assert not app.exception
        assert "No paper account" in app.info[-1].value
        assert app.dataframe[0].value.iloc[0]["status"] == "unavailable"


def test_operations_journal_filters_and_paper_account(tmp_path, monkeypatch):
    path = tmp_path / "ops.sqlite3"
    monkeypatch.setenv("BOT_OPERATIONS_DB", str(path))
    monkeypatch.setenv("TRADING_SYMBOLS", "BTCUSDT")
    journal = DecisionJournal(path)
    journal.record(
        symbol="BTCUSDT",
        mode="paper",
        outcome="blocked",
        reason="technical: stale",
        signal=signal(),
    )
    PaperStore(path, 1234).close()
    with (
        patch("dashboard.operations_view.operations_engine"),
        patch("dashboard.operations_view.FreshnessMonitor") as monitor,
    ):
        monitor.return_value.check.return_value = [SourceStatus("technical", NOW, 0, "fresh", True)]
        app = AppTest.from_file(PAGES / "1_Operations.py").run(timeout=15)
        assert not app.exception
        assert app.metric[0].value == "1,234.00"
        assert app.dataframe[1].value.iloc[0]["reason"] == "technical: stale"
        app.selectbox[1].select("live").run()
        assert not app.exception
        assert "No decisions" in app.info[0].value


def test_experiment_page_portfolio_symbol_and_window_warning(tmp_path, monkeypatch):
    write_experiment(tmp_path, "one")
    write_experiment(tmp_path, "two", "2026-02-01T00:00:00+00:00")
    monkeypatch.setenv("EXPERIMENT_RESULTS_DIR", str(tmp_path))
    app = AppTest.from_file(PAGES / "2_Experiments.py").run(timeout=15)
    assert not app.exception
    assert "different evaluation windows" in app.warning[0].value
    assert len(app.dataframe[0].value) == 2
    app.selectbox[0].select("BTCUSDT").run()
    assert not app.exception
    assert app.dataframe[0].value.iloc[0]["pnl"] == 10
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert "Choose at least one" in app.info[0].value


def test_experiment_page_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_RESULTS_DIR", str(tmp_path))
    app = AppTest.from_file(PAGES / "2_Experiments.py").run(timeout=15)
    assert not app.exception
    assert "No complete experiments" in app.info[0].value
