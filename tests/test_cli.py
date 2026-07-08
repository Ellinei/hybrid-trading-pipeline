"""Unit tests for the unified CLI dispatcher (python -m cli <subcommand>).

Every subcommand except `features` is a thin subprocess forward to an
existing module's own entry point -- these tests assert on the
constructed command, never on real subprocess execution (real execution
is exercised manually / in the golden-parity integration test).

Run:
    poetry run pytest tests/test_cli.py -v
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from cli.__main__ import _dbt_env, main


class TestSubcommandForwarding:

    @pytest.mark.parametrize("subcommand,module", [
        ("backfill", "ingestion.rest_backfill"),
        ("validate", "ingestion.validate"),
        ("train", "signals.ml_model"),
        ("backtest", "backtest.run_backtest"),
        ("sweep", "backtest.sweep"),
        ("trade", "execution.bot_runner"),
    ])
    def test_forwards_to_module_with_python_dash_m(self, subcommand, module):
        with patch("cli.__main__.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            code = main([subcommand, "--foo", "bar"])

        mock_run.assert_called_once_with([sys.executable, "-m", module, "--foo", "bar"])
        assert code == 0

    def test_propagates_nonzero_exit_code(self):
        with patch("cli.__main__.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            code = main(["backtest"])

        assert code == 1

    def test_verify_parity_forwards_to_golden_pytest(self):
        with patch("cli.__main__.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            main(["verify-parity"])

        args = mock_run.call_args[0][0]
        assert args[0] == sys.executable
        assert "pytest" in args
        assert "tests/test_backtest_golden_parity.py" in args

    def test_dashboard_forwards_to_streamlit_module(self):
        with patch("cli.__main__.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            main(["dashboard"])

        mock_run.assert_called_once_with(
            [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"]
        )


class TestUnknownAndEmptyInvocation:

    def test_unknown_subcommand_returns_nonzero_without_subprocess(self):
        with patch("cli.__main__.subprocess.run") as mock_run:
            code = main(["not-a-real-command"])

        mock_run.assert_not_called()
        assert code != 0

    def test_no_args_returns_nonzero_without_subprocess(self):
        with patch("cli.__main__.subprocess.run") as mock_run:
            code = main([])

        mock_run.assert_not_called()
        assert code != 0

    def test_help_flag_returns_zero_without_subprocess(self):
        with patch("cli.__main__.subprocess.run") as mock_run:
            code = main(["-h"])

        mock_run.assert_not_called()
        assert code == 0


class TestDbtEnv:

    def test_maps_postgres_vars_to_dbt_postgres_vars(self, monkeypatch):
        monkeypatch.setenv("POSTGRES_HOST", "localhost")
        monkeypatch.setenv("POSTGRES_PORT", "5433")
        monkeypatch.setenv("POSTGRES_USER", "trader")
        monkeypatch.setenv("POSTGRES_PASSWORD", "test")
        monkeypatch.setenv("POSTGRES_DB", "trading")

        env = _dbt_env()

        assert env["DBT_POSTGRES_HOST"] == "localhost"
        assert env["DBT_POSTGRES_PORT"] == "5433"
        assert env["DBT_POSTGRES_USER"] == "trader"
        assert env["DBT_POSTGRES_PASSWORD"] == "test"
        assert env["DBT_POSTGRES_DB"] == "trading"

    def test_falls_back_to_defaults_when_unset(self, monkeypatch):
        for var in ("POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_USER",
                    "POSTGRES_PASSWORD", "POSTGRES_DB"):
            monkeypatch.delenv(var, raising=False)

        env = _dbt_env()

        assert env["DBT_POSTGRES_HOST"] == "localhost"
        assert env["DBT_POSTGRES_PORT"] == "5433"
        assert env["DBT_POSTGRES_USER"] == "trader"
        assert env["DBT_POSTGRES_DB"] == "trading"


class TestFeaturesSubcommand:

    def test_runs_deps_run_test_in_order_in_dbt_dir(self):
        with patch("cli.__main__.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            code = main(["features"])

        assert mock_run.call_count == 3
        calls = [c.args[0] for c in mock_run.call_args_list]
        assert calls == [
            ["dbt", "deps", "--profiles-dir", "."],
            ["dbt", "run", "--profiles-dir", "."],
            ["dbt", "test", "--profiles-dir", "."],
        ]
        for c in mock_run.call_args_list:
            assert c.kwargs["cwd"] == "dbt"
        assert code == 0

    def test_stops_after_first_failing_step(self):
        with patch("cli.__main__.subprocess.run") as mock_run:
            mock_run.side_effect = [MagicMock(returncode=1)]
            code = main(["features"])

        assert mock_run.call_count == 1
        assert code == 1
