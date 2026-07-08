"""Golden parity regression test for the walk-forward cache-replay path.

Phase 11a speeds up walk-forward retraining (frozen hyperparameters,
n_jobs fixes, eventually parallel windows and a default cache/replay run
mode). None of that code runs during a --load-cache replay — the prob
cache already holds every ML prediction, so replay is pure in-memory
fusion + bar simulation. This test is the tripwire: every Phase 11a change
must still reproduce the exp007 trade log byte-for-byte when replayed from
the existing cache/wf_540d_h8.pkl. If it doesn't, a "performance" change
silently altered strategy behavior.

Requires a live TimescaleDB with the same OHLCV/feature data the golden
baseline was captured against, and the existing prob cache — skips
cleanly (not fails) when either is unavailable, e.g. in an environment
with no DB.

Run:
    poetry run pytest tests/test_backtest_golden_parity.py -v -m golden
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote_plus

import pytest

pytestmark = pytest.mark.golden

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = REPO_ROOT / "cache" / "wf_540d_h8.pkl"
GOLDEN_CSV = REPO_ROOT / "results" / "golden" / "trades_exp007_golden.csv"


def _db_available() -> bool:
    try:
        from dotenv import load_dotenv
        from sqlalchemy import create_engine, text

        load_dotenv(REPO_ROOT / ".env")
        engine = create_engine(
            "postgresql+psycopg2://{u}:{p}@{h}:{port}/{db}".format(
                u=quote_plus(os.getenv("POSTGRES_USER", "trader")),
                p=quote_plus(os.getenv("POSTGRES_PASSWORD", "")),
                h=os.getenv("POSTGRES_HOST", "localhost"),
                port=os.getenv("POSTGRES_PORT", "5432"),
                db=os.getenv("POSTGRES_DB", "trading"),
            )
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not CACHE_PATH.exists(), reason="prob cache cache/wf_540d_h8.pkl not present")
@pytest.mark.skipif(not GOLDEN_CSV.exists(), reason="golden baseline not captured — see results/golden/")
def test_exp007_cache_replay_matches_golden_baseline(tmp_path):
    """Re-run the exact exp007 --load-cache command; diff output byte-for-byte."""
    if not _db_available():
        pytest.skip("TimescaleDB not reachable — cannot load bars/ATR for replay")

    out_csv = tmp_path / "trades_replay.csv"
    proc = subprocess.run(
        [
            sys.executable, "-m", "backtest.run_backtest",
            "--load-cache", str(CACHE_PATH),
            "--ml-buy-threshold", "0.55",
            "--max-hold-bars", "24",
            "--output", str(out_csv),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )

    assert proc.returncode == 0, f"replay run failed:\n{proc.stdout}\n{proc.stderr}"
    assert out_csv.exists(), "replay produced no trade log"
    assert out_csv.read_text() == GOLDEN_CSV.read_text(), (
        "cache-replay trade log diverged from the golden exp007 baseline — "
        "a Phase 11a change altered replay behavior, not just its speed"
    )
