import csv
import json

from dashboard.data import load_experiments


def test_dashboard_uses_declared_evaluation_window_even_without_trades(tmp_path):
    run = tmp_path / "empty_trial"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps({"trade_count": 0}))
    (run / "config.json").write_text(json.dumps({
        "symbols": ["BTCUSDT"],
        "start": "2026-09-24T00:00:00+00:00",
        "end": "2026-10-01T00:00:00+00:00",
    }))
    with (run / "equity_curve.csv").open("w", newline="") as stream:
        csv.writer(stream).writerow(["timestamp", "equity"])
    runs, warnings = load_experiments(tmp_path)
    assert not warnings
    assert runs[0]["start"] == "2026-09-24T00:00:00+00:00"
    assert runs[0]["end"] == "2026-10-01T00:00:00+00:00"
