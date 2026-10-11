"""Causal live-training rows, partitions, and persisted provenance."""

import hashlib
import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from signals.ml_model import MLSignalGenerator, parse_train_args

UTC = UTC


def _feature_table():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("ATTACH DATABASE ':memory:' AS features"))
        columns = ", ".join(f"{name} REAL" for name in MLSignalGenerator.FEATURES)
        conn.execute(text(
            "CREATE TABLE features.feat_technical "
            f"(symbol TEXT, timestamp TEXT, close REAL, {columns})"
        ))
    return engine


def _insert_feature(engine, symbol, timestamp, close):
    values = {name: 1.0 for name in MLSignalGenerator.FEATURES}
    values.update(symbol=symbol, timestamp=timestamp.isoformat(), close=close)
    columns = ", ".join(values)
    binds = ", ".join(f":{name}" for name in values)
    with engine.begin() as conn:
        conn.execute(text(
            f"INSERT INTO features.feat_technical ({columns}) VALUES ({binds})"
        ), values)


def _rows(hours=20, symbols=("BTCUSDT", "ETHUSDT")):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for hour in range(hours):
        for symbol in symbols:
            row = {
                name: float(hour + offset / 100)
                for offset, name in enumerate(MLSignalGenerator.FEATURES)
            }
            row.update(
                symbol=symbol,
                timestamp=start + timedelta(hours=hour),
                label_timestamp=start + timedelta(hours=hour + (10 if hour == 7 else 1)),
                target=(hour + (symbol == "ETHUSDT")) % 2,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_future_close_change_cannot_change_rows_available_at_cutoff(monkeypatch):
    monkeypatch.setattr(MLSignalGenerator, "LABEL_HORIZON_BARS", 1)
    engine = _feature_table()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for hour, close in [(0, 100), (1, 101), (10, 102)]:
        _insert_feature(engine, "BTCUSDT", start + timedelta(hours=hour), close)

    gen = MLSignalGenerator(engine)
    cutoff = start + timedelta(hours=2)
    first = gen.load_training_data("BTCUSDT", min_rows=1, as_of=cutoff)
    with engine.begin() as conn:
        conn.execute(text("UPDATE features.feat_technical SET close = 200 WHERE timestamp = :ts"),
                     {"ts": (start + timedelta(hours=10)).isoformat()})
    second = gen.load_training_data("BTCUSDT", min_rows=1, as_of=cutoff)

    assert list(first["timestamp"]) == [start]
    pd.testing.assert_frame_equal(first, second)
    engine.dispose()


def test_gappy_bar_label_requires_its_actual_future_timestamp(monkeypatch):
    monkeypatch.setattr(MLSignalGenerator, "LABEL_HORIZON_BARS", 1)
    engine = _feature_table()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    _insert_feature(engine, "BTCUSDT", start, 100)
    _insert_feature(engine, "BTCUSDT", start + timedelta(hours=10), 110)

    gen = MLSignalGenerator(engine)
    assert gen.load_training_data("BTCUSDT", min_rows=1,
                                  as_of=start + timedelta(hours=2)).empty
    # as_of exactly at the label candle's open (hour 10): its close isn't
    # available for another bar, so the row must still be excluded — this
    # is the look-ahead boundary the fix closes.
    assert gen.load_training_data("BTCUSDT", min_rows=1,
                                  as_of=start + timedelta(hours=10)).empty
    # as_of one bar later (hour 11): the label candle has closed, so the
    # row is now available.
    available = gen.load_training_data("BTCUSDT", min_rows=1,
                                       as_of=start + timedelta(hours=11))
    assert list(available["label_timestamp"]) == [start + timedelta(hours=10)]
    engine.dispose()


def test_training_window_keeps_only_recent_feature_timestamps(monkeypatch):
    monkeypatch.setattr(MLSignalGenerator, "LABEL_HORIZON_BARS", 1)
    engine = _feature_table()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for hour in (0, 24, 48, 49):
        _insert_feature(engine, "BTCUSDT", start + timedelta(hours=hour), 100 + hour)
    # as_of is one bar past the hour-48 row's label open (hour 49) so that
    # row's label has actually closed and reveal-time filtering doesn't
    # itself exclude it — this test is about the train_window_days floor,
    # not the reveal boundary (covered separately).
    rows = MLSignalGenerator(engine).load_training_data(
        "BTCUSDT", min_rows=1, as_of=start + timedelta(hours=50),
        train_window_days=1,
    )
    assert list(rows["timestamp"]) == [start + timedelta(hours=48)]
    engine.dispose()


def test_outer_and_cv_partitions_keep_simultaneous_symbols_together_and_purge():
    train, holdout, cv = MLSignalGenerator.partition_training_rows(_rows())
    assert train["timestamp"].max() < holdout["timestamp"].min()
    assert set(train["timestamp"]).isdisjoint(set(holdout["timestamp"]))
    assert train["label_timestamp"].max() < holdout["timestamp"].min()
    assert len(cv) == 5
    for training_idx, validation_idx in cv:
        training = train.iloc[training_idx]
        validation = train.iloc[validation_idx]
        assert training["timestamp"].max() < validation["timestamp"].min()
        assert training["label_timestamp"].max() < validation["timestamp"].min()
        assert validation.groupby("timestamp")["symbol"].nunique().eq(2).all()


def test_partition_purge_uses_reveal_time_not_label_open_time():
    """A label whose OPEN time is before a purge boundary but whose CLOSE
    (reveal) time lands after it must still be purged — otherwise the model
    would train on an outcome not knowable until after that boundary. The
    old rule (`label_timestamp < boundary`) kept such a row purely because
    its open time looked early enough; this pins the fix for both the
    holdout purge and the CV purge. Uses off-hour labels so hour-aligned
    data can't coincidentally make the old and new rules agree.
    """
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for hour in range(10):
        if hour == 1:
            # Opens at 1:30 (before validation_start=2:00 in CV fold 0) but
            # only reveals at 2:30 (after it) — a real leak into fold 0.
            label_timestamp = start + timedelta(hours=1, minutes=30)
        elif hour == 7:
            # Opens at 7:30 (before holdout_start=8:00) but only reveals at
            # 8:30 (after holdout_start) — a real leak into holdout.
            label_timestamp = start + timedelta(hours=7, minutes=30)
        else:
            label_timestamp = start + timedelta(hours=hour + 1)
        rows.append({
            "symbol": "BTCUSDT",
            "timestamp": start + timedelta(hours=hour),
            "label_timestamp": label_timestamp,
        })
    full = pd.DataFrame(rows)

    train, holdout, cv = MLSignalGenerator.partition_training_rows(full)

    holdout_start = start + timedelta(hours=8)  # 9th of 10 distinct timestamps
    assert holdout["timestamp"].min() == holdout_start
    assert (start + timedelta(hours=7)) not in set(train["timestamp"])
    # hour 6's label reveals exactly at holdout_start (7:00 open + 1h = 8:00)
    # — known before holdout starts, so it's correctly retained.
    assert (start + timedelta(hours=6)) in set(train["timestamp"])

    # CV fold 0 (TimeSeriesSplit over 7 remaining train timestamps 0..6):
    # train=[0,1], validation_start=hour 2. hour 1's reveal (2:30) is after
    # validation_start (2:00), so it must be purged from fold 0's training
    # rows even though hour 0 (reveal 2:00, exactly at the boundary) stays.
    fold0_train_idx, _ = cv[0]
    fold0_timestamps = set(train.loc[fold0_train_idx, "timestamp"])
    assert (start + timedelta(hours=1)) not in fold0_timestamps
    assert start in fold0_timestamps


def test_train_cutoff_excludes_row_at_label_open_and_includes_one_bar_later(tmp_path, monkeypatch):
    """The same look-ahead bug at the train()-level cutoff, not just
    load_training_data(): as_of exactly at a label's own open time must
    exclude that row from the frame handed to partition_training_rows, while
    as_of one bar later (once the label candle has closed) includes it."""
    monkeypatch.setattr(MLSignalGenerator, "MODEL_PATH", str(tmp_path / "model.pkl"))
    monkeypatch.setattr(MLSignalGenerator, "SCALER_PATH", str(tmp_path / "scaler.pkl"))
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for hour in range(15):
        row = {
            name: float(hour + offset / 100)
            for offset, name in enumerate(MLSignalGenerator.FEATURES)
        }
        row.update(
            symbol="BTCUSDT",
            timestamp=start + timedelta(hours=hour),
            label_timestamp=start + timedelta(hours=hour + 1),
            target=hour % 2,
        )
        rows.append(row)
    full = pd.DataFrame(rows)
    gen = MLSignalGenerator(None)
    monkeypatch.setattr(gen, "load_training_data",
                        lambda symbol, **kwargs: full[full.symbol == symbol].copy())
    params = {"max_depth": 2, "n_estimators": 2, "learning_rate": 0.1}

    seen_timestamps = []
    real_partition = MLSignalGenerator.partition_training_rows

    def capture(frame):
        seen_timestamps.append(set(frame["timestamp"]))
        return real_partition(frame)

    monkeypatch.setattr(gen, "partition_training_rows", capture)

    # Row for hour 13 has label_timestamp = hour 14 (its own open time).
    at_open = gen.train(["BTCUSDT"], as_of=start + timedelta(hours=14),
                        train_window_days=540, frozen_params=params)
    at_close = gen.train(["BTCUSDT"], as_of=start + timedelta(hours=15),
                         train_window_days=540, frozen_params=params)

    assert at_open and at_close
    row13 = start + timedelta(hours=13)
    assert row13 not in seen_timestamps[0]
    assert row13 in seen_timestamps[1]


def test_frozen_training_persists_provenance_and_legacy_pickle_loads(tmp_path, monkeypatch):
    monkeypatch.setattr(MLSignalGenerator, "MODEL_PATH", str(tmp_path / "model.pkl"))
    monkeypatch.setattr(MLSignalGenerator, "SCALER_PATH", str(tmp_path / "scaler.pkl"))
    rows = _rows(hours=40)
    gen = MLSignalGenerator(None)
    monkeypatch.setattr(
        gen, "load_training_data",
        lambda symbol, **kwargs: rows[rows.symbol == symbol].copy(),
    )
    cutoff = datetime(2026, 1, 3, tzinfo=UTC)
    params = {"max_depth": 2, "n_estimators": 2, "learning_rate": 0.1}
    metrics = gen.train(["ETHUSDT", "BTCUSDT"], as_of=cutoff,
                        train_window_days=540, frozen_params=params)

    metadata = json.loads((tmp_path / "model.metadata.json").read_text())
    assert metrics["train_size"] > 0 and metrics["test_size"] > 0
    assert metadata["as_of"] == "2026-01-03T00:00:00+00:00"
    assert metadata["train_window_days"] == 540
    assert metadata["feature_schema"] == MLSignalGenerator.FEATURES
    assert metadata["label_horizon_bars"] == 8
    assert metadata["chosen_params"] == params
    assert metadata["parameter_mode"] == "frozen"
    assert len(metadata["data_fingerprint_sha256"]) == 64
    assert "source_revision" in metadata
    assert len(metadata["source_file_sha256"]) == 64
    assert isinstance(metadata["source_dirty"], bool)
    model_sha = hashlib.sha256((tmp_path / "model.pkl").read_bytes()).hexdigest()
    scaler_sha = hashlib.sha256((tmp_path / "scaler.pkl").read_bytes()).hexdigest()
    assert metadata["model_sha256"] == model_sha
    assert metadata["scaler_sha256"] == scaler_sha
    # Reveal time is exactly one bar after open time — labelled distinctly
    # so a consumer can't mistake the open time for the moment it was known.
    open_ts = datetime.fromisoformat(metadata["training_last_label_open_timestamp"])
    reveal_ts = datetime.fromisoformat(metadata["training_last_label_reveal_timestamp"])
    assert reveal_ts - open_ts == MLSignalGenerator.BAR_DURATION

    (tmp_path / "model.metadata.json").unlink()  # pre-provenance artifacts remain loadable
    loaded = MLSignalGenerator(None)
    assert loaded.load_model()
    scaled = loaded.scaler.transform(np.ones((1, len(gen.FEATURES))))
    assert loaded.model.predict_proba(scaled).shape == (1, 2)


def test_training_rejects_naive_cutoff():
    with pytest.raises(ValueError, match="timezone-aware"):
        MLSignalGenerator(None).train(["BTCUSDT"], as_of=datetime(2026, 1, 1))


def test_fingerprint_changes_when_historical_feature_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(MLSignalGenerator, "MODEL_PATH", str(tmp_path / "model.pkl"))
    monkeypatch.setattr(MLSignalGenerator, "SCALER_PATH", str(tmp_path / "scaler.pkl"))
    rows = _rows(hours=25, symbols=("BTCUSDT",))
    gen = MLSignalGenerator(None)
    monkeypatch.setattr(gen, "load_training_data", lambda symbol, **kwargs: rows.copy())
    params = {"max_depth": 2, "n_estimators": 2, "learning_rate": 0.1}
    cutoff = datetime(2026, 1, 2, 2, tzinfo=UTC)
    gen.train(["BTCUSDT"], as_of=cutoff, frozen_params=params)
    first = json.loads((tmp_path / "model.metadata.json").read_text())["data_fingerprint_sha256"]
    rows.loc[0, "rsi_14_proxy"] += 1
    gen.train(["BTCUSDT"], as_of=cutoff, frozen_params=params)
    second = json.loads((tmp_path / "model.metadata.json").read_text())["data_fingerprint_sha256"]
    assert first != second
    rows.loc[0, "timestamp"] += timedelta(minutes=1)
    gen.train(["BTCUSDT"], as_of=cutoff, frozen_params=params)
    third = json.loads((tmp_path / "model.metadata.json").read_text())["data_fingerprint_sha256"]
    assert second != third


def test_training_cli_normalizes_aware_cutoff_and_rejects_naive():
    args = parse_train_args(["--as-of", "2026-01-01T07:00:00+07:00"])
    assert args.as_of == datetime(2026, 1, 1, tzinfo=UTC)
    assert args.train_window_days == 540
    with pytest.raises(SystemExit):
        parse_train_args(["--as-of", "2026-01-01T00:00:00"])
