"""XGBoost k-bar forward-return predictor.

Trains on ``features.feat_technical`` and predicts whether the close
LABEL_HORIZON_BARS ahead exceeds the current close by more than
LABEL_MIN_RETURN (binary classification). The horizon matches the
strategy's observed median hold time and the minimum return matches the
round-trip fee+slippage cost, so a positive label means a move worth
trading, not next-bar noise.

Usage:
    python -m signals.ml_model   # train and print metrics
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import structlog
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from xgboost import XGBClassifier

from signals.base import Signal, SignalDirection

log = structlog.get_logger()


def _parse_aware_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--as-of must be an ISO 8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("--as-of must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the cutoff-bounded live ML model")
    parser.add_argument("--as-of", type=_parse_aware_utc,
                        help="UTC-aware information cutoff (ISO 8601; defaults to now)")
    parser.add_argument("--train-window-days", type=int, default=540,
                        help="Lookback ending at --as-of (default: 540)")
    parser.add_argument("--frozen-params", type=Path,
                        help="JSON object of max_depth, n_estimators, learning_rate; skips tuning")
    args = parser.parse_args(argv)
    if args.train_window_days <= 0:
        parser.error("--train-window-days must be positive")
    return args


class MLSignalGenerator:
    """XGBoost classifier: predicts k-bar forward return above cost."""

    MODEL_PATH  = "models/xgboost_signal.pkl"
    SCALER_PATH = "models/scaler.pkl"

    # Label definition — single source of truth, shared with the backtest
    # engine's _load_all_features so the two target computations can't drift.
    LABEL_HORIZON_BARS = 8     # ≈ observed median hold time on 1h bars
    LABEL_MIN_RETURN   = 0.003 # ≈ round-trip fees + slippage

    # features.feat_technical.timestamp is the OHLCV candle's OPEN time
    # (ingestion/binance_client.py sets it from Binance's kline "open_time";
    # ingestion/rest_backfill.py and the staging/intermediate dbt models pass
    # it through unchanged). A candle's outcome — its close — is a full bar
    # later, so a label isn't actually knowable until its own timestamp plus
    # this duration.
    BAR_DURATION = timedelta(hours=1)

    @classmethod
    def label_reveal_time(cls, label_timestamp):
        """The instant a label's outcome is actually knowable.

        ``label_timestamp`` (LEAD(timestamp, LABEL_HORIZON_BARS)) is the
        label candle's OPEN time; its close isn't available until
        BAR_DURATION later. Comparing ``label_timestamp`` itself against a
        cutoff is the look-ahead bug this fixes. Accepts a scalar datetime
        or a pandas Series.
        """
        return label_timestamp + cls.BAR_DURATION

    FEATURES = [
        # Original momentum/volatility features
        "rsi_14_proxy", "macd_line", "bb_zscore",
        "pct_change_1h", "pct_change_24h", "log_return_1h",
        "range_14", "volume",
        # Extended: multi-horizon returns
        "log_return_7d", "pct_change_4h",
        # Extended: trend position
        "close_vs_sma20", "close_vs_sma50",
        # Extended: volatility
        "bb_width", "atr_normalized",
        # Extended: volume conviction
        "vol_ma_ratio",
        # Extended: candle structure
        "candle_body_pct", "upper_wick_pct", "stoch_k",
    ]

    def __init__(
        self,
        db_engine,
        buy_prob_threshold: float = 0.65,
        sell_prob_threshold: float = 0.35,
    ) -> None:
        self.engine = db_engine
        self.model:  XGBClassifier   | None = None
        self.scaler: StandardScaler  | None = None
        self.buy_prob_threshold = buy_prob_threshold
        self.sell_prob_threshold = sell_prob_threshold
        self.last_grid_params_: dict | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def compute_target(cls, close: pd.Series, future_close: pd.Series) -> pd.Series:
        """Vectorized label: 1 when the k-bar forward net return strictly
        exceeds LABEL_MIN_RETURN. Shared by SQL-path training and the
        backtest engine's feature pre-loader so the two can't drift.

        Postgres numeric columns arrive as Decimal, which can't mix with
        float arithmetic — coerce both series first."""
        close = close.astype(float)
        future_close = future_close.astype(float)
        return (future_close / close - 1.0 > cls.LABEL_MIN_RETURN).astype(int)

    def load_training_data(
        self, symbol: str, min_rows: int = 200, *,
        as_of: datetime | None = None, train_window_days: int = 540,
    ) -> pd.DataFrame:
        """Load labels whose *actual* future bar is known at the UTC cutoff.

        The SQL window runs over the complete symbol history before applying
        the cutoff and window. This retains labels at the window's left edge
        while excluding labels revealed after the cutoff, even across gaps.

        "Known at the cutoff" means the label candle has *closed* by then —
        see ``label_reveal_time``. Comparing ``label_timestamp`` (the label
        candle's open) directly against ``as_of`` would keep a row whose
        outcome is not yet observable at the cutoff.
        """
        as_of = self._utc_cutoff(as_of)
        if train_window_days <= 0:
            raise ValueError("train_window_days must be positive")
        query = text(f"""
            SELECT
                symbol, timestamp,
                rsi_14_proxy, macd_line, bb_zscore,
                pct_change_1h, pct_change_24h, log_return_1h,
                range_14, volume,
                log_return_7d, pct_change_4h,
                close_vs_sma20, close_vs_sma50,
                bb_width, atr_normalized,
                vol_ma_ratio,
                candle_body_pct, upper_wick_pct, stoch_k,
                close,
                LEAD(timestamp, {int(self.LABEL_HORIZON_BARS)}) OVER (
                    PARTITION BY symbol ORDER BY timestamp
                ) AS label_timestamp,
                LEAD(close, {int(self.LABEL_HORIZON_BARS)}) OVER (
                    PARTITION BY symbol ORDER BY timestamp
                ) AS future_close
            FROM features.feat_technical
            WHERE symbol = :symbol
            ORDER BY timestamp
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query, {"symbol": symbol})
            rows   = result.fetchall()
            df     = pd.DataFrame(rows, columns=list(result.keys()))

        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["label_timestamp"] = pd.to_datetime(df["label_timestamp"], utc=True)
        df = df.loc[
            (df["timestamp"] >= as_of - timedelta(days=train_window_days))
            & (df["timestamp"] <= as_of)
            & (self.label_reveal_time(df["label_timestamp"]) <= as_of)
        ].copy()
        df = df.dropna(subset=["future_close", "label_timestamp"])
        df["target"] = self.compute_target(df["close"], df["future_close"])
        df = df.dropna(subset=self.FEATURES + ["target"])

        if len(df) < min_rows:
            log.warning("insufficient_training_data", symbol=symbol, rows=len(df))
            return pd.DataFrame()

        return df[["symbol", "timestamp", "label_timestamp"] + self.FEATURES + ["target"]]

    @staticmethod
    def _utc_cutoff(as_of: datetime | None) -> datetime:
        cutoff = as_of if as_of is not None else datetime.now(timezone.utc)
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        return cutoff.astimezone(timezone.utc)

    @staticmethod
    def _time_grouped_purged_cv(
        frame: pd.DataFrame, label_col: str, n_splits: int = 5,
    ) -> tuple[pd.DataFrame, list]:
        """Causal CV folds shared by the disk-model outer split and
        train_in_memory's walk-forward tuning path.

        Sorts *frame* globally by (timestamp[, symbol]) so simultaneous
        rows across symbols stay adjacent, groups ``n_splits`` folds by
        distinct timestamp via TimeSeriesSplit, and purges each fold's
        training rows whose *label_col* reveal time (see
        ``label_reveal_time``) falls after that fold's own validation
        start — the same rule ``partition_training_rows`` applies at the
        outer holdout boundary, applied here per-fold so no fold trains on
        a label revealed after its own validation window begins.

        Returns (sorted_frame, cv) where sorted_frame is *frame* re-sorted
        and reset_index(drop=True), and cv is a list of positional
        (train_idx, val_idx) numpy arrays indexing into sorted_frame.
        Raises ValueError if there aren't at least n_splits+1 distinct
        timestamps, or if a fold's training rows are purged away entirely.
        """
        sort_cols = ["timestamp", "symbol"] if "symbol" in frame.columns else ["timestamp"]
        frame = frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        times = frame["timestamp"].drop_duplicates().sort_values().to_numpy()
        if len(times) < n_splits + 1:
            raise ValueError(
                f"at least {n_splits + 1} distinct timestamps are required for "
                f"{n_splits}-fold CV"
            )
        cv = []
        for train_group_idx, validation_group_idx in TimeSeriesSplit(n_splits=n_splits).split(times):
            validation_start = times[validation_group_idx[0]]
            train_idx = frame.index[
                frame["timestamp"].isin(times[train_group_idx])
                & (MLSignalGenerator.label_reveal_time(frame[label_col]) <= validation_start)
            ].to_numpy()
            validation_idx = frame.index[
                frame["timestamp"].isin(times[validation_group_idx])
            ].to_numpy()
            if not len(train_idx):
                raise ValueError("no training rows remain in a fold after CV horizon purge")
            cv.append((train_idx, validation_idx))
        return frame, cv

    @staticmethod
    def partition_training_rows(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list]:
        """Split global UTC time groups and purge labels crossing each boundary."""
        full = full.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)
        times = full["timestamp"].drop_duplicates().sort_values().to_list()
        if len(times) < 8:
            raise ValueError("at least eight distinct training timestamps are required")
        holdout_start = times[int(len(times) * 0.8)]
        # Purge on reveal (close) time, not label_timestamp's open time: a
        # label candle that closes exactly when holdout begins is fully
        # known beforehand and doesn't share a candle with holdout, so `<=`
        # here is correct — see label_reveal_time. This is at least as
        # strict as the old `label_timestamp < holdout_start`, since
        # `label_timestamp + BAR_DURATION <= holdout_start` implies
        # `label_timestamp < holdout_start`.
        train = full.loc[
            (full["timestamp"] < holdout_start)
            & (MLSignalGenerator.label_reveal_time(full["label_timestamp"]) <= holdout_start)
        ].reset_index(drop=True)
        holdout = full.loc[full["timestamp"] >= holdout_start].reset_index(drop=True)
        if holdout.empty:
            raise ValueError("insufficient distinct timestamps after horizon purge")
        # CV folds over the outer training split reuse the same causal
        # sort/group/purge machinery train_in_memory's tuning path uses for
        # walk-forward frames (_time_grouped_purged_cv), so the two code
        # paths can't drift apart.
        try:
            train, cv = MLSignalGenerator._time_grouped_purged_cv(train, "label_timestamp", n_splits=5)
        except ValueError as exc:
            raise ValueError(f"insufficient distinct timestamps after horizon purge ({exc})") from exc
        return train, holdout, cv

    def train(
        self, symbols: list[str], *, as_of: datetime | None = None,
        train_window_days: int = 540, frozen_params: dict | None = None,
    ) -> dict:
        """Train a cutoff-bounded disk model; tune unless params are supplied."""
        as_of = self._utc_cutoff(as_of)
        if train_window_days <= 0:
            raise ValueError("train_window_days must be positive")
        frames = [self.load_training_data(
            sym, as_of=as_of, train_window_days=train_window_days,
        ) for sym in symbols]
        frames = [f for f in frames if not f.empty]

        if not frames:
            log.warning("no_training_data_for_any_symbol")
            return {}

        full = pd.concat(frames, ignore_index=True)
        full["timestamp"] = pd.to_datetime(full["timestamp"], utc=True)
        full["label_timestamp"] = pd.to_datetime(full["label_timestamp"], utc=True)
        full = full.loc[
            (full["timestamp"] >= as_of - timedelta(days=train_window_days))
            & (full["timestamp"] <= as_of)
            & (self.label_reveal_time(full["label_timestamp"]) <= as_of)
        ].copy()
        try:
            training, holdout, cv = self.partition_training_rows(full)
        except ValueError as exc:
            log.warning("insufficient_training_timestamps", error=str(exc))
            return {}
        X_tr, X_te = training[self.FEATURES].values, holdout[self.FEATURES].values
        y_tr, y_te = training["target"].to_numpy(), holdout["target"].to_numpy()

        # Pipeline ensures StandardScaler is fit only on each CV training fold,
        # preventing scale leakage into validation folds. Classifier runs
        # single-threaded (n_jobs=1): GridSearchCV already parallelizes
        # across the 40 fold fits (n_jobs=-1) below, so per-fit all-core
        # threading would nest and oversubscribe.
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1)),
        ])
        param_grid = {
            "clf__max_depth":     [3, 4],
            "clf__n_estimators":  [100, 200],
            "clf__learning_rate": [0.05, 0.1],
        }
        # 8 combos × 5 folds = 40 fits; typically finishes in < 2 minutes.
        # roc_auc, not accuracy: the cost-aware label is imbalanced (only
        # ~35-45% of bars clear the +0.3% hurdle), so accuracy rewards
        # always-predict-negative models.
        if frozen_params is None:
            grid = GridSearchCV(
                pipe, param_grid, cv=cv, scoring="roc_auc", refit=True, n_jobs=-1,
            )
            grid.fit(X_tr, y_tr)
            fitted = grid.best_estimator_
            chosen_params = {key.removeprefix("clf__"): value
                             for key, value in grid.best_params_.items()}
            parameter_mode = "tuned"
        else:
            allowed = {"max_depth", "n_estimators", "learning_rate"}
            if set(frozen_params) != allowed:
                raise ValueError(f"frozen_params must contain exactly {sorted(allowed)}")
            fitted = pipe.set_params(**{f"clf__{key}": value
                                          for key, value in frozen_params.items()})
            fitted.fit(X_tr, y_tr)
            chosen_params = dict(frozen_params)
            parameter_mode = "frozen"

        # Extract fitted scaler and classifier from the winning pipeline.
        # Preserves the existing two-file pickle format and inference path.
        scaler: StandardScaler = fitted.named_steps["scaler"]
        model: XGBClassifier   = fitted.named_steps["clf"]
        self.scaler = scaler
        self.model  = model
        X_te_s = scaler.transform(X_te)

        y_pred = model.predict(X_te_s)
        metrics = {
            "accuracy":  float(accuracy_score(y_te, y_pred)),
            "precision": float(precision_score(y_te, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_te, y_pred, zero_division=0)),
            "f1":        float(f1_score(y_te, y_pred, zero_division=0)),
            "train_size": int(len(X_tr)),
            "test_size":  int(len(X_te)),
            "best_params": chosen_params,
            "feature_importance": dict(
                zip(self.FEATURES, model.feature_importances_.tolist())
            ),
        }

        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        with open(self.MODEL_PATH,  "wb") as f: pickle.dump(self.model,  f)
        with open(self.SCALER_PATH, "wb") as f: pickle.dump(self.scaler, f)

        # Hash the canonical ordered training input, including label reveal
        # times, so a changed historical row or target changes the identity.
        fingerprint_input = full.sort_values(["timestamp", "symbol"], kind="stable")
        fingerprint_input = fingerprint_input[
            ["symbol", "timestamp", "label_timestamp"] + self.FEATURES + ["target"]
        ]
        fingerprint = hashlib.sha256(fingerprint_input.to_json(
            orient="records", date_format="iso", date_unit="ns", double_precision=15,
        ).encode("utf-8")).hexdigest()
        try:
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                text=True, cwd=Path(__file__).resolve().parents[1],
            ).strip()
            source_dirty = bool(subprocess.check_output(
                ["git", "status", "--porcelain", "--", "signals/ml_model.py"],
                stderr=subprocess.DEVNULL, text=True,
                cwd=Path(__file__).resolve().parents[1],
            ).strip())
        except (OSError, subprocess.CalledProcessError):
            revision = None
            source_dirty = None
        metadata = {
            "as_of": as_of.isoformat(),
            "train_window_days": train_window_days,
            "symbols": sorted(set(symbols)),
            "feature_schema": self.FEATURES,
            "label_horizon_bars": self.LABEL_HORIZON_BARS,
            "label_min_return": self.LABEL_MIN_RETURN,
            "chosen_params": chosen_params,
            "parameter_mode": parameter_mode,
            "data_fingerprint_sha256": fingerprint,
            "source_revision": revision,
            "source_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_dirty": source_dirty,
            "model_sha256": hashlib.sha256(Path(self.MODEL_PATH).read_bytes()).hexdigest(),
            "scaler_sha256": hashlib.sha256(Path(self.SCALER_PATH).read_bytes()).hexdigest(),
            "train_rows": len(training),
            "validation_rows": len(holdout),
            "validation_start": holdout["timestamp"].min().isoformat(),
            # Renamed from the ambiguous "training_last_label_timestamp"
            # (no code consumers found — grepped before renaming) now that
            # open time and reveal time are both tracked. The open time
            # identifies the label candle; the reveal time is the value
            # that must be <= as_of.
            "training_last_label_open_timestamp": training["label_timestamp"].max().isoformat(),
            "training_last_label_reveal_timestamp":
                self.label_reveal_time(training["label_timestamp"]).max().isoformat(),
        }
        metadata_path = Path(self.MODEL_PATH).with_suffix(".metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        log.info("model_trained", **{k: v for k, v in metrics.items()
                                      if k != "feature_importance"})
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def is_trained(self) -> bool:
        return (os.path.exists(self.MODEL_PATH)
                and os.path.exists(self.SCALER_PATH))

    def load_model(self) -> bool:
        """Load persisted model + scaler.  Returns True on success."""
        if not self.is_trained():
            return False
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")   # suppress xgboost/sklearn pickle version noise
                with open(self.MODEL_PATH,  "rb") as f: self.model  = pickle.load(f)
                with open(self.SCALER_PATH, "rb") as f: self.scaler = pickle.load(f)
            return True
        except Exception as exc:
            log.warning("model_load_failed", error=str(exc))
            return False

    def get_latest_features(
        self, symbol: str, as_of: datetime | None = None
    ) -> dict | None:
        """Return the most recent feature row for *symbol*, or None.

        When *as_of* is given, returns the most recent row at or before that
        timestamp instead of the live latest row — used by the backtester to
        replay signals without leaking future data.
        """
        params: dict = {"symbol": symbol}
        cutoff_clause = ""
        if as_of is not None:
            params["as_of"] = as_of
            cutoff_clause = "AND timestamp <= :as_of"

        query = text(f"""
            SELECT rsi_14_proxy, macd_line, bb_zscore,
                   pct_change_1h, pct_change_24h, log_return_1h,
                   range_14, volume,
                   log_return_7d, pct_change_4h,
                   close_vs_sma20, close_vs_sma50,
                   bb_width, atr_normalized,
                   vol_ma_ratio,
                   candle_body_pct, upper_wick_pct, stoch_k
            FROM features.feat_technical
            WHERE symbol = :symbol
            {cutoff_clause}
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        with self.engine.connect() as conn:
            row = conn.execute(query, params).fetchone()
        return dict(row._mapping) if row else None

    # ------------------------------------------------------------------
    # Walk-forward in-memory training
    # ------------------------------------------------------------------

    def set_walk_forward_mode(self, enabled: bool) -> None:
        """When True, generate_signal uses only the in-memory model.
        During warm-up (before first train_in_memory call), returns HOLD
        instead of falling back to the disk-based model."""
        self._walk_forward_mode = enabled

    def is_trained_in_memory(self) -> bool:
        return getattr(self, "_trained_in_memory", False) and self.model is not None

    def train_in_memory(self, df: pd.DataFrame, params: dict | None = None) -> bool:
        """Train on a pre-loaded feature DataFrame without writing to disk.

        df must contain self.FEATURES columns plus 'target'.
        Returns True on success, False if df has insufficient rows for CV
        (requires n_splits+1 = 6 minimum) or missing columns.

        When *params* is given (e.g. {"max_depth": 3, "n_estimators": 100,
        "learning_rate": 0.05}), skips GridSearchCV entirely and fits a
        single classifier with those hyperparameters — the walk-forward
        precompute loop tunes once on the first window via a params=None
        call, reads back ``last_grid_params_``, and passes it to every
        subsequent window to avoid re-running an 8-combo x 5-fold search
        ~35 times per run.

        The params=None (GridSearchCV) branch needs causal CV folds: real
        walk-forward frames (backtest.signal_cache.available_training_rows
        / backtest.engine._load_all_features) are sorted (symbol,
        timestamp) — symbol-grouped, not globally time-ordered — and carry
        no purge, so a naive TimeSeriesSplit over that row order can train
        one symbol's newest rows against another symbol's older,
        contemporaneous validation rows, and can leave a training label
        revealed after its own fold's validation start (Phase 11a,
        commit 7b73f06). When the frame carries 'timestamp' plus a
        label-open column ('future_timestamp' for walk-forward frames,
        'label_timestamp' for the disk path's frames), folds are built by
        _time_grouped_purged_cv instead — the same sort/group/purge logic
        partition_training_rows uses for the disk model's outer split.
        A frame without timestamps (bare FEATURES+target, as unit tests
        pass) can't be ordered causally at all; it falls back to the old
        plain TimeSeriesSplit over row order with a warning. Every real
        walk-forward caller supplies timestamps, so that fallback only
        fires for synthetic test frames.
        """
        n_splits = 5
        required = self.FEATURES + ["target"]
        if not all(c in df.columns for c in required):
            missing = [c for c in required if c not in df.columns]
            log.warning("train_in_memory_missing_columns", missing=missing)
            return False

        label_col = ("future_timestamp" if "future_timestamp" in df.columns
                     else "label_timestamp" if "label_timestamp" in df.columns
                     else None)
        causal = "timestamp" in df.columns and label_col is not None

        dropna_cols = required + (["timestamp", label_col] if causal else [])
        data = df.dropna(subset=dropna_cols)
        # Same floor for both paths: the frozen-params fit doesn't need CV
        # folds, but a near-empty window (data gap, aggressive
        # train_window_days) shouldn't silently produce a degenerate model
        # either way.
        min_rows = n_splits + 1
        if len(data) < min_rows:
            log.warning("train_in_memory_insufficient_data",
                        rows=len(data), required=min_rows)
            return False

        if params is not None:
            X = data[self.FEATURES].values
            y = data["target"].to_numpy()
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(eval_metric="logloss", random_state=42,
                                       n_jobs=-1, **params)),
            ])
            try:
                pipe.fit(X, y)
            except Exception as exc:
                log.warning("train_in_memory_failed", error=str(exc))
                return False

            self.scaler = pipe.named_steps["scaler"]
            self.model  = pipe.named_steps["clf"]
            self._trained_in_memory = True
            log.info("train_in_memory_complete_frozen_params", rows=len(data),
                     params=params)
            return True

        # GridSearchCV tuning path only: causal folds when the frame allows
        # it, else the old row-order TimeSeriesSplit (legacy callers only).
        if causal:
            try:
                data, cv = self._time_grouped_purged_cv(data, label_col, n_splits=n_splits)
            except ValueError as exc:
                log.warning("train_in_memory_causal_cv_unavailable", error=str(exc))
                return False
        else:
            log.warning("train_in_memory_legacy_frame_lacks_timestamps")
            cv = TimeSeriesSplit(n_splits=n_splits)

        X = data[self.FEATURES].values
        y = data["target"].to_numpy()

        # Classifier runs single-threaded (n_jobs=1) here: GridSearchCV
        # already parallelizes across the 8x5=40 fold fits (n_jobs=-1), so
        # letting each individual fit also spawn all-core threads causes
        # nested oversubscription that slows the whole search down.
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1)),
        ])
        param_grid = {
            "clf__max_depth":     [3, 4],
            "clf__n_estimators":  [100, 200],
            "clf__learning_rate": [0.05, 0.1],
        }
        grid = GridSearchCV(
            pipe, param_grid,
            cv=cv,
            scoring="roc_auc",
            refit=True,
            n_jobs=-1,
        )
        try:
            grid.fit(X, y)
        except Exception as exc:
            log.warning("train_in_memory_failed", error=str(exc))
            return False

        self.scaler = grid.best_estimator_.named_steps["scaler"]
        self.model  = grid.best_estimator_.named_steps["clf"]
        self._trained_in_memory = True
        self.last_grid_params_ = {
            k.replace("clf__", ""): v for k, v in grid.best_params_.items()
        }
        log.info("train_in_memory_complete", rows=len(data),
                 best_params=grid.best_params_)
        return True

    def train_walk_forward(
        self, slice_df: pd.DataFrame, frozen_params: dict | None
    ) -> tuple[bool, dict | None]:
        """One walk-forward retrain boundary: tunes via GridSearchCV only
        when *frozen_params* is None (the first successful window), else
        reuses it. Returns (trained, params_to_pass_in_next_call) — callers
        thread the second value back in on every subsequent boundary instead
        of each re-implementing the freeze/reuse bookkeeping themselves.
        """
        trained = self.train_in_memory(slice_df, params=frozen_params)
        if trained and frozen_params is None:
            frozen_params = self.last_grid_params_
        return trained, frozen_params

    def generate_signal(self, symbol: str, as_of: datetime | None = None) -> Signal:
        now = as_of if as_of is not None else datetime.now(timezone.utc)

        if self.model is None:
            if getattr(self, "_walk_forward_mode", False):
                return Signal(
                    direction=SignalDirection.HOLD, confidence=0.5,
                    source="ml", symbol=symbol, timestamp=now,
                    metadata={"reason": "walk_forward_warmup"},
                )
            if not self.load_model():
                return Signal(
                    direction=SignalDirection.HOLD, confidence=0.5,
                    source="ml", symbol=symbol, timestamp=now,
                    metadata={"reason": "model_not_trained"},
                )

        features = self.get_latest_features(symbol, as_of=as_of)

        if features is None:
            return Signal(
                direction=SignalDirection.HOLD, confidence=0.5,
                source="ml", symbol=symbol, timestamp=now,
                metadata={"reason": "no_features_available"},
            )

        if any(features.get(f) is None for f in self.FEATURES):
            return Signal(
                direction=SignalDirection.HOLD, confidence=0.5,
                source="ml", symbol=symbol, timestamp=now,
                metadata={"reason": "null_features"},
            )

        feat_values = [float(features[f]) for f in self.FEATURES]
        assert self.scaler is not None
        assert self.model is not None
        X = self.scaler.transform([feat_values])
        prob_up = float(self.model.predict_proba(X)[0][1])

        if prob_up > self.buy_prob_threshold:
            direction, confidence = SignalDirection.BUY,  prob_up
        elif prob_up < self.sell_prob_threshold:
            direction, confidence = SignalDirection.SELL, 1.0 - prob_up
        else:
            direction, confidence = SignalDirection.HOLD, 0.5

        return Signal(
            direction=direction, confidence=confidence,
            source="ml", symbol=symbol, timestamp=now,
            metadata={
                "probability_up": prob_up,
                "features": dict(zip(self.FEATURES, feat_values)),
            },
        )


# ---------------------------------------------------------------------------
# Entry point — train and report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from urllib.parse import quote_plus

    from dotenv import load_dotenv
    from sqlalchemy import create_engine

    args = parse_train_args()
    frozen_params = None
    if args.frozen_params is not None:
        frozen_params = json.loads(args.frozen_params.read_text(encoding="utf-8"))
        if not isinstance(frozen_params, dict):
            raise ValueError("--frozen-params must contain a JSON object")
    load_dotenv()

    engine = create_engine(
        "postgresql+psycopg2://{u}:{p}@{h}:{port}/{db}".format(
            u=quote_plus(os.getenv("POSTGRES_USER",     "trader")),
            p=quote_plus(os.getenv("POSTGRES_PASSWORD", "")),
            h=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            db=os.getenv("POSTGRES_DB",   "trading"),
        )
    )

    syms = [s.strip() for s in os.getenv("TRADING_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")]
    gen  = MLSignalGenerator(engine)
    metrics = gen.train(syms, as_of=args.as_of,
                        train_window_days=args.train_window_days,
                        frozen_params=frozen_params)

    if not metrics:
        print("No training data. Run: rest_backfill -> dbt run -> signals.ml_model")
        sys.exit(1)

    print("\n=== Model Training Results ===")
    for k, v in metrics.items():
        if k == "feature_importance":
            continue
        print(f"  {k:15}: {v:.4f}" if isinstance(v, float) else f"  {k:15}: {v}")

    print("\n  Feature Importance:")
    for feat, imp in sorted(metrics["feature_importance"].items(), key=lambda x: -x[1]):
        print(f"    {feat:25}: {imp:.4f}")

    engine.dispose()
