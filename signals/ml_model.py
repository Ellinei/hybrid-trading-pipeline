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

import os
import pickle
import sys
from datetime import datetime, timezone

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


class MLSignalGenerator:
    """XGBoost classifier: predicts k-bar forward return above cost."""

    MODEL_PATH  = "models/xgboost_signal.pkl"
    SCALER_PATH = "models/scaler.pkl"

    # Label definition — single source of truth, shared with the backtest
    # engine's _load_all_features so the two target computations can't drift.
    LABEL_HORIZON_BARS = 8     # ≈ observed median hold time on 1h bars
    LABEL_MIN_RETURN   = 0.003 # ≈ round-trip fees + slippage

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

    def load_training_data(self, symbol: str, min_rows: int = 200) -> pd.DataFrame:
        """Load features + k-bar forward-return target from the feature table."""
        query = text(f"""
            SELECT
                rsi_14_proxy, macd_line, bb_zscore,
                pct_change_1h, pct_change_24h, log_return_1h,
                range_14, volume,
                log_return_7d, pct_change_4h,
                close_vs_sma20, close_vs_sma50,
                bb_width, atr_normalized,
                vol_ma_ratio,
                candle_body_pct, upper_wick_pct, stoch_k,
                close,
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

        df = df.dropna(subset=["future_close"])
        df["target"] = self.compute_target(df["close"], df["future_close"])
        df = df.dropna(subset=self.FEATURES + ["target"])

        if len(df) < min_rows:
            log.warning("insufficient_training_data", symbol=symbol, rows=len(df))
            return pd.DataFrame()

        return df[self.FEATURES + ["target"]]

    def train(self, symbols: list[str]) -> dict:
        """Train on all symbols combined.  Returns metrics dict, empty if no data."""
        frames = [self.load_training_data(sym) for sym in symbols]
        frames = [f for f in frames if not f.empty]

        if not frames:
            log.warning("no_training_data_for_any_symbol")
            return {}

        full = pd.concat(frames, ignore_index=True)
        X = full[self.FEATURES].values
        y = full["target"].to_numpy()

        # Outer 80/20 holdout — temporal order preserved, no shuffle
        split = int(len(X) * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

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
        grid = GridSearchCV(
            pipe, param_grid,
            cv=TimeSeriesSplit(n_splits=5),
            scoring="roc_auc",
            refit=True,
            n_jobs=-1,
        )
        grid.fit(X_tr, y_tr)

        # Extract fitted scaler and classifier from the winning pipeline.
        # Preserves the existing two-file pickle format and inference path.
        scaler: StandardScaler = grid.best_estimator_.named_steps["scaler"]
        model: XGBClassifier   = grid.best_estimator_.named_steps["clf"]
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
            "best_params": grid.best_params_,
            "feature_importance": dict(
                zip(self.FEATURES, model.feature_importances_.tolist())
            ),
        }

        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        with open(self.MODEL_PATH,  "wb") as f: pickle.dump(self.model,  f)
        with open(self.SCALER_PATH, "wb") as f: pickle.dump(self.scaler, f)

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
        """
        n_splits = 5
        required = self.FEATURES + ["target"]
        if not all(c in df.columns for c in required):
            missing = [c for c in required if c not in df.columns]
            log.warning("train_in_memory_missing_columns", missing=missing)
            return False

        data = df[required].dropna()
        # Same floor for both paths: the frozen-params fit doesn't need CV
        # folds, but a near-empty window (data gap, aggressive
        # train_window_days) shouldn't silently produce a degenerate model
        # either way.
        min_rows = n_splits + 1
        if len(data) < min_rows:
            log.warning("train_in_memory_insufficient_data",
                        rows=len(data), required=min_rows)
            return False

        X = data[self.FEATURES].values
        y = data["target"].to_numpy()

        if params is not None:
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
            cv=TimeSeriesSplit(n_splits=n_splits),
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
    metrics = gen.train(syms)

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
