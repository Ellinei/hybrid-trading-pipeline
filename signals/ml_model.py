"""XGBoost next-hour direction predictor.

Trains on ``features.feat_technical`` and predicts whether the next-hour
close will be higher than the current close (binary classification).

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
    """XGBoost classifier: predicts next-hour price direction."""

    MODEL_PATH  = "models/xgboost_signal.pkl"
    SCALER_PATH = "models/scaler.pkl"

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

    def __init__(self, db_engine) -> None:
        self.engine = db_engine
        self.model:  XGBClassifier   | None = None
        self.scaler: StandardScaler  | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def load_training_data(self, symbol: str, min_rows: int = 200) -> pd.DataFrame:
        """Load features + next-hour target from the feature table."""
        query = text("""
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
                LEAD(close, 1) OVER (PARTITION BY symbol ORDER BY timestamp) AS next_close
            FROM features.feat_technical
            WHERE symbol = :symbol
            ORDER BY timestamp
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query, {"symbol": symbol})
            rows   = result.fetchall()
            df     = pd.DataFrame(rows, columns=list(result.keys()))

        df = df.dropna(subset=["next_close"])
        df["target"] = (df["next_close"] > df["close"]).astype(int)
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
        # preventing scale leakage into validation folds.
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(eval_metric="logloss", random_state=42)),
        ])
        param_grid = {
            "clf__max_depth":     [3, 4],
            "clf__n_estimators":  [100, 200],
            "clf__learning_rate": [0.05, 0.1],
        }
        # 12 combos × 5 folds = 60 fits; typically finishes in < 2 minutes.
        grid = GridSearchCV(
            pipe, param_grid,
            cv=TimeSeriesSplit(n_splits=5),
            scoring="accuracy",
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

    def generate_signal(self, symbol: str, as_of: datetime | None = None) -> Signal:
        now = as_of if as_of is not None else datetime.now(timezone.utc)

        if self.model is None and not self.load_model():
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

        if prob_up > 0.65:
            direction, confidence = SignalDirection.BUY,  prob_up
        elif prob_up < 0.35:
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
        print("No training data. Run: rest_backfill → dbt run → signals.ml_model")
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
