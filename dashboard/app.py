"""Streamlit monitoring dashboard for the Hybrid Trading Bot.

Tabs:
  Live        - live prices, latest signals, open positions
  Performance - P&L metrics, cumulative chart, trade history
  Signals     - RSI / MACD / Bollinger Band charts per symbol
  Model       - XGBoost feature importances + dbt table status

Run locally:
    poetry run streamlit run dashboard/app.py

Or via Docker (port 8501 after `docker compose up`).
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from datetime import UTC, datetime
from urllib.parse import quote_plus

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Ensure the project root is on sys.path when running from the dashboard/ dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Hybrid Trading Bot",
    page_icon=None,
    layout="wide",
)

SYMBOLS    = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
MODEL_PATH = "models/xgboost_signal.pkl"

# ---------------------------------------------------------------------------
# DB engine — cached as a resource (shared, never serialised by st.cache_data)
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_engine():
    """Create a SQLAlchemy engine once per Streamlit session."""
    url = "postgresql+psycopg2://{u}:{p}@{h}:{port}/{db}".format(
        u=quote_plus(os.getenv("POSTGRES_USER",     "trader")),
        p=quote_plus(os.getenv("POSTGRES_PASSWORD", "")),
        h=os.getenv("POSTGRES_HOST",  "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        db=os.getenv("POSTGRES_DB",   "trading"),
    )
    return create_engine(url)


# ---------------------------------------------------------------------------
# DB helpers — all use SQLAlchemy engine + pd.read_sql (no psycopg2 warnings)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_latest_trades(limit: int = 10) -> pd.DataFrame:
    try:
        with _get_engine().connect() as conn:
            return pd.read_sql(
                text("""
                    SELECT id, symbol, direction, entry_price, quantity,
                           signal_confidence, status, executed_at, pnl
                    FROM raw.trades
                    ORDER BY executed_at DESC
                    LIMIT :limit
                """),
                conn, params={"limit": limit},
            )
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def get_open_positions() -> pd.DataFrame:
    try:
        with _get_engine().connect() as conn:
            return pd.read_sql(
                text("""
                    SELECT id, symbol, direction, entry_price, quantity,
                           stop_loss, take_profit, signal_confidence, executed_at
                    FROM raw.trades
                    WHERE status = 'open'
                """),
                conn,
            )
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def get_closed_trades() -> pd.DataFrame:
    try:
        with _get_engine().connect() as conn:
            return pd.read_sql(
                text("""
                    SELECT * FROM raw.trades
                    WHERE status = 'closed' AND pnl IS NOT NULL
                    ORDER BY closed_at
                """),
                conn,
            )
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def get_technical_features(symbol: str, limit: int = 200) -> pd.DataFrame:
    try:
        with _get_engine().connect() as conn:
            df = pd.read_sql(
                text("""
                    SELECT timestamp, close, rsi_14_proxy, macd_line,
                           bb_upper, bb_middle, bb_lower, bb_zscore
                    FROM features.feat_technical
                    WHERE symbol = :symbol
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """),
                conn, params={"symbol": symbol, "limit": limit},
            )
        return df.sort_values("timestamp")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_dbt_tables() -> pd.DataFrame:
    try:
        with _get_engine().connect() as conn:
            return pd.read_sql(
                text("""
                    SELECT t.table_schema, t.table_name, t.table_type,
                           COALESCE(s.n_live_tup, 0) AS approx_rows
                    FROM information_schema.tables t
                    LEFT JOIN pg_stat_user_tables s
                        ON s.schemaname = t.table_schema
                       AND s.relname    = t.table_name
                    WHERE t.table_schema IN ('staging', 'features')
                    ORDER BY t.table_schema, t.table_name
                """),
                conn,
            )
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Live price helper (Binance testnet public REST — no auth needed)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=15)
def get_live_price(symbol: str) -> float:
    try:
        resp = requests.get(
            "https://testnet.binance.vision/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=5,
        )
        return float(resp.json()["price"])
    except Exception:
        return 0.0


# ===========================================================================
# Tabs
# ===========================================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["Live", "Performance", "Signals", "Model"]
)


# ---------------------------------------------------------------------------
# Tab 1 — Live
# ---------------------------------------------------------------------------

with tab1:
    now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(
        f"<h1 style='color:#e05252;'>Hybrid Trading Bot</h1>"
        f"<p style='color:#888;'>{now_utc} UTC</p>",
        unsafe_allow_html=True,
    )

    # Live prices
    prices: dict[str, float] = {}
    c1, c2, c3 = st.columns(3)
    for col, sym in zip([c1, c2, c3], SYMBOLS, strict=False):
        p = get_live_price(sym)
        prices[sym] = p
        col.metric(sym, f"${p:,.2f}" if p else "—")

    # Latest signals
    st.markdown("<h3 style='color:#e05252;'>Latest Signals</h3>", unsafe_allow_html=True)
    trades_df = get_latest_trades(limit=10)
    if trades_df.empty:
        st.info("No trades yet. Run `python -m execution.bot_runner --dry-run --once` first.")
    else:
        def _dir_color(val: str) -> str:
            return {
                "BUY":  "background-color:#ccffcc",
                "SELL": "background-color:#ffcccc",
            }.get(val, "background-color:#eeeeee")

        styled = trades_df.style.map(_dir_color, subset=["direction"])
        st.dataframe(styled, use_container_width=True)

    # Open positions + unrealised P&L
    st.markdown("<h3 style='color:#e05252;'>Open Positions</h3>", unsafe_allow_html=True)
    open_df = get_open_positions()
    if open_df.empty:
        st.info("No open positions.")
    else:
        open_df = open_df.copy()

        def _upnl(row: pd.Series) -> float:
            cp = prices.get(row["symbol"], 0.0)
            if cp <= 0:
                return 0.0
            mult = 1 if row["direction"] == "BUY" else -1
            return mult * (cp - float(row["entry_price"])) * float(row["quantity"])

        open_df["unrealized_pnl"] = open_df.apply(_upnl, axis=1)
        st.dataframe(open_df, use_container_width=True)

    # Auto-refresh
    if st.checkbox("Auto-refresh every 60 s", value=False):
        time.sleep(60)
        st.rerun()


# ---------------------------------------------------------------------------
# Tab 2 — Performance
# ---------------------------------------------------------------------------

with tab2:
    st.markdown("<h2 style='color:#4a90d9;'>Performance</h2>", unsafe_allow_html=True)
    closed_df = get_closed_trades()

    if closed_df.empty:
        st.info("No closed trades yet.")
    else:
        total   = len(closed_df)
        wins    = int((closed_df["pnl"] > 0).sum())
        wr      = wins / total * 100 if total else 0.0
        tot_pnl = float(closed_df["pnl"].sum())
        cum     = closed_df["pnl"].cumsum()
        max_dd  = float(cum.min()) if len(cum) else 0.0

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total Trades", total)
        mc2.metric("Win Rate",     f"{wr:.1f} %")
        mc3.metric("Total P&L",    f"${tot_pnl:+.2f}")
        mc4.metric("Max Drawdown", f"${max_dd:.2f}")

        closed_df["cumulative_pnl"] = cum
        fig_pnl = px.line(
            closed_df, x="closed_at", y="cumulative_pnl",
            title="Cumulative P&L",
            labels={"cumulative_pnl": "P&L ($)", "closed_at": "Date"},
        )
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_pnl, use_container_width=True)

        st.markdown("<h3 style='color:#4a90d9;'>Trade History</h3>", unsafe_allow_html=True)
        st.dataframe(closed_df, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3 — Signals
# ---------------------------------------------------------------------------

with tab3:
    st.markdown("<h2 style='color:#7b5ea7;'>Technical Signals</h2>", unsafe_allow_html=True)
    symbol  = st.selectbox("Symbol", SYMBOLS, key="sig_sym")
    feat_df = get_technical_features(symbol, limit=200)

    if feat_df.empty:
        st.info(f"No feature data for {symbol}. Run `dbt run` first.")
    else:
        # RSI
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=feat_df["timestamp"], y=feat_df["rsi_14_proxy"],
            name="RSI-14", line=dict(color="orange"),
        ))
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green",
                          annotation_text="Oversold 30")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",
                          annotation_text="Overbought 70")
        fig_rsi.update_layout(title=f"RSI-14 — {symbol}", height=300,
                               yaxis_range=[0, 100])
        st.plotly_chart(fig_rsi, use_container_width=True)

        # MACD
        fig_macd = px.line(
            feat_df, x="timestamp", y="macd_line",
            title=f"MACD Line — {symbol}",
            labels={"macd_line": "MACD"},
        )
        fig_macd.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_macd.update_layout(height=280)
        st.plotly_chart(fig_macd, use_container_width=True)

        # Bollinger Bands
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(
            x=feat_df["timestamp"], y=feat_df["bb_upper"],
            name="Upper Band", line=dict(color="blue", dash="dash", width=1),
        ))
        fig_bb.add_trace(go.Scatter(
            x=feat_df["timestamp"], y=feat_df["bb_lower"],
            name="Lower Band", line=dict(color="blue", dash="dash", width=1),
            fill="tonexty", fillcolor="rgba(0,100,200,0.08)",
        ))
        fig_bb.add_trace(go.Scatter(
            x=feat_df["timestamp"], y=feat_df["close"],
            name="Close", line=dict(color="black", width=2),
        ))
        fig_bb.update_layout(title=f"Bollinger Bands — {symbol}", height=350)
        st.plotly_chart(fig_bb, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4 — Model
# ---------------------------------------------------------------------------

with tab4:
    st.markdown("<h2 style='color:#2eaa6e;'>Model & Data Status</h2>", unsafe_allow_html=True)
    mc_left, mc_right = st.columns(2)

    with mc_left:
        st.subheader("XGBoost Model")
        if os.path.exists(MODEL_PATH):
            stat     = os.stat(MODEL_PATH)
            size_kb  = stat.st_size / 1024
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            st.success(f"Model loaded  •  {size_kb:.1f} KB  •  trained {modified}")

            with open(MODEL_PATH, "rb") as fh:
                model = pickle.load(fh)

            features = [
                "rsi_14_proxy", "macd_line", "bb_zscore",
                "pct_change_1h", "pct_change_24h", "log_return_1h",
                "range_14", "volume",
            ]
            fi_df = (
                pd.DataFrame({
                    "Feature":    features,
                    "Importance": model.feature_importances_,
                })
                .sort_values("Importance", ascending=True)
            )
            fig_fi = px.bar(
                fi_df, x="Importance", y="Feature", orientation="h",
                title="Feature Importances",
            )
            fig_fi.update_layout(height=350)
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.warning(
                "No model found. Train it first:\n\n"
                "```\npoetry run python -m signals.ml_model\n```"
            )

    with mc_right:
        st.subheader("dbt Tables")
        dbt_df = get_dbt_tables()
        if dbt_df.empty:
            st.info("No dbt tables found. Run `dbt run` first.")
        else:
            st.dataframe(dbt_df, use_container_width=True)
