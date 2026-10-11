"""Operations page: freshness, decisions, and the persistent simulated account."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict

import pandas as pd
import streamlit as st
from sqlalchemy import URL, create_engine

from dashboard.data import read_operations
from execution.freshness import FreshnessMonitor
from execution.journal import operations_path


@st.cache_resource
def operations_engine():
    return create_engine(
        URL.create(
            "postgresql+psycopg2",
            username=os.getenv("POSTGRES_USER", "trader"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "trading"),
        ),
        connect_args={"connect_timeout": 3},
    )


def render():
    st.title("Operations")
    st.caption("Source freshness, decision history, and a separate simulated account.")
    st.button("Refresh")
    symbols = [
        s.strip()
        for s in os.getenv("TRADING_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
        if s.strip()
    ]
    st.subheader("Data freshness")
    candle_age = st.number_input("Candle age limit (seconds)", min_value=1, value=10800)
    feature_age = st.number_input("Feature age limit (seconds)", min_value=1, value=10800)
    st.caption(
        "These controls preview thresholds. Set the bot's --candle-max-age and "
        "--feature-max-age to change entry checks. Streaming status is informational."
    )
    monitor = FreshnessMonitor(operations_engine(), candle_age, feature_age)
    rows = [
        {"symbol": symbol, **asdict(status)}
        for symbol in symbols
        for status in monitor.check(symbol)
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("Decision journal")
    symbol = st.selectbox("Symbol", ["All", *symbols])
    mode = st.selectbox("Mode", ["All", "paper", "dry_run", "live"])
    try:
        data = read_operations(
            operations_path(), None if symbol == "All" else symbol, None if mode == "All" else mode
        )
    except (OSError, ValueError, sqlite3.Error):
        st.error("The operations store is unavailable.")
        return
    if data["decisions"]:
        frame = pd.DataFrame(data["decisions"])
        st.dataframe(frame.drop(columns="details"), hide_index=True, use_container_width=True)
        selected = st.selectbox("Inspect decision", frame["id"].tolist())
        payload = next(row["details"] for row in data["decisions"] if row["id"] == selected)
        st.json(json.loads(payload))
        st.download_button(
            "Download decisions", frame.to_csv(index=False), "decisions.csv", "text/csv"
        )
    else:
        st.info("No decisions recorded for these filters. Run the bot to populate the journal.")

    st.subheader("Paper account")
    st.caption(
        "Account totals include all symbols. Stops are checked at bot-cycle quotes; "
        "moves between cycles may be missed. This is not an exchange fill replay."
    )
    if not data["account"]:
        st.info("No paper account yet. Start with: python -m cli trade --paper --once")
        return
    account = data["account"][0]
    positions = data["positions"]
    equity = account["cash"] + sum(p["entry_price"] * p["quantity"] for p in positions)
    left, middle, right = st.columns(3)
    left.metric("Available USDT", f"{account['cash']:,.2f}")
    middle.metric("Equity at entry cost", f"{equity:,.2f}")
    right.metric("Net P&L including open entry fees", f"{equity - account['initial_balance']:,.2f}")
    st.write("Open positions")
    st.dataframe(pd.DataFrame(positions), hide_index=True, use_container_width=True)
    st.write("Recently closed positions")
    st.dataframe(pd.DataFrame(data["closed"]), hide_index=True, use_container_width=True)
