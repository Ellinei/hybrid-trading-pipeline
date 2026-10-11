"""Compare existing experiment artifacts without retraining or database access."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from dashboard.data import comparison_rows, load_experiments


def render():
    st.title("Experiment comparison")
    root = st.text_input("Results directory", value=os.getenv("EXPERIMENT_RESULTS_DIR", "results"))
    runs, warnings = load_experiments(root)
    for warning in warnings:
        st.warning(warning)
    if not runs:
        st.info("No complete experiments found. Run python -m cli experiment --id <name> first.")
        return
    names = [run["id"] for run in runs]
    selected = st.multiselect("Experiments", names, default=names[:5])
    available_symbols = sorted({s for run in runs for s in run["metrics"].get("per_symbol", {})})
    symbol = st.selectbox("Breakdown", ["Portfolio", *available_symbols])
    selected_runs = [run for run in runs if run["id"] in selected]
    if not selected_runs:
        st.info("Choose at least one experiment.")
        return
    windows = {(r["start"], r["end"]) for r in selected_runs}
    if len(windows) > 1:
        st.warning(
            "Selected runs use different evaluation windows. Compare matching windows "
            "before drawing conclusions."
        )
    rows = comparison_rows(selected_runs, None if symbol == "Portfolio" else symbol)
    frame = pd.DataFrame(rows)
    if frame.empty:
        st.info("The selected experiments have no results for this symbol.")
        return
    st.caption(
        "Return, drawdown, and win rate are fractions: 0.05 means 5%. "
        "Missing metrics remain blank."
    )
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.download_button(
        "Download comparison", frame.to_csv(index=False), "comparison.csv", "text/csv"
    )
    if symbol == "Portfolio":
        st.bar_chart(frame.set_index("experiment")[["total_return", "max_drawdown"]])
    else:
        st.bar_chart(frame.set_index("experiment")[["pnl"]])
    with st.expander("Configurations and full metrics"):
        for run in selected_runs:
            st.write(run["id"])
            st.json({"config": run["config"], "metrics": run["metrics"]})
