import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from dashboard.experiments_view import render

st.set_page_config(page_title="Experiment comparison", layout="wide")
render()
