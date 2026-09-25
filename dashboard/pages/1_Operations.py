import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
from dotenv import load_dotenv

from dashboard.operations_view import render

load_dotenv()
st.set_page_config(page_title="Operations", layout="wide")
render()
