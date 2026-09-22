import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ML_SRC_DIR = PROJECT_ROOT / "ml" / "src"
if str(ML_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SRC_DIR))

from dashboard import render_dashboard
from input_handler import load_test_sequence, read_manual_sequence, read_uploaded_sequence
from ml.src.inference import predict_risk

st.set_page_config(page_title="EV Thermal Risk Advisory", layout="wide")
st.title("EV Battery Thermal Risk Advisory System")
st.caption("4 sensor channels × 30 time steps = 30×4 sequence for the LSTM/GRU model")
st.markdown("This app expects the recent battery window as a time-series matrix: Temperature, Current, Voltage, and Motor RPM over the last 30 observations.")

source = st.radio("Choose input mode", ["Upload CSV", "Manual entry", "Use test dataset sequence"])
sequence = None

if source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV with Battery_Temperature, Battery_Current, Battery_Voltage, Motor_RPM", type=["csv"])
    if uploaded_file is not None:
        try:
            sequence = read_uploaded_sequence(uploaded_file)
            st.success("30×4 sequence loaded successfully.")
        except Exception as exc:
            st.error(f"Upload error: {exc}")
elif source == "Manual entry":
    st.write("Enter the last 30 battery observations. Each row is one timestamp and has 4 sensor values.")
    default_rows = [
        {
            "Battery_Temperature": 30.0 + i * 0.25,
            "Battery_Current": -25.0 + i * 0.15,
            "Battery_Voltage": 360.0 - i * 0.10,
            "Motor_RPM": 1800 + i * 4,
        }
        for i in range(30)
    ]
    manual_df = st.data_editor(
        pd.DataFrame(default_rows),
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Battery_Temperature": st.column_config.NumberColumn("Temperature (°C)", min_value=-100.0, max_value=200.0, step=0.1),
            "Battery_Current": st.column_config.NumberColumn("Current (A)", min_value=-500.0, max_value=500.0, step=0.1),
            "Battery_Voltage": st.column_config.NumberColumn("Voltage (V)", min_value=0.0, max_value=500.0, step=0.1),
            "Motor_RPM": st.column_config.NumberColumn("Motor RPM", min_value=0.0, max_value=20000.0, step=1.0),
        },
    )
    if st.button("Predict thermal risk"):
        try:
            sequence = read_manual_sequence(manual_df.to_dict("records"))
            st.success("30×4 time-series window assembled.")
        except Exception as exc:
            st.error(f"Manual entry error: {exc}")
else:
    if st.button("Load test sequence"):
        try:
            sequence = load_test_sequence()
            st.success("Loaded historical 30×4 test window.")
        except Exception as exc:
            st.error(f"Sequence load error: {exc}")

if sequence is not None:
    try:
        result = predict_risk(sequence)
        st.session_state["probs_df"] = pd.DataFrame(
            [{"Risk": k, "Probability": v} for k, v in result["probabilities"].items()]
        ).set_index("Risk")
        render_dashboard(sequence, result)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
