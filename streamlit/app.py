import json
import sys
import time
from pathlib import Path

import pandas as pd
import serial
import serial.tools.list_ports
import streamlit as st

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ML_SRC_DIR = PROJECT_ROOT / "ml" / "src"

if str(ML_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SRC_DIR))

# ============================================================
# IMPORTS
# ============================================================

from dashboard import render_dashboard
from input_handler import (
    load_test_sequence,
    read_manual_sequence,
    read_uploaded_sequence,
)
from ml.src.inference import predict_risk

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="EV Battery Thermal Risk Advisory",
    layout="wide",
)

st.title("EV Battery Thermal Risk Advisory System")

st.caption("Live ESP32 → 30×4 Sequence → GRU Prediction")

# ============================================================
# BATTERY INFORMATION
# ============================================================

st.header("Battery Information")

col1, col2 = st.columns(2)

with col1:
    battery_max = st.number_input(
        "Maximum Battery Voltage (V)",
        value=420.0,
        step=1.0,
    )

    battery_normal = st.number_input(
        "Normal Battery Voltage (V)",
        value=360.0,
        step=1.0,
    )

with col2:
    battery_min = st.number_input(
        "Minimum Battery Voltage (V)",
        value=300.0,
        step=1.0,
    )

    battery_capacity = st.number_input(
        "Battery Capacity (Ah)",
        value=100.0,
        step=1.0,
    )

st.divider()

# ============================================================
# INPUT MODE
# ============================================================

source = st.radio(
    "Choose Input Mode",
    (
        "Upload CSV",
        "Manual Entry",
        "Use Test Dataset",
        "Live ESP32",
    ),
)

sequence = None

# ============================================================
# UPLOAD CSV
# ============================================================

if source == "Upload CSV":

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    if uploaded is not None:

        try:
            sequence = read_uploaded_sequence(uploaded)
            st.success("CSV loaded successfully.")

        except Exception as e:
            st.error(e)

# ============================================================
# MANUAL ENTRY
# ============================================================

elif source == "Manual Entry":

    default = [
        {
            "Battery_Temperature": 298 + i * 0.3,
            "Battery_Current": 20 + i * 0.2,
            "Battery_Voltage": battery_normal,
            "Motor_RPM": 1500 + i * 5,
        }
        for i in range(30)
    ]

    df = st.data_editor(
        pd.DataFrame(default),
        use_container_width=True,
    )

    if st.button("Predict"):

        sequence = read_manual_sequence(
            df.to_dict("records")
        )

# ============================================================
# TEST DATA
# ============================================================

elif source == "Use Test Dataset":

    if st.button("Load Test Sequence"):

        sequence = load_test_sequence()

# ============================================================
# LIVE ESP32
# ============================================================

elif source == "Live ESP32":

    st.subheader("ESP32 Live Acquisition")

    ports = serial.tools.list_ports.comports()
    port_names = [p.device for p in ports]

    if len(port_names) == 0:

        st.warning("No ESP32 COM Port Detected")

    else:

        com = st.selectbox("Select COM Port", port_names)

        st.info(f"ESP32 Connected : {com}")

        st.write("Prototype Battery Voltage = 14.8 V")

        if st.button("Start Live Prediction"):

            progress = st.progress(0, text="Connecting ESP32...")

            live_table = st.empty()

            timer = st.empty()

            samples = []

            try:

                ser = serial.Serial(
                    com,
                    115200,
                    timeout=0.1,
                )

                time.sleep(1)

                ser.reset_input_buffer()

                start = time.time()

                while len(samples) < 30:

                    line = (
                        ser.readline()
                        .decode("utf-8", errors="ignore")
                        .strip()
                    )

                    if not line.startswith("{"):
                        continue

                    try:
                        data = json.loads(line)

                    except:
                        continue

                    # -------------------------------------------------
                    # CONVERSION
                    # -------------------------------------------------

                    prototype_voltage = float(data["voltage"])

                    # Celsius → Kelvin
                    temperature_kelvin = (
                        float(data["temperature"]) + 273.15
                    )

                    # Prototype (14.8V) → User Battery
                    mapped_voltage = (
                        prototype_voltage / 14.8
                    ) * battery_normal

                    row = {
                        "Battery_Temperature": temperature_kelvin,
                        "Battery_Current": float(data["current"]),
                        "Battery_Voltage": mapped_voltage,
                        "Motor_RPM": float(data["rpm"]),
                    }

                    samples.append(row)

                    df_live = pd.DataFrame(samples)

                    live_table.dataframe(
                        df_live,
                        use_container_width=True,
                        height=260,
                    )

                    progress.progress(
                        len(samples) / 30,
                        text=f"Collecting {len(samples)}/30 Samples",
                    )

                    timer.write(
                        f"Elapsed : {time.time()-start:.1f} sec"
                    )

                ser.close()

                progress.progress(
                    1.0,
                    text="30 Samples Collected Successfully",
                )

                sequence = read_manual_sequence(samples)

                result = predict_risk(sequence)

                st.session_state["probs_df"] = (
                    pd.DataFrame(
                        [
                            {
                                "Risk": k,
                                "Probability": v,
                            }
                            for k, v in result[
                                "probabilities"
                            ].items()
                        ]
                    ).set_index("Risk")
                )

                st.success("GRU Prediction Completed")

                render_dashboard(sequence, result)

            except Exception as e:

                st.error(f"ESP32 Error : {e}")

# ============================================================
# NORMAL PREDICTION
# ============================================================

if sequence is not None and source != "Live ESP32":

    try:

        result = predict_risk(sequence)

        st.session_state["probs_df"] = (
            pd.DataFrame(
                [
                    {
                        "Risk": k,
                        "Probability": v,
                    }
                    for k, v in result[
                        "probabilities"
                    ].items()
                ]
            ).set_index("Risk")
        )

        render_dashboard(sequence, result)

    except Exception as e:

        st.error(f"Prediction Error : {e}")