from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"


def read_uploaded_sequence(file_obj):
    df = pd.read_csv(file_obj)
    required = ["Battery_Temperature", "Battery_Current", "Battery_Voltage", "Motor_RPM"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    seq = df[required].tail(30).copy()
    if len(seq) < 30:
        raise ValueError("At least 30 rows are required to form a time-series input window.")
    return seq[required].to_numpy(dtype=float)


def read_manual_sequence(entries):
    data = []
    for row in entries:
        data.append([
            float(row["Battery_Temperature"]),
            float(row["Battery_Current"]),
            float(row["Battery_Voltage"]),
            float(row["Motor_RPM"]),
        ])
    if len(data) < 30:
        raise ValueError("Please provide 30 sensor rows before prediction.")
    return pd.DataFrame(data, columns=["Battery_Temperature", "Battery_Current", "Battery_Voltage", "Motor_RPM"]).tail(30).to_numpy(dtype=float)


def load_test_sequence(limit=30):
    df = pd.read_csv(DATASET_PATH)
    features = ["Battery_Temperature", "Battery_Current", "Battery_Voltage", "Motor_RPM"]
    if len(df) < limit:
        raise ValueError("The processed test dataset is shorter than the required 30-sample window.")
    return df[features].tail(limit).to_numpy(dtype=float)
