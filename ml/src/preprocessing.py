import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from data_loader import ARTIFACTS_DIR, PROCESSED_DIR, RAW_DATA_PATH, ensure_directories
from risk_labeling import derive_project_risk_labels, load_risk_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES = [
    "Battery_Temperature",
    "Battery_Current",
    "Battery_Voltage",
    "Motor_RPM",
]


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["Timestamp"] = pd.to_datetime(cleaned["Timestamp"], errors="coerce")
    cleaned = cleaned.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    numeric_columns = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    for column in numeric_columns:
        if cleaned[column].isna().any():
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())
    return cleaned


def _validate_required_columns(df: pd.DataFrame):
    missing = [feature for feature in FEATURES if feature not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")


def save_feature_config(features=None, window_size=30, horizon_steps=1):
    config = {
        "features": features or FEATURES,
        "window_size": int(window_size),
        "horizon_steps": int(horizon_steps),
    }
    path = ARTIFACTS_DIR / "feature_config.json"
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2)
    return config


def save_model_config():
    config = {
        "window_size": 30,
        "num_features": 4,
        "features": FEATURES,
        "risk_classes": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        "selected_model": "lstm",
    }
    path = ARTIFACTS_DIR / "model_config.json"
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2)
    return config


def preprocess_and_split_data(raw_path: str | Path = RAW_DATA_PATH, config_path: str | Path | None = None):
    ensure_directories()
    config = load_risk_config(config_path)
    df = pd.read_csv(raw_path)
    df = _clean_dataframe(df)
    _validate_required_columns(df)
    df["risk_label"] = derive_project_risk_labels(df, config)

    class_order = config.get("class_order", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    label_encoder = LabelEncoder()
    label_encoder.fit(class_order)
    joblib.dump(label_encoder, ARTIFACTS_DIR / "label_encoder.pkl")

    train_size = int(len(df) * 0.70)
    validation_size = int(len(df) * 0.15)
    train_end = train_size
    val_end = train_size + validation_size

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    for name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        split_df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)

    scaler = StandardScaler()
    scaler.fit(train_df[FEATURES])
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")
    save_feature_config(FEATURES, config["window_size"], config["horizon_steps"])
    save_model_config()

    return train_df, val_df, test_df, scaler, label_encoder


if __name__ == "__main__":
    preprocess_and_split_data()
    print("Preprocessing complete.")
