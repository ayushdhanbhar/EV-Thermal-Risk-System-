import json
from pathlib import Path

import joblib
import numpy as np
from tensorflow.keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"


def advisory_from_risk(risk_level: str) -> str:
    mapping = {
        "LOW": "NORMAL OPERATION",
        "MEDIUM": "MONITOR BATTERY",
        "HIGH": "REDUCE LOAD",
        "CRITICAL": "CRITICAL WARNING",
    }
    return mapping.get(str(risk_level).upper(), "MONITOR BATTERY")


def _load_feature_order():
    with open(ARTIFACTS_DIR / "feature_config.json", "r", encoding="utf-8") as fp:
        return json.load(fp)["features"]


def predict_risk(sequence):
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.shape != (30, 4):
        raise ValueError(f"Expected sequence shape (30, 4); got {seq.shape}")

    feature_order = _load_feature_order()
    if list(feature_order) != ["Battery_Temperature", "Battery_Current", "Battery_Voltage", "Motor_RPM"]:
        raise ValueError(f"Feature order mismatch: expected {feature_order}")

    scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
    scaled = scaler.transform(seq)

    model_config = json.load(open(ARTIFACTS_DIR / "model_config.json", "r", encoding="utf-8"))
    selected_model = model_config.get("selected_model", "lstm")
    model_path = MODELS_DIR / f"{selected_model}_model.keras"
    if not model_path.exists():
        fallback = MODELS_DIR / "lstm_model.keras"
        if fallback.exists():
            model_path = fallback
        else:
            raise FileNotFoundError("No trained model found. Train a model first.")

    model = load_model(model_path)
    prob_vector = model.predict(scaled[np.newaxis, ...], verbose=0)[0]
    label_encoder = joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")
    class_index = int(np.argmax(prob_vector))
    risk_level = str(label_encoder.inverse_transform([class_index])[0])
    risk_probability = float(prob_vector[class_index])
    probabilities = {label: float(prob_vector[idx]) for idx, label in enumerate(label_encoder.classes_)}

    return {
        "risk_level": risk_level,
        "risk_probability": risk_probability,
        "probabilities": probabilities,
        "advisory": advisory_from_risk(risk_level),
    }
