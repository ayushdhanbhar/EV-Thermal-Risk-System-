import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "risk_config.json"


def load_risk_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Risk config not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fp:
        config = json.load(fp)
    return config


def _risk_from_temperature(temp_value: float, temp_delta: float, config: dict) -> int:
    class_order = config.get("class_order", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    risk_levels = config.get("risk_levels", {})

    critical_cfg = risk_levels.get("CRITICAL", {})
    high_cfg = risk_levels.get("HIGH", {})
    medium_cfg = risk_levels.get("MEDIUM", {})

    if temp_value >= float(critical_cfg.get("min_temperature_c", 55.0)) or temp_delta >= float(critical_cfg.get("min_delta_c", 4.0)):
        return class_order.index("CRITICAL")
    if temp_value >= float(high_cfg.get("min_temperature_c", 46.0)) or temp_delta >= float(high_cfg.get("max_delta_c", 6.0)):
        return class_order.index("HIGH")
    if temp_value >= float(medium_cfg.get("min_temperature_c", 34.0)) or temp_delta >= float(medium_cfg.get("max_delta_c", 3.5)):
        return class_order.index("MEDIUM")
    return class_order.index("LOW")


def derive_project_risk_labels(df: pd.DataFrame, config: dict | None = None) -> pd.Series:
    config = config or load_risk_config()
    feature_temp = "Battery_Temperature"
    if feature_temp not in df.columns:
        raise KeyError(f"Required temperature column '{feature_temp}' not found in dataframe.")

    temp_series = pd.to_numeric(df[feature_temp], errors="coerce")
    prev_temp = temp_series.shift(1).fillna(temp_series)
    temp_delta = (temp_series - prev_temp).fillna(0.0)

    computed = []
    for temp_value, temp_trend in zip(temp_series, temp_delta):
        computed.append(_risk_from_temperature(float(temp_value), float(temp_trend), config))

    return pd.Series(computed, index=df.index, name="risk_label")


def encode_risk_labels(risk_series: pd.Series, classes: list[str] | None = None) -> pd.Series:
    from sklearn.preprocessing import LabelEncoder

    if classes is None:
        classes = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    encoder = LabelEncoder()
    encoder.fit(classes)
    encoded = encoder.transform(risk_series.astype(str))
    return pd.Series(encoded, index=risk_series.index, name="risk_label_index")
