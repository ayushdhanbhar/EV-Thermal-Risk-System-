import numpy as np
import pandas as pd


def build_sequences(dataframe: pd.DataFrame, features: list[str], label_col: str = "risk_label", window_size: int = 30):
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if label_col not in dataframe.columns:
        raise KeyError(f"Label column '{label_col}' not found.")

    sequences = []
    labels = []
    for idx in range(window_size, len(dataframe) + 1):
        window = dataframe[features].iloc[idx - window_size:idx].values
        label = int(dataframe[label_col].iloc[idx - 1])
        sequences.append(window)
        labels.append(label)

    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.int32)
