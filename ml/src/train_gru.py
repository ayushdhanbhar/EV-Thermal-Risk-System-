import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from data_loader import ARTIFACTS_DIR, MODELS_DIR
from preprocessing import FEATURES, preprocess_and_split_data
from sequence_builder import build_sequences
from gru_model import build_gru_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_random_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


if __name__ == "__main__":
    set_random_seed(42)
    train_df, val_df, test_df, scaler, label_encoder = preprocess_and_split_data()
    with open(ARTIFACTS_DIR / "feature_config.json", "r", encoding="utf-8") as fp:
        feature_config = json.load(fp)
    window_size = int(feature_config["window_size"])

    X_train, y_train = build_sequences(train_df[FEATURES + ["risk_label"]], FEATURES, label_col="risk_label", window_size=window_size)
    X_val, y_val = build_sequences(val_df[FEATURES + ["risk_label"]], FEATURES, label_col="risk_label", window_size=window_size)

    X_train = scaler.transform(X_train.reshape(-1, 4)).reshape(-1, window_size, 4)
    X_val = scaler.transform(X_val.reshape(-1, 4)).reshape(-1, window_size, 4)

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_map = {int(cls): float(weight) for cls, weight in zip(classes, class_weights)}

    model = build_gru_model((window_size, len(FEATURES)), len(np.unique(y_train)))
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(MODELS_DIR / "gru_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=64,
        class_weight=class_weight_map,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1,
    )

    model.save(MODELS_DIR / "gru_model.keras")
    report_dir = PROJECT_ROOT / "reports" / "gru"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "training_history.json", "w", encoding="utf-8") as fp:
        json.dump(history.history, fp, indent=2)

    print("GRU training complete.")
