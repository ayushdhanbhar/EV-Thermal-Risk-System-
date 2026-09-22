import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from data_loader import MODELS_DIR, PLOTS_DIR, REPORTS_DIR
from preprocessing import FEATURES
from sequence_builder import build_sequences

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def compute_metrics(y_true, y_pred, classes):
    report = classification_report(y_true, y_pred, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    result = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "confusion_matrix": cm,
        "high_recall": float(report.get("HIGH", {}).get("recall", 0.0)),
        "critical_recall": float(report.get("CRITICAL", {}).get("recall", 0.0)),
    }
    return result


def save_confusion_matrix_plot(cm, classes, save_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def evaluate_model(model_path, test_file, scaler_path, label_encoder_path, feature_names=FEATURES, window_size=30, label_name="risk_label"):
    test_df = pd.read_csv(test_file)
    X_seq, y_true = build_sequences(test_df[feature_names + [label_name]], feature_names, label_col=label_name, window_size=window_size)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    X_scaled = scaler.transform(X_seq.reshape(-1, len(feature_names))).reshape(-1, window_size, len(feature_names))

    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    probs = model.predict(X_scaled, verbose=0)
    pred_idx = probs.argmax(axis=1)
    pred_labels = label_encoder.inverse_transform(pred_idx)
    true_labels = label_encoder.inverse_transform(y_true)

    classes = list(label_encoder.classes_)
    metrics = compute_metrics(
        np.array([classes.index(lbl) for lbl in true_labels]),
        np.array([classes.index(lbl) for lbl in pred_labels]),
        classes,
    )
    return metrics, X_scaled, y_true, pred_idx, true_labels, pred_labels


def compare_models():
    project_root = PROJECT_ROOT
    classes = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    test_file = project_root / "data" / "processed" / "test.csv"
    scaler_path = project_root / "ml" / "artifacts" / "scaler.pkl"
    encoder_path = project_root / "ml" / "artifacts" / "label_encoder.pkl"
    
    metrics_list = []
    for name, model_path in [("LSTM", project_root / "ml" / "models" / "lstm_model.keras"), ("GRU", project_root / "ml" / "models" / "gru_model.keras")]:
        metrics, *_ = evaluate_model(model_path, test_file, scaler_path, encoder_path, FEATURES, 30, "risk_label")
        metrics_list.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "Macro F1": metrics["macro_f1"],
            "HIGH Recall": metrics["high_recall"],
            "CRITICAL Recall": metrics["critical_recall"],
            "Parameters": "N/A",
            "Model Size": "N/A",
            "Training Time": "N/A",
            "Inference Time": "N/A",
        })

    comparison_df = pd.DataFrame(metrics_list)
    comparison_path = project_root / "reports" / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    
    plot_data = comparison_df[["Model", "Accuracy", "Precision", "Recall", "Macro F1"]].copy()
    ax = plot_data.set_index("Model").plot(kind="bar", figsize=(10, 6))
    plt.title("LSTM vs GRU Model Performance")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(project_root / "plots" / "model_comparison.png", dpi=200)
    plt.close()

    return comparison_df


if __name__ == "__main__":
    compare_models()
    print("Model comparison complete.")
