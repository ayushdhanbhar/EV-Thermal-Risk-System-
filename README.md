# EV Thermal Risk Advisory System

This project implements a time-series deep learning workflow for battery thermal-risk prediction using a 30-sample window of four EV telemetry signals:

- Battery Temperature
- Battery Current
- Battery Voltage
- Motor RPM

## Project phases

1. Phase 1: train and compare LSTM and GRU on the recorded EV telemetry dataset.
2. Phase 2: validate the saved model through a Streamlit frontend with historical or uploaded sequences.
3. Phase 3: keep inference independent from the data source and use a single reusable prediction pipeline.

## Dataset note

The primary and current training dataset is the raw EV telemetry CSV located in data/raw/EV_Predictive_Maintenance_Dataset_15min.csv. No additional merged datasets are used.

## Risk labeling strategy

The dataset contains degradation indicators such as Failure_Probability, RUL, TTF, and Component_Health_Score, but these are not a direct 4-class thermal-risk target. For a leakage-safe project-specific risk label, the project derives risk from the current battery temperature and its recent thermal trend only, using thresholds defined in risk_config.json.

This keeps the label generation aligned with prediction-time information and avoids future-data leakage.

## Directory structure

- data/raw/ : raw source dataset
- data/processed/ : programmatically split train/validation/test CSVs
- ml/src/ : preprocessing, modeling, training, evaluation, inference modules
- ml/models/ : trained LSTM and GRU models
- ml/artifacts/ : scaler, encoder, and config artifacts
- streamlit/ : validation frontend
- plots/ : EDA and model visualization output
- reports/ : evaluation summaries and comparison outputs
