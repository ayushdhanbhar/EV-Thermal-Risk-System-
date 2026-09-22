from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "EV_Predictive_Maintenance_Dataset_15min.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOTS_DIR = PROJECT_ROOT / "plots"


def ensure_directories():
    for directory in [PROCESSED_DIR, ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR, PLOTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_raw_dataset(path: str | Path | None = None) -> pd.DataFrame:
    dataset_path = Path(path) if path is not None else RAW_DATA_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {dataset_path}")
    df = pd.read_csv(dataset_path)
    return df


def save_processed_split(df: pd.DataFrame, name: str):
    ensure_directories()
    output_path = PROCESSED_DIR / f"{name}.csv"
    df.to_csv(output_path, index=False)
    return output_path
