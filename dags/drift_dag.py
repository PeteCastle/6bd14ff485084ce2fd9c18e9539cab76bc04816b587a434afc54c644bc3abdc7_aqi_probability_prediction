from src.monitoring.generate_drift import detect_drift
import mlflow
from src.constants import MODELS_DIR, OUTPUT_DIR, DATASET_DIR


def run_drift_detection():
    test_drift_results = detect_drift(
        DATASET_DIR / "processed" / "val_dataset.parquet",
        DATASET_DIR / "processed" / "drifted_val_dataset.parquet",
    )

    mlflow.log_param("test_drift_detected", test_drift_results["drift_detected"])
    mlflow.log_param(
        "test_overall_drift_score", test_drift_results["overall_drift_score"]
    )
