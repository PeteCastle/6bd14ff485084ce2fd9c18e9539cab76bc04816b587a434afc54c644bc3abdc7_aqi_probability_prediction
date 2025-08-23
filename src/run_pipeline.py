import argparse

import mlflow
from src.drift_detection import detect_drift
from src.utils import setup_mlflow_tracking

from .data_preprocessing import get_preprocessed_data, get_raw_data
from .evaluation import run_evaluation
from .feature_engineering import get_feature_engineered_data
from .model_training import run_training
from .constants import DATASET_DIR
import os

# Call the function for compliance only.  Duplicated code
setup_mlflow_tracking()

def main():
    parser = argparse.ArgumentParser(
        description="Run the data preprocessing and model training pipeline."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the script in dry run mode (1 epoch and 1 tries per model).",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=30,
        help="Number of trials to run (default: 30)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of epochs to train each model (default: 30)",
    )

    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate a report after training and evaluation.",
    )

    args = parser.parse_args()

    os.system("cls" if os.name == "nt" else "clear")

    data = get_raw_data()
    data = get_preprocessed_data(data)
    data = get_feature_engineered_data(data)

    if args.dry_run:
        print("Running in dry run mode. Only 1 epoch and 1 trial will be executed.")
        studies = run_training(data, num_trials=1, num_epochs=1)
    else:
        studies = run_training(
            data, num_trials=args.num_trials, num_epochs=args.num_epochs
        )

    run_evaluation(studies, data, args.generate_report)

    print("Running drift detection on test set...")
    test_drift_results = detect_drift(
        DATASET_DIR / "processed" / "val_dataset.parquet", 
        DATASET_DIR / "processed" / "drifted_val_dataset.parquet", 
    )

    mlflow.log_param("test_drift_detected", test_drift_results["drift_detected"])
    mlflow.log_param("test_overall_drift_score", test_drift_results["overall_drift_score"])

    if test_drift_results["drift_detected"]:
        raise ValueError("Data drift detected in teset set! Model retraining required.")

        print("Data drift detected. Doing retraining...")
        if args.dry_run:
            studies = run_training(data, num_trials=1, num_epochs=1)
        else:
            studies = run_training(
                data, num_trials=args.num_trials, num_epochs=args.num_epochs
            )


if __name__ == "__main__":
    main()
