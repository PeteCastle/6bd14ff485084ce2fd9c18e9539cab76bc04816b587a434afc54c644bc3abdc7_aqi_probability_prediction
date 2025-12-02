import pandas as pd
from typing import Dict, Any, Tuple
from src.constants import OUTPUT_DIR
import json
from pathlib import Path
import logging
from src.utils import setup_mlflow_tracking
from src.constants import DATASET_DIR, OUTPUT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

setup_mlflow_tracking()

def detect_drift(reference_data_path: str, current_data_path: str) -> Dict[str, Any]:
    from evidently import Report
    from evidently.presets import DataDriftPreset
    from evidently.metrics import ValueDrift

    reference_df = pd.read_parquet(DATASET_DIR /"reference.parquet")
    current_df = pd.read_parquet(DATASET_DIR / "current.parquet")

    assert set(reference_df.columns) == set(current_df.columns), "Reference and current datasets must have the same columns."

    target_cols = [col for col in reference_df.columns if "target" in col.lower()]
    feature_cols = [col for col in reference_df.columns if col not in target_cols]

    target_cols = [col for col in reference_df.columns if "target" in col.lower()]
    feature_cols = [col for col in reference_df.columns if col not in target_cols]

    threshold = 0.2

    report = Report(metrics=[
        DataDriftPreset(drift_share=threshold), # Data Drift
        *[ValueDrift(column=col) for col in feature_cols] # Concept Draft
    ])

    report_run = report.run(reference_data=reference_df, current_data=current_df)
    html_file = str(OUTPUT_DIR / f"data_drift_report.html")
    report_run.save_html(html_file)

    report_dict = report_run.dict()
    drift_detected = report_dict["metrics"][0]["value"]["share"] > threshold

    feature_drifts = {}
    for metric in report_dict["metrics"]:
        if "ValueDrift(column=" in metric["metric_id"]:
            column_name = metric["metric_id"].split("column=")[1].rstrip(")")
            drift_score = float(metric["value"])
            feature_drifts[column_name] = drift_score

    sorted_features = sorted(feature_drifts.items(), key=lambda x: x[1], reverse=True)
    all_features = dict(sorted_features)
    selected_features = dict(sorted_features[:3])

    overall_drift_score = sum(all_features.values()) / len(all_features) if all_features else 0

    output = {
        "drift_detected": drift_detected,
        "feature_drifts": selected_features,
        "overall_drift_score": overall_drift_score,
    }

    json_file = OUTPUT_DIR / "drift_report.json"

    with open(json_file, "w") as f:
        import json
        json.dump(output, f, indent=4)

    return html_file, json_file
        

