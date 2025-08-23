import pandas as pd
from typing import Dict, Any
from evidently import Report
from evidently.presets import DataDriftPreset
from src.constants import OUTPUT_DIR
import json

def detect_drift(reference_data_path: str, current_data_path: str) -> Dict[str, Any]:
    reference_df = pd.read_parquet(reference_data_path)
    current_df = pd.read_parquet(current_data_path)

    target_cols_ref = [col for col in reference_df.columns if "target" in col.lower()]
    target_cols_curr = [col for col in current_df.columns if "target" in col.lower()]

    if target_cols_ref:
        reference_df = reference_df.drop(columns=target_cols_ref)
    if target_cols_curr:
        current_df = current_df.drop(columns=target_cols_curr)


    threshold = 0.2
    report = Report(metrics=[DataDriftPreset(drift_share=threshold)])

    report_run = report.run(reference_data=reference_df, current_data=current_df)

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

    with open(OUTPUT_DIR / "drift_report.json", "w") as f:
        json.dump(output, f, indent=2)

    return output