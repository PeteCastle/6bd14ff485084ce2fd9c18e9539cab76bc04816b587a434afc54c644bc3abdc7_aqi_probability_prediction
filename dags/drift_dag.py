import os
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.constants import DATASET_DIR
from src.features.transform import get_feature_engineered_data

def run_drift_detection():
    import mlflow
    from mlflow.tracking import MlflowClient
    from src.monitoring.generate_drift import detect_drift

    ref = DATASET_DIR / "reference.parquet"
    cur = DATASET_DIR / "current.parquet"
    if not ref.exists() or not cur.exists():
        get_feature_engineered_data()

    # detect_drift now returns (html_file, json_file)
    html_file, json_file = detect_drift(ref, cur)

    client = MlflowClient()

    def has_artifact(run_id: str, filename: str) -> bool:
        stack = [""]
        while stack:
            p = stack.pop()
            for item in client.list_artifacts(run_id, p):
                if item.is_dir:
                    stack.append(item.path)
                elif os.path.basename(item.path) == filename:
                    return True
        return False

    def log_if_missing(run_id: str, path: Path):
        fname = os.path.basename(path)
        if not has_artifact(run_id, fname):
            client.log_artifact(run_id, str(path))

    # iterate all runs of all experiments
    for exp in client.search_experiments():
        page_token = None
        while True:
            runs = client.search_runs(
                [exp.experiment_id],
                max_results=1000,
                page_token=page_token,
            )
            for run in runs:
                rid = run.info.run_id
                log_if_missing(rid, html_file)
                log_if_missing(rid, json_file)
            page_token = runs.token
            if not page_token:
                break


with DAG(
    dag_id="drift_dag",
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    render_template_as_native_obj=True,
) as dag:
    PythonOperator(
        task_id="drift_detection",
        python_callable=run_drift_detection,
    )