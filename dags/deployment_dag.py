from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.deployment.promote import promote_best_model


def post_reload(url: str = "http://fastapi:8000/reload_model", timeout: int = 15):
    # Lazy import to avoid DAG-parse deps
    import requests

    resp = requests.post(url, timeout=timeout)
    resp.raise_for_status()
    try:
        print("Reload response:", resp.json())
    except Exception:
        print("Reload response (text):", resp.text)


with DAG(
    dag_id="promote_model_dag",
    start_date=datetime(2025, 1, 1),
    schedule=None,  # only runs when triggered
    catchup=False,
    max_active_runs=1,
    render_template_as_native_obj=True,
) as dag:
    promote_task = PythonOperator(
        task_id="promote_best_model",
        python_callable=promote_best_model,
        op_kwargs={
            "experiment_name": "aqi_mdn_experiment",
            "model_name": "champion",
            "model_artifact_path": "model",
            "approve": True,
        },
    )

    reload_task = PythonOperator(
        task_id="reload_fastapi_model",
        python_callable=post_reload,
        op_kwargs={"url": "http://fastapi:8000/reload_model", "timeout": 15},
    )

    promote_task >> reload_task
