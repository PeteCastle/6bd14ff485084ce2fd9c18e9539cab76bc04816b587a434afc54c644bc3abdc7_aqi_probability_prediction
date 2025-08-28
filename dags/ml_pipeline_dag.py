# from airflow import DAG
# from airflow.operators.python import PythonOperator, BranchPythonOperator
# from airflow.operators.empty import EmptyOperator
# from datetime import datetime
# from src.features.data_preprocessing import get_raw_data, get_preprocessed_data
# from src.features.transform import get_feature_engineered_data
# from src.models.train import (
#     lstm_mdn_objective,
#     gru_mdn_objective,
#     rnn_mdn_objective,
#     tcn_mdn_objective,
#     transformer_mdn_objective,
# )
# from airflow.sdk import Param
# from src.models.validate import run_evaluation
# from src.constants import MODELS_DIR, OUTPUT_DIR, DATASET_DIR
# from src.monitoring.generate_drift import detect_drift
# import mlflow
# import optuna
# import pandas as pd
# import os
# import json


# def prepare_data(**context):
#     data = get_raw_data()
#     data = get_preprocessed_data(data)
#     context["ti"].xcom_push(key="preprocessed_df", value=data)


# def feature_engineering(**context):
#     data = context["ti"].xcom_pull(task_ids="preprocess_data", key="preprocessed_df")
#     data = get_feature_engineered_data(data)
#     context["ti"].xcom_push(key="dataset_df", value=data)


# def _optimize_model(objective_func, study_name, dataset_df, num_trials, num_epochs):
#     study = optuna.create_study(
#         direction="minimize",
#         study_name=study_name,
#         storage=os.getenv("OPTUNA_DATABASE_URL"),
#         load_if_exists=True,
#     )
#     study.optimize(
#         lambda trial: objective_func(trial, dataset_df, num_epochs), n_trials=num_trials
#     )
#     return study


# def train_model(model_name, objective_func, num_trials, num_epochs, **context):
#     dataset_df = context["ti"].xcom_pull(
#         task_ids="feature_engineering", key="dataset_df"
#     )
#     _optimize_model(
#         objective_func,
#         f"{model_name}_mdn_hyperparam_search",
#         dataset_df,
#         int(num_trials),
#         int(num_epochs),
#     )


# def evaluate_model(**context):
#     study_names = ["lstm", "gru", "rnn", "tcn", "transformer"]
#     studies = {
#         name: optuna.create_study(
#             study_name=f"{name}_mdn_hyperparam_search",
#             storage=os.getenv("OPTUNA_DATABASE_URL"),
#             load_if_exists=True,
#         )
#         for name in study_names
#     }
#     dataset_df = context["ti"].xcom_pull(
#         task_ids="feature_engineering", key="dataset_df"
#     )

#     report_folder = "dry_runs/" if context["params"]["dry_run"] else ""
#     report_folder += context["dag_run"].run_id

#     run_evaluation(
#         studies, dataset_df, generate_report=True, report_folder=report_folder
#     )


# def run_drift_detection():
#     test_drift_results = detect_drift(
#         DATASET_DIR / "reference.parquet",
#         DATASET_DIR / "current.parquet",
#     )

#     mlflow.log_param("test_drift_detected", test_drift_results["drift_detected"])
#     mlflow.log_param(
#         "test_overall_drift_score", test_drift_results["overall_drift_score"]
#     )


# def branch_on_drift():
#     with open(OUTPUT_DIR / "drift_report.json") as f:
#         results = json.load(f)
#     if results.get("drift_detected"):
#         return [
#             "retrain_lstm",
#             "retrain_gru",
#             "retrain_rnn",
#             "retrain_tcn",
#             "retrain_transformer",
#         ]
#     return "pipeline_complete"


# with DAG(
#     dag_id="ml_pipeline_dag",
#     params={
#         "num_trials": Param(30, type="integer", minimum=1),
#         "num_epochs": Param(30, type="integer", minimum=1),
#         "dry_run": Param(False, type="boolean"),
#     },
#     catchup=False,
#     max_active_runs=1,
#     render_template_as_native_obj=True,
# ) as dag:
#     preprocess_data = PythonOperator(
#         task_id="preprocess_data",
#         python_callable=prepare_data,
#     )

#     feature_engineering_task = PythonOperator(
#         task_id="feature_engineering",
#         python_callable=feature_engineering,
#     )

#     train_lstm = PythonOperator(
#         task_id="train_lstm",
#         python_callable=train_model,
#         op_kwargs={
#             "model_name": "lstm",
#             "objective_func": lstm_mdn_objective,
#             "num_trials": "{{ 1 if params.dry_run else params.num_trials }}",
#             "num_epochs": "{{ 1 if params.dry_run else params.num_epochs }}",
#         },
#         retries=2,
#     )

#     train_gru = PythonOperator(
#         task_id="train_gru",
#         python_callable=train_model,
#         op_kwargs={
#             "model_name": "gru",
#             "objective_func": gru_mdn_objective,
#             "num_trials": "{{ 1 if params.dry_run else params.num_trials }}",
#             "num_epochs": "{{ 1 if params.dry_run else params.num_epochs }}",
#         },
#         retries=2,
#     )

#     train_rnn = PythonOperator(
#         task_id="train_rnn",
#         python_callable=train_model,
#         op_kwargs={
#             "model_name": "rnn",
#             "objective_func": rnn_mdn_objective,
#             "num_trials": "{{ 1 if params.dry_run else params.num_trials }}",
#             "num_epochs": "{{ 1 if params.dry_run else params.num_epochs }}",
#         },
#         retries=2,
#     )

#     train_tcn = PythonOperator(
#         task_id="train_tcn",
#         python_callable=train_model,
#         op_kwargs={
#             "model_name": "tcn",
#             "objective_func": tcn_mdn_objective,
#             "num_trials": "{{ 1 if params.dry_run else params.num_trials }}",
#             "num_epochs": "{{ 1 if params.dry_run else params.num_epochs }}",
#         },
#         retries=2,
#     )

#     train_transformer = PythonOperator(
#         task_id="train_transformer",
#         python_callable=train_model,
#         op_kwargs={
#             "model_name": "transformer",
#             "objective_func": transformer_mdn_objective,
#             "num_trials": "{{ 1 if params.dry_run else params.num_trials }}",
#             "num_epochs": "{{ 1 if params.dry_run else params.num_epochs }}",
#         },
#         retries=2,
#     )

#     evaluate_results = PythonOperator(
#         task_id="evaluate_results",
#         python_callable=evaluate_model,
#     )

#     drift_detection = PythonOperator(
#         task_id="drift_detection",
#         python_callable=run_drift_detection,
#     )

#     branch = BranchPythonOperator(
#         task_id="branch_on_drift",
#         python_callable=branch_on_drift,
#     )

#     retrain_lstm = PythonOperator(
#         task_id="retrain_lstm",
#         python_callable=train_model,
#         op_kwargs={
#             "model_name": "lstm",
#             "objective_func": lstm_mdn_objective,
#             "num_trials": "{{ 1 if params.dry_run else params.num_trials }}",
#             "num_epochs": "{{ 1 if params.dry_run else params.num_epochs }}",
#         },
#     )

#     retrain_gru = PythonOperator(
#         task_id="retrain_gru",
#         python_callable=train_model,
#         op_kwargs={
#             "model_name": "gru",
#             "objective_func": gru_mdn_objective,
#             "num_trials": "{{ 1 if params.dry_run else params.num_trials }}",
#             "num_epochs": "{{ 1 if params.dry_run else params.num_epochs }}",
#         },
#     )

#     retrain_rnn = PythonOperator(
#         task_id="retrain_rnn",
#         python_callable=train_model,
#         op_kwargs={
#             "model_name": "rnn",
#             "objective_func": rnn_mdn_objective,
#             "num_trials": "{{ 1 if params.dry_run else params.num_trials }}",
#             "num_epochs": "{{ 1 if params.dry_run else params.num_epochs }}",
#         },
#     )

#     retrain_tcn = PythonOperator(
#         task_id="retrain_tcn",
#         python_callable=train_model,
#         op_kwargs={
#             "model_name": "tcn",
#             "objective_func": tcn_mdn_objective,
#             "num_trials": "{{ 1 if params.dry_run else params.num_trials }}",
#             "num_epochs": "{{ 1 if params.dry_run else params.num_epochs }}",
#         },
#     )

#     retrain_transformer = PythonOperator(
#         task_id="retrain_transformer",
#         python_callable=train_model,
#         op_kwargs={
#             "model_name": "transformer",
#             "objective_func": transformer_mdn_objective,
#             "num_trials": "{{ 1 if params.dry_run else params.num_trials }}",
#             "num_epochs": "{{ 1 if params.dry_run else params.num_epochs }}",
#         },
#     )

#     pipeline_complete = EmptyOperator(task_id="pipeline_complete")

#     # Main pipeline flow
#     (
#         preprocess_data
#         >> feature_engineering_task
#         >> [train_lstm, train_gru, train_rnn, train_tcn, train_transformer]
#         >> evaluate_results
#         >> drift_detection
#         >> branch
#     )

#     # Branch outcomes
#     branch >> [retrain_lstm, retrain_gru, retrain_rnn, retrain_tcn, retrain_transformer]
#     branch >> pipeline_complete
