from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.data_preprocessing import get_raw_data, get_preprocessed_data
from src.feature_engineering import get_feature_engineered_data
from src.model_training import (
    lstm_mdn_objective,
    gru_mdn_objective,
    rnn_mdn_objective,
    tcn_mdn_objective,
    transformer_mdn_objective,
)
from airflow.sdk import Param
from src.evaluation import run_evaluation
from src.constants import MODELS_DIR

import optuna
import pandas as pd
import os

def prepare_data(**context):
    data = get_raw_data()
    data = get_preprocessed_data(data)
    data = get_feature_engineered_data(data)
    context['ti'].xcom_push(key='dataset_df', value=data)

def _optimize_model(objective_func, study_name, dataset_df, num_trials, num_epochs):
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=os.getenv("OPTUNA_DATABASE_URL"),
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective_func(trial, dataset_df, num_epochs), n_trials=num_trials)
    return study

def train_model(model_name, objective_func, num_trials, num_epochs, **context):
    dataset_df = context['ti'].xcom_pull(task_ids='prepare_data', key='dataset_df')
    _optimize_model(objective_func, f"{model_name}_mdn_hyperparam_search", dataset_df, int(num_trials), int(num_epochs))

def evaluate_all(**context):
    study_names = ["lstm", "gru", "rnn", "tcn", "transformer"]
    studies = {
        name: optuna.load_study(
            study_name=f"{name}_mdn_hyperparam_search",
            storage=os.getenv("OPTUNA_DATABASE_URL"),
        )
        for name in study_names
    }
    dataset_df = context['ti'].xcom_pull(task_ids='prepare_data', key='dataset_df')

    report_folder = "dry_runs/" if context['params']['dry_run'] else "/"
    report_folder += context['dag_run'].run_id

    run_evaluation(studies, dataset_df, generate_report=True, report_folder=report_folder)

with DAG(
    dag_id="model_training_pipeline",
    params={
        "num_trials": Param(30, type="integer", minimum=1),
        "num_epochs": Param(30, type="integer", minimum=1),
        "dry_run": Param(False, type="boolean"),
    },
    catchup=False,
    max_active_runs=1,
    render_template_as_native_obj=True
) as dag:

    prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data,
    )

    train_lstm = PythonOperator(
        task_id="train_lstm",
        python_callable=train_model,
        op_kwargs={
            "model_name": "lstm",
            "objective_func": lstm_mdn_objective,
            "num_trials": "{{ 1 if  v else params.num_trials }}",
            "num_epochs": "{{ 1 if params.dry_run else params.num_epochs }}"
        },
        retries=2,
    )

    train_gru = PythonOperator(
        task_id="train_gru",
        python_callable=train_model,
        op_kwargs={
            "model_name": "gru",
            "objective_func": gru_mdn_objective,
            "num_trials": "{{ 1 if params.dry_run else params.num_trials }}",
            "num_epochs": "{{ 1 if params.dry_run else params.num_epochs }}"
        },
        retries=2,
    )

    train_rnn = PythonOperator(
        task_id="train_rnn",
        python_callable=train_model,
        op_kwargs={
            "model_name": "rnn",
            "objective_func": rnn_mdn_objective,
            "num_trials": "{{ 1 if params.dry_run else params.num_trials }}",
            "num_epochs": "{{ 1 if params.dry_run else params.num_epochs }}"
        },
        retries=2,
    )

    train_tcn = PythonOperator(
        task_id="train_tcn",
        python_callable=train_model,
        op_kwargs={
            "model_name": "tcn",
            "objective_func": tcn_mdn_objective,
            "num_trials": "{{ 1 if params.dry_run else params.num_trials }}",
            "num_epochs": "{{ 1 if params.dry_run else params.num_epochs }}"
        },
        retries=2,
    )

    train_transformer = PythonOperator(
        task_id="train_transformer",
        python_callable=train_model,
        op_kwargs={
            "model_name": "transformer",
            "objective_func": transformer_mdn_objective,
            "num_trials": "{{ 1 if params.dry_run else params.num_trials }}",
            "num_epochs": "{{ 1 if params.dry_run else params.num_epochs }}"
        },
        retries=2,
    )

    evaluate = PythonOperator(
        task_id="evaluate_results",
        python_callable=evaluate_all,
    )

    prepare >> [train_lstm, train_gru, train_rnn, train_tcn, train_transformer] >> evaluate