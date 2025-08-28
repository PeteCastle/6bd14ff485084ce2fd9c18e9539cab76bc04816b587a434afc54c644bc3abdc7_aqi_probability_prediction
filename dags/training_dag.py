from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from src.features.data_preprocessing import get_raw_data, get_preprocessed_data
from src.features.transform import get_feature_engineered_data
from src.models.train import (
    lstm_mdn_objective,
    gru_mdn_objective,
    rnn_mdn_objective,
    tcn_mdn_objective,
    transformer_mdn_objective,
    load_config,
)
from src.data.ingest import ingest_data
from airflow.sdk import Param
from src.models.validate import run_evaluation
import optuna
from optuna.samplers import TPESampler
import os


def prepare_data(**context):
    data = get_raw_data()
    data = get_preprocessed_data(data)
    context["ti"].xcom_push(key="preprocessed_df", value=data)


def load_training_config(**context):
    """Load and validate training configuration."""
    config_path = "config/config.yaml"
    try:
        config = load_config(config_path)

        # Log configuration details
        print("🔧 Training Configuration Loaded:")
        print(f"   Config file: {config_path}")

        optuna_config = config.get("optuna", {})
        sampler_config = optuna_config.get("sampler", {})
        print(
            f"   Optuna sampler: {sampler_config.get('type', 'Unknown')} (seed={sampler_config.get('seed', 'Unknown')})"
        )

        hyperparams = config.get("hyperparameters", {})
        common_params = hyperparams.get("common", {})
        model_params = hyperparams.get("models", {})

        print(f"   Common hyperparameters: {list(common_params.keys())}")
        print(f"   Model-specific configs: {list(model_params.keys())}")

        # Push config to XCom for other tasks
        context["ti"].xcom_push(key="training_config", value=config)

        return config

    except Exception as e:
        print(f"❌ Error loading training configuration: {e}")
        print("⚠️ Will proceed with default settings")
        return None


def feature_engineering(**context):
    data = context["ti"].xcom_pull(task_ids="preprocess_data", key="preprocessed_df")
    data = get_feature_engineered_data(data)
    context["ti"].xcom_push(key="dataset_df", value=data)


def _optimize_model(
    objective_func,
    study_name,
    dataset_df,
    num_trials,
    num_epochs,
    config_path="config/config.yaml",
):
    """Optimize model using config-based sampler settings."""
    # Load configuration for sampler settings
    try:
        config = load_config(config_path)
        optuna_config = config.get("optuna", {})
        sampler_config = optuna_config.get("sampler", {})

        if sampler_config.get("type") == "TPESampler":
            seed = sampler_config.get("seed", 10)
            sampler = TPESampler(seed=seed)
            print(f"🎯 Using TPESampler with seed={seed} from config")
        else:
            sampler = TPESampler(seed=10)  # Default fallback
            print("⚠️ Using default TPESampler with seed=10")
    except Exception as e:
        print(f"⚠️ Could not load config, using default sampler: {e}")
        sampler = TPESampler(seed=42)  # Original fallback

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=os.getenv("OPTUNA_DATABASE_URL"),
        load_if_exists=True,
        sampler=sampler,
    )
    study.optimize(
        lambda trial: objective_func(trial, dataset_df, num_epochs), n_trials=num_trials
    )
    return study


def train_model(model_name, objective_func, num_trials, num_epochs, **context):
    dataset_df = context["ti"].xcom_pull(
        task_ids="feature_engineering", key="dataset_df"
    )
    _optimize_model(
        objective_func,
        f"{model_name}_mdn_hyperparam_search",
        dataset_df,
        int(num_trials),
        int(num_epochs),
    )


def evaluate_model(**context):
    study_names = ["lstm", "gru", "rnn", "tcn", "transformer"]
    studies = {
        name: optuna.create_study(
            study_name=f"{name}_mdn_hyperparam_search",
            storage=os.getenv("OPTUNA_DATABASE_URL"),
            load_if_exists=True,
        )
        for name in study_names
    }
    dataset_df = context["ti"].xcom_pull(
        task_ids="feature_engineering", key="dataset_df"
    )

    report_folder = "dry_runs/" if context["params"]["dry_run"] else ""
    report_folder += context["dag_run"].run_id

    run_evaluation(
        studies, dataset_df, generate_report=True, report_folder=report_folder
    )


with DAG(
    dag_id="training_dag",
    params={
        "num_trials": Param(30, type="integer", minimum=1),
        "num_epochs": Param(30, type="integer", minimum=1),
        "dry_run": Param(False, type="boolean"),
    },
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    render_template_as_native_obj=True,
) as dag:
    ingest_data_ = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
    )

    load_config_task = PythonOperator(
        task_id="load_training_config",
        python_callable=load_training_config,
    )

    preprocess_data = PythonOperator(
        task_id="preprocess_data",
        python_callable=prepare_data,
    )

    feature_engineering_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering,
    )

    train_lstm = PythonOperator(
        task_id="train_lstm",
        python_callable=train_model,
        op_kwargs={
            "model_name": "lstm",
            "objective_func": lstm_mdn_objective,
            "num_trials": "{{ 1 if params.dry_run else params.num_trials }}",
            "num_epochs": "{{ 1 if params.dry_run else params.num_epochs }}",
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
            "num_epochs": "{{ 1 if params.dry_run else params.num_epochs }}",
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
            "num_epochs": "{{ 1 if params.dry_run else params.num_epochs }}",
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
            "num_epochs": "{{ 1 if params.dry_run else params.num_epochs }}",
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
            "num_epochs": "{{ 1 if params.dry_run else params.num_epochs }}",
        },
        retries=2,
    )

    evaluate_results = PythonOperator(
        task_id="evaluate_results",
        python_callable=evaluate_model,
    )

    trigger_promote = TriggerDagRunOperator(
        task_id="trigger_promote_dag",
        trigger_dag_id="promote_model_dag",
        wait_for_completion=False,  # set True if you want training_dag to block until promotion finishes
    )

    (
        ingest_data_
        >> load_config_task
        >> preprocess_data
        >> feature_engineering_task
        >> [train_lstm, train_gru, train_rnn, train_tcn, train_transformer]
        >> evaluate_results
        >> trigger_promote
    )
