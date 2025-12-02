import time
import os
import json
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from src.utils import setup_mlflow_tracking

setup_mlflow_tracking()

def promote_best_model(
    experiment_name: str = "aqi_mdn_experiment",
    model_name: str = "champion",
    model_artifact_path: str = "model",
    approve: bool = True,
    poll_interval: float = 2.0,
):
    print(f"Promoting best model from experiment '{experiment_name}' to registered model '{model_name}'")
    client = MlflowClient()
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found.")

    # Get best run by final_val_loss
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="metrics.final_val_loss < 0",
        order_by=["metrics.final_val_loss ASC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No finished runs with metric 'final_val_loss' found.")

    best = runs[0]
    run_id = best.info.run_id
    best_loss = best.data.metrics["final_val_loss"]
    model_uri = f"runs:/{run_id}/{model_artifact_path}"

    # Register model
    mv = mlflow.register_model(model_uri=model_uri, name=model_name, tags=best.data.params)
    version = mv.version

    # Wait until model is ready
    while True:
        mv = client.get_model_version(name=model_name, version=version)
        if mv.status == "READY":
            break
        time.sleep(poll_interval)

    # Promote to Production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )

    # # Optional: set alias
    # try:
    #     client.set_registered_model_alias(name=model_name, alias="champion", version=version)
    # except RestException:
    #     pass

    # # Tag the source run
    # client.set_tag(run_id, "promoted_to_production", "true")
    # client.set_tag(run_id, "registered_model_name", model_name)
    # client.set_tag(run_id, "registered_model_version", str(version))

    # return {
    #     "approved": True,
    #     "run_id": run_id,
    #     "final_val_loss": best_loss,
    #     "model_name": model_name,
    #     "version": version,
    #     "stage": "Production",
    #     "model_uri": model_uri,
    # }