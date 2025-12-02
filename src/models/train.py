import os
import time
import warnings
import typing
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm

import mlflow
import mlflow.pyfunc
import mlflow.tracking
import shap

from src.constants import MODELS_DIR
from src.models.loss import mdn_loss
from src.data.torch_datasets import generate_datasets
from src.utils import setup_mlflow_tracking
from src.models import GRU_MDN, LSTM_MDN, RNN_MDN, TCN_MDN, Transformer_MDN

setup_mlflow_tracking()


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def suggest_hyperparameter(trial, param_name: str, param_config: dict):
    """Suggest hyperparameter based on configuration."""
    param_type = param_config["type"]

    if param_type == "categorical":
        return trial.suggest_categorical(param_name, param_config["choices"])
    elif param_type == "int":
        return trial.suggest_int(param_name, param_config["low"], param_config["high"])
    elif param_type == "float":
        log = param_config.get("log", False)
        return trial.suggest_float(
            param_name, param_config["low"], param_config["high"], log=log
        )
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


class Trainer(mlflow.pyfunc.PythonModel):
    MLFLOW_EXPERIMENT = "aqi_mdn_experiment"

    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
        criterion,
        optimizer,
        train_dataset,
        val_dataset=None,
        batch_size: int = 256,
        params: dict = None,
    ):
        self.batch_size = batch_size

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if self.device == torch.device("cpu"):
            warnings.warn(
                "Running on CPU. Training may be slow. Consider using MPS or CUDA if available.",
            )

        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = (
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            if val_dataset
            else None
        )
        self.history = {"train_loss": [], "val_loss": [], "training_time": []}
        self.params = params or {}
        self.start_epoch = 0
        self.name = name
        self.best_val_loss = float("inf")
        self.is_best_model = False

    @property
    def was_saved(self) -> bool:
        """Returns True if the model was saved (i.e., it was the best for its class)."""
        return self.is_best_model

    def load_context(self, context):
        """
        Load model context when the model is loaded for serving.
        This method is called by MLflow when loading the model.
        """
        # If we have artifacts, load the model state
        if hasattr(context, "artifacts") and "model_state" in context.artifacts:
            model_state_path = context.artifacts["model_state"]
            self._load_model_from_artifacts(model_state_path)

    def _load_model_from_artifacts(self, model_state_path):
        """Load model state from artifacts when serving."""
        try:
            checkpoint = torch.load(model_state_path, map_location=self.device)

            # Get model class and hyperparams from checkpoint
            model_class_name = checkpoint.get("model_class_name")
            hyperparams = checkpoint.get("hyperparams", {})

            # Import the model class dynamically
            model_classes = {
                "LSTM_MDN": LSTM_MDN,
                "GRU_MDN": GRU_MDN,
                "RNN_MDN": RNN_MDN,
                "TCN_MDN": TCN_MDN,
                "Transformer_MDN": Transformer_MDN,
            }

            if model_class_name in model_classes:
                model_class = model_classes[model_class_name]
                # Initialize the model with saved hyperparams
                model_hyperparams = {
                    k: v
                    for k, v in hyperparams.items()
                    if k not in ["learning_rate", "lookback_days", "step"]
                }
                self.model = model_class(
                    input_dim=21, output_dim=7, **model_hyperparams
                ).to(self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()
            else:
                raise ValueError(f"Unknown model class: {model_class_name}")

        except Exception as e:
            print(f"Warning: Could not load model from artifacts: {e}")
            # Model will remain None, which will cause predict() to fail appropriately

    def predict(self, model_input):
        """
        Generate predictions directly using the trained model (without MLflow context).

        Args:
            model_input (pd.DataFrame or np.ndarray or torch.Tensor): Input data for prediction.

        Returns:
            np.ndarray: Predictions from the model.
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained.")

        self.model.eval()

        if isinstance(model_input, pd.DataFrame):
            input_tensor = torch.tensor(model_input.values, dtype=torch.float32)
        elif isinstance(model_input, np.ndarray):
            input_tensor = torch.tensor(model_input, dtype=torch.float32)
        else:
            input_tensor = model_input

        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            mu, sigma, alpha = self.model(input_tensor)

            batch_size = input_tensor.shape[0]
            num_mixtures = self.model.num_mixtures
            output_dim = mu.shape[1] // num_mixtures

            mu = mu.view(batch_size, num_mixtures, output_dim)
            alpha = alpha.view(batch_size, num_mixtures)

            predictions = torch.sum(mu * alpha.unsqueeze(-1), dim=1)

            return predictions.cpu().numpy()

    def _is_best_model_for_class(self, current_val_loss: float) -> bool:
        """
        Check if the current model is the best for its class by querying MLflow.

        Args:
            current_val_loss (float): The current validation loss to compare

        Returns:
            bool: True if this is the best model for its class, False otherwise
        """
        try:
            # Get the experiment
            experiment = mlflow.get_experiment_by_name(self.MLFLOW_EXPERIMENT)
            if experiment is None:
                # If experiment doesn't exist, this is the first model of its class
                return True

            # Search for runs with the same model class
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"params.model_name = '{self.model.__class__.__name__}'",
                order_by=["metrics.final_val_loss ASC"],
                max_results=1,
            )

            if len(runs) == 0:
                # No previous runs for this model class
                return True

            best_previous_loss = runs.iloc[0]["metrics.final_val_loss"]
            return current_val_loss < best_previous_loss

        except Exception as e:
            # If there's any error querying MLflow, default to saving
            print(f"Warning: Could not query MLflow for best model comparison: {e}")
            return True

    def _generate_run_name(self) -> str:
        """
        Generate a meaningful run name for MLflow based on model type and key hyperparameters.

        Returns:
            str: A formatted run name containing model info and key hyperparameters
        """
        model_name = self.model.__class__.__name__

        # Extract key hyperparameters for the run name
        key_params = []

        # Add hidden dimension if available
        if "hidden_dim" in self.params:
            key_params.append(f"h{self.params['hidden_dim']}")

        # Add number of layers if available
        if "num_layers" in self.params:
            key_params.append(f"l{self.params['num_layers']}")

        # Add number of mixtures if available
        if "num_mixtures" in self.params:
            key_params.append(f"m{self.params['num_mixtures']}")

        # Add lookback days if available
        if "lookback_days" in self.params:
            key_params.append(f"lb{self.params['lookback_days']}")

        # Add step if available
        if "step" in self.params:
            key_params.append(f"s{self.params['step']}")

        # Add learning rate if available
        if "learning_rate" in self.params:
            lr_str = f"{self.params['learning_rate']:.0e}".replace("e-0", "e-").replace(
                "e+0", "e+"
            )
            key_params.append(f"lr{lr_str}")

        # Add dropout if available and not zero
        if "dropout" in self.params and self.params["dropout"] > 0:
            key_params.append(f"dr{self.params['dropout']:.2f}")

        # Add number of heads for transformer models
        if "num_heads" in self.params:
            key_params.append(f"heads{self.params['num_heads']}")

        # Combine model name with parameters
        if key_params:
            param_str = "_".join(key_params)
            run_name = f"{model_name}_{param_str}"
        else:
            run_name = f"{model_name}_{self.name}"

        return run_name

    def train(self, num_epochs=30, save=True):
        # Create a meaningful run name
        run_name = self._generate_run_name()
        experiment = mlflow.set_experiment(self.MLFLOW_EXPERIMENT)

        with mlflow.start_run(
            run_name=run_name, experiment_id=experiment.experiment_id
        ):
            mlflow.log_param("model_name", self.model.__class__.__name__)
            mlflow.log_param("name", self.name)
            mlflow.log_params(self.params)

            try:
                for epoch in (pbar := tqdm(range(self.start_epoch, num_epochs))):
                    start_time = time.time()
                    self.model.train()
                    train_loss = 0.0

                    for inputs, targets in tqdm(
                        self.train_loader,
                        desc=f"Epoch {epoch + 1}",
                        unit="batch",
                        disable=True,
                    ):
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)

                        self.optimizer.zero_grad()
                        mu, sigma, alpha = self.model(inputs)

                        loss = self.criterion(
                            targets, mu, sigma, alpha, self.model.num_mixtures
                        )
                        loss.backward()
                        self.optimizer.step()
                        train_loss += loss.detach().item()

                    avg_train_loss = train_loss / len(self.train_loader)
                    self.history["train_loss"].append(avg_train_loss)

                    if self.val_loader:
                        self.model.eval()
                        val_loss = 0.0
                        with torch.no_grad():
                            for inputs, targets in tqdm(
                                self.val_loader,
                                desc="Validation",
                                unit="batch",
                                disable=True,
                            ):
                                inputs = inputs.to(self.device)
                                targets = targets.to(self.device)

                                mu, sigma, alpha = self.model(inputs)
                                loss = self.criterion(
                                    targets, mu, sigma, alpha, self.model.num_mixtures
                                )

                                val_loss += loss.item()

                        avg_val_loss = val_loss / len(self.val_loader)
                        self.history["val_loss"].append(avg_val_loss)

                        # Track the best validation loss for this training session
                        if avg_val_loss < self.best_val_loss:
                            self.best_val_loss = avg_val_loss

                        pbar.set_description(
                            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}"
                        )
                    else:
                        pbar.set_description(
                            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}"
                        )

                    self.history["training_time"].append(time.time() - start_time)

            except KeyboardInterrupt as e:
                print("\n🛑 Training interrupted by user. Saving checkpoint...")
                self.save(epoch + 1)
                raise KeyboardInterrupt from e

            # Check if this is the best model for its class
            if self.history["val_loss"]:
                final_val_loss = self.history["val_loss"][-1]
                self.is_best_model = self._is_best_model_for_class(final_val_loss)
            else:
                # If no validation dataset, we can't compare models properly
                print(
                    "⚠️ No validation dataset provided. Cannot determine if this is the best model."
                )
                self.is_best_model = False

            if save:
                if self.history["val_loss"] and self.is_best_model:
                    print(
                        "✅ Training completed. This is the best model for its class. Saving model..."
                    )
                    self.save(num_epochs)
                elif not self.history["val_loss"]:
                    print(
                        "⚠️ Training completed. No validation loss available for comparison. Skipping save."
                    )
                    self._log_metrics_only(num_epochs)
                else:
                    print(
                        f"⏭️ Training completed. Model did not beat the best {self.model.__class__.__name__} model. Skipping save."
                    )
                    # Still log metrics even if we don't save the model
                    self._log_metrics_only(num_epochs)

    def predict_from_val(self, indices: list[int]):
        """
        Generates predictions from specific samples in the validation dataset using a trained
        Mixture Density Network (MDN) model. Computes the expected value of the predicted
        mixture distribution and retrieves metadata for interpretability.

        Args:
            indices (list[int]): List of sample indices from the validation dataset to evaluate.

        Returns:
            dict: A dictionary containing:
                - "predictions" (np.ndarray): Expected values computed from the MDN outputs.
                - "ground_truths" (np.ndarray): True target values for the selected indices.
                - "mus" (np.ndarray): Mixture component means predicted by the MDN.
                - "sigmas" (np.ndarray): Mixture component standard deviations predicted by the MDN.
                - "alphas" (np.ndarray): Mixture weights (probabilities) for each Gaussian component.
                - "timestamps" (np.ndarray): Timestamps corresponding to each data point, from metadata.
                - "unnormalized_targets" (np.ndarray): Original target values before normalization.
                - "means" (np.ndarray): Mean values used for normalization of each data point.
                - "stds" (np.ndarray): Standard deviations used for normalization of each data point.

        Raises:
            ValueError: If the validation dataset is not provided during initialization.

        Notes:
            - This method sets the model to evaluation mode (`model.eval()`).
            - The MDN output is decomposed into μ, σ, and α, and the expected value is calculated.
            - Metadata retrieval assumes the dataset implements a `get_metadata(index)` method.
        """
        if self.val_loader is None:
            raise ValueError("Validation dataset not provided.")

        self.model.eval()
        filtered_data = [self.val_dataset[i] for i in indices]
        data_loader = DataLoader(filtered_data, batch_size=1, shuffle=False)

        predictions = []
        ground_truths = []
        mus, sigmas, alphas = [], [], []
        timestamps = []
        original_targets = []
        means = []
        stds = []

        with torch.no_grad():
            for i, (inputs, targets) in zip(indices, data_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                mu, sigma, alpha = self.model(inputs)

                batch_size = inputs.shape[0]
                num_mixtures = self.model.num_mixtures
                output_dim = mu.shape[1] // num_mixtures

                mu = mu.view(batch_size, num_mixtures, output_dim)
                sigma = sigma.view(batch_size, num_mixtures, output_dim)
                alpha = alpha.view(batch_size, num_mixtures)

                pred = torch.sum(mu * alpha.unsqueeze(-1), dim=1)

                predictions.append(pred.squeeze(0).cpu().numpy())
                ground_truths.append(targets.squeeze(0).cpu().numpy())
                mus.append(mu.squeeze(0).cpu().numpy())
                sigmas.append(sigma.squeeze(0).cpu().numpy())
                alphas.append(alpha.squeeze(0).cpu().numpy())

                # Retrieve metadata using index
                meta = self.val_dataset.get_metadata(i)
                timestamps.append(meta["timestamp"])
                original_targets.append(meta["original_target"])
                means.append(meta["mean"])
                stds.append(meta["std"])

        return {
            "predictions": np.array(predictions),
            "ground_truths": np.array(ground_truths),
            "mus": np.array(mus),
            "sigmas": np.array(sigmas),
            "alphas": np.array(alphas),
            "timestamps": np.array(timestamps),
            "unnormalized_targets": np.array(original_targets),
            "means": np.array(means),
            "stds": np.array(stds),
        }

    def save(self, epoch=None):
        mlflow.log_metric("final_epoch", epoch or len(self.history["train_loss"]))
        mlflow.log_metric("final_train_loss", self.history["train_loss"][-1])
        if self.history["val_loss"]:
            mlflow.log_metric("final_val_loss", self.history["val_loss"][-1])
        for metric_name, values in self.history.items():
            for step, value in enumerate(values):
                mlflow.log_metric(metric_name, value, step=step)

        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)

        # Save model state as artifact for reloading
        model_state_path = f"models/{self.name}_state.pth"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "hyperparams": self.params,
                "model_class_name": self.model.__class__.__name__,
            },
            model_state_path,
        )

        # Add SHAP explainer if this is the best model
        # if self.is_best_model and self.val_dataset is not None:
        #     self._generate_shap_explainer()

        # Log the Trainer itself as a PyFunc model
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=self,
            artifacts={"model_state": model_state_path},
            registered_model_name=self.model.__class__.__name__,
        )

    def _generate_shap_explainer(self):
        """Generate SHAP explainer and feature importance plots for the best model."""
        try:
            print("🔍 Generating SHAP explainer for best model...")

            # Get current MLflow run ID for unique file naming
            run_id = mlflow.active_run().info.run_id
            model_id = f"{self.model.__class__.__name__}_{run_id[:8]}"

            # Get a sample of validation data for SHAP analysis
            sample_size = min(100, len(self.val_dataset))  # Use up to 100 samples
            indices = np.random.choice(
                len(self.val_dataset), sample_size, replace=False
            )

            # Extract input features from validation dataset
            background_data = []
            test_data = []

            for i in range(min(50, len(indices))):  # Background data
                inputs, _ = self.val_dataset[indices[i]]
                background_data.append(inputs.numpy())

            for i in range(50, min(100, len(indices))):  # Test data
                inputs, _ = self.val_dataset[indices[i]]
                test_data.append(inputs.numpy())

            if not test_data:  # If we don't have enough data, use background for test
                test_data = background_data[:10]

            background_data = np.array(background_data)
            test_data = np.array(test_data)

            # Create a wrapper function for SHAP that returns expected values
            def model_predict(X):
                """Wrapper function for SHAP that returns model predictions."""
                self.model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                    mu, sigma, alpha = self.model(X_tensor)

                    # Calculate expected value (prediction) for each output dimension
                    batch_size = X_tensor.shape[0]
                    num_mixtures = self.model.num_mixtures
                    output_dim = mu.shape[1] // num_mixtures

                    mu = mu.view(batch_size, num_mixtures, output_dim)
                    alpha = alpha.view(batch_size, num_mixtures)

                    # Compute expected value: sum(alpha_i * mu_i)
                    predictions = torch.sum(mu * alpha.unsqueeze(-1), dim=1)

                    return predictions.cpu().numpy()

            # Create SHAP explainer
            explainer = shap.DeepExplainer(model_predict, background_data)
            shap_values = explainer.shap_values(test_data)

            # If shap_values is a list (multi-output), we need to handle each output
            if isinstance(shap_values, list):
                # For multi-output, create plots for each output dimension
                output_names = ["PM2.5", "PM10", "O3", "NO2", "SO2", "CO", "AQI"]

                for output_idx, output_name in enumerate(output_names):
                    if output_idx < len(shap_values):
                        # Create feature importance plot for this output
                        plt.figure(figsize=(12, 8))
                        shap.summary_plot(
                            shap_values[output_idx],
                            test_data,
                            show=False,
                            feature_names=[
                                f"feature_{i}" for i in range(test_data.shape[1])
                            ],
                        )
                        plt.title(f"SHAP Feature Importance - {output_name}")
                        plt.tight_layout()

                        # Save plot with unique model ID
                        plot_path = f"models/shap_plot_{model_id}_{output_name.lower().replace('.', '_')}.png"
                        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                        plt.close()

                        # Log as MLflow artifact
                        mlflow.log_artifact(plot_path, "shap_plots")

                        # Calculate and log feature importances
                        feature_importance = np.abs(shap_values[output_idx]).mean(
                            axis=0
                        )
                        importance_dict = {
                            f"feature_{i}_importance_{output_name}": float(imp)
                            for i, imp in enumerate(feature_importance)
                        }
                        mlflow.log_metrics(importance_dict)

                # Create overall summary plot
                plt.figure(figsize=(15, 10))
                shap.summary_plot(
                    shap_values[0],  # Use first output for overall summary
                    test_data,
                    show=False,
                    feature_names=[f"feature_{i}" for i in range(test_data.shape[1])],
                )
                plt.title("SHAP Feature Importance - Overall Summary")
                plt.tight_layout()

            else:
                # Single output case
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    shap_values,
                    test_data,
                    show=False,
                    feature_names=[f"feature_{i}" for i in range(test_data.shape[1])],
                )
                plt.title("SHAP Feature Importance")
                plt.tight_layout()

                # Calculate and log feature importances
                feature_importance = np.abs(shap_values).mean(axis=0)
                importance_dict = {
                    f"feature_{i}_importance": float(imp)
                    for i, imp in enumerate(feature_importance)
                }
                mlflow.log_metrics(importance_dict)

            # Save overall plot as shap_plot.png with unique model ID
            plot_path = f"models/shap_plot_{model_id}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            # Log the main plot as MLflow artifact
            mlflow.log_artifact(plot_path, "shap_explainer")

            # Save SHAP values as numpy arrays with unique model ID
            shap_values_path = f"models/{model_id}_shap_values.npy"
            test_data_path = f"models/{model_id}_test_data.npy"

            if isinstance(shap_values, list):
                # Save as a compressed archive for multi-output
                np.savez_compressed(
                    shap_values_path.replace(".npy", ".npz"), *shap_values
                )
            else:
                np.save(shap_values_path, shap_values)

            np.save(test_data_path, test_data)

            # Log SHAP data as artifacts
            mlflow.log_artifact(
                (
                    shap_values_path.replace(".npy", ".npz")
                    if isinstance(shap_values, list)
                    else shap_values_path
                ),
                "shap_explainer",
            )
            mlflow.log_artifact(test_data_path, "shap_explainer")

            print("✅ SHAP explainer generated and logged successfully!")

        except Exception as e:
            print(f"⚠️ Failed to generate SHAP explainer: {str(e)}")
            import traceback

            traceback.print_exc()

    def _log_metrics_only(self, epoch=None):
        """Log metrics to MLflow without saving the model."""
        mlflow.log_metric("final_epoch", epoch or len(self.history["train_loss"]))
        mlflow.log_metric("final_train_loss", self.history["train_loss"][-1])
        if self.history["val_loss"]:
            mlflow.log_metric("final_val_loss", self.history["val_loss"][-1])
        for metric_name, values in self.history.items():
            for step, value in enumerate(values):
                mlflow.log_metric(metric_name, value, step=step)

    def load(self, path):
        raise NotImplementedError("Refactor")

    @staticmethod
    def from_template(
        dataset_df: pd.DataFrame,
        model_class: torch.nn.Module,
        hyperparams: dict,
        _id: int,
    ):
        _hyperparams = hyperparams.copy()
        hyperparams = hyperparams.copy()

        training_dataset, validation_dataset = generate_datasets(
            dataset_df,
            lookback=hyperparams.pop("lookback_days") * 24,
            delay=24,
            step=hyperparams.pop("step"),
        )

        learning_rate = hyperparams.pop("learning_rate")
        model = model_class(input_dim=21, output_dim=7, **hyperparams)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        name = f"{model_class.__name__}_{_id}"
        return Trainer(
            name=name,
            model=model,
            criterion=mdn_loss,
            optimizer=optimizer,
            train_dataset=training_dataset,
            val_dataset=validation_dataset,
            batch_size=256,
            params=_hyperparams,
        )

    @classmethod
    def from_best_mlflow_run(
        cls,
        df: pd.DataFrame,
        model_class: torch.nn.Module,
        experiment_name: str = None,
    ):
        """
        Create a Trainer instance from the best MLflow run (lowest validation loss) for a given model class.

        Args:
            df (pd.DataFrame): The dataset to use for training/validation
            model_class (torch.nn.Module): The model class to find the best run for (e.g., LSTM_MDN, GRU_MDN)
            experiment_name (str, optional): The MLflow experiment name. Defaults to the class experiment.

        Returns:
            Trainer: A new Trainer instance configured with the best run's parameters and loaded history

        Raises:
            ValueError: If no runs found for the specified model class
        """
        experiment_name = experiment_name or cls.MLFLOW_EXPERIMENT

        # Get the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found in MLflow.")

        # Search for runs with the specified model class, ordered by validation loss
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"params.model_name = '{model_class.__name__}'",
            order_by=["metrics.final_val_loss ASC"],
            max_results=1,
        )

        if len(runs) == 0:
            raise ValueError(
                f"No MLflow runs found for model class '{model_class.__name__}' with validation loss in experiment '{experiment_name}'."
            )

        best_run = runs.iloc[0]
        run_id = best_run["run_id"]
        run_name = best_run.get("tags.mlflow.runName", "Unknown")
        # print("BEST RUN")
        # print(best_run)
        val_loss = best_run["metrics.final_val_loss"]

        print(
            f"Found best {model_class.__name__} run: {run_name} (val_loss: {val_loss:.6f})"
        )

        # Extract hyperparameters from the run
        hyperparams = {}
        param_columns = [col for col in best_run.index if col.startswith("params.")]
        for col in param_columns:
            param_name = col.replace("params.", "")
            param_value = best_run[col]

            # Skip non-hyperparameter params
            if param_name in [
                "model_name",
                "name",
                "trial_number",
                "trial_value",
                "trial_state",
                "study_name",
            ]:
                continue

            # Convert string values back to appropriate types
            if param_value is not None:
                try:
                    # Try to evaluate as Python literal (handles int, float, list, etc.)
                    hyperparams[param_name] = (
                        eval(param_value)
                        if isinstance(param_value, str)
                        else param_value
                    )
                except Exception:
                    # If evaluation fails, keep as string
                    hyperparams[param_name] = param_value

        # Create a synthetic ID from the run_id for naming
        synthetic_id = run_id[:8]  # Use first 8 chars of run_id

        # Use the existing from_template method to create the trainer
        trainer = cls.from_template(
            dataset_df=df,
            model_class=model_class,
            hyperparams=hyperparams,
            _id=synthetic_id,
        )

        # Add MLflow run information for better tracking
        trainer.params["source_run_id"] = run_id
        trainer.params["source_run_name"] = run_name
        trainer.params["source_val_loss"] = val_loss
        trainer.params["loaded_from_mlflow"] = True

        # Load training history from MLflow using the specific run
        trainer._load_history_from_mlflow_run(run_id)

        return trainer

    def _load_history_from_mlflow_run(self, run_id: str):
        """
        Load training history from a specific MLflow run ID.

        Args:
            run_id (str): The MLflow run ID to load history from
        """
        try:
            # Get all metrics for this run
            client = mlflow.tracking.MlflowClient()

            # Try to get metric history (step-by-step training metrics)
            try:
                train_metrics = client.get_metric_history(run_id, "train_loss")
                val_metrics = client.get_metric_history(run_id, "val_loss")
                time_metrics = client.get_metric_history(run_id, "training_time")

                # Reconstruct history from metric history
                self.history = {
                    "train_loss": [m.value for m in train_metrics],
                    "val_loss": [m.value for m in val_metrics],
                    "training_time": [m.value for m in time_metrics],
                }
            except Exception:
                # Fallback: use final metrics if step-by-step history is not available
                self.history = {"train_loss": [], "val_loss": [], "training_time": []}

            # Get the run info for fallback metrics
            run_info = client.get_run(run_id)
            run_data = run_info.data

            # Fill in missing data with final metrics if available
            if (
                not self.history["train_loss"]
                and "final_train_loss" in run_data.metrics
            ):
                final_train_loss = run_data.metrics["final_train_loss"]
                if final_train_loss is not None:
                    self.history["train_loss"] = [final_train_loss]

            if not self.history["val_loss"] and "final_val_loss" in run_data.metrics:
                final_val_loss = run_data.metrics["final_val_loss"]
                if final_val_loss is not None:
                    self.history["val_loss"] = [final_val_loss]

            print(f"✅ Loaded training history from MLflow run: {run_id}")
            print(
                f"   Train loss: {self.history['train_loss'][-1] if self.history['train_loss'] else 'N/A':.6f}"
            )
            print(
                f"   Val loss: {self.history['val_loss'][-1] if self.history['val_loss'] else 'N/A':.6f}"
            )

        except Exception as e:
            print(
                f"Warning: Could not load training history from MLflow run {run_id}: {e}"
            )
            # Keep default empty history
            self.history = {"train_loss": [], "val_loss": [], "training_time": []}


def create_objective_function(model_class: torch.nn.Module, config: dict):
    """Create an objective function for Optuna optimization based on config."""

    def objective(trial, dataset_df: pd.DataFrame, num_epochs: int = 30):
        try:
            # Get hyperparameter configurations
            common_params = config.get("hyperparameters", {}).get("common", {})
            model_params = (
                config.get("hyperparameters", {})
                .get("models", {})
                .get(model_class.__name__, {})
            )

            # Suggest common hyperparameters
            for param_name, param_config in common_params.items():
                suggest_hyperparameter(trial, param_name, param_config)

            # Suggest model-specific hyperparameters
            for param_name, param_config in model_params.items():
                suggest_hyperparameter(trial, param_name, param_config)

            trainer = Trainer.from_template(
                dataset_df, model_class, trial.params, trial.number
            )
            trainer.train(num_epochs=num_epochs)

            return trainer.history["val_loss"][-1]

        except KeyboardInterrupt as e:
            raise KeyboardInterrupt from e
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error during trial {trial.number}: {e}")
            return float("inf")

    return objective


# Legacy objective functions for backward compatibility
def lstm_mdn_objective(trial, dataset_df: pd.DataFrame, num_epochs: int = 30):
    config = load_config()
    objective_func = create_objective_function(LSTM_MDN, config)
    return objective_func(trial, dataset_df, num_epochs)


def gru_mdn_objective(trial, dataset_df: pd.DataFrame, num_epochs: int = 30):
    config = load_config()
    objective_func = create_objective_function(GRU_MDN, config)
    return objective_func(trial, dataset_df, num_epochs)


def rnn_mdn_objective(trial, dataset_df: pd.DataFrame, num_epochs: int = 30):
    config = load_config()
    objective_func = create_objective_function(RNN_MDN, config)
    return objective_func(trial, dataset_df, num_epochs)


def tcn_mdn_objective(trial, dataset_df: pd.DataFrame, num_epochs: int = 30):
    config = load_config()
    objective_func = create_objective_function(TCN_MDN, config)
    return objective_func(trial, dataset_df, num_epochs)


def transformer_mdn_objective(trial, dataset_df: pd.DataFrame, num_epochs: int = 30):
    config = load_config()
    objective_func = create_objective_function(Transformer_MDN, config)
    return objective_func(trial, dataset_df, num_epochs)


def run_training(
    dataset_df: pd.DataFrame,
    config_path: str = "config/config.yaml",
    num_trials: int = 30,
    num_epochs: int = 30,
) -> typing.Dict[str, optuna.Study]:
    """Run hyperparameter optimization for all models using configuration."""

    # Load configuration for hyperparameters only
    config = load_config(config_path)

    # Create sampler from config
    optuna_config = config.get("optuna", {})
    sampler_config = optuna_config.get("sampler", {})

    if sampler_config.get("type") == "TPESampler":
        seed = sampler_config.get("seed", 10)
        sampler = TPESampler(seed=seed)
    else:
        sampler = TPESampler(seed=10)  # Default fallback

    # Fixed study configurations (not from config)
    storage_path = f"sqlite:///{MODELS_DIR}/optuna_study.db"
    direction = "minimize"

    # Model configurations
    models = [
        (LSTM_MDN, lstm_mdn_objective, "lstm_mdn_hyperparam_search"),
        (GRU_MDN, gru_mdn_objective, "gru_mdn_hyperparam_search"),
        (RNN_MDN, rnn_mdn_objective, "rnn_mdn_hyperparam_search"),
        (TCN_MDN, tcn_mdn_objective, "tcn_mdn_hyperparam_search"),
        (
            Transformer_MDN,
            transformer_mdn_objective,
            "transformer_mdn_hyperparam_search",
        ),
    ]

    results = {}

    for model_class, objective_func, study_name in models:
        print(f"Starting study for {model_class.__name__}...")

        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage_path,
            load_if_exists=True,
            sampler=sampler,
        )

        study.optimize(
            lambda trial: objective_func(trial, dataset_df, num_epochs),
            n_trials=num_trials,
        )

        # Store result with lowercase key for backward compatibility
        model_key = model_class.__name__.lower().replace("_mdn", "")
        results[model_key] = study

    return results
