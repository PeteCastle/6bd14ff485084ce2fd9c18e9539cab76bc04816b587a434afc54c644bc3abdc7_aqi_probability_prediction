import os
import time
import warnings
import json
import hashlib
import typing

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

import optuna
from tqdm import tqdm

import mlflow

from src.constants import MODELS_DIR
from src.loss import mdn_loss
from src.torch_datasets import generate_datasets
from .models import GRU_MDN, LSTM_MDN, RNN_MDN, TCN_MDN, Transformer_MDN

mlflow.set_tracking_uri("http://mlflow:5000")

class Trainer:
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
        params :dict = None, 
    ):
        """
        Initializes the training pipeline for a PyTorch model, including device configuration,
        data loading, and model checkpoint handling.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            criterion: The loss function used for training (e.g., nn.MSELoss, custom NLL).
            optimizer: The optimization algorithm (e.g., torch.optim.Adam).
            train_dataset (Dataset): The dataset used for training.
            val_dataset (Dataset, optional): The dataset used for validation. Defaults to None.
            batch_size (int, optional): Batch size used for training and validation loaders. Defaults to 256.
            params (dict, optional): Additional parameters for logging. Defaults to None.

        Attributes:
            device (torch.device): The device on which the model will be trained (MPS, CUDA, or CPU).
            model (torch.nn.Module): The model moved to the selected device.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader or None): DataLoader for the validation dataset if provided.
            history (dict): Dictionary to store training history including loss and timing.
            start_epoch (int): The starting epoch index, updated if a checkpoint exists.
            best_val_loss (float): The best validation loss achieved during training.
            is_best_model (bool): Whether this model is the best for its class.
        """

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
        self.best_val_loss = float('inf')
        self.is_best_model = False

    @property
    def was_saved(self) -> bool:
        """Returns True if the model was saved (i.e., it was the best for its class)."""
        return self.is_best_model

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
                max_results=1
            )
            
            if len(runs) == 0:
                # No previous runs for this model class
                return True
            
            best_previous_loss = runs.iloc[0]['metrics.final_val_loss']
            return current_val_loss < best_previous_loss
            
        except Exception as e:
            # If there's any error querying MLflow, default to saving
            print(f"Warning: Could not query MLflow for best model comparison: {e}")
            return True

    def train(self, num_epochs=30, save=True):
        with mlflow.start_run() as run:
            mlflow.set_experiment(self.MLFLOW_EXPERIMENT)
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
                        desc=f"Epoch {epoch+1}",
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
                            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}"
                        )
                    else:
                        pbar.set_description(
                            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}"
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
                print("⚠️ No validation dataset provided. Cannot determine if this is the best model.")
                self.is_best_model = False

            if save:
                if self.history["val_loss"] and self.is_best_model:
                    print("✅ Training completed. This is the best model for its class. Saving model...")
                    self.save(num_epochs)
                elif not self.history["val_loss"]:
                    print("⚠️ Training completed. No validation loss available for comparison. Skipping save.")
                    self._log_metrics_only(num_epochs)
                else:
                    print(f"⏭️ Training completed. Model did not beat the best {self.model.__class__.__name__} model. Skipping save.")
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
        mlflow.pytorch.log_model(
            self.model,
            name=self.name,
            registered_model_name=self.model.__class__.__name__,
        )

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

def lstm_mdn_objective(trial, dataset_df: pd.DataFrame, num_epochs: int = 30):
    
    try:
        trial.suggest_categorical("hidden_dim", [64, 128, 256])
        trial.suggest_int("num_layers", 1, 3)
        trial.suggest_int("num_mixtures", 3, 8)
        trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        trial.suggest_categorical("lookback_days", [1, 2, 4])
        trial.suggest_categorical("step", [1, 2, 3, 4, 6])  # hours between timesteps
        trial.suggest_float("dropout", 0.0, 0.5)

        # run = _start_mlflow_run("LSTM-MDN", trial)

        trainer = Trainer.from_template(
            dataset_df, LSTM_MDN, trial.params, trial.number
        )
        trainer.train(num_epochs=num_epochs)

        # _log_training_history(trainer)
        # _save_mlflow_run(trainer)

        return trainer.history["val_loss"][-1]

    except KeyboardInterrupt as e:
        raise KeyboardInterrupt from e
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during trial {trial.number}: {e}")
        return float("inf")


def gru_mdn_objective(trial, dataset_df: pd.DataFrame, num_epochs: int = 30):
    try:
        trial.suggest_categorical("hidden_dim", [64, 128, 256])
        trial.suggest_int("num_layers", 1, 3)
        trial.suggest_int("num_mixtures", 3, 8)
        trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        trial.suggest_categorical("lookback_days", [1, 2, 4])
        trial.suggest_categorical("step", [1, 2, 3, 4, 6])  # hours between timesteps
        trial.suggest_float("dropout", 0.0, 0.5)

        trainer = Trainer.from_template(dataset_df, GRU_MDN, trial.params,trial.number)
        trainer.train(num_epochs=num_epochs)

        return trainer.history["val_loss"][-1]
    except KeyboardInterrupt as e:
        raise KeyboardInterrupt from e
    except Exception as e:
        print(f"Error during trial {trial.number}: {e}")
        return float("inf")


def rnn_mdn_objective(trial, dataset_df: pd.DataFrame, num_epochs: int = 30):
    try:
        trial.suggest_categorical("hidden_dim", [64, 128, 256])
        trial.suggest_int("num_layers", 1, 3)
        trial.suggest_int("num_mixtures", 3, 8)
        trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        trial.suggest_categorical("lookback_days", [1, 2, 4])
        trial.suggest_categorical("step", [1, 2, 3, 4, 6])  # hours between timesteps
        trial.suggest_float("dropout", 0.0, 0.5)

        # run = _start_mlflow_run("RNN-MDN", trial)

        trainer = Trainer.from_template(dataset_df, RNN_MDN, trial.params,trial.number)
        trainer.train(num_epochs=num_epochs)

        # _log_training_history(trainer)
        # _save_mlflow_run(trainer)

        return trainer.history["val_loss"][-1]
    except KeyboardInterrupt as e:
        raise KeyboardInterrupt from e
    except Exception as e:
        print(f"Error during trial {trial.number}: {e}")
        return float("inf")


def tcn_mdn_objective(trial, dataset_df: pd.DataFrame, num_epochs: int = 30):
    try:
        trial.suggest_categorical("hidden_dim", [64, 128, 256])
        trial.suggest_int("num_layers", 2, 5)
        trial.suggest_int("num_mixtures", 3, 8)
        trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        trial.suggest_categorical("lookback_days", [1, 2, 4])
        trial.suggest_categorical("step", [1, 2, 3, 4, 6])

        trainer = Trainer.from_template(dataset_df, TCN_MDN, trial.params,trial.number)
        trainer.train(num_epochs=num_epochs)

        return trainer.history["val_loss"][-1]
    except KeyboardInterrupt as e:
        raise KeyboardInterrupt from e
    except Exception as e:
        print(f"Error during trial {trial.number}: {e}")
        return float("inf")


def transformer_mdn_objective(trial, dataset_df: pd.DataFrame, num_epochs: int = 30):
    try:
        trial.suggest_categorical("hidden_dim", [64, 128, 256])
        trial.suggest_int("num_layers", 1, 4)
        trial.suggest_categorical("num_heads", [2, 4, 8])
        trial.suggest_int("num_mixtures", 3, 8)
        trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        trial.suggest_float("dropout", 0.0, 0.5)
        trial.suggest_categorical("lookback_days", [1, 2, 4])
        trial.suggest_categorical("step", [1, 2, 3, 4, 6])

        # run = _start_mlflow_run("Transformer-MDN", trial)

        trainer = Trainer.from_template(
            dataset_df, Transformer_MDN, trial.params,trial.number
        )
        trainer.train(num_epochs=num_epochs)

        # _log_training_history(trainer)
        # _save_mlflow_run(trainer)

        return trainer.history["val_loss"][-1]
    except KeyboardInterrupt as e:
        raise KeyboardInterrupt from e
    except Exception as e:
        print(f"Error during trial {trial.number}: {e}")
        return float("inf")


def run_training(
    dataset_df: pd.DataFrame, num_trials: int = 30, num_epochs: int = 30
) -> typing.Dict[str, optuna.Study]:
    lstm_study = optuna.create_study(
        direction="minimize",
        study_name="lstm_mdn_hyperparam_search",
        storage=f"sqlite:///{MODELS_DIR}/optuna_study.db",
        load_if_exists=True,
    )
    lstm_study.optimize(
        lambda trial: lstm_mdn_objective(trial, dataset_df, num_epochs),
        n_trials=num_trials,
    )

    gru_study = optuna.create_study(
        direction="minimize",
        study_name="gru_mdn_hyperparam_search",
        storage=f"sqlite:///{MODELS_DIR}/optuna_study.db",
        load_if_exists=True,
    )
    gru_study.optimize(
        lambda trial: gru_mdn_objective(trial, dataset_df, num_epochs),
        n_trials=num_trials,
    )

    rnn_study = optuna.create_study(
        direction="minimize",
        study_name="rnn_mdn_hyperparam_search",
        storage=f"sqlite:///{MODELS_DIR}/optuna_study.db",
        load_if_exists=True,
    )
    rnn_study.optimize(
        lambda trial: rnn_mdn_objective(trial, dataset_df, num_epochs),
        n_trials=num_trials,
    )

    tcn_study = optuna.create_study(
        direction="minimize",
        study_name="tcn_mdn_hyperparam_search",
        storage=f"sqlite:///{MODELS_DIR}/optuna_study.db",
        load_if_exists=True,
    )
    tcn_study.optimize(
        lambda trial: tcn_mdn_objective(trial, dataset_df, num_epochs),
        n_trials=num_trials,
    )

    transformer_study = optuna.create_study(
        direction="minimize",
        study_name="transformer_mdn_hyperparam_search",
        storage=f"sqlite:///{MODELS_DIR}/optuna_study.db",
        load_if_exists=True,
    )
    transformer_study.optimize(
        lambda trial: transformer_mdn_objective(trial, dataset_df, num_epochs),
        n_trials=num_trials,
    )

    return {
        "lstm": lstm_study,
        "gru": gru_study,
        "rnn": rnn_study,
        "tcn": tcn_study,
        "transformer": transformer_study,
    }
