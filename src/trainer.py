import os
import time
import warnings

import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import MODELS_DIR
from src.loss import mdn_loss
from src.torch_datasets import generate_datasets


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion,
        optimizer,
        train_dataset,
        val_dataset=None,
        model_pth: str = "model_checkpoint.pth",
        batch_size: int = 256,
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
            model_pth (str, optional): Path to the model checkpoint file. Defaults to "model_checkpoint.pth".
            batch_size (int, optional): Batch size used for training and validation loaders. Defaults to 256.

        Attributes:
            device (torch.device): The device on which the model will be trained (MPS, CUDA, or CPU).
            model (torch.nn.Module): The model moved to the selected device.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader or None): DataLoader for the validation dataset if provided.
            history (dict): Dictionary to store training history including loss and timing.
            start_epoch (int): The starting epoch index, updated if a checkpoint exists.
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
        self.model_pth = MODELS_DIR / model_pth

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

        self.start_epoch = (
            self.load(self.model_pth) if os.path.exists(self.model_pth) else 0
        )

    def train(self, num_epochs=30, save=True):
        """
        Trains the model for a specified number of epochs using the provided training and
        validation datasets, loss function, and optimizer. Supports checkpointing and
        interruption handling.

        Args:
            num_epochs (int, optional): The total number of training epochs.
                                        Training will resume from `self.start_epoch` if a checkpoint is found. Defaults to 30.

        Behavior:
            - Iterates over training data and computes the loss using the specified MDN loss function.
            - Performs backpropagation and optimization on each batch.
            - If a validation dataset is provided, evaluates the model after each epoch.
            - Tracks training and validation losses, and epoch durations in `self.history`.
            - Displays a progress bar with real-time loss reporting.
            - Saves a checkpoint and safely exits if interrupted (e.g., via Ctrl+C).

        Raises:
            KeyboardInterrupt: If training is manually interrupted, a checkpoint is saved and the exception is re-raised.
        """
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

                    # print("Mu shape:", mu.shape)
                    # print(mu[:, 1])
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

        if save:
            print("✅ Training completed. Saving final model checkpoint...")
            self.save(num_epochs)

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
        """
        Saves the current state of the model, optimizer, and training history to disk.

        Args:
            epoch (int, optional): The training epoch to record. If not provided, it defaults
            to the current length of the training loss history.

        Saves:
            A dictionary containing:
                - 'model_state_dict': Current weights of the model.
                - 'optimizer_state_dict': Current state of the optimizer.
                - 'history': Training and validation loss history.
                - 'epoch': The epoch number at the time of saving.

        Effects:
            - Serializes the checkpoint to `self.model_pth` using `torch.save`.
            - Prints a confirmation message indicating the file path and epoch.

        """
        if epoch is None:
            epoch = len(self.history["train_loss"])

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
                "epoch": epoch,
            },
            self.model_pth,
        )
        print(f"✅ Saved model to {self.model_pth} at epoch {epoch}")

    def load(self, path):
        """
        Loads a saved model checkpoint from the specified file path.

        Args:
            path (str): Path to the saved checkpoint file.

        Loads:
            - 'model_state_dict': Restores the model weights.
            - 'optimizer_state_dict': Restores the optimizer state.
            - 'history': Restores the training history (if available).
            - 'epoch': The epoch at which the checkpoint was saved.

        Returns:
            int: The epoch number stored in the checkpoint. Defaults to 0 if not found.

        Effects:
            - Loads the model and optimizer states into the current instance.
            - Updates the training history if present in the checkpoint.
            - Prints a confirmation message with the path and epoch number.

        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        print(
            f"✅ Loaded checkpoint from '{path}' (epoch {checkpoint.get('epoch', 0)})"
        )
        return checkpoint.get("epoch", 0)

    @staticmethod
    def from_template(
        dataset_df: pd.DataFrame,
        model_class: torch.nn.Module,
        hyperparams: dict,
        _id : str,
    ):
        import hashlib
        import json

        hyperparams = hyperparams.copy()

        model_pth = f"{model_class.__name__.lower()}_{_id}_checkpoint.pth"

        training_dataset, validation_dataset = generate_datasets(
            dataset_df,
            lookback=hyperparams.pop("lookback_days") * 24,
            delay=24,
            step=hyperparams.pop("step"),
        )

        learning_rate = hyperparams.pop("learning_rate")

        model = model_class(input_dim=21, output_dim=7, **hyperparams)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        return Trainer(
            model=model,
            criterion=mdn_loss,
            optimizer=optimizer,
            train_dataset=training_dataset,
            val_dataset=validation_dataset,
            model_pth=model_pth,
            batch_size=256,
        )

    @staticmethod
    def from_best_optuna_trial(
        study: optuna.Study,
        dataset_df: pd.DataFrame,
        model_class: torch.nn.Module,
    ):
        best_trial = study.best_trial
        hyperparams = best_trial.params

        return Trainer.from_template(
            dataset_df=dataset_df, model_class=model_class, hyperparams=hyperparams, _id=best_trial.number
        )
