import hashlib
import json
import typing

import optuna
import pandas as pd
import torch

from src.constants import MODELS_DIR

from .loss import mdn_loss
from .models import GRU_MDN, LSTM_MDN, RNN_MDN, TCN_MDN, Transformer_MDN
from .torch_datasets import generate_datasets
from .trainer import Trainer


def lstm_mdn_objective(trial, dataset_df: pd.DataFrame, num_epochs: int = 30):
    try:
        trial.suggest_categorical("hidden_dim", [64, 128, 256])
        trial.suggest_int("num_layers", 1, 3)
        trial.suggest_int("num_mixtures", 3, 8)
        trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        trial.suggest_categorical("lookback_days", [1, 2, 4])
        trial.suggest_categorical("step", [1, 2, 3, 4, 6])  # hours between timesteps
        trial.suggest_float("dropout", 0.0, 0.5)

        trainer = Trainer.from_template(
            dataset_df, LSTM_MDN, trial.params, trial.number
        )
        trainer.train(num_epochs=num_epochs)

        return trainer.history["val_loss"][-1]

    except KeyboardInterrupt as e:
        raise KeyboardInterrupt from e
    except Exception as e:
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

        trainer = Trainer.from_template(dataset_df, GRU_MDN, trial.params, trial.number)
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

        trainer = Trainer.from_template(dataset_df, RNN_MDN, trial.params, trial.number)
        trainer.train(num_epochs=num_epochs)
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
        # dropout = trial.suggest_float("dropout", 0.0, 0.5)

        trainer = Trainer.from_template(dataset_df, TCN_MDN, trial.params, trial.number)
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

        trainer = Trainer.from_template(
            dataset_df, Transformer_MDN, trial.params, trial.number
        )
        trainer.train(num_epochs=num_epochs)

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
