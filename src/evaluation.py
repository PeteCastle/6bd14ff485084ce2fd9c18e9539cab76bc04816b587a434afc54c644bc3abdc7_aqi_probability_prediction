import math
from io import StringIO
import os
import numpy as np
import pandas as pd
import torch

from src.constants import OUTPUT_DIR, POLLUTANT_COLUMNS
from src.model_training import GRU_MDN, LSTM_MDN, RNN_MDN, TCN_MDN, Transformer_MDN
from src.trainer import Trainer
from src.visualizers import (
    MDNVisualizer,
    compare_model_performance,
    save_model_performance,
)


def calculate_baseline(dataset_df: pd.DataFrame):
    nlls = []
    for _, group in dataset_df.groupby("city_name"):
        group = group.copy()
        y = group.drop(columns=["city_name"])[POLLUTANT_COLUMNS]
        y = torch.tensor(y.values, dtype=torch.float32)
        mu = y.mean()
        sigma = y.std()

        eps = 1e-6
        sigma = torch.clamp(sigma, min=eps)
        z = (y - mu) / sigma

        log_prob = (
            -0.5 * 7 * math.log(2 * math.pi)
            - torch.sum(torch.log(sigma))
            - 0.5 * torch.sum(z**2, dim=1)
        )
        nll = -log_prob.mean()

        nlls.append(nll.item())
        return np.average(nlls)


def run_evaluation(
    study: dict, dataset_df: pd.DataFrame, generate_report: bool = False, report_folder: str = ""
):
    trainers = {
        "LSTM-MDN": Trainer.from_best_optuna_trial(study["lstm"], dataset_df, LSTM_MDN),
        "GRU-MDN": Trainer.from_best_optuna_trial(study["gru"], dataset_df, GRU_MDN),
        "RNN-MDN": Trainer.from_best_optuna_trial(study["rnn"], dataset_df, RNN_MDN),
        "TCN-MDN": Trainer.from_best_optuna_trial(study["tcn"], dataset_df, TCN_MDN),
        "Transformer-MDN": Trainer.from_best_optuna_trial(
            study["transformer"], dataset_df, Transformer_MDN
        ),
    }

    results = []
    for model_name, trainer in trainers.items():
        results.append(
            {
                "Model": model_name,
                "Training Loss": trainer.history["train_loss"][-1],
                "Validation Loss": trainer.history["val_loss"][-1],
                "Training Time per Epoch (s)": np.average(
                    trainer.history["training_time"]
                ),
                "Trainer": trainer,
            }
        )

    best_results_df = pd.DataFrame(results).drop(columns=["Trainer"])
    best_results_df.set_index("Model", inplace=True)
    best_row = min(results, key=lambda x: x["Validation Loss"])
    best_trainer = best_row["Trainer"]

    sample_indeces = range(0, 100)

    os.makedirs(OUTPUT_DIR / report_folder, exist_ok=True)

    visualizer = MDNVisualizer(best_trainer, report_folder)

    if generate_report:
        compare_model_performance(*[trainer for trainer in trainers.values()], report_folder=report_folder)
        save_model_performance(best_trainer, report_folder=report_folder)
        visualizer.save_timeseries_from_val(
            sample_indeces, num_targets=None, title="Example Timeseries"
        )
        visualizer.generate_mixture_gif(0, 30, num_targets=2)

    buffer = StringIO()

    buffer.write("\n\n\nMODEL RESULTS\n\n")
    buffer.write(best_results_df.to_string(index=True))
    buffer.write("\n\n")

    baseline_nll = calculate_baseline(dataset_df)
    buffer.write(f"Baseline NLL: {baseline_nll:.4f}\n")
    buffer.write(
        f"Best model: {best_row['Model']} with val loss: {best_row['Validation Loss']:.4f}\n\n"
    )

    insights = visualizer.generate_insights_from_timestep(0)
    buffer.write("Example Insights:\n")
    buffer.write(insights["text_report"][0])
    buffer.write("\n")

    print(buffer.getvalue())

    if generate_report:
        with open(OUTPUT_DIR / report_folder / "metrics.txt", "w") as f:
            f.write(buffer.getvalue())

        print(
            f"Evaluation complete. Results and visualizations saved to '{OUTPUT_DIR}'"
        )
    else:
        print(
            "Evaluation complete. Use --generate-report to save results and visualizations."
        )

    return best_trainer, best_results_df
