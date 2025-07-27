# Air Quality Index Probability Prediction (MLOps Version)

This repository is a modified version of the [original Air Quality Index (AQI) Probability Prediction project](https://github.com/PeteCastle/aqi-mdn), tailored for an activity as part of the requirements for Machine Learning Operations (MLOps) course.  Intellectual property rights for the original project are retained by the original authors: Francis Mark Cayco, Andgrel Heber Jison, Angela Elaine Pelayo, and Eros Paul Estante.

**Francis Mark Cayco**

Masters of Science in Data Science

Asian Institute of Management

## Project Overview
*How can Mixture Density Networks improve air pollution forecasting by providing uncertainty-aware predictions that support more informed and reliable business or policy decisions?*

Traditional air quality forecasting models frequently provide deterministic, single-point predictions without quantifying the associated uncertainty. This limitation can be problematic, especially when decision-making requires an understanding of the range of possible outcomes. Recent studies have highlighted this issue. For instance, research has shown that most current data-driven air quality forecasting solutions lack proper quantifications of model uncertainty, which is crucial for communicating the confidence in forecasts. This gap underscores the need for models that can provide probabilistic forecasts, offering a distribution of possible outcomes rather than a single deterministic prediction.

Incorporating uncertainty quantification into air quality forecasts allows for better risk assessment and more informed decision-making. Probabilistic models, such as those using deep learning techniques, have been developed to address this need, providing more reliable uncertainty estimates and improving the practical applicability of air quality forecasts.

The goal of this project is to develop a probabilistic air quality forecasting model that captures a full range of possible pollutant concentrations, rather than relying on single-point predictions.  Use Mixture Density Networks (MDNs) to model predictive uncertainty.

Train and evaluate the MDN framework using various sequence modeling architectures:
- LSTM-MDN
- GRU-MDN
- Classic RNN-MDN
- TCN-MDN
- Transformer-MDN


## Data Sources
This project uses air quality index (AQI) data sourced from the **OpenWeatherMap API**, covering 138 cities globally from 2023 to 2025. For consistency and regional focus, we limit the scope to cities within Metro Manila, Philippines. The dataset contains hourly measurements of seven key air pollutants: **sulfur dioxide (SO₂), nitrogen dioxide (NO₂), particulate matter (PM10 and PM2.5), ozone (O₃), and carbon monoxide (CO)**. These pollutants are commonly monitored in environmental health studies and serve as the prediction targets for our probabilistic forecasting models.

Sources:
- [2023 to 2024 Data (Kaggle)](https://www.kaggle.com/datasets/bwandowando/philippine-major-cities-air-quality-data)
- [2025 Data (Kaggle)](https://www.kaggle.com/datasets/bwandowando/philippine-cities-air-quality-index-data-2025/data)

Note: The data structure in Kaggle might've changed and updated since we last accessed it.

## Setup Instructions

### 1. Create and activate a virtual environment using `uv`
Ensure that UV is [installed in your computer](https://docs.astral.sh/uv/getting-started/installation/).

**(For Linux/MacOS)**
```bash
uv venv
source .venv/bin/activate
```

**(For Windows)**
```bash
uv venv
source .venv/Scripts/activate.ps1
```

### 2. Install dependencies
Use either of the following, depending on your system’s hardware:
- For Apple Silicon / Metal backend: `uv pip install '.[metal]'`
- For NVIDIA GPU / CUDA backend: `uv pip install '.[cuda]'`
- CPU Only: `uv pip install '.[cpu]'`

### 3. Run Pre-Commit Hooks  (optional but recommended)
Install pre-commit hooks to ensure code quality and consistency:
```bash
pre-commit install
```
Run all pre-commit hooks on all files.
```bash
pre-commit run --all-files
```

This will apply formatting (e.g., Black), validate configs, strip Jupyter outputs, and check for large files.

### 4. Run the Pipeline
To execute the training and evaluation pipeline:
```bash
python -m src.run_pipeline
```

**Command-Line Arguments**

| Argument              | Type      | Default | Description                                                                                     |
|-----------------------|-----------|---------|-------------------------------------------------------------------------------------------------|
| `--num_trials`        | `int`     | `30`    | Number of Optuna trials to run for each model.                                                  |
| `--num_epochs`        | `int`     | `30`    | Number of training epochs per trial.                                                            |
| `--dry-run`           | `flag`    | `False` | Runs a fast version of the pipeline with **1 trial** and **1 epoch** per model. Ignores other training args. |
| `--generate-report`   | `flag`    | `False` | If set, generates a markdown report after training and evaluation.                             |

**Examples:**

To run the pipeline with 50 trials and 20 epochs, and generate a report:
```bash
python -m src.run_pipeline --num_trials 50 --num_epochs 20 --generate-report
```

To run a quick dry run with minimal settings, and generate a sample report:
```bash
python -m src.run_pipeline --dry-run --generate-report
```

## 🧠 Reflection

One of the key challenges I encountered during this project was related to compatibility issues with some `pre-commit` hooks. A few hooks initially failed to run due to unknown errors, which disrupted the development workflow. After some debugging, I resolved the issue by updating the affected hooks to their latest versions, which restored compatibility and allowed the hooks to execute correctly across different environments. In contrast, the `uv` setup process was smooth, as we had already adopted it in other projects. Similarly, there were no major issues with data preprocessing since the core machine learning pipeline had already been implemented in a previous iteration of the project. However, I did face some issues when reworking the caching logic. The original logic was tightly coupled with the notebook-based workflow, and I had to redesign it to support a more modular and reusable pipeline structure. This required careful consideration to ensure cache hits/misses behaved correctly in a script-based MLOps setup.

A key area for future improvement is enabling parallel training of the five deep learning models (LSTM-MDN, GRU-MDN, RNN-MDN, TCN-MDN, Transformer-MDN). Since we are using Optuna for hyperparameter optimization, we can leverage its distributed capabilities to train each model concurrently across multiple machines or processes.

## Folder Structure
This project follows a modular and reproducible structure tailored for machine learning workflows.
```
aqi-probability-prediction/
│
├── .vscode/
│   └── Editor-specific settings for VSCode.
│
├── cache/
│   └── Temporary files and intermediate artifacts such as checkpoints and cached datasets.
│
├── data/
│   ├── raw/
│   │   └── Original datasets as collected or received. Keeping them unmodified ensures full reproducibility.
│   └── processed/
│       └── Cleaned, transformed, and feature-engineered datasets ready for modeling. Separating these avoids accidental overwrites and aids in debugging.
│
├── models/
│   └── Trained model artifacts, including weights and saved checkpoints. Used for reloading and evaluation without retraining.
│
├── notebooks/
│   └── Jupyter notebooks for exploratory data analysis (EDA), prototyping, and result visualization.
│
├── reports/
│   └── Generated charts, logs, and markdown/PDF reports.
│
├── src/
│   └── All core logic is encapsulated in the `src` module for modularity and ease of testing:
│       ├── __init__.py              # Declares src as a Python package
│       ├── constants.py             # Global constants (directories, column names, etc.)
│       ├── data_preprocessing.py   # Functions for loading and cleaning raw datasets
│       ├── evaluation.py           # Evaluation metrics and model performance summaries
│       ├── feature_engineering.py  # Transformations like lookbacks, scaling, or encodings
│       ├── loss.py                 # Contains MDN loss.
│       ├── model_training.py       # Contains training logic.
│       ├── models.py               # Model class definitions (LSTM, GRU, TCN, etc.)
│       ├── run_pipeline.py         # Entrypoint script to execute full training pipeline
│       ├── torch_datasets.py       # Dataset loading Pandas datasets to Torch datasets
│       ├── trainer.py              # Training loop, validation logic, and checkpointing
│       └── visualizers.py          # Utilities for visualizing predictions, insights, and losses
│
├── .gitignore
├── pre-commit-config.yaml
│   └── Ensures code quality.  See Pre-Commit Configuration section below.
│
└── pyproject.toml
    └── Declares project metadata, dependencies, and build system in a standard, tool-compatible format.
```

## Pre-Commit Configuration
This project uses [**pre-commit**](https://pre-commit.com/) to ensure consistent formatting and prevent common mistakes before code is committed.

| Hook ID                   | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `black`                  | Formats Python code using [Black](https://github.com/psf/black), a strict code formatter. Ensures consistent style across all `.py` files. |
| `trailing-whitespace`    | Removes trailing whitespace from all files to keep diffs clean.             |
| `end-of-file-fixer`      | Ensures that files end with a single newline character.                     |
| `check-yaml`             | Validates YAML syntax for files like GitHub Actions, pre-commit configs, etc. |
| `nbstripout`             | Strips output and metadata from Jupyter notebooks to prevent committing large diffs. |
| `check-added-large-files`| Blocks accidentally committed large files (over 5MB) to avoid bloating the repo. |
| `check-toml`             | Validates that `pyproject.toml` and other TOML files are correctly formatted and parseable. |

Run all hooks on all files manually using `pre-commit run --all-files`

```
docker build -f docker/Dockerfile \
  -t 6bd14ff485084ce2fd9c18e9539cab76bc04816b587a434afc54c644bc3abdc7-ml-pipeline:latest \
  .
```

```
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" -v "$(pwd)/cache:/app/cache" -v "$(pwd)/reports:/app/reports" 6bd14ff485084ce2fd9c18e9539cab76bc04816b587a434afc54c644bc3abdc7-ml-pipeline:latest
```


```
docker-compose -f docker/docker-compose.yml run airflow-webserver airflow db migrate
```

```
docker-compose -f docker/docker-compose.yml run airflow-webserver airflow cli users create \
  --username admin \
  --password admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com
```

```
docker compose -f docker/docker-compose.yml up --build  
```


