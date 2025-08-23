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

## Setup Instructions
> Notes for Methods 1 and 2:
> - CUDA GPU support in Docker is currently a work in progress.
> - Apple Silicon MPS will never be supported in this Dockerized setup.
> - This method supports only CPU execution.
> - If you require GPU acceleration, use Method 3: Local Setup instead.

### Method 1: Airflow Containerized Setup

This method runs the entire pipeline using **Apache Airflow** in a Dockerized environment. It is the recommended way to orchestrate scheduled training, evaluation, and report generation.

#### 1. Install Docker Compose

Make sure you have the following installed:
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
  (v2 preferred: `docker compose` instead of `docker-compose`)

To verify installation:
```bash
docker --version
docker compose version
```

#### 2. Clone the repository
Clone the repository to your local machine:
```bash
git clone https://github.com/PeteCastle/6bd14ff485084ce2fd9c18e9539cab76bc04816b587a434afc54c644bc3abdc7_aqi_probability_prediction aqi-probability-prediction
```

#### 3. Setup the Airflow environment variables
- Navigate to the project directory.
- Inside the `config` directory, create an `.env` file using `config/.env.example` as a reference.
- Inside the `config` directory, create an `airflow.cfg` file using `config/airflow.cfg.example` as a reference.

#### 4. Build the services
Build all services defined in the Docker Compose file:
```bash
docker compose -f deploy/docker/docker-compose.yml build
```

#### 5. Start the Airflow services
Start the Airflow webserver, scheduler, and other services in detached mode:
```bash
docker compose -f deploy/docker/docker-compose.yml up -d
```

Optional:
Get the logs of the Airflow worker to monitor the pipeline execution:
```bash
docker compose -f deploy/docker/docker-compose.yml logs -f airflow-worker
```

#### 6. Running the Pipeline
Access the Airflow web interface and DAGs at `http://localhost:8080/dags`.
![Airflow DAGs](docs/assets/airflow_dags.png)
Click on `ml_pipeline_dag` to view the DAG details.
![Model Training Pipeline](docs/assets/ml_pipeline_dag.png)
Click on `Trigger` to run the pipeline manually
![Pipeline Trigger](docs/assets/pipeline_trigger.png)
Modify parameters if you want to specify the number of trials, epochs, or in dry run.  Note that in Airflow setup, the script will always generate a report after training and evaluation.

#### 7.  Test the Pipeline
To test the pipeline, you can trigger the `prepare_data` task directly from the Airflow UI or use the command line:
```bash
docker compose -f deploy/docker/docker-compose.yml exec airflow-webserver \
  airflow dags test ml_pipeline_dag
```
You may also se the `--conf` flag to pass in parameters:
```bash
docker compose -f deploy/docker/docker-compose.yml exec airflow-webserver \
  airflow dags test ml_pipeline_dag \
  --conf '{"dry_run": true}'
```
You may also test the `prepare_data` task directly using the command line:
```
docker compose -f deploy/docker/docker-compose.yml exec airflow-webserver airflow tasks test ml_pipeline_dag prepare_data 2025-07-31
```

### Method 2: Docker Setup
#### 1. Install Docker

Follow the instructions for your operating system:

- [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
- [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
- [Docker Engine for Linux](https://docs.docker.com/engine/install/)


After installation, verify Docker is running
```bash
docker --version
```

#### 2. Clone the repository
Clone the repository to your local machine:
```bash
git clone https://github.com/PeteCastle/6bd14ff485084ce2fd9c18e9539cab76bc04816b587a434afc54c644bc3abdc7_aqi_probability_prediction aqi-probability-prediction
```

#### 3. Build the Docker Image
Use the following command to build the Docker image from the provided pipeline.Dockerfile. Run this in the project root directory:
```bash
docker build -f deploy/docker/pipeline.Dockerfile \
  -t 6bd14ff485084ce2fd9c18e9539cab76bc04816b587a434afc54c644bc3abdc7-ml-pipeline:latest \
  --build-arg BACKEND=cpu \
  .
```

**Build Arguments**:
The Dockerfile accepts a build argument `--build-arg`:
- `BACKEND` — defines the dependency group to install.  Possible values: `cpu` (default), `cuda`, or `mps`.  Note that while `cuda` and `mps` are accepted, they are not currently functional in this Dockerized setup.

#### 4. Run the Docker Container
To run the pipeline, execute the following command in the project root directory:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/cache:/app/cache" \
  -v "$(pwd)/reports:/app/reports" \
  6bd14ff485084ce2fd9c18e9539cab76bc04816b587a434afc54c644bc3abdc7-ml-pipeline:latest
```

##### Command-Line Arguments
The docker run command accepts the following command-line arguments:
| Argument              | Type      | Default | Description                                                                                     |
|-----------------------|-----------|---------|-------------------------------------------------------------------------------------------------|
| `--num_trials`        | `int`     | `30`    | Number of Optuna trials to run for each model.                                                  |
| `--num_epochs`        | `int`     | `30`    | Number of training epochs per trial.                                                            |
| `--dry-run`           | `flag`    | `False` | Runs a shorter version of the pipeline with **1 trial** and **1 epoch** per model. If set, it ignores `num_trials` and `num_epochs`. |
| `--generate-report`   | `flag`    | `False` | If set, generates a markdown report after training and evaluation.        |

Example to dry run the pipeline and generate report:
```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/cache:/app/cache" \
  -v "$(pwd)/reports:/app/reports" \
  6bd14ff485084ce2fd9c18e9539cab76bc04816b587a434afc54c644bc3abdc7-ml-pipeline:latest \
  --dry-run \
  --generate-report
```

### Method 3: Local Setup
#### 1. Clone the repository
Clone the repository to your local machine:
```bash
git clone https://github.com/PeteCastle/6bd14ff485084ce2fd9c18e9539cab76bc04816b587a434afc54c644bc3abdc7_aqi_probability_prediction aqi-probability-prediction
```

#### 2. Create and activate a virtual environment using `uv`
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

#### 3. Install dependencies
Use either of the following, depending on your system’s hardware:
- For Apple Silicon / Metal backend: `uv pip install '.[metal]'`
- For NVIDIA GPU / CUDA backend: `uv pip install '.[cuda]'`
- CPU Only: `uv pip install '.[cpu]'`

#### 4. Run Pre-Commit Hooks  (optional but recommended)
Install pre-commit hooks to ensure code quality and consistency:
```bash
pre-commit install
```
Run all pre-commit hooks on all files.
```bash
pre-commit run --all-files
```

This will apply formatting (e.g., Black), validate configs, strip Jupyter outputs, and check for large files.

#### 5. Run the Pipeline
To execute the training and evaluation pipeline:
```bash
python -m src.run_pipeline
```

Note that the pipeline also accepts [command-line arguments](#command-line-arguments).

**Examples:**

To run the pipeline with 50 trials and 20 epochs, and generate a report:
```bash
python -m src.run_pipeline --num_trials 50 --num_epochs 20 --generate-report
```

To run a quick dry run with minimal settings, and generate a sample report:
```bash
python -m src.run_pipeline --dry-run --generate-report
```

## MLFlow Integration
This project integrates MLflow as the primary MLOps platform for experiment tracking, model management, and deployment pipeline orchestration.  It follows a distributed approach where tracking capabilities are embedded throughout the ML pipeline rather than centralized in a single location.

All model training activities are consolidated under a single experiment named `"aqi_mdn_experiment"`.

The MLflow tracking server is configured to use a containerized Mlflow server, or an Sqlite if not available.

The tracking URI can be found in `http://localhost:5000` for Dockerized Airflow setup, and `sqlite:///mlflow.db` for local or non-Airflow setups.

Contrary to the given requirements to log exactly 3 hyperparameters, I logged all hyperparameters since they are relevant for reproducibility and model retraining.

### Experiment Tracking
The Trainer class was designed to inherit from `mlflow.pyfunc.PythonModel`, making every trained model inherently deployable through MLflow's serving infrastructure.
- Automatically generates descriptive run names.
- Implements conditional saving logic that only persists models achieving new best performance for their specific architecture class.
- Saves complete model state including weights, hyperparameters, etc.
- The `val_loss` and `train_loss` metrics are logged in Mlflow, as well as the `name` of the model architecture.

### Best Model Selection
- During model evaluation, the program queries MLflow's experiment database to compare current model performance against historical runs.
- In logging,
- Only saves models that achieve superior validation performance.

### Registration Implementation:**
The performance criterion is where the `val_loss` (negative log likelihood) is less than `-3`.  This value is the baseline negative log likelihood.  A metric to determine if the model outbeats the random guessing scenario.  When a model meets the performance criteria, the system automatically:
1. Generates a standardized model name following the pattern `aqi_prediction_{model_type}`.
2. Registers the model using the current MLflow run ID as the source.

### Model Retrieval
Implements a "best-of-breed" approach where the system automatically identifies and loads the best performing model for each architecture type from MLflow's experiment database.

## Model Drift Detection
The drift detection system is integrated into both the standalone pipeline (`run_pipeline.py`) and the orchestrated Airflow DAG (`ml_pipeline_dag.py`). It operates as a quality gate that validates model performance by comparing reference data against current validation data.

In the function `detect_drift(reference_data_path, current_data_path)` in `src/drift_detection.py`, the system uses the Evidently library with DataDriftPreset for statistical drift analysis. It is configured with a drift threshold of 0.2 (20%) and automatically excludes the target column to focus on feature drift. There is feature-level drift scoring with ranking of the most drifted features, and it generates a JSON report for detailed analysis.

The output is saved in `reports/drift_report.json` and includes:
- `drift_detected`: Boolean indicating if overall drift exceeds threshold
- `feature_drifts`: Dictionary of top 3 most drifted features with scores
- `overall_drift_score`: Average drift score across all features

### Integration Strategy
#### Pipeline Integration
In standalone pipeline (`run_pipeline.py`), drift detection is integrated as a post-evaluation step.  If drift is detected, it raises an exception stating that a model retraining is required.

### DAG Integration
The Airflow DAG provides a more sophisticated orchestration with conditional branching. Drift detection runs after model evaluation If drift is detected, it triggers a retraining of the model.

### Drifted Data Generation
Drifted data generation occurs in the `feature_engineering` step rather than `data_preprocessing`. This design decision ensures that:
- The model receives fully processed data for time series transformation
- Drift simulation happens on feature-engineered data that matches production input format
- Time series preprocessing maintains temporal relationships

The `preprocess_data` function does not return separate drifted and original datasets as tuples because:
- Train-test split occurs during model training phase
- This aligns with the project's time series nature where temporal splits are more appropriate
- Allows for more flexible data handling in the training pipeline.

## Folder Structure
The following table shows the new files and modifications added to support drift detection and MLflow integration compared to the previous assignment:

| File/Directory | Type | Description |
|----------------|------|-------------|
| `deploy/airflow/dags/ml_pipeline_dag.py` | Modified | Enhanced DAG with drift detection task, conditional branching logic, and automated retraining when drift is detected. Separated data preparation into two distinct tasks: preprocessing and feature engineering. |
| `deploy/docker/Dockerfile.mlflow` | New | Dockerfile for MLflow service deployment to support experiment tracking and model registry functionality. |
| `deploy/docker/airflow.Dockerfile` | Modified | Updated to use CUDA base image instead of standard Airflow image to enable GPU support for model training. Previous version renamed to `airflow.old.Dockerfile`. |
| `deploy/docker/airflow.old.Dockerfile` | Renamed | Original Airflow Dockerfile preserved for reference. |
| `deploy/docker/docker-compose.yml` | Modified | Added MLflow service configuration and improved startup behavior with proper service dependencies to ensure correct initialization order. |
| `notebooks/00_download_data.ipynb` | New | Jupyter notebook for downloading data from Kaggle with interactive data exploration capabilities. |
| `notebooks/03_drift_detection.ipynb` | New | Interactive notebook demonstrating drift detection analysis, visualization, and experimentation with different drift scenarios. |
| `pyproject.toml` | Modified | Added new dependencies for drift detection (Evidently), MLflow integration, and enhanced data processing capabilities. |
| `scripts/init-mlflow-db.sh` | New | Shell script for initializing MLflow database and setting up the tracking server environment. |
| `src/data_ingestion.py` | New | Script version of data download functionality from Kaggle, providing programmatic access to dataset retrieval. |
| `src/drift_detection.py` | New | Core drift detection module implementing statistical analysis, feature-level drift scoring, and automated reporting using Evidently library. |
| `src/evaluation.py` | Modified | Enhanced with MLflow integration for experiment logging, model result persistence, and automated model registration in the MLflow model registry. |
| `src/feature_engineering.py` | Modified | Added functionality for generating drifted datasets for testing purposes and utility functions for converting PyTorch datasets to pandas DataFrames. |
| `src/run_pipeline.py` | Modified | Integrated MLflow experiment tracking and drift detection logic with conditional pipeline execution based on drift analysis results. |
| `src/utils.py` | Modified | Added helper functions for MLflow server initialization and configuration management to ensure consistent tracking setup across different environments. |

## Testing Instructions
### Test Standalone Pipeline
```bash
python -m src.run_pipeline --dry-run --generate-report
```
*Expected Behavior: Pipeline should raise "Data drift detected in test set! Model retraining required." when running with drifted data*

### Test Airflow Pipeline
```
AIRFLOW_HOME=/home/ec2-user/aqi_probability_prediction/deploy/airflow PYTHONPATH="${PYTHONPATH}:$(pwd)" airflow dags test ml_pipeline_dag 2025-08-02 --conf '{"dry_run": true}'
```
Note: ensure that you are running the command in the project root directory.

### Verify MLflow Tracking
Access the MLflow tracking UI at `http://localhost:5000` to verify that experiments.

### Access the Drift Report
The drift detection report is saved as a JSON file at `reports/drift_report.json`.

## Reflection
The biggest challenge I faced was enabling GPU support in the Dockerized environment. My previous assignment used a standard Airflow image without CUDA capabilities, making model training painfully slow. I spent considerable time migrating to a CUDA-enabled image, completely reconstructing the Docker environment and debugging dependency conflicts. Additionally, most assignment instructions were designed for conventional ML workflows and didn't apply to my time series forecasting approach with mixture density networks. Rather than forcing generic patterns, I adapted the core objectives to fit my existing architecture. The most technically challenging aspect was refactoring my custom trainer class for MLflow compatibility, requiring significant architectural changes to separate logging concerns from training logic.

Despite these challenges, I developed custom solutions including time series-specific drift detection, parallel model training with centralized MLflow tracking, and intelligent Airflow branching that automatically responds to detected drift. This experience taught me that establishing robust infrastructure early prevents rework, modular design facilitates integration of new requirements, and specialized use cases often require thoughtful adaptation of standard MLOps patterns. The final system seamlessly integrates drift detection and MLflow tracking, providing automated quality monitoring essential for production deployment.
