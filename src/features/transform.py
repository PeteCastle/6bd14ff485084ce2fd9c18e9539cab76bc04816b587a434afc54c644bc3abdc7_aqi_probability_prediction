import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from src.constants import DATASET_DIR
from .data_preprocessing import get_preprocessed_data
from src.data.torch_datasets import generate_datasets
from src.data.torch_datasets import ConcatDatasetWithMetadata


def dataset_to_df(
    dataset: ConcatDatasetWithMetadata, max_samples: int = 2500
) -> pd.DataFrame:
    """
    Convert PyTorch dataset to pandas DataFrame for analysis and validation.

    This function transforms tensor-based datasets back to tabular format, enabling
    easy inspection, validation, and traditional data analysis workflows.

    Args:
        dataset: PyTorch dataset containing features and targets
        max_samples: Maximum number of samples to convert (for memory efficiency)

    Returns:
        pd.DataFrame: Flattened dataset with features and targets as columns

    Transformation Steps:
        1. Sample reduction - Subsample dataset to manage memory usage
        2. Feature flattening - Convert multi-dimensional tensors to flat features
        3. Target extraction - Handle both scalar and multi-dimensional targets
        4. DataFrame construction - Create structured tabular representation
    """
    rows = []

    # Step 1: Sample Reduction Strategy
    # Rationale: Calculate step size to ensure we don't exceed max_samples limit
    # This prevents memory overflow when dealing with large datasets while maintaining
    # representative sampling across the entire dataset
    step = max(1, len(dataset) // max_samples)

    # Step 2: Iterative Data Extraction
    # Rationale: Process samples in chunks with progress tracking to handle large datasets
    # efficiently without loading everything into memory simultaneously
    for i in tqdm(range(0, len(dataset), step), desc="Converting dataset to DataFrame"):
        x, y = dataset[i]  # Extract features and target
        meta = dataset.get_metadata(i)  # Get associated metadata
        row = {}

        # Step 3: Feature Flattening
        # Rationale: Convert multi-dimensional tensor features into flat structure
        # This enables compatibility with traditional ML algorithms and simplifies analysis
        # Each feature dimension becomes a separate column (f0, f1, f2, ...)
        for j, val in enumerate(x.flatten()):
            row[f"f{j}"] = float(val)

        # Step 4: Target Processing
        # Rationale: Handle both multi-dimensional and scalar targets flexibly
        # Multi-dimensional targets are flattened into separate columns (target_0, target_1, ...)
        # Scalar targets are stored as single 'target' column
        if hasattr(y, "shape") and getattr(y, "ndim", 0) > 0 and y.shape[0] > 1:
            # Multi-dimensional target handling
            for k, val in enumerate(y.flatten()):
                row[f"target_{k}"] = float(val)  # Ensure scalar conversion
        else:
            # Scalar target handling with robust type conversion
            try:
                row["target"] = float(y.item())  # Extract scalar from tensor
            except Exception:
                row["target"] = float(y)  # Direct conversion fallback
        rows.append(row)

    # Step 5: DataFrame Construction
    # Rationale: Convert collected rows into structured DataFrame for analysis
    # This enables standard pandas operations and compatibility with visualization tools
    return pd.DataFrame(rows)


def get_feature_engineered_data(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Apply comprehensive feature engineering transformations to create ML-ready dataset.

    This function performs advanced data transformations including temporal interpolation,
    categorical encoding, and dataset generation for time series forecasting models.

    Args:
        df: Preprocessed DataFrame. If None, loads from preprocessing pipeline.

    Returns:
        pd.DataFrame: Feature-engineered dataset ready for model training

    Feature Engineering Pipeline:
        1. Temporal gap filling - Create complete hourly time series
        2. Missing value interpolation - Fill gaps with linear interpolation
        3. Categorical encoding - Convert city names to one-hot encoded features
        4. Dataset persistence - Save processed features for reuse
        5. Validation dataset generation - Create evaluation datasets with/without drift
    """
    # Load preprocessed data if not provided
    if df is None:
        df = get_preprocessed_data()

    filled_dfs = []

    # Step 1: City-wise Temporal Processing
    # Rationale: Process each city separately to maintain location-specific patterns
    # while ensuring consistent temporal structure across all locations
    for city, group in tqdm(df.groupby("city_name")):

        # Step 1a: Set datetime as index for time series operations
        # Rationale: DateTime indexing enables efficient time series operations
        # and facilitates temporal resampling and interpolation
        group = group.set_index("datetime")

        # Step 1b: Create Complete Temporal Grid
        # Rationale: Generate continuous hourly timestamps to identify missing periods
        # This ensures uniform temporal spacing required for time series models
        # and prevents issues with irregular sampling intervals
        full_index = pd.date_range(
            start=group.index.min(), end=group.index.max(), freq="h"
        )

        # Step 1c: Handle Duplicate Timestamps
        # Rationale: Average multiple measurements at the same timestamp
        # This resolves potential data collection overlaps and ensures one value per hour
        # numeric_only=True prevents issues with non-numeric columns during aggregation
        group = group.groupby(level=0).mean(numeric_only=True)

        # Step 1d: Temporal Reindexing
        # Rationale: Align data to complete hourly grid, creating NaN for missing periods
        # This standardizes the temporal structure across all cities and time periods
        group = group.reindex(full_index)

        # Step 2: Missing Value Interpolation
        # Rationale: Linear interpolation fills temporal gaps with realistic estimates
        # - Preserves temporal trends and patterns in air quality data
        # - More accurate than forward-fill or mean imputation for continuous variables
        # - Maintains smooth transitions that reflect atmospheric processes
        # - Essential for time series models that require complete sequences
        group.interpolate(method="linear", inplace=True)

        # Step 2a: Restore City Identifier
        # Rationale: Re-add city name after reindexing since it was lost during the process
        # This maintains the geographic information needed for location-specific modeling
        group["city_name"] = city
        filled_dfs.append(group)

    # Step 3: Combine City Datasets
    # Rationale: Concatenate all city datasets into unified structure
    # This creates a comprehensive dataset while preserving temporal alignment
    filled_dfs = pd.concat(filled_dfs)

    # Step 4: Categorical Feature Engineering
    # Rationale: Preserve original city names for reference while creating encoded features
    city_column = filled_dfs["city_name"].copy()

    # Step 4a: One-Hot Encoding of Cities
    # Rationale: Convert categorical city names into numerical features for ML models
    # - One-hot encoding prevents ordinal relationships between cities
    # - Each city becomes a binary feature (0/1) enabling location-specific learning
    # - dtype=np.int8 saves memory while maintaining precision for binary features
    # - Essential for models that cannot handle categorical inputs directly
    dataset_df = pd.get_dummies(filled_dfs, prefix="city", dtype=np.int8)

    # Step 4b: Restore Original City Column
    # Rationale: Keep original city names for interpretability and debugging
    # This allows easy identification of samples by location in analysis
    dataset_df["city_name"] = city_column

    # Step 5: Feature Persistence
    # Rationale: Save processed features to parquet format for efficient reuse
    # - Parquet provides fast I/O and compression for time series data
    # - Avoids recomputing expensive transformations in subsequent runs
    # - Enables reproducible experiments and model training
    dataset_df.to_parquet(DATASET_DIR / "processed" / f"features.parquet", index=True)

    # Step 6: Validation Dataset Generation
    # Rationale: Create specialized datasets for model evaluation and drift detection

    # Step 6a: Standard Validation Dataset
    # Rationale: Generate validation set without distributional drift
    # - lookback=96: Use 4 days (96 hours) of historical data for prediction
    # - delay=24: Predict 1 day (24 hours) ahead, relevant for air quality planning
    # - step=1: Use every hour for maximum data utilization
    # - drift_strength=0: No artificial drift for baseline evaluation
    if not os.path.exists(DATASET_DIR / f"reference.parquet"):
        train_dataset, _ = generate_datasets(
            dataset_df, lookback=96, delay=24, step=1, drift_strength=0
        )
        train_dataset = dataset_to_df(train_dataset)
        train_dataset.to_parquet(
            DATASET_DIR / f"reference.parquet", index=False
        )
        del train_dataset  # Free memory after saving

    # Step 6b: Drift-Affected Validation Dataset
    # Rationale: Generate validation set with simulated distributional drift
    # - drift_strength=0.1: Moderate drift simulation for robustness testing
    # - Tests model performance under changing environmental conditions
    # - Evaluates model stability and adaptation capabilities
    # - Critical for production deployment where data distribution may shift
    if not os.path.exists(DATASET_DIR / f"current.parquet"):
        drifted_train_dataset, _ = generate_datasets(
            dataset_df, lookback=96, delay=24, step=1, drift_strength=0.1
        )

        drifted_train_dataset = dataset_to_df(drifted_train_dataset)
        drifted_train_dataset.to_parquet(
            DATASET_DIR / f"current.parquet", index=False
        )
        del drifted_train_dataset  # Free memory after saving

    return dataset_df
