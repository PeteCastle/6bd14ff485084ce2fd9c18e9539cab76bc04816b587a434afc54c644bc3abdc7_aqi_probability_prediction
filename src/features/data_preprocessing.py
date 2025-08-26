import glob

import pandas as pd
from tqdm import tqdm

from src.constants import CITY_NAMES, DATASET_DIR


def get_raw_data() -> pd.DataFrame:
    """
    Load and combine raw air quality data from multiple CSV files.

    This function reads all CSV files from the raw AQI dataset directory and
    concatenates them into a single DataFrame for further processing.

    Returns:
        pd.DataFrame: Combined raw air quality data from all available cities

    Transformation Steps:
        1. Glob pattern matching to find all CSV files in raw/aqi subdirectories
        2. Iterative reading of each CSV file with progress tracking
        3. Concatenation of all DataFrames with index reset
    """
    """
    Load and combine raw air quality data from multiple CSV files.

    This function reads all CSV files from the raw AQI dataset directory and
    concatenates them into a single DataFrame for further processing.

    Returns:
        pd.DataFrame: Combined raw air quality data from all available cities

    Transformation Steps:
        1. Glob pattern matching to find all CSV files in raw/aqi subdirectories
        2. Iterative reading of each CSV file with progress tracking
        3. Concatenation of all DataFrames with index reset
    """
    dfs = []

    # Step 1: Find all CSV files in the raw AQI dataset directory
    # Rationale: Using glob pattern to dynamically discover all available city data files
    # This allows the system to automatically include new cities without code changes
    for file in (
        pbar := tqdm(glob.glob(str(DATASET_DIR / "raw" / "aqi" / "*" / "*.csv")))
    ):
        pbar.set_description(f"Reading {file}")

        # Step 2: Read each CSV file
        # Rationale: Loading data incrementally to handle large datasets efficiently
        # and provide progress feedback to users
        df = pd.read_csv(file)
        dfs.append(df)

    # Step 3: Combine all DataFrames into a single dataset
    # Rationale: Concatenating with ignore_index=True ensures a continuous index
    # across all data sources, preventing index conflicts
    return pd.concat(dfs, ignore_index=True)


def get_preprocessed_data(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Apply comprehensive data preprocessing transformations to air quality data.

    This function performs essential data cleaning and preparation steps to ensure
    the dataset is suitable for machine learning model training and evaluation.

    Args:
        df (pd.DataFrame, optional): Input DataFrame. If None, loads raw data automatically.

    Returns:
        pd.DataFrame: Preprocessed and cleaned air quality data

    Transformation Pipeline:
        1. City filtering - Keep only cities with sufficient data availability
        2. Feature selection - Extract relevant pollutant measurements and metadata
        3. Data sorting - Organize by city and temporal order
        4. Datetime normalization - Standardize temporal data for consistency
    """
    """
    Apply comprehensive data preprocessing transformations to air quality data.

    This function performs essential data cleaning and preparation steps to ensure
    the dataset is suitable for machine learning model training and evaluation.

    Args:
        df (pd.DataFrame, optional): Input DataFrame. If None, loads raw data automatically.

    Returns:
        pd.DataFrame: Preprocessed and cleaned air quality data

    Transformation Pipeline:
        1. City filtering - Keep only cities with sufficient data availability
        2. Feature selection - Extract relevant pollutant measurements and metadata
        3. Data sorting - Organize by city and temporal order
        4. Datetime normalization - Standardize temporal data for consistency
    """
    # Load raw data if not provided
    if df is None:
        df = get_raw_data()

    # Step 1: City Filtering
    # Rationale: Filter to include only cities from Metro Manila with reliable data
    # This ensures data quality and focuses the model on a geographically coherent region
    # CITY_NAMES contains pre-validated cities with sufficient historical data
    df = df[df["city_name"].isin(CITY_NAMES)]

    # Step 2: Feature Selection
    # Rationale: Select only the essential features needed for AQI prediction
    # - datetime: Temporal information for time series analysis
    # - components.*: Core pollutant measurements (CO, NO, NO2, O3, SO2, PM2.5, PM10, NH3)
    # - city_name: Geographic identifier for location-specific patterns
    # Excluding unnecessary columns reduces noise and computational complexity
    df = df[
        [
            "datetime",  # Temporal feature for time series modeling
            "components.co",  # Carbon monoxide concentration
            "components.no",  # Nitric oxide concentration
            "components.no2",  # Nitrogen dioxide concentration
            "components.o3",  # Ozone concentration
            "components.so2",  # Sulfur dioxide concentration
            "components.pm2_5",  # Fine particulate matter (≤2.5μm)
            "components.pm10",  # Coarse particulate matter (≤10μm)
            "components.nh3",  # Ammonia concentration
            "city_name",  # Geographic identifier
        ]
    ]

    # Step 3: Data Sorting
    # Rationale: Sort by city_name first, then datetime to ensure chronological order
    # This is crucial for time series analysis and enables efficient data access patterns
    # In-place sorting saves memory and improves downstream processing performance
    df.sort_values(by=["city_name", "datetime"], inplace=True)

    # Step 4: Datetime Normalization
    # Rationale: Convert to UTC timezone and round to nearest hour for several reasons:
    # - UTC standardization eliminates timezone confusion across different data sources
    # - Hourly rounding aggregates sub-hourly measurements and reduces noise
    # - Creates consistent temporal granularity for time series forecasting
    # - Aligns with typical air quality monitoring and reporting standards
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.round("h")

    return df
