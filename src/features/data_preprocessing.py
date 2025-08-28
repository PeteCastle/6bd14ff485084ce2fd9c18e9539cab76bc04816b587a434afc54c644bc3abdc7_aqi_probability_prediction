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
    files = glob.glob(str(DATASET_DIR / "raw" / "aqi" / "*" / "*.csv"))
    if len(files) == 0:
        from src.data.ingest import ingest_data

        ingest_data()
    files = glob.glob(str(DATASET_DIR / "raw" / "aqi" / "*" / "*.csv"))

    for file in (pbar := tqdm(files)):
        pbar.set_description(f"Reading {file}")
        print(f"Loading data from {file}")

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
    if df is None:
        df = get_raw_data()

    df = df[df["city_name"].isin(CITY_NAMES)]
    df = df[
        [
            "datetime",
            "components.co",
            "components.no",
            "components.no2",
            "components.o3",
            "components.so2",
            "components.pm2_5",
            "components.pm10",
            "components.nh3",
            "city_name",
        ]
    ]

    df.sort_values(by=["city_name", "datetime"], inplace=True)

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.round("h")

    # Step 5: Clamp negative pollutant values to zero
    pollutant_cols = [
        "components.co",
        "components.no",
        "components.no2",
        "components.o3",
        "components.so2",
        "components.pm2_5",
        "components.pm10",
        "components.nh3",
    ]
    df[pollutant_cols] = df[pollutant_cols].clip(lower=0)

    return df
