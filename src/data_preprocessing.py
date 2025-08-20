import glob

import pandas as pd
from tqdm import tqdm

from src.constants import CITY_NAMES, DATASET_DIR


def get_raw_data() -> pd.DataFrame:
    dfs = []
    for file in (
        pbar := tqdm(glob.glob(str(DATASET_DIR / "raw" / "aqi" / "*" / "*.csv")))
    ):
        pbar.set_description(f"Reading {file}")
        df = pd.read_csv(file)
        dfs.append(df)

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
    return df
