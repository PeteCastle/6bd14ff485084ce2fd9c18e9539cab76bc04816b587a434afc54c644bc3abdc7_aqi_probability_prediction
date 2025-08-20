import numpy as np
import pandas as pd
from tqdm import tqdm

from .constants import DATASET_DIR
from .data_preprocessing import get_preprocessed_data


def get_feature_engineered_data(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        df = get_preprocessed_data()

    filled_dfs = []
    for city, group in tqdm(df.groupby("city_name")):
        group = group.set_index("datetime")
        full_index = pd.date_range(
            start=group.index.min(), end=group.index.max(), freq="h"
        )
        group = group.groupby(level=0).mean(numeric_only=True)
        group = group.reindex(full_index)

        group.interpolate(method="linear", inplace=True)
        group["city_name"] = city  # Re-add city name after reindex
        filled_dfs.append(group)

    filled_dfs = pd.concat(filled_dfs)

    city_column = filled_dfs["city_name"].copy()
    dataset_df = pd.get_dummies(filled_dfs, prefix="city", dtype=np.int8)
    dataset_df["city_name"] = city_column

    dataset_df.to_csv(
        DATASET_DIR / "processed" / f"feature_engineered_data.csv", index=True
    )

    return dataset_df
