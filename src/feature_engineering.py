import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from .constants import DATASET_DIR
from .data_preprocessing import get_preprocessed_data
from src.torch_datasets import generate_datasets
from src.torch_datasets import ConcatDatasetWithMetadata


def dataset_to_df(dataset: ConcatDatasetWithMetadata, max_samples: int = 2500) -> pd.DataFrame:
    rows = []
    step = max(1, len(dataset) // max_samples)
    # raise NotImplementedError
    for i in tqdm(range(0, len(dataset), step), desc="Converting dataset to DataFrame"):
        x, y = dataset[i]  # assuming (features, target)
        meta = dataset.get_metadata(i)
        row = {}
        # flatten features if needed
        for j, val in enumerate(x.flatten()):
            row[f"f{j}"] = float(val)
        if hasattr(y, "shape") and getattr(y, "ndim", 0) > 0 and y.shape[0] > 1:
            for k, val in enumerate(y.flatten()):
                row[f"target_{k}"] = float(val)  # force scalar
        else:
            try:
                row["target"] = float(y.item())
            except Exception:
                row["target"] = float(y)
        rows.append(row)
    return pd.DataFrame(rows)

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


    if not os.path.exists(DATASET_DIR / "processed" / f"val_dataset.parquet"):
        _, val_dataset = generate_datasets(
            dataset_df, lookback=96, delay=24, step=1, drift_strength=0
        )
        val_dataset = dataset_to_df(val_dataset)
        val_dataset.to_parquet(
            DATASET_DIR / "processed" / f"val_dataset.parquet", index=False
        )
        del val_dataset

    if not os.path.exists(DATASET_DIR / "processed" / f"drifted_val_dataset.parquet"):
        _, drifted_val_dataset = generate_datasets(
            dataset_df, lookback=96, delay=24, step=1, drift_strength=0.1
        )

        drifted_val_dataset = dataset_to_df(drifted_val_dataset)
        drifted_val_dataset.to_parquet(
            DATASET_DIR / "processed" / f"drifted_val_dataset.parquet", index=False
        )
        del drifted_val_dataset
    
    return dataset_df
