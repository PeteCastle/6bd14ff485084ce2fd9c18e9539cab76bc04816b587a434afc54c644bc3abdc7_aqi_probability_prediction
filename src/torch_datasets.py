import hashlib
import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm

from src.constants import CACHE_DIR, POLLUTANT_COLUMNS


class AQIDataset(Dataset):
    def __init__(self, data, lookback, delay, min_index, max_index, step, mean, std):
        import time

        start_time = time.time()
        self.data = data
        self.lookback = lookback
        self.delay = delay
        self.min_index = min_index
        self.max_index = min(
            max_index if max_index is not None else len(data) - delay - 1,
            len(data) - delay - 1,
        )
        self.step = step
        self.mean = mean
        self.std = std

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        param_str = f"{self._data_hash()}_lb{lookback}_d{delay}_min{min_index}_max{self.max_index}_s{step}"
        samples_path = os.path.join(CACHE_DIR, f"{param_str}_samples.pt")
        targets_path = os.path.join(CACHE_DIR, f"{param_str}_targets.pt")
        metadata_path = os.path.join(CACHE_DIR, f"{param_str}_metadata.pkl")

        if all(os.path.exists(p) for p in [samples_path, targets_path, metadata_path]):
            self.samples = torch.load(samples_path)
            self.targets = torch.load(targets_path)

            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            self._timestamps = metadata["timestamps"]
            self.mean = metadata["mean"]
            self.std = metadata["std"]
        else:
            samples, targets, timestamps = [], [], []

            for idx in (pbar := tqdm(range(self.__len__()), disable=True)):
                i = self.min_index + idx + self.lookback
                indices = range(i - self.lookback, i, self.step)

                sample = self.data.iloc[
                    indices
                ].values  # shape: (lookback, num_features)
                target_row = self.data.iloc[i + self.delay]

                target = target_row[POLLUTANT_COLUMNS].values.astype(np.float32)
                samples.append(sample)
                targets.append(target)
                timestamps.append(pd.to_datetime(target_row.name))

            self.samples = torch.tensor(
                np.array(samples), dtype=torch.float32, device=self.device
            )
            self.targets = torch.tensor(
                np.array(targets), dtype=torch.float32, device=self.device
            )
            self._timestamps = timestamps

            torch.save(self.samples, samples_path)
            torch.save(self.targets, targets_path)
            with open(metadata_path, "wb") as f:
                pickle.dump(
                    {
                        "timestamps": self._timestamps,
                        "mean": self.mean,
                        "std": self.std,
                    },
                    f,
                )

    def _data_hash(self) -> str:
        content_hash = pd.util.hash_pandas_object(self.data, index=True).values
        combined = content_hash.tobytes()
        return hashlib.md5(combined).hexdigest()

    def __len__(self):
        return self.max_index - self.min_index - self.lookback + 1

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]

    def get_metadata(self, index: int):
        normalized_target = self.targets[index].cpu().numpy()
        original_target = normalized_target * self.std + self.mean
        return {
            "timestamp": self._timestamps[index],
            "original_target": original_target,
            "mean": self.mean,
            "std": self.std,
        }


class ConcatDatasetWithMetadata(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self._offsets = self._compute_offsets()

    def _compute_offsets(self):
        offsets = []
        offset = 0
        for d in self.datasets:
            offsets.append(offset)
            offset += len(d)
        return offsets

    def get_metadata(self, index: int):
        for i in range(len(self.datasets)):
            if index < self._offsets[i] + len(self.datasets[i]):
                local_index = index - self._offsets[i]
                return self.datasets[i].get_metadata(local_index)
        raise IndexError(
            "Index out of range for get_metadata in ConcatDatasetWithMetadata."
        )


def generate_datasets(
    dataset_df: pd.DataFrame, lookback=96, delay=24, step=1
) -> tuple[ConcatDatasetWithMetadata, ConcatDatasetWithMetadata]:
    training_datasets = []
    validation_datasets = []
    grouped_items = list(dataset_df.groupby("city_name"))
    random.shuffle(grouped_items)

    for city, group in (pbar := tqdm(grouped_items)):
        pbar.set_description(f"Processing {city}")
        group.sort_index(ascending=True, inplace=True)

        last_index = group.index[-1]
        one_year_ago = last_index - pd.DateOffset(months=6)
        min_index = group.index.get_loc(one_year_ago)

        group.drop(columns=["city_name"], inplace=True)
        training_df = group.iloc[:min_index].copy()
        validation_df = group.iloc[min_index:].copy()

        training_mean = training_df[POLLUTANT_COLUMNS].mean()
        training_std = training_df[POLLUTANT_COLUMNS].std()

        training_df[POLLUTANT_COLUMNS] = (
            training_df[POLLUTANT_COLUMNS] - training_mean
        ) / training_std
        validation_df[POLLUTANT_COLUMNS] = (
            validation_df[POLLUTANT_COLUMNS] - training_mean
        ) / training_std

        training_dataset = AQIDataset(
            training_df,
            lookback=lookback,
            delay=delay,
            min_index=0,
            max_index=None,
            step=step,
            mean=training_mean,
            std=training_std,
        )

        validation_dataset = AQIDataset(
            validation_df,
            lookback=lookback,
            delay=delay,
            min_index=0,
            max_index=None,
            step=step,
            mean=training_mean,
            std=training_std,
        )

        training_datasets.append(training_dataset)
        validation_datasets.append(validation_dataset)

    training_dataset = ConcatDatasetWithMetadata(training_datasets)
    validation_dataset = ConcatDatasetWithMetadata(validation_datasets)

    return training_dataset, validation_dataset
