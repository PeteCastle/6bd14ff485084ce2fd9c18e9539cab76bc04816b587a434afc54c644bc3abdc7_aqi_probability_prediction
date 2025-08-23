import kagglehub
import os
from src.constants import DATASET_DIR
from pathlib import Path
import shutil


def download_and_extract_data():
    path = kagglehub.dataset_download(
        "bwandowando/philippine-major-cities-air-quality-data"
    )
    print("Path to dataset files:", path)

    raw_v2_dir = Path(DATASET_DIR) / "raw"
    year_2023_dir = raw_v2_dir / "aqi" / "2023"
    year_2024_dir = raw_v2_dir / "aqi" / "2024"

    year_2023_dir.mkdir(parents=True, exist_ok=True)
    year_2024_dir.mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)

                year = file[:4]

                if year == "2023":
                    destination = year_2023_dir / file
                    shutil.copy2(file_path, destination)
                elif year == "2024":
                    destination = year_2024_dir / file
                    shutil.copy2(file_path, destination)

    path = kagglehub.dataset_download(
        "bwandowando/philippine-cities-air-quality-index-data-2025"
    )

    year_2025_dir = raw_v2_dir / "aqi" / "2025"
    year_2025_dir.mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower() == "cities.csv":
                file_path = os.path.join(root, file)
                shutil.copy2(file_path, raw_v2_dir / "cities.csv")

            if file.endswith(".csv"):
                file_path = os.path.join(root, file)

                year = file[:4]

                if year == "2025":
                    destination = year_2025_dir / file
                    shutil.copy2(file_path, destination)


if __name__ == "__main__":
    download_and_extract_data()
