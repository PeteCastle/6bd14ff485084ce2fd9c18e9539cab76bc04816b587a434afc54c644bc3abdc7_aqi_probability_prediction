import os
from pathlib import Path

__all__ = [
    "DATASET_DIR",
    "CACHE_DIR",
    "MODELS_DIR",
    "TQDM_DISABLE",
    "CITY_NAMES",
    "POLLUTANT_COLUMNS",
]

# Directories
DATASET_DIR = Path(__file__).parent.parent / "data"
if not DATASET_DIR.exists():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path(__file__).parent.parent / "cache"
if not CACHE_DIR.exists():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = Path(__file__).parent.parent / "models"
if not MODELS_DIR.exists():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path(__file__).parent.parent / "reports"
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Environment Variables
TQDM_DISABLE = os.getenv("TQDM_DISABLE", "0") == "1"


CITY_NAMES = [
    "Caloocan City",
    "Las Piñas",
    "Makati City",
    "Malabon",
    "Mandaluyong City",
    "Manila",
    "Marikina",  # Data not Available
    "Muntinlupa",  # Data not Available
    "Navotas",
    "Paranaque City",
    "Pasay",  # Data not Available
    "Pasig",
    "Pateros",  # Data not Available
    "Quezon City",
    "San Juan",
    "Taguig",
    "Valenzuela",
]

POLLUTANT_COLUMNS = [
    "components.co",
    "components.no",
    "components.no2",
    "components.o3",
    "components.so2",
    "components.pm2_5",
    "components.pm10",
]
