# tests/test_features_parquet.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# tests/conftest.py
import pytest
import pandas as pd
from src.constants import DATASET_DIR
from src.features.transform import get_feature_engineered_data


@pytest.fixture(scope="session")
def features_path():
    return DATASET_DIR / "processed" / "features.parquet"


@pytest.fixture(scope="session")
def features_df(features_path):
    if not features_path.exists():
        print(f"{features_path} does not exist, generating...")
        get_feature_engineered_data()
    df = pd.read_parquet(features_path)
    yield df


def test_features_parquet_not_empty(features_df):
    assert not features_df.empty
    assert len(features_df) > 100


def test_features_columns(features_df):
    required = [
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
    for col in required:
        assert col in features_df.columns, f"missing {col}"


def test_features_no_missing_values(features_df):
    assert not features_df.isna().any().any(), "NaNs present"


def test_features_value_ranges(features_df):
    cols = [
        "components.co",
        "components.no",
        "components.no2",
        "components.o3",
        "components.so2",
        "components.pm2_5",
        "components.pm10",
        "components.nh3",
    ]
    for c in cols:
        negatives = features_df[features_df[c] < 0]
        if not negatives.empty:
            print(f"\nNegative values in {c}:")
            print(negatives[[c, "city_name"]].head(10))  # limit for readability
        assert (features_df[c] >= 0).all(), f"{c} has negatives"


def test_datetime_integrity(features_df):
    # sorted within city
    by_city = features_df.groupby("city_name")
    for city, g in by_city:
        dt = pd.to_datetime(g.index, utc=True)
        assert dt.is_monotonic_increasing, f"datetime not sorted in {city}"
        # hourly step or at least no duplicates (post-interp file may gap if source sparse)
        assert not dt.duplicated().any(), f"duplicate timestamps in {city}"


def test_reasonable_upper_bounds(features_df):
    bounds = {
        "components.co": 10000,
        "components.no": 1000,
        "components.no2": 1000,
        "components.o3": 1000,
        "components.so2": 1000,
        "components.pm2_5": 2000,
        "components.pm10": 3000,
        "components.nh3": 1000,
    }
    for c, ub in bounds.items():
        assert (features_df[c] <= ub).all(), f"{c} exceeded bound {ub}"


def test_dtypes(features_df):
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
    for c in pollutant_cols:
        assert pd.api.types.is_numeric_dtype(features_df[c]), f"{c} not numeric"
    assert features_df["city_name"].dtype == object or pd.api.types.is_string_dtype(
        features_df["city_name"]
    )
