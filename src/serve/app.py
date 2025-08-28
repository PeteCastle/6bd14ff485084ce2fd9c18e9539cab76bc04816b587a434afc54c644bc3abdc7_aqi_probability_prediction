from fastapi import FastAPI, HTTPException
import logging
from typing import Dict, Union, List
from src.utils import setup_mlflow_tracking
import mlflow
import mlflow.tracking
import pandas as pd
import numpy as np
import torch
from src.constants import DATASET_DIR
from pydantic import BaseModel, validator
from mlflow.tracking import MlflowClient


dataset = pd.read_parquet(DATASET_DIR / "processed" / "features.parquet")
CITY_COLS = [col for col in dataset.columns if col.startswith("city_") and col != "city_name"]
CITY_NAMES = [col.replace("city_", "") for col in CITY_COLS]
COLUMNS = [
    'components.co', 'components.no', 'components.no2', 'components.o3',
    'components.so2', 'components.pm2_5', 'components.pm10', 'components.nh3',
    *CITY_COLS, 'city_name'
]

class AQIRequest(BaseModel):
    co: float
    no: float
    no2: float
    o3: float
    so2: float
    pm2_5: float
    pm10: float
    nh3: float
    city: str

    @validator("city")
    def city_must_be_valid(cls, v):
        if v not in CITY_NAMES:
            raise ValueError(f"Invalid city '{v}'. Allowed values are: {', '.join(CITY_NAMES)}")
        return v

    def to_dataframe(self) -> pd.DataFrame:
        # One-hot encoding for city columns
        city_one_hot = {f"city_{c}": 1 if self.city == c else 0 for c in CITY_NAMES}
        # Compose the row as a dict
        row = {
            'components.co': self.co,
            'components.no': self.no,
            'components.no2': self.no2,
            'components.o3': self.o3,
            'components.so2': self.so2,
            'components.pm2_5': self.pm2_5,
            'components.pm10': self.pm10,
            'components.nh3': self.nh3,
            **{f"city_{c}": city_one_hot[f"city_{c}"] for c in CITY_NAMES},
            'city_name': self.city
        }
        # Ensure all columns present (missing city columns default to 0)
        for col in COLUMNS:
            if col not in row:
                row[col] = 0
        # Arrange the row in the correct order
        df = pd.DataFrame([[row[col] for col in COLUMNS]], columns=COLUMNS)
        df.drop(columns=['city_name'], inplace=True)
        return df

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

setup_mlflow_tracking()

app = FastAPI(
    title="AQI Probability Prediction API",
    description="API for predicting Air Quality Index probabilities using Mixture Density Networks",
    version="1.0.0"
)

@app.get("/model", summary="Get model schema and hyperparameters", description="Returns the AQIRequest schema and the MLflow model hyperparameters.")
def get_model_info():
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {
        "schema": AQIRequest.schema(),
        "hyperparameters": app.state.model_params,
        "importance": None
    }

def _load_best_model():
    client = MlflowClient()
    model_name = "champion"

    mv = client.get_latest_versions(model_name, stages=["Production"])
    if not mv:
        raise RuntimeError(f"No Production version found for {model_name}")

    version = mv[0].version
    model_uri = f"models:/{model_name}/{version}"

    model = mlflow.pyfunc.load_model(model_uri)

    model_parameters = client.get_model_version(name=model_name, version=version).tags

    # Store model + parameters in FastAPI state
    app.state.model = model
    app.state.model_name = model_name
    app.state.model_version = version
    app.state.model_params = model_parameters   # guaranteed field

@app.post("/reload_model")
async def reload_model():
    try:
        info = _load_best_model()
        return {"status": "success", **info}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    
@app.on_event("startup")
async def startup_load():
    print("Loading model on startup…")
    _load_best_model()


@app.get("/", response_model=Dict[str, str], summary="Root health check", description="Returns API information and documentation links.")
async def root():
    """Root endpoint"""
    return {
        "message": "AQI Probability Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.post("/predict", summary="Predict air quality components", description="Accepts one or multiple AQIRequest objects with air component values and city, validates inputs, and returns predictions mapped to component keys.")
async def predict(aqi_requests: Union[AQIRequest, List[AQIRequest]]):
    # Accept either a single AQIRequest or a list of them
    if isinstance(aqi_requests, AQIRequest):
        requests_list = [aqi_requests]
    elif isinstance(aqi_requests, list):
        requests_list = aqi_requests
    else:
        raise HTTPException(status_code=400, detail="Invalid input format")

    dfs = [req.to_dataframe() for req in requests_list]
    input_df = pd.concat(dfs, ignore_index=True)
    x = input_df.to_numpy().astype(np.float32)
    x = x.reshape(x.shape[0], 1, x.shape[1])  # (batch, seq_len=1, features)

    model = app.state.model

    prediction = model.predict(x)
    output_keys = [
        "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"
    ]
    predictions_list = []
    for row in prediction:
        # Map each output dimension to the corresponding key
        pred_dict = {key: float(row[idx]) for idx, key in enumerate(output_keys) if idx < len(row)}
        predictions_list.append(pred_dict)

    return {"predictions": predictions_list}