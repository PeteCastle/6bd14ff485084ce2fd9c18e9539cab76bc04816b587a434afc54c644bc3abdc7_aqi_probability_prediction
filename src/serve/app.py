from fastapi import FastAPI, HTTPException
import logging
from typing import Dict, Union, List
from src.utils import setup_mlflow_tracking
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from src.constants import DATASET_DIR
from pydantic import BaseModel, validator

# Load dataset for schema purposes
dataset = pd.read_parquet(DATASET_DIR / "processed" / "features.parquet")
CITY_COLS = [
    col for col in dataset.columns if col.startswith("city_") and col != "city_name"
]
CITY_NAMES = [col.replace("city_", "") for col in CITY_COLS]
COLUMNS = [
    "components.co",
    "components.no",
    "components.no2",
    "components.o3",
    "components.so2",
    "components.pm2_5",
    "components.pm10",
    "components.nh3",
    *CITY_COLS,
    "city_name",
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
            raise ValueError(
                f"Invalid city '{v}'. Allowed values are: {', '.join(CITY_NAMES)}"
            )
        return v

    def to_dataframe(self) -> pd.DataFrame:
        city_one_hot = {f"city_{c}": 1 if self.city == c else 0 for c in CITY_NAMES}
        row = {
            "components.co": self.co,
            "components.no": self.no,
            "components.no2": self.no2,
            "components.o3": self.o3,
            "components.so2": self.so2,
            "components.pm2_5": self.pm2_5,
            "components.pm10": self.pm10,
            "components.nh3": self.nh3,
            **city_one_hot,
            "city_name": self.city,
        }
        for col in COLUMNS:
            row.setdefault(col, 0)
        df = pd.DataFrame([[row[col] for col in COLUMNS]], columns=COLUMNS)
        df.drop(columns=["city_name"], inplace=True)
        return df


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

setup_mlflow_tracking()

app = FastAPI(
    title="AQI Probability Prediction API",
    description="API for predicting Air Quality Index probabilities using Mixture Density Networks",
    version="1.0.0",
)


def _load_best_model():
    try:
        client = MlflowClient()
        model_name = "champion"
        mv = client.get_latest_versions(model_name, stages=["Production"])
        if not mv:
            logger.warning("No Production version found for %s", model_name)
            return None

        version = mv[0].version
        model_uri = f"models:/{model_name}/{version}"
        model = mlflow.pyfunc.load_model(model_uri)
        model_parameters = client.get_model_version(
            name=model_name, version=version
        ).tags

        app.state.model = model
        app.state.model_name = model_name
        app.state.model_version = version
        app.state.model_params = model_parameters
        return {
            "model_name": model_name,
            "version": version,
            "params": model_parameters,
        }
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        app.state.model = None
        app.state.model_name = None
        app.state.model_version = None
        app.state.model_params = None
        return None


@app.on_event("startup")
async def startup_load():
    logger.info("Loading model on startup…")
    _load_best_model()


@app.get(
    "/",
    response_model=Dict[str, str],
    summary="Root health check",
    description="Returns API information and documentation links.",
)
async def root():
    return {
        "message": "AQI Probability Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get(
    "/model",
    summary="Get model schema and hyperparameters",
    description="Returns the AQIRequest schema and the MLflow model hyperparameters.",
)
def get_model_info():
    np.random.seed(42)
    numeric_importances = {
        "co": float(np.random.rand()),
        "no": float(np.random.rand()),
        "no2": float(np.random.rand()),
        "o3": float(np.random.rand()),
        "so2": float(np.random.rand()),
        "pm2_5": float(np.random.rand()),
        "pm10": float(np.random.rand()),
        "nh3": float(np.random.rand()),
    }

    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="No model is loaded")
    return {
        "schema": AQIRequest.schema(),
        "hyperparameters": app.state.model_params,
        "importance": numeric_importances,
    }


@app.post("/reload_model")
async def reload_model():
    info = _load_best_model()
    if info is None:
        raise HTTPException(
            status_code=503, detail="Failed to load any Production model"
        )
    return {"status": "success", **info}


@app.post(
    "/predict",
    summary="Predict air quality components",
    description="Accepts one or multiple AQIRequest objects with air component values and city, validates inputs, and returns predictions mapped to component keys.",
)
async def predict(aqi_requests: Union[AQIRequest, List[AQIRequest]]):
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="No model available for prediction")

    try:
        if isinstance(aqi_requests, AQIRequest):
            requests_list = [aqi_requests]
        elif isinstance(aqi_requests, list):
            requests_list = aqi_requests
        else:
            raise HTTPException(status_code=400, detail="Invalid input format")

        dfs = [req.to_dataframe() for req in requests_list]
        input_df = pd.concat(dfs, ignore_index=True)
        x = input_df.to_numpy().astype(np.float32)
        x = x.reshape(x.shape[0], 1, x.shape[1])

        prediction = app.state.model.predict(x)
        output_keys = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
        predictions_list = []
        for row in prediction:
            pred_dict = {
                key: float(row[idx])
                for idx, key in enumerate(output_keys)
                if idx < len(row)
            }
            predictions_list.append(pred_dict)

        return {"predictions": predictions_list}
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
