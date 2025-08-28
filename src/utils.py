import requests
import mlflow
import os
from src.constants import MODELS_DIR

def setup_mlflow_tracking():
    """
    Set up MLflow tracking URI with fallback mechanism.
    Try to connect to the MLflow server first, if it fails, fall back to local SQLite.
    """
    primary_uri = "http://mlflow:5000"
    fallback_uri = f"sqlite:///{MODELS_DIR}/mlflow.db"
    
    try:
        response = requests.get(f"{primary_uri}/health", timeout=5)
        if response.status_code == 200:
            mlflow.set_tracking_uri(primary_uri)
            print(f"✅ Connected to MLflow server at {primary_uri}")
            return primary_uri
    except (requests.exceptions.RequestException, requests.exceptions.Timeout):
        pass
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    mlflow.set_tracking_uri(fallback_uri)
    print(f"⚠️ MLflow server not available, using local SQLite database: {fallback_uri}")
    return fallback_uri