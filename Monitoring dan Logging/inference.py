from fastapi import FastAPI
from pydantic import BaseModel
import requests
import time

from prometheus_client import Counter, Histogram, make_asgi_app

app = FastAPI()

# === METRICS ===
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds"
)

REQUEST_ERRORS = Counter(
    "prediction_errors_total",
    "Total number of prediction errors"
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# === SCHEMA ===
class PredictRequest(BaseModel):
    features: list[float]

# === MLflow Serve Endpoint ===
MLFLOW_SERVE_URL = "http://127.0.0.1:5001/invocations"

@app.post("/predict")
def predict(request: PredictRequest):
    start = time.time()

    try:
        payload = {
            "inputs": [request.features]
        }

        response = requests.post(
            MLFLOW_SERVE_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        REQUEST_COUNT.inc()
        REQUEST_LATENCY.observe(time.time() - start)

        return response.json()

    except Exception:
        REQUEST_ERRORS.inc()
        raise