from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import time

from prometheus_client import Counter, Histogram, make_asgi_app

app = FastAPI()

# ===== LOAD MODEL =====
model = joblib.load("model_rf.pkl")
scaler = joblib.load("scaler.pkl")

# ===== PROMETHEUS METRICS =====
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

# ===== PROMETHEUS ENDPOINT =====
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ===== REQUEST SCHEMA =====
class PredictRequest(BaseModel):
    features: list[float]

# ===== ROUTES =====
@app.get("/")
def home():
    return {"message": "Heart Disease Model is running"}

@app.post("/predict")
def predict(request: PredictRequest):
    start = time.time()
    try:
        X = np.array(request.features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)

        REQUEST_COUNT.inc()
        REQUEST_LATENCY.observe(time.time() - start)

        return {"prediction": int(prediction[0])}

    except Exception as e:
        REQUEST_ERRORS.inc()
        raise e