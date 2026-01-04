from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import time

from prometheus_exporter import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    PREDICTION_COUNT
)

MODEL_URI = "file:///D:/Kuliah/Asah/Project/MSML/Monitoring dan Logging/mlruns/1/models/m-084de3830b4444138fc3ddd1dcfdd223/artifacts"

model = mlflow.pyfunc.load_model(MODEL_URI)

app = FastAPI()

class InputData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/predict")
def predict(data: InputData):
    start_time = time.time()
    REQUEST_COUNT.inc()

    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)

    latency = time.time() - start_time
    REQUEST_LATENCY.observe(latency)
    PREDICTION_COUNT.inc()

    return {
        "prediction": int(prediction[0]),
        "latency": latency
    }
