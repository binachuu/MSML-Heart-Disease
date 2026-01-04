from prometheus_client import Counter, Histogram, start_http_server

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests"
)

PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total predictions made"
)

REQUEST_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency"
)

# expose metrics di :8001
start_http_server(8001)