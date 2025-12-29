from prometheus_client import start_http_server, Counter, Histogram
import time

prediction_requests_total = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds"
)

prediction_errors_total = Counter(
    "prediction_errors_total",
    "Total number of prediction errors"
)

if __name__ == "__main__":
    start_http_server(8001)
    print("Prometheus exporter running on port 8001")
    while True:
        time.sleep(5)