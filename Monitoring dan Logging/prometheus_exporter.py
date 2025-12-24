from prometheus_client import Counter, Histogram, start_http_server
import time

# === METRICS ===
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

def start_exporter():
    start_http_server(8001)
    print("Prometheus Exporter running on port 8001")

if __name__ == "__main__":
    start_exporter()
    while True:
        time.sleep(1)