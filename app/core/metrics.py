from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)


class Metrics:
    """Prometheus metrics, kept in a per-app registry so that several
    apps (as in the tests) never register the same metric twice."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()

        self.http_requests = Counter(
            "http_requests_total",
            "HTTP requests by route and status.",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.http_request_duration = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration by route.",
            ["method", "route"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.inferences = Counter(
            "inference_total",
            "Image inferences by outcome.",
            ["outcome"],
            registry=self.registry,
        )
        self.inference_stage_duration = Histogram(
            "inference_stage_duration_seconds",
            "Inference duration by stage (decode, inference, total).",
            ["stage"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.detections = Histogram(
            "inference_detections",
            "Detections per inference.",
            buckets=(0, 1, 2, 5, 10, 20, 50, 100),
            registry=self.registry,
        )
        self.model_loaded = Gauge(
            "model_loaded",
            "1 when the YOLO model is loaded.",
            registry=self.registry,
        )
