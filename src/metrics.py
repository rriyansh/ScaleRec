"""Prometheus metrics instrumentation."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response


HTTP_REQUESTS_TOTAL = Counter(
    "scalerec_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

HTTP_REQUEST_DURATION = Histogram(
    "scalerec_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path", "status"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

HTTP_REQUEST_SIZE = Histogram(
    "scalerec_http_request_size_bytes",
    "HTTP request body size",
    buckets=(128, 512, 1024, 4096, 16384, 65536, 262144, 1048576),
)

RECOMMENDATIONS_TOTAL = Counter(
    "scalerec_recommendations_total",
    "Total recommendation requests",
    ["status"],
)

RECOMMENDATION_LATENCY = Histogram(
    "scalerec_recommendation_latency_seconds",
    "Recommendation pipeline latency",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

CACHE_HITS = Counter("scalerec_cache_hits_total", "Redis cache hits")
CACHE_MISSES = Counter("scalerec_cache_misses_total", "Redis cache misses")
CACHE_ERRORS = Counter("scalerec_cache_errors_total", "Redis cache errors")

FAISS_SEARCH_LATENCY = Histogram(
    "scalerec_faiss_search_seconds",
    "FAISS search latency",
    buckets=(0.0005, 0.001, 0.005, 0.01, 0.05, 0.1),
)

RANKER_LATENCY = Histogram(
    "scalerec_ranker_seconds",
    "XGBoost ranking latency",
    buckets=(0.0005, 0.001, 0.005, 0.01, 0.05),
)

MODEL_LOADED = Gauge(
    "scalerec_model_loaded", "1 if model is loaded, else 0"
)

CIRCUIT_STATE = Gauge(
    "scalerec_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=half-open, 2=open)",
    ["name"],
)


async def metrics_endpoint(request: Request) -> Response:
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type="text/plain")
