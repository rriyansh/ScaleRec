# ScaleRec API Reference

Base URL (local): `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs` (Swagger UI)  
Alternative: `http://localhost:8000/redoc`

---

## Endpoints

### `GET /health`

Liveness probe. Always returns `200` once the service is up.

**Response**

```json
{
  "status": "ok"
}
```

**Status codes**

| Code | Meaning |
|------|---------|
| 200  | Service healthy |

---

### `GET /recommend/{user_id}`

Returns top-K personalized recommendations for a user.

**Path parameters**

| Name      | Type    | Required | Description                     |
|-----------|---------|----------|---------------------------------|
| `user_id` | integer | Yes      | User ID (must be in `[0, N_users)`) |

**Query parameters**

None.

**Response**

```json
{
  "user_id": 42,
  "recommendations": [
    { "item_id": 123, "score": 0.8712 },
    { "item_id": 456, "score": 0.7431 },
    { "item_id": 789, "score": 0.6218 }
  ],
  "cached": false
}
```

| Field             | Type    | Description |
|-------------------|---------|-------------|
| `user_id`         | integer | Echo of the requested user |
| `recommendations` | array   | Ordered list (highest score first). Length ≤ `top_k` from config |
| `recommendations[].item_id` | integer | Item ID |
| `recommendations[].score`   | float   | CTR probability from XGBoost, in `[0, 1]` |
| `cached`          | boolean | `true` if served from Redis cache, `false` if computed fresh |

**Status codes**

| Code | Meaning |
|------|---------|
| 200  | Success |
| 400  | `user_id` out of range |
| 503  | Service not ready (models still loading) |

**Example — success**

```bash
curl http://localhost:8000/recommend/42
```

```json
{
  "user_id": 42,
  "recommendations": [
    {"item_id": 123, "score": 0.8712},
    {"item_id": 456, "score": 0.7431}
  ],
  "cached": false
}
```

**Example — invalid user**

```bash
curl -i http://localhost:8000/recommend/999999
```

```
HTTP/1.1 400 Bad Request
Content-Type: application/json

{"detail": "user_id 999999 out of range"}
```

**Example — cached hit**

```bash
curl http://localhost:8000/recommend/42
```

```json
{
  "user_id": 42,
  "recommendations": [ ... ],
  "cached": true
}
```

---

## Behavior Notes

### Caching

- Results are cached in **Redis** per `user_id` for `serving.cache_ttl` seconds (default: **300**).
- Cache key format: `rec:{user_id}`
- If Redis is unavailable at startup, the service **still runs** — caching is simply disabled. A single warning is logged.
- To force fresh computation, either wait for TTL expiry or flush Redis: `redis-cli FLUSHDB`.

### Latency Profile

On a fresh request (cache miss):

| Stage                         | Typical latency (CPU) |
|-------------------------------|-----------------------|
| User tower forward pass       | ~2 ms                 |
| FAISS search (top-50)         | ~1–5 ms               |
| XGBoost scoring (50 items)    | ~1 ms                 |
| **Total (p50)**               | **~5–10 ms**          |

Cached request: **< 1 ms** (single Redis GET).

### Error Handling

- All endpoints return JSON (`Content-Type: application/json`)
- Errors use FastAPI's standard envelope: `{ "detail": "<message>" }`
- No authentication is enforced — add an API gateway or middleware for production

---

## Adding Authentication (Production)

Wrap the router or app with middleware:

```python
from fastapi import FastAPI, Depends, HTTPException, Header

def verify_token(x_api_key: str = Header(...)):
    if x_api_key != "your-secret":
        raise HTTPException(status_code=401, detail="Unauthorized")

app = FastAPI()
# Apply to specific routes:
# @app.get("/recommend/{user_id}", dependencies=[Depends(verify_token)])
```

---

## Rate Limiting (Production)

Add `slowapi`:

```bash
pip install slowapi
```

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/recommend/{user_id}")
@limiter.limit("100/minute")
def recommend(user_id: int, request: Request):
    ...
```

---

## Batch Recommendations (Optional Extension)

The current API is single-user. To support batch requests, add a `POST /recommend/batch` endpoint:

```python
from pydantic import BaseModel

class BatchRequest(BaseModel):
    user_ids: list[int]

@app.post("/recommend/batch")
def recommend_batch(req: BatchRequest) -> dict:
    results = {}
    for uid in req.user_ids:
        results[uid] = recommender.recommend(uid)
    return results
```

---

## Monitoring

Recommended metrics to export (Prometheus format):

| Metric | Type | Description |
|--------|------|-------------|
| `scalerec_requests_total` | Counter | Total requests, labeled by `endpoint` and `status` |
| `scalerec_latency_seconds` | Histogram | End-to-end latency |
| `scalerec_cache_hits_total` | Counter | Redis cache hits |
| `scalerec_cache_misses_total` | Counter | Redis cache misses |
| `scalerec_faiss_search_seconds` | Histogram | FAISS query latency |
| `scalerec_ranker_seconds` | Histogram | XGBoost inference latency |

Add via `prometheus-fastapi-instrumentator`:

```bash
pip install prometheus-fastapi-instrumentator
```

```python
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

Metrics available at `GET /metrics`.

---

## Versioning

Current version: `0.1.0`

The API is not yet versioned by URL (`/v1/...`). When breaking changes are introduced, migrate to:

```
GET /v1/recommend/{user_id}
GET /v2/recommend/{user_id}
```

Deprecated endpoints should return a `Deprecation` header until removed.

---

## OpenAPI / Schema

FastAPI auto-generates the OpenAPI schema:

```
http://localhost:8000/openapi.json
```

Use this schema with:

- **Postman** — import via URL
- **OpenAPI Generator** — generate client SDKs in any language
- **Swagger UI** — already served at `/docs`

Example — generate a Python client:

```bash
openapi-generator-cli generate \
  -i http://localhost:8000/openapi.json \
  -g python \
  -o scalerec-client
```

---

## Quick Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness probe |
| `/recommend/{user_id}` | GET | Top-K recommendations for a user |
| `/docs` | GET | Interactive Swagger UI |
| `/redoc` | GET | ReDoc documentation |
| `/openapi.json` | GET | OpenAPI schema |
