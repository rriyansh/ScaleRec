# ScaleRec — Two-Tower Neural Recommender System

Production-grade two-stage recommendation engine for e-commerce.

Combines a **Two-Tower neural retrieval model** (PyTorch) with **FAISS ANN search** for candidate generation, and an **XGBoost ranker** for CTR prediction. Served via FastAPI (dev) or TorchServe (prod), with Redis caching.

---

## Architecture

```
User Request
     │
     ▼
┌──────────────┐     ┌────────────────┐     ┌───────────────┐
│  User Tower  │────▶│  FAISS ANN     │────▶│  XGBoost      │
│  (PyTorch)   │     │  Retrieval     │     │  Ranker       │
└──────────────┘     └────────────────┘     └───────────────┘
                                                    │
                                                    ▼
                                              Top-K Items
```

**Stage 1 — Retrieval:** Two-Tower network produces L2-normalized user & item embeddings, trained with in-batch softmax. FAISS stores item embeddings for sub-linear ANN search (top-50 candidates).

**Stage 2 — Ranking:** XGBoost classifier scores each candidate using `[user_emb | item_emb | dot_product]` features. Returns top-K ranked items.

---

## Project Structure

```
scalerec/
├── config/              # YAML configs (default + test)
├── data/                # Raw + processed data
├── models_store/        # Trained artifacts (.pt, .bin, .npy, .json)
├── scripts/             # Shell scripts for pipeline steps
├── src/
│   ├── data/            # Synthetic data + preprocessing
│   ├── models/          # Two-Tower + Ranker
│   ├── retrieval/       # Training + FAISS index
│   ├── ranking/         # Feature engineering + XGBoost training
│   ├── serving/         # FastAPI, TorchServe handler, Redis cache
│   └── utils/           # IO helpers
├── tests/               # Unit + integration tests
├── docs/                # Architecture + API reference
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

---

## Quickstart

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
chmod +x scripts/*.sh
make train
```

This runs:
1. Synthetic data generation
2. Preprocessing (normalization)
3. Two-Tower training
4. FAISS index build
5. XGBoost ranker training

### 3. Serve the API

```bash
make serve
```

Open: [http://localhost:8000/recommend/42](http://localhost:8000/recommend/42)

Or visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive Swagger UI.

### 4. Run tests

```bash
make test
```

---

## Step-by-Step Commands

```bash
# Generate synthetic data
bash scripts/generate_data.sh

# Train Two-Tower model
bash scripts/train_retrieval.sh

# Build FAISS index
bash scripts/build_index.sh

# Train XGBoost ranker
bash scripts/train_ranking.sh

# Serve with FastAPI
bash scripts/serve_api.sh
```

Or use the Makefile:

```bash
make data              # Generate + preprocess
make train-retrieval   # Train Two-Tower
make build-index       # Build FAISS
make train-rank        # Train XGBoost
make train             # All of the above
make serve             # Start API
make test              # Run pytest
make clean             # Remove artifacts
```

---

## Docker

```bash
docker-compose up --build
```

This starts:
- **scalerec** on `http://localhost:8000`
- **redis** on `localhost:6379` (for caching)

---

## Configuration

All parameters live in `config/config.yaml`:

```yaml
model:
  embedding_dim: 32
  hidden_dim: 64
  temperature: 0.07

training:
  batch_size: 256
  learning_rate: 0.001
  epochs: 5

faiss:
  index_type: "FlatIP"      # Use "IVFPQ" for large-scale
  nlist: 100
  nprobe: 10

ranking:
  n_candidates: 50          # Stage 1 output
  top_k: 10                 # Final recommendations
  xgb_params:
    n_estimators: 100
    max_depth: 4
    learning_rate: 0.1

serving:
  redis_host: "localhost"
  redis_port: 6379
  cache_ttl: 300
```

For fast test runs, use `config/config.test.yaml` (small data, 1 epoch):

```bash
CONFIG_PATH=config/config.test.yaml python -m src.data.synthetic
```

---

## Environment Variables

Copy `.env.example` to `.env` and adjust as needed:

```env
APP_ENV=development
LOG_LEVEL=INFO
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_TTL=300
API_HOST=0.0.0.0
API_PORT=8000
CONFIG_PATH=config/config.yaml
```

---

## Pipeline Overview

### Data

- **Synthetic generator** (`src/data/synthetic.py`) creates users, items, and interactions. Click probability is correlated with feature similarity so the model has a learnable signal.
- **Preprocessing** (`src/data/preprocess.py`) applies `StandardScaler` to features and splits train/test.

Swap synthetic data for real data (MovieLens, e-commerce logs) by replacing `src/data/synthetic.py` — the rest of the pipeline stays the same.

### Training

- **Two-Tower** (`src/retrieval/train.py`) uses in-batch negatives: for each positive `(user, item)` pair, all other items in the batch act as negatives. Loss: cross-entropy over similarity matrix with temperature scaling.
- **FAISS index** (`src/retrieval/build_index.py`) encodes all items once, stores in an Inner Product index.
- **Ranker** (`src/ranking/train.py`) samples positives + FAISS-retrieved candidates, labels them, and trains XGBoost for binary CTR prediction.

### Serving

- **Recommender** (`src/serving/recommender.py`) loads all artifacts once and exposes `recommend(user_id)`.
- **FastAPI** (`src/serving/api.py`) wraps it in `/recommend/{user_id}` with Redis caching.
- **TorchServe handler** (`src/serving/handler.py`) provides the same pipeline for production model servers.

---

## API Reference

Base URL (local): `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs` (Swagger UI)  
Alternative: `http://localhost:8000/redoc`

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

## Production Notes

### Scaling FAISS

For millions of items, switch `faiss.index_type` to `IVFPQ`:

```yaml
faiss:
  index_type: "IVFPQ"
  nlist: 4096       # ~sqrt(n_items)
  m: 32             # subquantizers
  nbits: 8
  nprobe: 32
```

Trade-off: **~95% recall** at 10–100x faster queries vs. FlatIP.

### Scaling Training

- Use **GPU**: the model automatically uses `cuda` if available.
- Increase `batch_size` for better in-batch negatives (256 → 1024+ works well).
- For extreme scale, explore **PyTorch DistributedDataParallel** + sharded DataLoader.

### TorchServe Deployment

```bash
bash scripts/serve_torchserve.sh
```

Model will be archived into `model_store/scalerec.mar` and served on port `8080`/`8081`.

---

## Testing

```bash
pytest -v                        # all tests
pytest tests/test_models.py -v   # single file
pytest -k "two_tower" -v         # by keyword
```

Test suite covers:
- Config loading
- Data generation + preprocessing
- Model forward pass + loss
- Full retrieval pipeline (train + index)
- Full ranking pipeline (feature build + train)
- End-to-end API (`/health`, `/recommend/{id}`, error paths)

---

## Troubleshooting

**`ModuleNotFoundError: src`**
Run from the project root: `cd scalerec && python -m src.data.synthetic`

**`Redis unavailable; cache disabled`**
Start Redis: `docker run -p 6379:6379 redis:7-alpine` or `docker-compose up redis`

**FAISS import error**
`pip install faiss-cpu==1.8.0`. On Apple Silicon, use `conda install -c conda-forge faiss-cpu`.

**CUDA out of memory**
Reduce `training.batch_size` in config.

**Ranker AUC stuck at ~0.5**
Increase `faiss.nprobe` or use `FlatIP` index during training to ensure positives are retrieved.

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

## License

MIT — see `LICENSE`.

---

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Make changes, run `make test`
4. Open a PR

---

## References

- Covington et al., *Deep Neural Networks for YouTube Recommendations* (RecSys 2016)
- Yi et al., *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations* (RecSys 2019)
- Johnson et al., *Billion-scale similarity search with GPUs* (FAISS)
