# ScaleRec Architecture

## Overview

ScaleRec is a **two-stage recommendation system** designed for e-commerce personalization. It separates the problem into:

1. **Retrieval (Candidate Generation)** — narrow down millions of items to a few hundred relevant candidates fast
2. **Ranking (Personalization)** — score and order those candidates precisely for the user

This separation is the industry standard (YouTube, Netflix, Amazon, Pinterest) because ranking every item with an expensive model is infeasible at scale.

---

## High-Level Flow

```
                    ┌────────────────────────────────┐
                    │       User Request             │
                    │   (user_id + context)          │
                    └───────────────┬────────────────┘
                                    │
                                    ▼
              ┌─────────────────────────────────────────┐
              │  STAGE 1: RETRIEVAL                     │
              │                                         │
              │   User Features ──▶ User Tower          │
              │                       │                 │
              │                       ▼                 │
              │                 User Embedding          │
              │                       │                 │
              │                       ▼                 │
              │              FAISS ANN Search           │
              │             (over item index)           │
              │                       │                 │
              │                       ▼                 │
              │            Top-N Candidates             │
              │              (e.g., N=50)               │
              └───────────────────┬─────────────────────┘
                                  │
                                  ▼
              ┌─────────────────────────────────────────┐
              │  STAGE 2: RANKING                       │
              │                                         │
              │   For each (user, candidate) pair:      │
              │   Build feature vector ──▶ XGBoost      │
              │                       │                 │
              │                       ▼                 │
              │               CTR Probability           │
              │                       │                 │
              │                       ▼                 │
              │            Sort + Take Top-K            │
              │              (e.g., K=10)               │
              └───────────────────┬─────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────────┐
                    │   Ranked Recommendations     │
                    │   [{item_id, score}, ...]    │
                    └──────────────────────────────┘
```

---

## Stage 1 — Retrieval (Two-Tower + FAISS)

### Why Two-Tower?

Classic collaborative filtering scales poorly (O(users × items) scoring). Two-Tower decomposes the problem:

- **User Tower** encodes user features → dense embedding
- **Item Tower** encodes item features → dense embedding
- **Score** = dot product (or cosine) between the two embeddings

Because towers are **independent** at inference time, we can:

1. Precompute all item embeddings offline → store in FAISS
2. Compute one user embedding online → query FAISS for nearest items

This gives **sub-linear** retrieval time (log-scale or constant, depending on index type).

### Architecture Diagram

```
       User Features                        Item Features
            │                                    │
            ▼                                    ▼
   ┌─────────────────┐                  ┌─────────────────┐
   │   User Tower    │                  │   Item Tower    │
   │                 │                  │                 │
   │  Linear(d→h)    │                  │  Linear(d→h)    │
   │       │         │                  │       │         │
   │       ▼         │                  │       ▼         │
   │    ReLU         │                  │    ReLU         │
   │       │         │                  │       │         │
   │       ▼         │                  │       ▼         │
   │   Dropout       │                  │   Dropout       │
   │       │         │                  │       │         │
   │       ▼         │                  │       ▼         │
   │  Linear(h→e)    │                  │  Linear(h→e)    │
   │       │         │                  │       │         │
   │       ▼         │                  │       ▼         │
   │  L2 Normalize   │                  │  L2 Normalize   │
   └────────┬────────┘                  └────────┬────────┘
            │                                    │
            ▼                                    ▼
      User Embedding                     Item Embedding
         (dim=32)                           (dim=32)
            │                                    │
            └────────────────┬───────────────────┘
                             │
                             ▼
                      Dot Product
                             │
                             ▼
                     Similarity Score
```

### Training Loss — In-Batch Softmax

For a batch of `B` positive `(user, item)` pairs:

```
        User emb (B × d)               Item emb (B × d)
              │                              │
              └──────────┬───────────────────┘
                         │
                         ▼
                Similarity Matrix (B × B)
                    ┌─────────────┐
                    │ P P N N N N │
                    │ N P P N N N │
                    │ N N P P N N │
                    │ N N N P P N │
                    │ N N N N P P │
                    └─────────────┘
                    P = positive, N = negative
                         │
                         ▼
              Cross-Entropy (diagonal = label)
```

- **Diagonal** entries are positives (matching user-item pairs)
- **Off-diagonal** are negatives (naturally in-batch)
- This is efficient: no separate negative sampling required

**Temperature scaling** (`τ = 0.07` by default) sharpens the softmax distribution.

### FAISS Index

- **Index type:** `IndexFlatIP` (dev) / `IndexIVFPQ` (prod)
- **Metric:** Inner Product (equivalent to cosine on L2-normalized vectors)
- **Size:** One vector per item (e.g., 1M items × 32 dims × 4 bytes ≈ 128 MB)

Production index types trade recall for speed:

| Index Type | Recall | Build Time | Query Time | Memory |
|------------|--------|------------|------------|--------|
| `FlatIP`   | 100%   | Fast       | Slow (linear) | High |
| `IVFFlat`  | ~95%   | Medium     | Fast       | Medium |
| `IVFPQ`    | ~90-95%| Slow       | Very fast  | Low  |

---

## Stage 2 — Ranking (XGBoost)

### Why a Separate Ranker?

Retrieval optimizes for **recall** (get all relevant items into the candidate set). It uses cheap dot products.

Ranking optimizes for **precision** (order candidates correctly). It can afford to use richer features since it only scores `N=50` candidates:

- Cross-features (user × item interactions)
- Contextual signals (time of day, device, session history)
- Business rules (price match, category preference, recency)

XGBoost is the industry favorite because:
- Fast inference (~0.1 ms per candidate)
- Handles missing values natively
- Interpretable feature importance
- Robust to scale differences

### Feature Engineering

For each `(user, candidate_item)` pair:

```
┌─────────────────────────────────────────────────┐
│  Ranker Features                                │
├─────────────────────────────────────────────────┤
│  user_embedding    (32 dims)   ← from User Tower│
│  item_embedding    (32 dims)   ← from Item Tower│
│  dot_product       (1 dim)     ← similarity     │
│  ─────────────────────────────────────────────  │
│  [Optional extensions]                          │
│  price_difference                               │
│  category_match                                 │
│  brand_affinity                                 │
│  recency_score                                  │
│  user_session_features                          │
└─────────────────────────────────────────────────┘
                    │
                    ▼
              XGBoost Classifier
                    │
                    ▼
             CTR Probability [0, 1]
```

### Training Data

1. Take all positive `(user, item)` pairs from training set
2. For each user, run FAISS retrieval to get top-N candidates
3. Label candidates: `1` if the candidate is the true positive, else `0`
4. Train XGBoost as binary classifier with `binary:logistic` objective
5. Evaluate with **AUC** and **NDCG@K**

---

## Serving Architecture

### Development Mode — FastAPI

```
┌──────────┐     HTTP      ┌──────────────┐
│  Client  │──────────────▶│   FastAPI    │
└──────────┘               │              │
                           │  /recommend  │
                           │              │
                           │  ┌─────────┐ │
                           │  │ Cache   │ │
                           │  │ (Redis) │ │
                           │  └────┬────┘ │
                           │       │      │
                           │       ▼      │
                           │  ┌─────────┐ │
                           │  │Recommender││
                           │  │Pipeline  ││
                           │  └─────────┘ │
                           └──────────────┘
```

- Single process
- Loads models into memory at startup (via `lifespan` context manager)
- Redis caching optional (graceful fallback)
- Good for: local dev, small deployments, testing

### Production Mode — TorchServe

```
┌──────────┐     HTTP      ┌──────────────┐
│  Client  │──────────────▶│  TorchServe  │
└──────────┘               │   Frontend   │
                           └──────┬───────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
              ┌─────────┐   ┌─────────┐   ┌─────────┐
              │ Worker  │   │ Worker  │   │ Worker  │
              │  +      │   │  +      │   │  +      │
              │ Model   │   │ Model   │   │ Model   │
              └────┬────┘   └────┬────┘   └────┬────┘
                   │             │             │
                   └─────────────┼─────────────┘
                                 │
                                 ▼
                          ┌─────────────┐
                          │    Redis    │
                          │    Cache    │
                          └─────────────┘
```

- Multiple workers (horizontal scaling)
- Built-in batching
- Model versioning
- Metrics endpoint (Prometheus compatible)
- Good for: production, high traffic, A/B testing

---

## Data Flow

### Offline (Training)

```
Raw Data ──▶ Preprocessing ──▶ Feature Store
                                      │
                                      ▼
                          ┌──────────────────────┐
                          │  Train Two-Tower     │
                          └──────────┬───────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  Encode Items        │
                          │  (Item Tower only)   │
                          └──────────┬───────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  Build FAISS Index   │
                          └──────────┬───────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  Build Ranker        │
                          │  Training Set        │
                          └──────────┬───────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  Train XGBoost       │
                          └──────────┬───────────┘
                                     │
                                     ▼
                            Model Artifacts
                         (.pt, .bin, .npy, .json)
```

### Online (Serving)

```
Request (user_id)
      │
      ▼
┌──────────────┐
│ Cache Check  │── HIT ──▶ Return cached result
└──────┬───────┘
       │ MISS
       ▼
┌──────────────┐
│ User Tower   │  (~2 ms)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ FAISS Search │  (~1-5 ms)
└──────┬───────┘
       │  top-50 candidates
       ▼
┌──────────────┐
│ Feature Build│  (~0.5 ms)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ XGBoost Score│  (~1 ms)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Sort + Top-K │  (<0.1 ms)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Cache Write  │
└──────┬───────┘
       │
       ▼
   Return top-10
```

**Total p50 latency (cache miss): ~5–10 ms**  
**Cache hit: < 1 ms**

---

## Design Decisions & Trade-offs

### Why In-Batch Softmax vs. BCE?

| Loss | Pros | Cons |
|------|------|------|
| **BCE** (binary cross-entropy) | Simple, can add explicit negatives | Poor gradients with many negatives, requires careful sampling |
| **In-Batch Softmax** ✅ | Efficient, uses all other batch items as negatives, strong gradients | Requires large batch size (≥256), sensitive to class imbalance within batch |

We chose in-batch softmax because it's the standard approach for two-tower retrieval at scale (YouTube, Pinterest, Spotify all use variants).

### Why XGBoost vs. a Neural Ranker?

| Ranker | Pros | Cons |
|--------|------|------|
| **Neural (DLRM)** | Can learn complex interactions | Slow inference, heavy memory, complex to train |
| **XGBoost** ✅ | Fast (~0.1 ms), interpretable, robust to missing data, small memory | Limited to tree-structured feature interactions |

For a first-production ranker, XGBoost is a proven sweet spot. Upgrade to a neural ranker only when you have enough traffic to justify the complexity.

### Why FAISS vs. Alternatives?

| Library | Pros | Cons |
|---------|------|------|
| **FAISS** ✅ | Battle-tested, GPU support, many index types, extreme scale | C++ bindings (not pure Python) |
| Annoy | Simple, memory-mapped | Slower, no GPU, limited index types |
| ScaNN | Fast, Google-backed | Less community, harder tuning |
| HNSWlib | Great recall/speed trade-off | No product quantization (memory heavy) |

FAISS is the default choice for two-tower systems.

---

## Scaling Considerations

### When to Switch Index Types

| Item count | Recommended Index |
|------------|-------------------|
| < 100K     | `FlatIP` |
| 100K – 10M | `IVFFlat` or `HNSW` |
| 10M – 1B   | `IVFPQ` |
| > 1B       | `IVFPQ` + sharding |

### Distributed Training

For large datasets:

- **DataParallel** (single node, multi-GPU) — easy, ~2x speedup
- **DistributedDataParallel (DDP)** (multi-node) — recommended for billions of interactions
- **Sharded data loading** — split items across workers

### Model Serving at Scale

- **Horizontal scaling:** Add TorchServe workers
- **Caching:** Redis for hot users (power-law distribution → 20% of users = 80% of traffic)
- **Model versioning:** Keep multiple `.mar` files for A/B testing
- **Circuit breakers:** Fall back to popularity baseline if models fail

---

## Failure Modes & Mitigations

| Failure | Mitigation |
|---------|-----------|
| FAISS index missing/corrupt | Fall back to popularity-based recommendations |
| Redis down | Serve uncached (graceful degradation) |
| Model file missing | Fail fast at startup (health check returns 503) |
| Slow user tower | Cache user embeddings per session |
| XGBoost latency spike | Reduce `n_candidates` dynamically |
| Cold-start users | Use side features (age, region) in user tower |

---

## Future Improvements

1. **Add contextual features** — time of day, device, session history to both towers
2. **Neural ranker (DLRM / DCN)** — replace XGBoost once traffic justifies complexity
3. **Real-time feature store** — integrate Redis / Feast for fresher user signals
4. **Multi-objective ranking** — score for CTR + CVR + revenue simultaneously
5. **A/B testing framework** — shadow deployment of new models
6. **Model monitoring** — drift detection on embeddings and scores
7. **Sequential models** — replace user tower with transformer-based session encoder (SASRec, BERT4Rec)

---

## References

- **Covington et al., 2016** — *Deep Neural Networks for YouTube Recommendations* (RecSys)
- **Yi et al., 2019** — *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations* (RecSys)
- **Johnson et al., 2017** — *Billion-scale similarity search with GPUs* (FAISS paper)
- **Guo et al., 2017** — *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*
- **Huang et al., 2019** — *Embedding-based Retrieval in Facebook Search*
