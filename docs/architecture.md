# ScaleRec Architecture

## Two-Stage Recommender

### Stage 1 — Retrieval
- **Two-Tower neural network** produces L2-normalized user & item embeddings
- Trained with **in-batch softmax** (contrastive loss)
- **FAISS** stores item embeddings for sub-linear ANN search
- Returns top-N candidates (default: 50)

### Stage 2 — Ranking
- **XGBoost** classifier scores each candidate
- Features: `[user_emb | item_emb | dot_product]`
- Returns top-K ranked items (default: 10)

### Serving
- **FastAPI** for direct HTTP access
- **TorchServe** for production model server
- **Redis** for caching per-user recommendations
