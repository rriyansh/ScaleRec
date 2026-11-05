# ScaleRec: Two-Tower Neural Recommender Engine

## System Overview
ScaleRec is a production-grade recommendation engine built to handle millions of items at sub-millisecond latencies. It implements a decoupled two-tower deep learning architecture for rapid candidate generation, followed by a multi-stage tree-based ranking pipeline.

## System Architecture
1. **Retrieval Phase (Candidate Generation):** 
   - Utilizes a Two-Tower Neural Network (User Tower and Item Tower). Both towers project categorical and numerical features into a shared dense embedding space.
   - Item embeddings are pre-computed offline and indexed in FAISS (Facebook AI Similarity Search). 
   - At inference, the User Tower generates the user embedding in real-time, and a dot-product similarity search in FAISS instantly retrieves the top 1,000 candidates from a pool of millions.
2. **Ranking Phase:**
   - The top 1,000 candidates are passed to a heavy ranking model (XGBoost). This model utilizes dense interaction features (e.g., click-through rates, session context) to score and re-rank the candidates, outputting the final top 10 items for the user.
3. **Serving Layer:**
   - Deployed via TorchServe with a Redis caching layer to store pre-computed item embeddings and frequently requested user recommendations to minimize database hits.

## Engineering Trade-offs
- **Cold Start Problem:** New items lack interaction data for the ranking model. Implemented a fallback heuristic utilizing content-based filtering (item metadata embeddings) until sufficient interaction data is collected.

## Tech Stack
- **Deep Learning:** PyTorch, TorchServe
- **Vector Retrieval:** FAISS
- **Ranking:** XGBoost
- **Infrastructure:** Redis, FastAPI
