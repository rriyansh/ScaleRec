#!/usr/bin/env bash
set -euo pipefail
torch-model-archiver \
    --model-name scalerec \
    --version "${MODEL_VERSION:-1.0}" \
    --serialized-file models_store/two_tower.pt \
    --handler src/serving/handler.py \
    --extra-files "models_store/faiss_index.bin,models_store/item_embeddings.npy,models_store/xgb_ranker.json,config/config.yaml" \
    --export-path model_store \
    --force
exec torchserve --start --ncs --model-store model_store --models scalerec.mar
