#!/usr/bin/env bash
set -euo pipefail
bash scripts/generate_data.sh
bash scripts/train_retrieval.sh
bash scripts/build_index.sh
bash scripts/train_ranking.sh
bash scripts/serve_api.sh
