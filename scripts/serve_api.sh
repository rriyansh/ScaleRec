#!/usr/bin/env bash
set -euo pipefail
exec uvicorn src.serving.api:app \
    --host "${API_HOST:-0.0.0.0}" \
    --port "${API_PORT:-8000}" \
    --workers "${API_WORKERS:-1}"
