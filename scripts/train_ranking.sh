#!/usr/bin/env bash
set -euo pipefail
python -m src.ranking.train
python -m src.models.versioning
