#!/usr/bin/env bash
set -euo pipefail
python -m src.data.synthetic
python -m src.data.preprocess
