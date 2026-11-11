.PHONY: help install install-dev data train-retrieval build-index train-rank train \
        serve serve-prod test test-cov lint format typecheck security clean docker \
        docker-prod benchmark precommit

PY := python
PIP := pip

help:
	@echo "ScaleRec — Available targets:"
	@echo "  install         Install production dependencies"
	@echo "  install-dev     Install dev dependencies"
	@echo "  data            Generate synthetic data + preprocess"
	@echo "  train-retrieval Train Two-Tower model"
	@echo "  build-index     Build FAISS index"
	@echo "  train-rank      Train XGBoost ranker"
	@echo "  train           Run full training pipeline"
	@echo "  serve           Start FastAPI (dev)"
	@echo "  serve-prod      Start FastAPI (production workers)"
	@echo "  test            Run tests"
	@echo "  test-cov        Run tests with coverage"
	@echo "  lint            Run ruff"
	@echo "  format          Run black + ruff --fix"
	@echo "  typecheck       Run mypy"
	@echo "  security        Run security checks (ruff S rules)"
	@echo "  benchmark       Run API benchmark"
	@echo "  docker          Build docker image"
	@echo "  docker-prod     Build production docker image"
	@echo "  clean           Remove artifacts"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	pre-commit install

data:
	$(PY) -m src.data.synthetic
	$(PY) -m src.data.preprocess

train-retrieval:
	$(PY) -m src.retrieval.train

build-index:
	$(PY) -m src.retrieval.build_index

train-rank:
	$(PY) -m src.ranking.train

train: train-retrieval build-index train-rank

serve:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --workers 4

test:
	pytest

test-cov:
	pytest --cov=src --cov-report=term-missing --cov-report=html

lint:
	ruff check src tests

format:
	black src tests
	ruff check --fix src tests

typecheck:
	mypy src

security:
	ruff check --select S src

benchmark:
	bash scripts/benchmark.sh

docker:
	docker build -t scalerec:dev .

docker-prod:
	docker build -f Dockerfile.prod -t scalerec:prod .

precommit:
	pre-commit run --all-files

clean:
	rm -rf models_store/*.pt models_store/*.pkl models_store/*.json \
	       models_store/*.bin models_store/*.npy models_store/*.version
	rm -rf data/processed/*
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
