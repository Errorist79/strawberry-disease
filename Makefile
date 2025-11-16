.PHONY: help install download-dataset train-model docker-build docker-up docker-down migrate test clean

help:
	@echo "Strawberry Disease Detection System - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          - Install Python dependencies"
	@echo "  make download-dataset - Download Kaggle dataset"
	@echo "  make train-model      - Train YOLOv8 model"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build Docker images"
	@echo "  make docker-up        - Start all services"
	@echo "  make docker-down      - Stop all services"
	@echo "  make docker-logs      - Show service logs"
	@echo ""
	@echo "Database:"
	@echo "  make migrate          - Run database migrations"
	@echo "  make migrate-create   - Create new migration"
	@echo ""
	@echo "Development:"
	@echo "  make run-api          - Run API locally"
	@echo "  make run-collector    - Run camera collector locally"
	@echo "  make run-inference    - Run inference service locally"
	@echo "  make test             - Run tests"
	@echo "  make clean            - Clean generated files"

install:
	pip install -r requirements.txt

download-dataset:
	@echo "Downloading dataset from Kaggle..."
	./scripts/download_dataset.sh

train-model:
	@echo "Training YOLO model..."
	python scripts/train_model.py --epochs 100 --batch 16

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

migrate:
	docker-compose exec api alembic upgrade head

migrate-create:
	@read -p "Enter migration message: " msg; \
	docker-compose exec api alembic revision --autogenerate -m "$$msg"

run-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-collector:
	python -m src.services.camera_collector

run-inference:
	python -m src.services.inference_service

run-aggregator:
	python -m src.services.risk_aggregator

run-notifier:
	python -m src.services.telegram_notifier

test:
	pytest tests/ -v --cov=src --cov-report=html

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
