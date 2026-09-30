.PHONY: install dev qdrant ingest ask eval eval-retrieval ablation serve test lint docker-up docker-down

install:
	pip install -e .

dev:
	pip install -e ".[dev,local]"

qdrant:
	docker compose up -d qdrant

ingest:
	regrag ingest

ask:
	regrag ask "$(Q)"

eval:
	regrag eval --name $(or $(NAME),run)

eval-retrieval:
	regrag eval --retrieval-only --name $(or $(NAME),retrieval)

ablation:
	regrag ablation --ingest

serve:
	regrag serve

test:
	pytest --cov=regrag --cov-report=term-missing

lint:
	ruff check src tests
	ruff format --check src tests

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down
