# hybrid-rag Makefile
# Usage: make <target>
#   make fetch       - Download all docs from llms.txt sources
#   make clean       - Strip MDX/JSX from raw markdown
#   make chunk       - Chunk cleaned docs (fixed + heading-aware)
#   make embed       - Embed chunks and upsert to Qdrant
#   make retrieve    - Test retrieval on a sample question
#   make generate    - Test generation on a sample question
#   make eval-gen    - Run full evaluation (fixed vs heading vs baseline)
#   make up          - Start Qdrant (docker-compose up -d)
#   make down        - Stop Qdrant
#   make test        - Run unit tests
#   make lint        - Run ruff + mypy
#   make format      - Run black + ruff --fix

.PHONY: fetch clean chunk embed retrieve generate eval-gen up down test lint format help

# Default target
help:
	@echo "hybrid-rag - Portfolio RAG over MCP, A2A, LangGraph"
	@echo ""
	@echo "Targets:"
	@echo "  make up          - Start Qdrant"
	@echo "  make down        - Stop Qdrant"
	@echo "  make fetch       - Download docs from llms.txt"
	@echo "  make clean       - Clean raw markdown (strip MDX/JSX)"
	@echo "  make chunk       - Chunk docs (fixed + heading-aware)"
	@echo "  make embed       - Embed chunks to Qdrant"
	@echo "  make retrieve    - Test retrieval"
	@echo "  make generate    - Test generation"
	@echo "  make eval-gen    - Full evaluation pipeline"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run ruff + mypy"
	@echo "  make format      - Format code"

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

up:
	docker-compose up -d
	@echo "Waiting for Qdrant to be ready..."
	@until curl -sf http://localhost:6333/health > /dev/null; do sleep 1; done
	@echo "Qdrant ready at http://localhost:6333"

down:
	docker-compose down

# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------

# Fetch all sources
fetch: up
	python -m scripts.fetch_docs

# Clean raw markdown
clean:
	python -m scripts.clean_md

# Chunk (both strategies)
chunk:
	python -m scripts.chunk

# Embed and index
embed: up
	python -m scripts.embed

# Full pipeline: fetch -> clean -> chunk -> embed
pipeline: fetch clean chunk embed

# ---------------------------------------------------------------------------
# Retrieval & Generation (for quick testing)
# ---------------------------------------------------------------------------

retrieve:
	@read -p "Question: " q; \
	python -m scripts.retrieve "$$q"

generate:
	@read -p "Question: " q; \
	python -m scripts.generate "$$q"

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

eval-gen:
	python -m scripts.evaluate

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	ruff check .
	mypy hybrid_rag/

format:
	ruff check --fix .
	black .

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

# Show Qdrant collections info
collections:
	curl -s http://localhost:6333/collections | jq .

# Count chunks per collection
counts:
	@for c in fixed_mcp fixed_a2a fixed_langgraph heading_mcp heading_a2a heading_langgraph; do \
		echo -n "$$c: "; \
		curl -s -X POST http://localhost:6333/collections/hybrid_rag_$$c/points/count | jq .result.count; \
	done

# Clean all generated data (careful!)
clean-data:
	rm -rf data/raw/* data/processed/* data/eval/results/*
	rm -f data/manifest.json

# Install dependencies
install:
	pip install -e ".[dev]"
	pre-commit install

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

env-check:
	@echo "Checking required env vars..."
	@test -n "$$GROQ_API_KEY" || (echo "ERROR: GROQ_API_KEY not set" && exit 1)
	@test -n "$$OPENAI_API_KEY" || (echo "ERROR: OPENAI_API_KEY not set" && exit 1)
	@echo "All required env vars present"

# Load .env if exists
ifneq (,$(wildcard .env))
include .env
export
endif