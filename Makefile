POSTGRES_USER ?= trader
POSTGRES_DB   ?= trading

.PHONY: up down restart logs ps clean test lint fmt psql help deps \
	backtest sweep train trade dashboard verify-parity features

help: ## List all targets with descriptions
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

up: ## Start all services in detached mode
	docker compose up -d

down: ## Stop all services
	docker compose down

restart: down up ## Restart all services

logs: ## Follow logs for all services (last 100 lines)
	docker compose logs -f --tail=100

ps: ## Show running service status
	docker compose ps

clean: ## Destroy all containers and volumes (irreversible)
	@read -p "WARNING: This destroys all data volumes. Continue? [y/N] " confirm && \
	[ "$$confirm" = "y" ] && docker compose down -v || echo "Aborted."

test: ## Run test suite with coverage
	poetry run pytest

lint: ## Run ruff and mypy checks
	poetry run ruff check . && poetry run mypy .

fmt: ## Format code with ruff
	poetry run ruff format .

psql: ## Open psql shell in postgres container
	docker compose exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

deps: ## Sync venv after pyproject.toml changes
	poetry lock && poetry install

backtest: ## Run a backtest (pass flags via ARGS, e.g. ARGS="--walk-forward 1")
	poetry run python -m cli backtest $(ARGS)

sweep: ## Run a risk-param sweep (pass flags via ARGS)
	poetry run python -m cli sweep $(ARGS)

train: ## Train the disk-persisted ML model
	poetry run python -m cli train

trade: ## Run the live/paper trading bot (pass flags via ARGS, e.g. ARGS="--dry-run --once")
	poetry run python -m cli trade $(ARGS)

dashboard: ## Launch the Streamlit dashboard
	poetry run python -m cli dashboard

features: ## Run dbt deps/run/test to (re)build the feature tables
	poetry run python -m cli features

verify-parity: ## Replay the golden exp007 baseline and diff against it
	poetry run python -m cli verify-parity