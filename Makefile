.DEFAULT_GOAL := help

.PHONY: help
help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

.PHONY: install
install: ## Install dependencies
	uv sync

.PHONY: dev
dev: ## Install development dependencies
	uv sync --group dev

.PHONY: test
test: ## Run test suite
	uv run pytest

.PHONY: lint
lint: ## Run linter
	uv run ruff check src/ tests/

.PHONY: format
format: ## Format code
	uv run ruff format src/ tests/

.PHONY: typecheck
typecheck: ## Run type checker
	uv run mypy src/

.PHONY: check
check: lint typecheck test ## Run all checks

.PHONY: serve
serve: ## Run Delta ACP server
	uv run delta serve

.PHONY: clean
clean: ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
