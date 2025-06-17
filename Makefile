# Makefile for Bank Churn Analysis Project
# Automates common development, testing, and deployment tasks

# Variables
PYTHON := python3
PIP := pip3
PROJECT_NAME := bank-churn-analysis
SRC_DIR := src
TEST_DIR := src/tests
DATA_DIR := data
MODELS_DIR := models
REPORTS_DIR := reports
DEPLOYMENT_DIR := deployment
NOTEBOOKS_DIR := notebooks

# Docker variables
DOCKER_IMAGE := $(PROJECT_NAME):latest
DOCKER_CONTAINER := $(PROJECT_NAME)-container
DOCKER_PORT := 8000

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

# Help target
.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)Bank Churn Analysis Project - Available Commands$(NC)"
	@echo "================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# Environment setup
.PHONY: install
install: ## Install project dependencies
	@echo "$(BLUE)Installing project dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

.PHONY: install-dev
install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov pytest-html pytest-xdist black flake8 mypy jupyter
	@echo "$(GREEN)Development dependencies installed successfully!$(NC)"

.PHONY: setup
setup: ## Setup project environment and directories
	@echo "$(BLUE)Setting up project environment...$(NC)"
	mkdir -p $(DATA_DIR)/raw $(DATA_DIR)/interim $(DATA_DIR)/processed
	mkdir -p $(MODELS_DIR)
	mkdir -p $(REPORTS_DIR)/figures $(REPORTS_DIR)/test_results
	mkdir -p $(NOTEBOOKS_DIR)
	mkdir -p $(DEPLOYMENT_DIR)/monitoring
	@echo "$(GREEN)Project environment setup complete!$(NC)"

.PHONY: clean
clean: ## Clean up temporary files and caches
	@echo "$(BLUE)Cleaning up temporary files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.log" -delete
	@echo "$(GREEN)Cleanup complete!$(NC)"

# Data pipeline
.PHONY: download-data
download-data: ## Download dataset from Kaggle
	@echo "$(BLUE)Downloading dataset from Kaggle...$(NC)"
	$(PYTHON) $(SRC_DIR)/data/download_data.py
	@echo "$(GREEN)Dataset downloaded successfully!$(NC)"

.PHONY: load-data
load-data: ## Load and validate raw data
	@echo "$(BLUE)Loading and validating raw data...$(NC)"
	$(PYTHON) $(SRC_DIR)/data/load_data.py
	@echo "$(GREEN)Data loaded and validated!$(NC)"

.PHONY: clean-data
clean-data: ## Clean and preprocess data
	@echo "$(BLUE)Cleaning and preprocessing data...$(NC)"
	$(PYTHON) $(SRC_DIR)/data/clean_data.py
	@echo "$(GREEN)Data cleaned and preprocessed!$(NC)"

.PHONY: create-features
create-features: ## Create engineered features
	@echo "$(BLUE)Creating engineered features...$(NC)"
	$(PYTHON) $(SRC_DIR)/features/create_features.py
	@echo "$(GREEN)Features created successfully!$(NC)"

.PHONY: data-pipeline
data-pipeline: download-data load-data clean-data create-features ## Run complete data pipeline
	@echo "$(GREEN)Complete data pipeline executed successfully!$(NC)"

# Model training
.PHONY: segment-customers
segment-customers: ## Run customer segmentation
	@echo "$(BLUE)Running customer segmentation...$(NC)"
	$(PYTHON) $(SRC_DIR)/models/segment.py
	@echo "$(GREEN)Customer segmentation complete!$(NC)"

.PHONY: train-churn
train-churn: ## Train churn prediction model
	@echo "$(BLUE)Training churn prediction model...$(NC)"
	$(PYTHON) $(SRC_DIR)/models/train_churn.py
	@echo "$(GREEN)Churn model training complete!$(NC)"

.PHONY: explain-model
explain-model: ## Generate model explanations
	@echo "$(BLUE)Generating model explanations...$(NC)"
	$(PYTHON) $(SRC_DIR)/models/explain.py
	@echo "$(GREEN)Model explanations generated!$(NC)"

.PHONY: train-models
train-models: segment-customers train-churn explain-model ## Train all models
	@echo "$(GREEN)All models trained successfully!$(NC)"

# Testing
.PHONY: test
test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	$(PYTHON) run_tests.py

.PHONY: test-unit
test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTHON) run_tests.py --unit

.PHONY: test-integration
test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTHON) run_tests.py --integration

.PHONY: test-api
test-api: ## Run API tests only
	@echo "$(BLUE)Running API tests...$(NC)"
	$(PYTHON) run_tests.py --api

.PHONY: test-performance
test-performance: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	$(PYTHON) run_tests.py --performance

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTHON) run_tests.py --coverage

.PHONY: test-fast
test-fast: ## Run fast tests (skip slow ones)
	@echo "$(BLUE)Running fast tests...$(NC)"
	$(PYTHON) run_tests.py --fast

# Code quality
.PHONY: lint
lint: ## Run code linting
	@echo "$(BLUE)Running code linting...$(NC)"
	flake8 $(SRC_DIR) --max-line-length=100 --ignore=E203,W503
	@echo "$(GREEN)Linting complete!$(NC)"

.PHONY: format
format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	black $(SRC_DIR) --line-length=100
	@echo "$(GREEN)Code formatting complete!$(NC)"

.PHONY: type-check
type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checking...$(NC)"
	mypy $(SRC_DIR) --ignore-missing-imports
	@echo "$(GREEN)Type checking complete!$(NC)"

.PHONY: quality
quality: format lint type-check ## Run all code quality checks
	@echo "$(GREEN)All code quality checks complete!$(NC)"

# Jupyter notebooks
.PHONY: notebook
notebook: ## Start Jupyter notebook server
	@echo "$(BLUE)Starting Jupyter notebook server...$(NC)"
	jupyter notebook --notebook-dir=$(NOTEBOOKS_DIR) --ip=0.0.0.0 --port=8888 --no-browser

.PHONY: lab
lab: ## Start JupyterLab server
	@echo "$(BLUE)Starting JupyterLab server...$(NC)"
	jupyter lab --notebook-dir=$(NOTEBOOKS_DIR) --ip=0.0.0.0 --port=8888 --no-browser

# API and deployment
.PHONY: run-api
run-api: ## Run the FastAPI application locally
	@echo "$(BLUE)Starting FastAPI application...$(NC)"
	cd $(DEPLOYMENT_DIR) && uvicorn app:app --host 0.0.0.0 --port $(DOCKER_PORT) --reload

.PHONY: build-docker
build-docker: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE) $(DEPLOYMENT_DIR)
	@echo "$(GREEN)Docker image built successfully!$(NC)"

.PHONY: run-docker
run-docker: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -d --name $(DOCKER_CONTAINER) -p $(DOCKER_PORT):8000 $(DOCKER_IMAGE)
	@echo "$(GREEN)Docker container started on port $(DOCKER_PORT)!$(NC)"

.PHONY: stop-docker
stop-docker: ## Stop and remove Docker container
	@echo "$(BLUE)Stopping Docker container...$(NC)"
	docker stop $(DOCKER_CONTAINER) || true
	docker rm $(DOCKER_CONTAINER) || true
	@echo "$(GREEN)Docker container stopped and removed!$(NC)"

.PHONY: docker-compose-up
docker-compose-up: ## Start all services with docker-compose
	@echo "$(BLUE)Starting services with docker-compose...$(NC)"
	cd $(DEPLOYMENT_DIR) && docker-compose up -d
	@echo "$(GREEN)All services started!$(NC)"

.PHONY: docker-compose-down
docker-compose-down: ## Stop all services with docker-compose
	@echo "$(BLUE)Stopping services with docker-compose...$(NC)"
	cd $(DEPLOYMENT_DIR) && docker-compose down
	@echo "$(GREEN)All services stopped!$(NC)"

.PHONY: docker-logs
docker-logs: ## View Docker container logs
	@echo "$(BLUE)Viewing Docker container logs...$(NC)"
	docker logs -f $(DOCKER_CONTAINER)

# Monitoring and health checks
.PHONY: health-check
health-check: ## Check API health
	@echo "$(BLUE)Checking API health...$(NC)"
	curl -f http://localhost:$(DOCKER_PORT)/health || echo "$(RED)API health check failed!$(NC)"

.PHONY: api-docs
api-docs: ## Open API documentation
	@echo "$(BLUE)Opening API documentation...$(NC)"
	open http://localhost:$(DOCKER_PORT)/docs

# Complete workflows
.PHONY: full-pipeline
full-pipeline: setup data-pipeline train-models test ## Run complete ML pipeline
	@echo "$(GREEN)Complete ML pipeline executed successfully!$(NC)"

.PHONY: deploy
deploy: build-docker run-docker health-check ## Build and deploy the application
	@echo "$(GREEN)Application deployed successfully!$(NC)"

.PHONY: ci
ci: install-dev quality test-coverage ## Run CI pipeline (install, quality checks, tests)
	@echo "$(GREEN)CI pipeline completed successfully!$(NC)"

.PHONY: dev-setup
dev-setup: install-dev setup ## Setup development environment
	@echo "$(GREEN)Development environment setup complete!$(NC)"

# Documentation
.PHONY: docs
docs: ## Generate project documentation
	@echo "$(BLUE)Generating project documentation...$(NC)"
	@echo "Documentation files:"
	@echo "- README.md: Project overview and setup instructions"
	@echo "- reports/project_summary.md: Comprehensive project report"
	@echo "- API docs: http://localhost:$(DOCKER_PORT)/docs (when API is running)"
	@echo "$(GREEN)Documentation references provided!$(NC)"

# Utility targets
.PHONY: status
status: ## Show project status
	@echo "$(BLUE)Project Status:$(NC)"
	@echo "=============="
	@echo "Data files:"
	@ls -la $(DATA_DIR)/ 2>/dev/null || echo "  No data directory found"
	@echo "\nModel files:"
	@ls -la $(MODELS_DIR)/ 2>/dev/null || echo "  No models directory found"
	@echo "\nReports:"
	@ls -la $(REPORTS_DIR)/ 2>/dev/null || echo "  No reports directory found"
	@echo "\nDocker containers:"
	@docker ps --filter name=$(DOCKER_CONTAINER) --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "  No Docker containers running"

.PHONY: requirements
requirements: ## Generate requirements.txt from current environment
	@echo "$(BLUE)Generating requirements.txt...$(NC)"
	$(PIP) freeze > requirements.txt
	@echo "$(GREEN)Requirements.txt updated!$(NC)"

.PHONY: check-deps
check-deps: ## Check if all dependencies are installed
	@echo "$(BLUE)Checking dependencies...$(NC)"
	$(PYTHON) run_tests.py --check-deps

# Cleanup targets
.PHONY: clean-data-files
clean-data-files: ## Remove all data files
	@echo "$(YELLOW)Warning: This will remove all data files!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf $(DATA_DIR)/*; \
		echo "\n$(GREEN)Data files removed!$(NC)"; \
	else \
		echo "\n$(BLUE)Operation cancelled.$(NC)"; \
	fi

.PHONY: clean-models
clean-models: ## Remove all model files
	@echo "$(YELLOW)Warning: This will remove all trained models!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf $(MODELS_DIR)/*; \
		echo "\n$(GREEN)Model files removed!$(NC)"; \
	else \
		echo "\n$(BLUE)Operation cancelled.$(NC)"; \
	fi

.PHONY: clean-all
clean-all: clean clean-data-files clean-models stop-docker ## Clean everything (use with caution)
	@echo "$(GREEN)Complete cleanup finished!$(NC)"

# Development shortcuts
.PHONY: dev
dev: dev-setup data-pipeline train-models ## Quick development setup and training
	@echo "$(GREEN)Development environment ready!$(NC)"

.PHONY: quick-test
quick-test: test-fast ## Run quick tests
	@echo "$(GREEN)Quick tests completed!$(NC)"

.PHONY: demo
demo: full-pipeline deploy api-docs ## Run complete demo (pipeline + deployment)
	@echo "$(GREEN)Demo setup complete! Check API docs at http://localhost:$(DOCKER_PORT)/docs$(NC)"

# Show available make targets
.PHONY: list
list: help ## Alias for help