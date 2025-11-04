.PHONY: help setup install-deps install-torch install-flash-attn install-verl clean all

# Use bash for all commands
SHELL := /bin/bash

# Virtual environment path
VENV_DIR := .venv

# Default target
help: ## Show available targets
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

all: setup ## Complete setup (recommended)

setup: check-uv create-venv install-basic-deps install-torch install-flash-attn install-project install-verl install-final-deps ## Complete project setup
	@echo "✓ Setup complete! Activate the environment with: source $(VENV_DIR)/bin/activate"

check-uv: ## Check if uv is installed
	@which uv > /dev/null || (echo "Error: uv is not installed. Please install it first." && exit 1)
	@echo "✓ uv is available"

create-venv: check-uv ## Create virtual environment
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment..."; \
		uv venv $(VENV_DIR); \
		echo "✓ Virtual environment created"; \
	else \
		echo "✓ Virtual environment already exists"; \
	fi

install-basic-deps: create-venv ## Install basic dependencies
	@echo "Installing basic dependencies..."
	@source $(VENV_DIR)/bin/activate && uv pip install packaging wheel setuptools
	@echo "✓ Basic dependencies installed"

install-torch: create-venv ## Install PyTorch with CUDA support
	@echo "Installing PyTorch with CUDA support..."
	@source $(VENV_DIR)/bin/activate && uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
	@echo "✓ PyTorch with CUDA installed"

install-flash-attn: create-venv ## Install flash-attn
	@echo "Installing flash-attn (this may take a while)..."
	@source $(VENV_DIR)/bin/activate && uv pip install flash-attn==2.7.4.post1 --no-build-isolation
	@echo "✓ flash-attn installed"

install-project: create-venv ## Install project dependencies
	@echo "Installing project dependencies..."
	@source $(VENV_DIR)/bin/activate && uv pip install -e .
	@echo "✓ Project dependencies installed"

install-verl: create-venv ## Install verl package with vLLM support
	@echo "Installing verl package with vLLM support..."
	@cd verl && source ../$(VENV_DIR)/bin/activate && uv pip install -e ".[vllm]" && cd ..
	@echo "✓ verl package with vLLM installed"

install-final-deps: create-venv ## Install final dependencies
	@echo "Installing final dependencies..."
	@source $(VENV_DIR)/bin/activate && uv pip install msgspec deepspeed trl
	@echo "✓ Final dependencies installed"

clean: ## Clean up build artifacts
	@echo "Cleaning up..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/
	@echo "✓ Cleaned up build artifacts"

clean-all: clean ## Remove virtual environment
	@echo "Removing virtual environment..."
	@rm -rf $(VENV_DIR)
	@echo "✓ Virtual environment removed" 