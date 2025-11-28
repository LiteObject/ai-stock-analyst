# Makefile for AI Stock Analyst Project
# Provides convenient commands for common development tasks

.PHONY: install install-dev test test-cov lint format clean run backtest ui help

# Python virtual environment
VENV := .venv
PYTHON := $(VENV)/Scripts/python
PIP := $(VENV)/Scripts/pip

# Default target
help:
	@echo "AI Stock Analyst - Available Commands"
	@echo "==================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run all tests"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint         - Run linting checks"
	@echo "  make format       - Format code with black and isort"
	@echo ""
	@echo "Running:"
	@echo "  make run TICKER=AAPL  - Run hedge fund for a ticker"
	@echo "  make backtest TICKER=AAPL  - Run backtest for a ticker"
	@echo "  make ui               - Launch Streamlit web interface"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        - Remove cache and temporary files"

# Install production dependencies
install:
	$(PIP) install -r requirements.txt

# Install development dependencies
install-dev: install
	$(PIP) install pytest pytest-cov black isort flake8 mypy pydantic-settings

# Run tests
test:
	$(PYTHON) -m pytest tests/ -v

# Run tests with coverage
test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Run linting (matches CI checks)
lint:
	$(PYTHON) -m black --check --line-length=120 src/ tests/ app/
	$(PYTHON) -m isort --check-only --profile=black src/ tests/ app/
	$(PYTHON) -m flake8 src/ tests/ app/ --max-line-length=120 --ignore=E501,W503,E203
	$(PYTHON) -m mypy src/ --ignore-missing-imports --no-error-summary

# Format code (auto-fix formatting issues)
format:
	$(PYTHON) -m black src/ tests/ app/ --line-length=120
	$(PYTHON) -m isort src/ tests/ app/ --profile=black

# Clean cache and temporary files
clean:
	@echo "Cleaning cache and temporary files..."
	@if exist "__pycache__" rmdir /s /q __pycache__
	@if exist ".pytest_cache" rmdir /s /q .pytest_cache
	@if exist "htmlcov" rmdir /s /q htmlcov
	@if exist ".mypy_cache" rmdir /s /q .mypy_cache
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	@echo "Done."

# Run the hedge fund
run:
ifndef TICKER
	@echo "Error: TICKER is required. Usage: make run TICKER=AAPL"
else
	$(PYTHON) src/main.py --ticker $(TICKER)
endif

# Run backtest
backtest:
ifndef TICKER
	@echo "Error: TICKER is required. Usage: make backtest TICKER=AAPL"
else
	$(PYTHON) src/backtester.py --ticker $(TICKER)
endif

# Launch Streamlit UI
ui:
	$(PYTHON) -m streamlit run app/streamlit_app.py
