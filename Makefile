VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: check-python install install-dev smoke-test clean

check-python:
	@$(PYTHON) -c "import sys; assert sys.version_info[:2] == (3, 11), f'Python 3.11 required, got {sys.version}'"

install: check-python
	$(PIP) install --upgrade pip
	$(PIP) install -e .

install-dev: check-python
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev,full]

smoke-test: check-python
	$(PYTHON) -m pytest -q

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
