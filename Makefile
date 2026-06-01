# tt_symbiote — developer convenience targets.
# All real configuration lives in pyproject.toml and .pre-commit-config.yaml.

.PHONY: install install-ttnn test lint smoke dist help

help:
	@echo "make install        # editable install with dev extras"
	@echo "make install-ttnn   # editable install with dev + ttnn extras"
	@echo "make lint           # run pre-commit on all files"
	@echo "make test           # run hardware-free tests (no device required)"
	@echo "make smoke MODEL=x  # run hardware smoke for tests/models/<MODEL>/ (e.g. MODEL=bailing_moe_v2)"
	@echo "make dist           # build sdist + wheel into dist/ (pre-tag gate)"

install:
	pip install -e ".[dev]"

install-ttnn:
	pip install -e ".[dev,ttnn]"

lint:
	pre-commit run --all-files

test:
	pytest tests/auto -q

smoke:
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make smoke MODEL=<model_name>"; \
		echo "Example: make smoke MODEL=bailing_moe_v2"; \
		exit 1; \
	fi
	pytest tests/models/$(MODEL) -q -s

dist:
	rm -rf dist/ build/ src/*.egg-info
	python -m build
	twine check dist/*
