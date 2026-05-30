# tt_symbiote — developer convenience targets.
# All real configuration lives in pyproject.toml and .pre-commit-config.yaml.

.PHONY: install test lint smoke help

help:
	@echo "make install        # editable install with dev extras"
	@echo "make lint           # run pre-commit on all files"
	@echo "make test           # run capability tests (no device required if synthetic-only)"
	@echo "make smoke MODEL=x  # run smoke test for tests/capabilities/<MODEL>/ (e.g. MODEL=bailing_moe_v2)"

install:
	pip install -e ".[dev]"

lint:
	pre-commit run --all-files

test:
	pytest tests/capabilities -q

smoke:
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make smoke MODEL=<model_name>"; \
		echo "Example: make smoke MODEL=bailing_moe_v2"; \
		exit 1; \
	fi
	pytest tests/capabilities/$(MODEL) -q -s
