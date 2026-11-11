# Every target is a thin wrapper around a `uv run python -m ...` command, so the same
# commands work without make (e.g. on Windows): copy the recipe line.
DATA_CONFIG ?= configs/data/swiss_dwellings.yaml

.PHONY: data viz fixtures test lint format

data:  ## raw dataset -> unified plans, filters, de-duplication, frozen split
	uv run python -m housegen.data.build --config $(DATA_CONFIG)

viz:  ## data report: reports/data_report.md + reports/figures/data/
	uv run python -m housegen.data.report --config $(DATA_CONFIG)

fixtures:  ## refresh tests/fixtures/plans.jsonl from the built dataset
	uv run python tests/fixtures/make_fixtures.py

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .
