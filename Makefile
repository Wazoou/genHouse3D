# Every target is a thin wrapper around a `uv run python -m ...` command, so the same
# commands work without make (e.g. on Windows): copy the recipe line.
DATA_CONFIG ?= configs/data/swiss_dwellings.yaml

.PHONY: data viz fixtures build3d render test lint format

data:  ## raw dataset -> unified plans, filters, de-duplication, frozen split
	uv run python -m housegen.data.build --config $(DATA_CONFIG)

viz:  ## data report: reports/data_report.md + reports/figures/data/
	uv run python -m housegen.data.report --config $(DATA_CONFIG)

fixtures:  ## refresh tests/fixtures/plans.jsonl from the built dataset
	uv run python tests/fixtures/make_fixtures.py

build3d:  ## real plans -> GLB houses in outputs/, validation and roof statistics
	uv run python -m housegen.geometry3d.cli --config $(DATA_CONFIG)

render:  ## gallery figures in reports/figures/3d/
	uv run python -m housegen.render --config $(DATA_CONFIG)

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .
